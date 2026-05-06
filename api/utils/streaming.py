"""Streaming utilities for real-time audio delivery.

Includes cooperative cancellation: when the FastAPI Request reports
disconnected (client closed the TCP connection), we set the cancel_event,
which the underlying tts_service uses to halt model.generate() and free
GPU resources via VibeVoice's `stop_check_fn` parameter.
"""

import asyncio
import json
import logging
import threading
from typing import AsyncIterator, Iterator, Optional, Union
from fastapi import Request
from fastapi.responses import StreamingResponse
import numpy as np
import torch

logger = logging.getLogger(__name__)

_SENTINEL = object()


async def _watch_for_disconnect(request: Request, cancel_event: threading.Event):
    """Background task that polls request.is_disconnected() and sets the
    cancel event when the client goes away."""
    try:
        while not cancel_event.is_set():
            await asyncio.sleep(0.1)
            if await request.is_disconnected():
                logger.info("Client disconnected — signalling cancellation")
                cancel_event.set()
                return
    except asyncio.CancelledError:
        # Normal shutdown when the response generator finishes
        raise


async def _aiter_sync(audio_stream: Iterator, cancel_event: threading.Event):
    """Pull from a sync iterator without blocking the event loop. Stops
    promptly when the cancel event fires."""
    loop = asyncio.get_running_loop()
    iterator = iter(audio_stream)
    while True:
        if cancel_event.is_set():
            return
        chunk = await loop.run_in_executor(None, next, iterator, _SENTINEL)
        if chunk is _SENTINEL:
            return
        yield chunk


async def audio_chunk_generator(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000,
    cancel_event: Optional[threading.Event] = None,
    request: Optional[Request] = None,
) -> AsyncIterator[bytes]:
    """
    Generate audio chunks for streaming response.

    Args:
        audio_stream: Iterator yielding audio chunks
        format: Audio format for encoding
        sample_rate: Sample rate of audio
        cancel_event: Threading event used to signal cancellation to the
            underlying generator (model.generate() exits, GPU is freed).
        request: FastAPI request — when provided we poll is_disconnected()
            in a background task and trip the cancel_event when the client
            closes the connection.

    Yields:
        Encoded audio chunk bytes
    """
    from api.utils.audio_utils import audio_to_bytes

    if cancel_event is None:
        cancel_event = threading.Event()

    disconnect_task: Optional[asyncio.Task] = None
    if request is not None:
        disconnect_task = asyncio.create_task(_watch_for_disconnect(request, cancel_event))

    try:
        async for chunk in _aiter_sync(audio_stream, cancel_event):
            chunk_bytes = audio_to_bytes(chunk, sample_rate=sample_rate, format=format)
            yield chunk_bytes
    finally:
        # Always trip the event so the producer thread exits.
        cancel_event.set()
        if disconnect_task is not None and not disconnect_task.done():
            disconnect_task.cancel()
            try:
                await disconnect_task
            except (asyncio.CancelledError, Exception):
                pass


async def sse_audio_generator(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000,
    cancel_event: Optional[threading.Event] = None,
    request: Optional[Request] = None,
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events for audio streaming. Same cancellation
    semantics as audio_chunk_generator.
    """
    from api.utils.audio_utils import audio_to_bytes
    import base64

    if cancel_event is None:
        cancel_event = threading.Event()

    disconnect_task: Optional[asyncio.Task] = None
    if request is not None:
        disconnect_task = asyncio.create_task(_watch_for_disconnect(request, cancel_event))

    chunk_id = 0

    try:
        async for chunk in _aiter_sync(audio_stream, cancel_event):
            chunk_bytes = audio_to_bytes(chunk, sample_rate=sample_rate, format=format)
            chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')
            event_data = {
                "chunk_id": chunk_id,
                "audio": chunk_base64,
                "format": format,
                "sample_rate": sample_rate,
            }
            yield f"data: {json.dumps(event_data)}\n\n"
            chunk_id += 1

        # Send completion event only if not cancelled
        if not cancel_event.is_set():
            yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        error_data = {"error": str(e), "type": type(e).__name__}
        yield f"data: {json.dumps(error_data)}\n\n"
    finally:
        cancel_event.set()
        if disconnect_task is not None and not disconnect_task.done():
            disconnect_task.cancel()
            try:
                await disconnect_task
            except (asyncio.CancelledError, Exception):
                pass


def create_streaming_response(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000,
    use_sse: bool = False,
    cancel_event: Optional[threading.Event] = None,
    request: Optional[Request] = None,
) -> StreamingResponse:
    """
    Create a FastAPI StreamingResponse for audio.

    Args:
        audio_stream: Iterator yielding audio chunks
        format: Audio format
        sample_rate: Sample rate
        use_sse: Whether to use Server-Sent Events format
        cancel_event: Threading event for cooperative cancellation. Passed
            through to the inner generator. Routes should also pass this
            same event into tts_service.generate_speech() so the model
            stops on disconnect.
        request: FastAPI Request, used to poll is_disconnected() and trip
            the cancel_event automatically.

    Returns:
        FastAPI StreamingResponse
    """
    from api.utils.audio_utils import get_content_type

    if use_sse:
        return StreamingResponse(
            sse_audio_generator(
                audio_stream, format, sample_rate,
                cancel_event=cancel_event, request=request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return StreamingResponse(
            audio_chunk_generator(
                audio_stream, format, sample_rate,
                cancel_event=cancel_event, request=request,
            ),
            media_type=get_content_type(format),
            headers={
                "Transfer-Encoding": "chunked",
                "Cache-Control": "no-cache",
            },
        )
