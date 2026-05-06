"""OpenAI-compatible TTS endpoint."""

import asyncio
import logging
import threading
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import Response, StreamingResponse

from api.models import OpenAITTSRequest, ErrorResponse
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.utils.audio_utils import audio_to_bytes, get_content_type, get_audio_duration
from api.utils.streaming import create_streaming_response
from api.config import settings

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/v1/audio", tags=["OpenAI Compatible"])

# Global service instances (initialized in main.py)
tts_service: TTSService = None
voice_manager: VoiceManager = None


def get_tts_service() -> TTSService:
    """Dependency to get TTS service."""
    if tts_service is None or not tts_service.is_loaded:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    return tts_service


def get_voice_manager() -> VoiceManager:
    """Dependency to get voice manager."""
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    return voice_manager


@router.post("/speech")
async def create_speech(
    body: OpenAITTSRequest,
    request: Request,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    Generate speech from text using OpenAI-compatible API.

    Cooperative cancellation: generation runs in a background worker thread
    so the request handler can poll request.is_disconnected(). On client
    disconnect we set a cancel_event which trips VibeVoice's stop_check_fn
    at the next outer-loop step (~100-500ms), the lock releases, GPU is
    freed, and the next request can proceed immediately. Returns HTTP 499
    (client closed request) on cancel — nginx convention.
    """
    try:
        # Try loading as OpenAI voice first, then as direct VibeVoice preset
        voice_audio = voices.load_voice_audio(body.voice, is_openai_voice=True)

        if voice_audio is None:
            voice_audio = voices.load_voice_audio(body.voice, is_openai_voice=False)

        if voice_audio is None:
            available_openai = ', '.join(voices.OPENAI_VOICE_MAPPING.keys())
            available_presets = ', '.join(sorted(voices.voice_presets.keys()))
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{body.voice}' not found. OpenAI voices: {available_openai}. VibeVoice presets: {available_presets}"
            )

        formatted_script = tts.format_script_for_single_speaker(body.input, speaker_id=0)

        cancel_event = threading.Event()
        result_holder: dict = {}

        def _run_generation():
            try:
                result_holder['audio'] = tts.generate_speech(
                    text=formatted_script,
                    voice_samples=[voice_audio],
                    cfg_scale=settings.default_cfg_scale,
                    stream=False,
                    cancel_event=cancel_event,
                )
            except Exception as e:
                result_holder['error'] = e

        start_time = time.time()
        gen_thread = threading.Thread(target=_run_generation, daemon=True)
        gen_thread.start()

        # Poll for client disconnect while generation runs.
        while gen_thread.is_alive():
            await asyncio.sleep(0.1)
            if await request.is_disconnected():
                logger.info("OpenAI TTS client disconnected — cancelling generation")
                cancel_event.set()
                # 5s is plenty: stop_check_fn fires within one outer step
                gen_thread.join(timeout=5.0)
                if gen_thread.is_alive():
                    logger.warning("Generation thread didn't exit within 5s after cancel")
                return Response(status_code=499)

        generation_time = time.time() - start_time

        if 'error' in result_holder:
            raise result_holder['error']

        audio = result_holder.get('audio')
        if audio is None:
            return Response(status_code=499)

        audio_duration = get_audio_duration(audio, sample_rate=24000)

        text_preview = body.input[:100] + "..." if len(body.input) > 100 else body.input
        logger.info(
            f"Generated speech - Text: {text_preview} | Voice: {body.voice} | "
            f"Model: {body.model} ({settings.vibevoice_model_path}) | "
            f"CFG: {settings.default_cfg_scale} | Audio Duration: {audio_duration:.2f}s | Generation Time: {generation_time:.2f}s"
        )

        audio_bytes = audio_to_bytes(
            audio,
            sample_rate=24000,
            format=body.response_format
        )

        return Response(
            content=audio_bytes,
            media_type=get_content_type(body.response_format),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{body.response_format}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices")
async def list_voices(
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    List all available voices in OpenAI-compatible format.
    """
    try:
        voice_list = []

        for openai_name, vibevoice_preset in voices.OPENAI_VOICE_MAPPING.items():
            if vibevoice_preset in voices.voice_presets:
                voice_list.append({
                    "id": openai_name,
                    "object": "voice",
                    "name": openai_name
                })

        all_voices = voices.list_available_voices()
        for voice in all_voices:
            if voice["name"] not in voices.OPENAI_VOICE_MAPPING.values():
                voice_list.append({
                    "id": voice["name"],
                    "object": "voice",
                    "name": voice["name"]
                })

        return {
            "object": "list",
            "data": voice_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
