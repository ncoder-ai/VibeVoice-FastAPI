"""VibeVoice-specific TTS endpoints with multi-speaker support."""

import asyncio
import base64
import logging
import threading
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import Response

from api.models import (
    VibeVoiceGenerateRequest,
    VibeVoiceGenerateResponse,
    VoiceListResponse,
    HealthResponse
)
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.utils.audio_utils import audio_to_bytes, get_audio_duration
from api.utils.streaming import create_streaming_response
from api.config import settings

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/v1/vibevoice", tags=["VibeVoice Extended"])

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


@router.post("/generate")
async def generate_speech(
    body: VibeVoiceGenerateRequest,
    request: Request,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    Generate multi-speaker speech with VibeVoice-specific features.

    Supports:
    - Multi-speaker dialogue (up to 4 speakers)
    - Custom voice samples via base64 or presets
    - CFG scale control
    - Inference step control
    - Real-time streaming via SSE
    - Cooperative cancellation on client disconnect via stop_check_fn
    """
    try:
        # Load voice samples for each speaker
        voice_samples = []

        for speaker_config in sorted(body.speakers, key=lambda s: s.speaker_id):
            if speaker_config.voice_sample_base64:
                try:
                    audio_bytes = base64.b64decode(speaker_config.voice_sample_base64)
                    import io
                    import soundfile as sf
                    audio_data, sr = sf.read(io.BytesIO(audio_bytes))

                    if sr != 24000:
                        import librosa
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)

                    if len(audio_data.shape) > 1:
                        import numpy as np
                        audio_data = np.mean(audio_data, axis=1)

                    voice_samples.append(audio_data.astype('float32'))

                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to decode voice sample for speaker {speaker_config.speaker_id}: {str(e)}"
                    )

            elif speaker_config.voice_preset:
                audio_data = voices.load_voice_audio(speaker_config.voice_preset, is_openai_voice=False)

                if audio_data is None:
                    available_voices = [v["name"] for v in voices.list_available_voices()]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Voice preset '{speaker_config.voice_preset}' not found. Available: {', '.join(available_voices)}"
                    )

                voice_samples.append(audio_data)

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Speaker {speaker_config.speaker_id} must have either voice_preset or voice_sample_base64"
                )

        voice_list = []
        for speaker_config in sorted(body.speakers, key=lambda s: s.speaker_id):
            if speaker_config.voice_preset:
                voice_list.append(f"speaker{speaker_config.speaker_id}={speaker_config.voice_preset}")
            else:
                voice_list.append(f"speaker{speaker_config.speaker_id}=base64_audio")
        voices_str = ", ".join(voice_list)

        actual_inference_steps = body.inference_steps if body.inference_steps is not None else settings.vibevoice_inference_steps

        if body.stream:
            # Streaming path: cancel_event is wired through tts_service into
            # model.generate() (stop_check_fn) AND into create_streaming_response
            # which polls request.is_disconnected() and trips the event when
            # the client closes the connection. Result: GPU memory is freed
            # within one outer-step on disconnect, lock released, next request
            # ready immediately.
            cancel_event = threading.Event()

            text_preview = body.script[:100] + "..." if len(body.script) > 100 else body.script
            logger.info(
                f"Generating speech (streaming) - Text: {text_preview} | Voices: {voices_str} | "
                f"Model: {settings.vibevoice_model_path} | CFG: {body.cfg_scale} | "
                f"Steps: {actual_inference_steps} | Seed: {body.seed if body.seed is not None else 'None'}"
            )

            audio_stream = tts.generate_speech(
                text=body.script,
                voice_samples=voice_samples,
                cfg_scale=body.cfg_scale,
                inference_steps=body.inference_steps,
                seed=body.seed,
                stream=True,
                cancel_event=cancel_event,
            )

            return create_streaming_response(
                audio_stream,
                format=body.response_format,
                sample_rate=24000,
                use_sse=True,
                cancel_event=cancel_event,
                request=request,
            )

        else:
            # Non-streaming: run generation in a thread so we can poll for
            # client disconnect. Same cancellation pattern as OpenAI endpoint.
            cancel_event = threading.Event()
            result_holder: dict = {}

            def _run_generation():
                try:
                    result_holder['audio'] = tts.generate_speech(
                        text=body.script,
                        voice_samples=voice_samples,
                        cfg_scale=body.cfg_scale,
                        inference_steps=body.inference_steps,
                        seed=body.seed,
                        stream=False,
                        cancel_event=cancel_event,
                    )
                except Exception as e:
                    result_holder['error'] = e

            start_time = time.time()
            gen_thread = threading.Thread(target=_run_generation, daemon=True)
            gen_thread.start()

            while gen_thread.is_alive():
                await asyncio.sleep(0.1)
                if await request.is_disconnected():
                    logger.info("VibeVoice native client disconnected — cancelling generation")
                    cancel_event.set()
                    gen_thread.join(timeout=5.0)
                    return Response(status_code=499)

            generation_time = time.time() - start_time

            if 'error' in result_holder:
                raise result_holder['error']

            audio = result_holder.get('audio')
            if audio is None:
                return Response(status_code=499)

            audio_duration = get_audio_duration(audio, sample_rate=24000)

            text_preview = body.script[:100] + "..." if len(body.script) > 100 else body.script
            logger.info(
                f"Generated speech - Text: {text_preview} | Voices: {voices_str} | "
                f"Model: {settings.vibevoice_model_path} | CFG: {body.cfg_scale} | "
                f"Steps: {actual_inference_steps} | Seed: {body.seed if body.seed is not None else 'None'} | "
                f"Audio Duration: {audio_duration:.2f}s | Generation Time: {generation_time:.2f}s"
            )

            audio_bytes = audio_to_bytes(
                audio,
                sample_rate=24000,
                format=body.response_format
            )

            duration = audio_duration

            from api.utils.audio_utils import get_content_type
            return Response(
                content=audio_bytes,
                media_type=get_content_type(body.response_format),
                headers={
                    "Content-Disposition": f"attachment; filename=vibevoice_output.{body.response_format}",
                    "X-Audio-Duration": str(duration),
                    "X-Audio-Format": body.response_format,
                    "X-Audio-Sample-Rate": "24000"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices", response_model=VoiceListResponse)
async def list_vibevoice_voices(
    voices: VoiceManager = Depends(get_voice_manager)
):
    """List all available voices for VibeVoice."""
    try:
        all_voices = voices.list_available_voices()
        return VoiceListResponse(voices=all_voices, count=len(all_voices))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def vibevoice_health(
    tts: TTSService = Depends(get_tts_service)
):
    """Health check for the VibeVoice service."""
    return HealthResponse(
        status="healthy" if tts.is_loaded else "loading",
        model_loaded=tts.is_loaded,
        model_path=settings.vibevoice_model_path,
        device=tts.device or "unknown"
    )
