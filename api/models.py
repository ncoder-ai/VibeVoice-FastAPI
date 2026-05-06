"""Pydantic models for API request/response schemas."""

from typing import Optional, Literal, List
from pydantic import BaseModel, Field, validator


# ============================================================
# OpenAI-Compatible TTS Models
# ============================================================

class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible TTS request schema."""
    
    model: str = Field(
        default="tts-1",
        description="Model to use: tts-1 or tts-1-hd (both map to VibeVoice)"
    )
    input: str = Field(
        ...,
        description="The text to generate audio for",
        max_length=4096
    )
    voice: str = Field(
        ...,
        description="Voice to use: OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer) or any VibeVoice preset name"
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "m4a"]] = Field(
        default="mp3",
        description="Audio format for the response"
    )
    speed: Optional[float] = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed of generated audio (0.25 to 4.0)"
    )
    
    # Note: Voice validation removed to allow any VibeVoice preset name
    # Validation happens in the endpoint with proper error messages


# ============================================================
# VibeVoice-Specific Models
# ============================================================

class SpeakerConfig(BaseModel):
    """Configuration for a single speaker."""
    
    speaker_id: int = Field(
        ...,
        ge=0,
        le=3,
        description="Speaker ID (0-3)"
    )
    voice_preset: Optional[str] = Field(
        default=None,
        description="Name of voice preset to use (from voices directory)"
    )
    voice_sample_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded audio sample for voice cloning"
    )
    
    @validator("voice_preset", "voice_sample_base64")
    def validate_voice_source(cls, v, values):
        """Ensure at least one voice source is provided."""
        if "voice_preset" in values and not values.get("voice_preset") and not v:
            raise ValueError("Either voice_preset or voice_sample_base64 must be provided")
        return v


class VibeVoiceGenerateRequest(BaseModel):
    """VibeVoice-specific generation request with multi-speaker support."""
    
    script: str = Field(
        ...,
        description="Multi-speaker script with format 'Speaker 0: text\\nSpeaker 1: text'",
        max_length=100000
    )
    speakers: List[SpeakerConfig] = Field(
        ...,
        min_items=1,
        max_items=4,
        description="Speaker configurations (1-4 speakers)"
    )
    cfg_scale: Optional[float] = Field(
        default=1.3,
        ge=1.0,
        le=2.0,
        description="Classifier-free guidance scale (1.0-2.0)"
    )
    inference_steps: Optional[int] = Field(
        default=None,
        ge=5,
        le=50,
        description=(
            "Number of diffusion inference steps. When None (default), the "
            "server-side VIBEVOICE_INFERENCE_STEPS env setting is used. "
            "Pass an explicit value (5-50) to override per-request."
        )
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "m4a"]] = Field(
        default="mp3",
        description="Audio format for the response"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Enable real-time streaming via Server-Sent Events"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    @validator("speakers")
    def validate_speaker_ids(cls, v):
        """Ensure speaker IDs are sequential starting from 0."""
        speaker_ids = sorted([s.speaker_id for s in v])
        expected_ids = list(range(len(v)))
        if speaker_ids != expected_ids:
            raise ValueError(f"Speaker IDs must be sequential starting from 0. Got: {speaker_ids}")
        return v


class VibeVoiceGenerateResponse(BaseModel):
    """Response for VibeVoice generation."""
    
    audio_url: Optional[str] = Field(
        default=None,
        description="URL to download the generated audio (for non-streaming)"
    )
    duration: Optional[float] = Field(
        default=None,
        description="Duration of generated audio in seconds"
    )
    format: str = Field(
        ...,
        description="Audio format of the response"
    )
    sample_rate: int = Field(
        default=24000,
        description="Sample rate of the audio"
    )


# ============================================================
# Common Models
# ============================================================

class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: dict = Field(
        ...,
        description="Error details"
    )
    
    @staticmethod
    def from_exception(exc: Exception, status_code: int = 500) -> "ErrorResponse":
        """Create error response from exception."""
        return ErrorResponse(
            error={
                "message": str(exc),
                "type": type(exc).__name__,
                "code": status_code
            }
        )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        default="healthy",
        description="Service health status"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    device: str = Field(
        ...,
        description="Device being used for inference"
    )
    model_path: str = Field(
        ...,
        description="Path to the loaded model"
    )


class VoiceListResponse(BaseModel):
    """Response listing available voices."""
    
    voices: List[dict] = Field(
        ...,
        description="List of available voice presets"
    )

