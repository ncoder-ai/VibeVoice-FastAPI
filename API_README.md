# VibeVoice FastAPI Documentation

Complete documentation for the VibeVoice FastAPI server with OpenAI-compatible endpoints.

## Table of Contents

- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Voice Management](#voice-management)
- [Models](#models)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

The easiest way to get started is the one-click installer:

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/ncoder-ai/VibeVoice-FastAPI/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ncoder-ai/VibeVoice-FastAPI/main/install.bat" -OutFile "install.bat"; .\install.bat
```

This clones the repo, detects your GPU, generates `.env`, and sets up everything automatically.

Alternatively, set up manually:

```bash
# Run setup script
./setup.sh

# Configure
cp env.example .env
# Edit .env as needed

# Start server
./start.sh
```

The API will be available at `http://localhost:8001`

### First API Call

```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello from VibeVoice!",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

## API Endpoints

### OpenAI-Compatible Endpoints

#### POST `/v1/audio/speech`

Generate speech from text (OpenAI-compatible).

**Request:**
```json
{
  "model": "tts-1",
  "input": "Your text here",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Response:** Audio file in requested format

**Example:**
```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello!","voice":"alloy","response_format":"mp3"}' \
  --output speech.mp3
```

#### GET `/v1/audio/voices`

List available voices.

**Query Parameters:**
- `show_all` (optional): If `true`, returns all voices including custom ones

**Example:**
```bash
# OpenAI voices only (6 voices)
curl http://localhost:8001/v1/audio/voices

# All voices including custom
curl 'http://localhost:8001/v1/audio/voices?show_all=true'
```

### VibeVoice Extended Endpoints

#### POST `/v1/vibevoice/generate`

Generate multi-speaker dialogue with advanced features.

**Request:**
```json
{
  "script": "Speaker 0: Welcome!\nSpeaker 1: Thanks!",
  "speakers": [
    {"speaker_id": 0, "voice_preset": "en-Alice_woman"},
    {"speaker_id": 1, "voice_preset": "en-Carter_man"}
  ],
  "cfg_scale": 1.3,
  "inference_steps": 10,
  "response_format": "mp3",
  "stream": false
}
```

#### GET `/v1/vibevoice/voices`

List all VibeVoice voice presets with detailed information.

#### GET `/v1/vibevoice/health`

Check service health and model status.

## Voice Management

### Built-in Voices

**OpenAI-Compatible Voices (6):**
- `alloy` → en-Alice_woman (English, Female)
- `echo` → en-Carter_man (English, Male)
- `fable` → en-Maya_woman (English, Female)
- `onyx` → en-Frank_man (English, Male)
- `nova` → en-Mary_woman_bgm (English, Female with BGM)
- `shimmer` → en-Alice_woman (English, Female)

**Additional Presets (3):**
- `in-Samuel_man` (Indian English, Male)
- `zh-Anchen_man_bgm` (Chinese, Male with BGM)
- `zh-Bowen_man` (Chinese, Male)

### Adding Custom Voices

The API **automatically discovers and loads all audio files** from your voices directory.

**Steps:**
1. Place audio files in `VOICES_DIR` (configured in `.env`)
2. Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`
3. Restart the server
4. Use the filename (without extension) as the voice name

**Example:**
```bash
# Add a custom voice
cp my-voice.wav demo/voices/custom-voice.wav

# Restart server
./start.sh

# Use it
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Testing custom voice",
    "voice": "custom-voice",
    "response_format": "mp3"
  }' \
  --output output.mp3
```

**Voice Sample Requirements:**
- Duration: 3-10 seconds recommended
- Quality: Clear speech, minimal background noise
- Format: Any supported audio format
- Sample rate: Automatically resampled to 24kHz

### Using Voices

**Method 1: OpenAI Voice Names**
```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello","voice":"alloy","response_format":"mp3"}'
```

**Method 2: Direct Preset Names**
```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello","voice":"in-Samuel_man","response_format":"mp3"}'
```

**Method 3: Custom Voice Names**
```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello","voice":"my-custom-voice","response_format":"mp3"}'
```

## Models

### Available Models

| Model | Size | Context | VRAM | RTF (RTX 3090) | Notes |
|-------|------|---------|------|---------------:|-------|
| **`ncoder-ai/VibeVoice-Large-AWQ`** | 7B (INT4) | 32K | **~8.4 GB** | **~0.70** | Recommended. AWQ-INT4 LLM + FP16 audio in one drop-in checkpoint. |
| `rsxdalv/VibeVoice-Large` | 7B | 32K | ~17 GB | ~0.54 | Full FP16, fastest on a 24 GB card. |
| `FabioSarracino/VibeVoice-Large-Q8` | 7B (INT8) | 32K | ~11 GB | ~1.22 | bnb-Q8. Slower than AWQ on Ampere; needs INFERENCE_STEPS=7+ to avoid audible noise. |
| `microsoft/VibeVoice-1.5B` | 1.5B | 64K | ~8 GB | n/a | Single-speaker only, no voice cloning, no multi-speaker dialogue. |
| `microsoft/VibeVoice-Realtime-0.5B` | 0.5B | — | ~4 GB | <0.3 | Single-speaker realtime variant. |

### Switching Models

Edit `.env`:
```bash
# Default — drop-in AWQ-INT4 (8.4 GB VRAM, RTF ~0.70)
VIBEVOICE_MODEL_PATH=ncoder-ai/VibeVoice-Large-AWQ

# Or full-precision Large
# VIBEVOICE_MODEL_PATH=rsxdalv/VibeVoice-Large

# Or a local path to any of the above (faster cold start than HF re-download)
# VIBEVOICE_MODEL_PATH=/path/to/your/model
```

Then restart the server.

## Configuration

### Environment Variables

Key settings in `.env`:

```bash
# Model Configuration
VIBEVOICE_MODEL_PATH=ncoder-ai/VibeVoice-Large-AWQ   # AWQ-INT4 drop-in, ~8.4 GB VRAM
VIBEVOICE_DEVICE=cuda                                # cuda, cpu, or mps
VIBEVOICE_INFERENCE_STEPS=7                          # 5-50; 7 is the sweet spot
VIBEVOICE_DTYPE=float16                              # match AWQ's internal compute path
VIBEVOICE_ATTN_IMPLEMENTATION=flash_attention_2      # falls back to sdpa if unavailable

# Voice Configuration
VOICES_DIR=demo/voices                               # Directory with voice files

# API Server
API_HOST=0.0.0.0
API_PORT=8001
API_CORS_ORIGINS=*

# Performance Optimization
TORCH_COMPILE=true                                   # ~40% speedup (first request slower)
TORCH_COMPILE_MODE=default                           # default, reduce-overhead, or max-autotune
TORCH_CACHE_DIR=~/.cache/torch_compile_vibevoice     # persist compile cache across restarts

# Generation Defaults
DEFAULT_CFG_SCALE=1.8                                # 1.0-3.0
DEFAULT_RESPONSE_FORMAT=mp3
DEFAULT_DO_SAMPLE=False
DEFAULT_TEMPERATURE=1.0
DEFAULT_TOP_P=1.0
DEFAULT_TOP_K=50
DEFAULT_REPETITION_PENALTY=1.0
```

> **Pre-quantized models** (like `ncoder-ai/VibeVoice-Large-AWQ` or `FabioSarracino/VibeVoice-Large-Q8`) embed their own `quantization_config` and are loaded directly by transformers — no extra `VIBEVOICE_QUANTIZATION` setting needed. `VIBEVOICE_QUANTIZATION` is only for runtime torchao quantization (`int8_torchao`, `int4_torchao`) on top of an FP16 base.

### Audio Formats

Supported output formats:
- **mp3** - MPEG Audio Layer III (default)
- **opus** - Opus codec (efficient for streaming)
- **aac** - Advanced Audio Coding
- **flac** - Free Lossless Audio Codec
- **wav** - Waveform Audio File Format
- **pcm** - Raw PCM audio data
- **m4a** - MPEG-4 Audio

## Python SDK

### Using OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"  # Not required for local server
)

# Generate speech
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",  # or any custom voice name
    input="Hello from VibeVoice!"
)

response.stream_to_file("speech.mp3")
```

### Using Requests

```python
import requests

response = requests.post(
    "http://localhost:8001/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello from VibeVoice!",
        "voice": "alloy",
        "response_format": "mp3"
    }
)

with open("speech.mp3", "wb") as f:
    f.write(response.content)
```

## Troubleshooting

### Voice Not Found

1. List all voices to see what's available:
   ```bash
   curl 'http://localhost:8001/v1/audio/voices?show_all=true'
   ```

2. Check voices directory:
   ```bash
   ls -la demo/voices/
   ```

3. Restart server to reload voices:
   ```bash
   ./start.sh
   ```

### Model Not Loading

1. Check model path in `.env`:
   ```bash
   cat .env | grep VIBEVOICE_MODEL_PATH
   ```

2. Verify model exists:
   ```bash
   ls -la ~/.cache/huggingface/hub/ | grep VibeVoice
   ```

3. Check server logs for "Model loaded successfully!" message

### CUDA Out of Memory

1. Switch to the AWQ-INT4 model (~8.4 GB VRAM, no quality loss):
   ```bash
   VIBEVOICE_MODEL_PATH=ncoder-ai/VibeVoice-Large-AWQ
   ```

2. Or fall back to the 1.5B single-speaker model (~8 GB VRAM):
   ```bash
   VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B
   ```

3. Reduce inference steps:
   ```bash
   VIBEVOICE_INFERENCE_STEPS=5
   ```

3. Use CPU (much slower):
   ```bash
   VIBEVOICE_DEVICE=cpu
   ```

### Flash Attention Installation Failed

The API automatically falls back to SDPA attention. For better performance:
```bash
source venv/bin/activate
pip install flash-attn --no-build-isolation
```

### Audio Format Issues

Ensure ffmpeg is installed:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html and add to PATH
```

### Server Won't Start

1. Check Python version:
   ```bash
   python3 --version   # Linux/macOS
   python --version     # Windows
   ```

2. Verify dependencies:
   ```bash
   pip list | grep torch
   ```

3. Check GPU availability:
   ```bash
   nvidia-smi
   ```

## Performance Tips

1. **Use CUDA with flash-attn** for best performance
2. **Adjust inference steps**: 5 (fast) to 20+ (high quality)
3. **Use streaming** for real-time applications
4. **Keep workers=1** when using GPU (model loading limitation)

## Interactive Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## Support

- **GitHub Issues**: https://github.com/ncoder-ai/VibeVoice-FastAPI/issues
- **Documentation**: See [README.md](README.md) and [DOCKER_GUIDE.md](DOCKER_GUIDE.md)
