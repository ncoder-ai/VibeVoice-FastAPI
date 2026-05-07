# VibeVoice FastAPI Server

A production-ready FastAPI server that exposes the VibeVoice TTS model as an OpenAI-compatible API, with Docker support and comprehensive voice management.

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange?logo=openai)](https://platform.openai.com/docs/api-reference/audio)

</div>

## 🚀 Features

- **Docker-First Deployment**: Production-ready Docker setup with GPU support (recommended)
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's TTS API (`/v1/audio/speech`)
- **Unlimited Custom Voices**: Automatically load any voice from a directory - just drop audio files and restart
- **Multi-Format Support**: MP3, WAV, FLAC, AAC, M4A, Opus, PCM
- **Streaming Support**: Real-time audio streaming for long-form content
- **Voice Management**: Dynamic voice loading, OpenAI voice mapping, and custom voice presets
- **AWQ-INT4 Quantization**: Drop-in [`ncoder-ai/VibeVoice-Large-AWQ`](https://huggingface.co/ncoder-ai/VibeVoice-Large-AWQ) — **8.4 GB VRAM** at **RTF ~0.70**
  on RTX 3090, faster + smaller than bnb-Q8 with no audible quality loss. See
  [AWQ quickstart](#-awq-quickstart-recommended-for-single-3090) below.
- **Production Ready**: Health checks, error handling, CORS support, and comprehensive logging

## ⚡ AWQ Quickstart (recommended for single 3090)

For best speed-to-VRAM ratio on a single 24 GB GPU, just point at the unified AWQ model:

```bash
# In your .env (or docker-env file):
VIBEVOICE_MODEL_PATH=ncoder-ai/VibeVoice-Large-AWQ
VIBEVOICE_INFERENCE_STEPS=7
TORCH_COMPILE=true
```

That's it — no extra env vars. Transformers reads `quantization_config` from the
checkpoint and wires AWQ kernels automatically. Start with `docker compose up -d --build`.
First request downloads the model (~9 GB); subsequent runs reuse the cache.

**Benchmarks** (RTX 3090, 277-char prompt, 7 steps):

| Setup | VRAM | Generation Time | RTF |
|---|---:|---:|---:|
| `rsxdalv/VibeVoice-Large` (FP16) | 17.4 GB | ~9 s for 16 s audio | 0.54 |
| `FabioSarracino/VibeVoice-Large-Q8` (bnb-Q8) | 10.8 GB | ~20 s for 16 s audio | 1.22 |
| **`ncoder-ai/VibeVoice-Large-AWQ`** | **8.4 GB** | **~11 s for 16 s audio** | **0.70** |

The unified model has its Qwen2 LLM quantized to INT4 with AWQ + Marlin GEMM kernels.
The audio tokenizer + diffusion head stay FP16 inside the same checkpoint, so audio
quality is indistinguishable from FP16 at INFERENCE_STEPS=7.

> **Legacy 2-step path:** the LLM-only [`ncoder-ai/VibeVoice-Large-AWQ-INT4`](https://huggingface.co/ncoder-ai/VibeVoice-Large-AWQ-INT4) repo plus
> `VIBEVOICE_QUANTIZATION=awq` + `VIBEVOICE_AWQ_LLM_PATH=…` still works for backwards
> compatibility. Prefer the single-model path above.

## 📋 Quick Start

### One-Click Install (Recommended)

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/ncoder-ai/VibeVoice-FastAPI/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ncoder-ai/VibeVoice-FastAPI/main/install.bat" -OutFile "install.bat"; .\install.bat
```

This clones the repo, detects your GPU, walks you through setup, and starts the server automatically.

Or if you already have the repo cloned:
```bash
python3 install.py
```

### Docker Deployment (Manual)

If you prefer to set things up manually with Docker:

```bash
# Clone the repository
git clone https://github.com/ncoder-ai/VibeVoice-FastAPI.git
cd VibeVoice-FastAPI

# Copy and configure environment
cp docker-env.example .env
# Edit .env - set VOICES_DIR to your voice files path

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

The API will be available at `http://localhost:8001`

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for detailed Docker instructions.

### Local Installation (Linux/macOS)

For development or if you prefer bare-metal installation:

```bash
# Clone the repository
git clone https://github.com/ncoder-ai/VibeVoice-FastAPI.git
cd VibeVoice-FastAPI

# Run setup script
./setup.sh

# Configure environment
cp env.example .env
# Edit .env with your settings

# Start server
./start.sh
```

### Local Installation (Windows)

Windows baremetal installation requires manual setup (the `.sh` scripts are Linux/macOS only):

```powershell
# Clone the repository
git clone https://github.com/ncoder-ai/VibeVoice-FastAPI.git
cd VibeVoice-FastAPI

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install VibeVoice and API dependencies
pip install -e .
pip install -r requirements-api.txt

# Configure environment
copy env.example .env
# Edit .env with your settings (use notepad, vscode, etc.)

# Start server
start.bat
```

> **Note:** Flash-attention does not have pre-built Windows wheels. The API will automatically fall back to SDPA attention, which works well. Also ensure [ffmpeg](https://ffmpeg.org/download.html) is installed and on your PATH for audio format conversion.

## 📖 Documentation

- **[API README](API_README.md)** - Complete API documentation with examples, voice management, and troubleshooting
- **[Docker Quickstart](DOCKER_QUICKSTART.md)** - Docker deployment quickstart guide

## 🎯 API Endpoints

### OpenAI-Compatible Endpoints

- `POST /v1/audio/speech` - Generate speech from text (OpenAI-compatible)
- `GET /v1/audio/voices` - List all available voices

### VibeVoice-Specific Endpoints

- `POST /v1/vibevoice/generate` - Advanced generation with multi-speaker support
- `GET /v1/vibevoice/voices` - List all voices with detailed info
- `GET /v1/vibevoice/health` - Detailed health check

### Example: Generate Speech

```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the VibeVoice API",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Example: List Voices

```bash
# List all available voices
curl http://localhost:8001/v1/audio/voices

# List with OpenAI format
curl http://localhost:8001/v1/audio/voices | jq
```
## MODEL MANAGEMENT
VibeVoice Large (AWQ-INT4, recommended): Huggingface: ncoder-ai/VibeVoice-Large-AWQ [https://huggingface.co/ncoder-ai/VibeVoice-Large-AWQ]
VibeVoice Large (FP16): Huggingface: rsxdalv/VibeVoice-Large [https://huggingface.co/rsxdalv/VibeVoice-Large]
VibeVoice Large (AWQ-INT4 LLM only, advanced): Huggingface: ncoder-ai/VibeVoice-Large-AWQ-INT4 [https://huggingface.co/ncoder-ai/VibeVoice-Large-AWQ-INT4]


VibeVoice 1.5B: Huggingface microsoft/VibeVoice-1.5B [https://huggingface.co/microsoft/VibeVoice-1.5B]
## 🎤 Voice Management

### Using OpenAI-Compatible Voices

The API includes 6 OpenAI-compatible voice mappings:
- `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### Using Custom Voices

Simply place audio files (`.wav`, `.mp3`, `.flac`, `.m4a`, etc.) in your `VOICES_DIR` and restart the server. All files are automatically loaded as voice presets!

```bash
# Add a custom voice
cp my_voice.wav /path/to/voices/custom_voice.wav
# Restart server - voice is now available!
```

### Direct Voice Usage

You can use any voice name directly in API requests:

```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Testing custom voice",
    "voice": "custom_voice",
    "response_format": "wav"
  }'
```

## ⚙️ Configuration

Key environment variables (see `env.example` for full list):

```bash
# Model Configuration
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B  # or local path
VIBEVOICE_DEVICE=cuda                           # cuda, cpu, or mps
VIBEVOICE_INFERENCE_STEPS=10                    # 5-50, higher = better quality

# Voice Configuration
VOICES_DIR=demo/voices                           # Directory with voice files

# API Configuration
API_PORT=8001
API_CORS_ORIGINS=*

# Performance Optimization
TORCH_COMPILE=true                               # 20-50% speedup (slower first request)
TORCH_COMPILE_MODE=max-autotune                  # default, reduce-overhead, or max-autotune
# VIBEVOICE_QUANTIZATION=int8_torchao            # Reduce VRAM ~40%

# Generation Defaults
DEFAULT_CFG_SCALE=1.8                            # 1.0-3.0
DEFAULT_RESPONSE_FORMAT=mp3
```

## 🐳 Docker Deployment

Docker is the **recommended and preferred** deployment method. It provides:
- ✅ Consistent environment across all systems
- ✅ No dependency conflicts
- ✅ Easy GPU configuration
- ✅ Production-ready setup
- ✅ Simplified updates and maintenance

### Requirements

- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with 8GB+ VRAM (for 1.5B model) or 16GB+ (for Large model)

### Quick Start

```bash
# Copy and configure environment
cp docker-env.example .env
# Edit .env - set VOICES_DIR to your voice files path

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for complete Docker deployment guide.

## 🔧 Development

### Setup Development Environment

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install -r requirements-api.txt

# Install PyTorch (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Optional: Install flash-attn for faster inference
# See setup.sh for pre-built wheel installation
```

### Running Tests

```bash
# Start server
./start.sh

# Test API
curl http://localhost:8001/health
curl http://localhost:8001/v1/audio/voices
```

## 📊 Supported Models

| Model | Size | Context | Max Length | VRAM Required |
|-------|------|---------|------------|---------------|
| VibeVoice-1.5B | 1.5B | 64K | ~90 min | 8GB+ |
| VibeVoice-Large | 7B | 32K | ~45 min | 16GB+ |

Models are automatically downloaded from HuggingFace on first use.

## 🛠️ System Requirements

**For Docker Deployment (Recommended):**
- **Docker**: Docker and Docker Compose installed
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for 1.5B model) or 16GB+ (for Large model)
- **NVIDIA Container Toolkit**: Required for GPU support
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB minimum, 50GB recommended (with model cache)
- **OS**: Linux (recommended), macOS, or Windows (with Docker Desktop + WSL2)

**For Local Installation:**
- **Python**: 3.12
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB minimum, 50GB recommended
- **OS**: Linux, macOS, or Windows
- **ffmpeg**: Required for audio format conversion ([download](https://ffmpeg.org/download.html))

## 🔐 Security Notes

- The API does not include authentication by default. For production use, add authentication middleware or deploy behind a reverse proxy with authentication.
- Voice files are loaded from the configured directory - ensure proper file permissions.
- Model weights are downloaded from HuggingFace - verify model integrity in production.

## 📝 License

This project maintains the original VibeVoice model codebase. Please refer to the original VibeVoice license for model usage terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- **VibeVoice Team** at Microsoft for the original model
- **[shijincai](https://github.com/shijincai)** for maintaining a backup of the original VibeVoice codebase
- **FastAPI** for the excellent web framework
- **HuggingFace** for model hosting and transformers library

## 📚 Additional Resources

- [VibeVoice Original Paper](https://arxiv.org/pdf/2508.19205)
- [VibeVoice HuggingFace Collection](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

## ⚠️ Limitations

- **Language Support**: Primarily English and Chinese. Other languages may produce unexpected results.
- **Non-Speech Audio**: The model focuses on speech synthesis and may generate background music or sounds spontaneously.
- **Commercial Use**: This model is intended for research and development. Test thoroughly before production use.

## 🆘 Troubleshooting

### Server won't start
- Check GPU availability: `nvidia-smi`
- Verify Python version: `python3 --version` (should be 3.12)
- Check dependencies: `pip list | grep torch`

### Out of memory errors
- Use smaller model: `VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B`
- Reduce inference steps: `VIBEVOICE_INFERENCE_STEPS=5`
- Use CPU mode: `VIBEVOICE_DEVICE=cpu` (much slower)

### Voice not found errors
- Verify `VOICES_DIR` path in `.env`
- Check file permissions
- Ensure audio files are in supported formats

For more help, see the [API README](API_README.md) or open an issue.

---

**Made with ❤️ for the VibeVoice community**
