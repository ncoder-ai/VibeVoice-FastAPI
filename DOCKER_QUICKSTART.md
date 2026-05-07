# Docker Quick Start Guide

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

**Windows users:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend enabled. GPU support requires the [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) driver.

## Quick Start

### Option A: One-Click Install (Easiest)

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/ncoder-ai/VibeVoice-FastAPI/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ncoder-ai/VibeVoice-FastAPI/main/install.bat" -OutFile "install.bat"; .\install.bat
```

Or if you already have the repo cloned: `python3 install.py`

The installer will auto-detect your GPU, walk you through configuration, generate `.env`, update `docker-compose.yml`, and start the container for you.

### Option B: Manual Setup

#### 1. Setup Environment

```bash
# Copy environment file
cp docker-env.example .env

# Edit .env - set your voice directory path
nano .env
```

**Required:** Set `VOICES_DIR` to the absolute path where your voice files are stored on the host.

#### 2. Build and Run

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8001/health

# API docs
open http://localhost:8001/docs
```

## Configuration

### Essential Settings in `.env`

```bash
# Model (HuggingFace ID or local path).
# Default = AWQ-INT4 drop-in (~8.4 GB VRAM, RTF ~0.70 on RTX 3090, multi-speaker).
# Other options: rsxdalv/VibeVoice-Large (FP16, 17 GB), microsoft/VibeVoice-1.5B
# (single-speaker, 8 GB), FabioSarracino/VibeVoice-Large-Q8 (bnb-Q8, slower).
VIBEVOICE_MODEL_PATH=ncoder-ai/VibeVoice-Large-AWQ

# Voice directory on HOST (required)
# Linux/macOS:
VOICES_DIR=/path/to/your/voices/on/host
# Windows:
# VOICES_DIR=C:\Users\username\voices

# Optional: HuggingFace cache for faster model loading
# Linux/macOS:
HF_CACHE_DIR=~/.cache/huggingface
# Windows:
# HF_CACHE_DIR=C:\Users\username\.cache\huggingface
```

> **Windows note:** Use full Windows paths (e.g., `C:\Users\...`). The `~` tilde shortcut does not expand on Windows. Docker Desktop will handle converting Windows paths to Linux mount paths automatically.

### GPU Configuration

Edit `docker-compose.yml` to specify which GPU to use:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Change to your GPU ID
          capabilities: [gpu]
```

## Volume Mounts

The following host paths are mounted into the container:

- **Voices**: `${VOICES_DIR}` → `/app/voices` (read-only)
- **HuggingFace Cache**: `${HF_CACHE_DIR}` → `/root/.cache/huggingface` (read-write, optional)
- **Models**: `${MODELS_DIR}` → `/app/models` (read-write, optional)

## Troubleshooting

### "Extra inputs are not permitted" error

If you see an error like `Extra inputs are not permitted [type=extra_forbidden, input_value='./models', input_type=str]`, you have env vars in your `.env` that the app doesn't recognize. This was fixed - pull the latest code. Any unknown env vars (like `MODELS_DIR`, `HF_CACHE_DIR`) are now safely ignored by the app.

### No voices available

Check that `VOICES_DIR` in `.env` points to the correct host path with voice files.

### GPU not detected

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

# Check container GPU access
docker exec vibevoice-api nvidia-smi
```

### Container won't start

```bash
# Check logs
docker-compose logs vibevoice-api
```

### Slow model loading

Mount your HuggingFace cache in `.env`:
```bash
HF_CACHE_DIR=~/.cache/huggingface
```

## Key Features

- ✅ **No compilation** - All packages installed from pre-built wheels
- ✅ **Fast builds** - Builds complete in minutes, not hours
- ✅ **Python 3.12 + CUDA 12.8** - Optimized for wheel availability
- ✅ **Flash-attention** - Pre-built wheel support (optional)

## Resource Requirements

- **Minimum**: 8GB GPU VRAM, 16GB RAM
- **Recommended**: 16GB+ GPU VRAM, 32GB RAM

## API Endpoints

- Health: `http://localhost:8001/health`
- API Docs: `http://localhost:8001/docs`
- OpenAI-compatible TTS: `POST /v1/audio/speech`
- List voices: `GET /v1/audio/voices`

