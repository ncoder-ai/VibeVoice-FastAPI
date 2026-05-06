FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ONLY_BINARY=:all:

# Install system dependencies.
# - python3.12 + dev: runtime + Triton needs Python.h to compile kernel launchers
# - gcc/g++: REQUIRED at runtime by Triton to compile fused CUDA kernels
#   when torch.compile fires. Without these, torch.compile silently falls
#   back to eager mode and the cache stays empty.
# - ffmpeg, libsndfile1: audio I/O
# - curl: healthcheck
# - git: required by some pip installs (kept; ~5MB)
# --no-install-recommends keeps the image lean (~50MB savings).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    gcc \
    g++ \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv /app/venv

# Activate venv and upgrade pip
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY pyproject.toml README.md ./
COPY requirements-api.txt ./

# Install PyTorch with CUDA support (CUDA 12.8 compatible)
RUN pip install torch==2.8.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install torchao for INT8 quantization support
# torchao 0.13.0 was built for PyTorch 2.8.0
RUN pip install torchao==0.13.0

# Install flash-attn from pre-built wheel (PyTorch 2.8 compatible).
# This MUST succeed — fall-through to source build would take ~30 min and
# requires nvcc which isn't in the runtime image. Fail the build instead.
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Install VibeVoice package. Fail-fast: if the binary wheels aren't available
# we don't want to silently fall through to a multi-minute source build.
COPY vibevoice/ ./vibevoice/
COPY demo/ ./demo/
RUN pip install --only-binary=:all: -e .

# Install API dependencies (binary-only)
RUN pip install --only-binary=:all: -r requirements-api.txt

# Copy API code
COPY api/ ./api/
COPY start.sh ./

# Create directories for voices and models
RUN mkdir -p /app/voices /app/models

# Expose API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the API (using venv python)
CMD ["sh", "-c", "/app/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8001}"]
