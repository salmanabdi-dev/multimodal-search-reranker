FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU build of PyTorch (works on x86_64 and arm64). For an NVIDIA GPU, swap this
# line for the matching CUDA wheel and set MMS_DEVICE=cuda.
ARG TORCH_VERSION=2.5.1
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dev]"

# Bake all three models (MiniLM bi-encoder, ms-marco cross-encoder, CLIP ViT-B/32) into
# the image so container start-up is offline and deterministic. CLIP adds ~600 MB.
# The resolved commit hashes are written to /models/manifest.json.
# Model names come from .env via compose build args, so changing a model in .env and
# running `docker compose up -d --build` bakes the new one (the runtime hub is offline).
ARG MMS_BI_ENCODER_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG MMS_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG MMS_CLIP_MODEL=openai/clip-vit-base-patch32
RUN MMS_BI_ENCODER_MODEL="$MMS_BI_ENCODER_MODEL" MMS_CROSS_ENCODER_MODEL="$MMS_CROSS_ENCODER_MODEL" \
    MMS_CLIP_MODEL="$MMS_CLIP_MODEL" python -m mmsearch.cli download-models

COPY sql ./sql
COPY data ./data
COPY tests ./tests
RUN mkdir -p /app/data/media

EXPOSE 8000
CMD ["uvicorn", "mmsearch.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
