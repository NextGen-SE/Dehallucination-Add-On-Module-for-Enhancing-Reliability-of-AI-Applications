# TruthLayer FastAPI backend (POST /verify).
# Build context is the repo root; .dockerignore allow-lists only what the API needs.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Embedding + NLI models download here on first start (mounted as a volume in compose).
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first: the default PyPI wheel bundles several GB of CUDA
# libraries the API never uses. requirements.txt then sees torch as satisfied.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Unprivileged user; owns the model cache so a fresh named volume is writable.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p "$HF_HOME" \
    && chown -R app:app /home/app

COPY api/ api/
COPY claim_extraction.py evidence_verification.py knowledge_base.json ./

USER app
EXPOSE 8000

# Configuration (API key hashes, database URL, ...) comes from the environment;
# the app refuses to start without it. See .env.example.
CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
