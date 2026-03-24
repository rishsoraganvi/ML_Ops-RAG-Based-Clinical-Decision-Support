# =============================================================================
# RAGOps FastAPI Service — Dockerfile
# Python 3.11 slim base, production-hardened
# =============================================================================

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user for security ────────────────────────────────────────────────
RUN groupadd --gid 1001 ragops && \
    useradd  --uid 1001 --gid ragops --shell /bin/bash --create-home ragops

WORKDIR /app

# ── Python deps (cached layer — copy requirements first) ─────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY --chown=ragops:ragops . .

USER ragops

EXPOSE 8080

# Uvicorn with 2 workers; override LOG_LEVEL via env
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--log-level", "info", \
     "--no-access-log"]
