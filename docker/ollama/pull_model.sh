#!/usr/bin/env bash
# =============================================================================
# RAGOps — Ollama model pull init script
# Runs inside the Ollama container on first boot.
# Idempotent: skips pull if model is already cached in ollama_data volume.
# =============================================================================

set -euo pipefail

MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
MAX_RETRIES=60
RETRY_DELAY=10

echo "[ragops/ollama] Checking for model: ${MODEL}"

retry_count=0
until ollama list > /dev/null 2>&1; do
    retry_count=$((retry_count + 1))
    if [ "${retry_count}" -ge "${MAX_RETRIES}" ]; then
        echo "[ragops/ollama] ERROR: Ollama server did not become ready after ${MAX_RETRIES} retries"
        exit 1
    fi
    echo "[ragops/ollama] Waiting for Ollama server... (attempt ${retry_count}/${MAX_RETRIES})"
    sleep "${RETRY_DELAY}"
done

echo "[ragops/ollama] Server is ready. Pulling ${MODEL} (cached if already present)..."
ollama pull "${MODEL}"

echo "[ragops/ollama] Model ${MODEL} is ready."

# RAGAS embedding metrics (context_precision, answer_relevancy) require a
# separate embedding model. Pull it inside the container so CI and dev are
# both guaranteed to have it before evaluate() runs.
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
echo "[ragops/ollama] Pulling embedding model: ${EMBED_MODEL}"
ollama pull "${EMBED_MODEL}"
echo "[ragops/ollama] Embedding model ${EMBED_MODEL} is ready."
