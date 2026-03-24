#!/usr/bin/env bash
# =============================================================================
# RAGOps — Ollama model pull init script
# Runs inside the Ollama container on first boot.
# Idempotent: skips pull if model is already cached in ollama_data volume.
# =============================================================================

set -euo pipefail

MODEL="${OLLAMA_MODEL:-llama3:8b}"
MAX_RETRIES=10
RETRY_DELAY=10

echo "[ragops/ollama] Checking for model: ${MODEL}"

retry_count=0
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
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
