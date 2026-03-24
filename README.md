# RAGOps — Docker Infrastructure Layer

Production Docker environment for the RAGOps clinical decision support framework.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ragops_net (bridge)                  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  ChromaDB    │  │   MLflow     │  │  Ollama          │  │
│  │  :8000       │  │   :5000      │  │  :11434          │  │
│  │              │  │              │  │  [NVIDIA GPU]    │  │
│  │  chroma_data │  │  mlflow_data │  │  ollama_data     │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
│  ┌──────┴─────────────────┴────────────────────┴─────────┐  │
│  │              FastAPI  :8080                           │  │
│  │   /health  /query  /evaluate  /drift/check  /xai/check│  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Services & Ports

| Service   | Internal DNS      | Host Port | Notes                        |
|-----------|-------------------|-----------|------------------------------|
| ChromaDB  | `chromadb:8000`   | 8000      | Vector store                 |
| MLflow    | `mlflow:5000`     | 5000      | SQLite backend + artifact UI |
| Ollama    | `ollama:11434`    | 11434     | LLaMA-3-8B on NVIDIA GPU     |
| FastAPI   | `fastapi:8080`    | 8080      | RAGOps orchestration layer   |

## Quick Start

```bash
# 1. Clone & configure
cp .env.example .env
# Edit .env as needed (NVIDIA_GPU_COUNT, OLLAMA_MODEL, thresholds)

# 2. Verify NVIDIA container toolkit is installed
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# 3. Start all services
docker compose up -d

# 4. Wait for healthy (Ollama pulls ~4.7 GB LLaMA-3 on first boot)
python scripts/healthcheck.py --timeout 300

# 5. Verify dashboards
open http://localhost:5000   # MLflow UI
open http://localhost:8080/docs  # FastAPI Swagger
```

## Named Volumes

```bash
# Inspect persistence
docker volume ls | grep ragops

# Nuclear reset (destroys all data including embeddings and model cache)
docker compose down -v
```

## Logs

```bash
docker compose logs -f fastapi     # RAGOps app logs
docker compose logs -f ollama      # Model download progress
docker compose logs -f mlflow      # Experiment tracking
```

## Environment Variables

See `.env.example` for all tunables including:
- PSI drift detection thresholds (`PSI_WARNING_THRESHOLD`, `PSI_ALERT_THRESHOLD`)
- XAI consistency thresholds (`XAI_WARNING_THRESHOLD`, `XAI_ALERT_THRESHOLD`)
- RAGAS quality gate minimums
- Ollama GPU/concurrency settings
