# RAGOps Startup Guide

A step-by-step guide to get the RAGOps Clinical Decision Support system running on your machine. Two sections below — pick the one that matches your comfort level.

---

## Section 1: Non-Technical Guide

This section assumes you have never worked with Docker, Python, or command-line tools before. Follow each step exactly as written.

### What You Need Before Starting

1. **A computer with an NVIDIA GPU** (at least 8 GB VRAM)
   - The system runs a large AI model (LLaMA-3-8B) that requires a GPU
   - Check: open a terminal and type `nvidia-smi` — you should see your GPU listed
   - If you see an error, you need to install NVIDIA drivers first

2. **Docker Desktop** installed and running
   - Download from https://www.docker.com/products/docker-desktop
   - After installation, open Docker Desktop and wait until it says "Docker is running"
   - On Windows: make sure WSL 2 is enabled (Docker Desktop will prompt you)

3. **Git** installed
   - Download from https://git-scm.com/downloads
   - This lets you download the project code

4. **NVIDIA Container Toolkit** installed
   - This lets Docker use your GPU
   - Installation guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Step-by-Step: Getting RAGOps Running

**Step 1 — Download the project**

Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:

```
git clone https://github.com/rishsoraganvi/ML_Ops-RAG-Based-Clinical-Decision-Support.git
cd ML_Ops-RAG-Based-Clinical-Decision-Support/ragops
```

**Step 2 — Create your configuration file**

```
cp env.example .env
```

This creates a file called `.env` with default settings. You do not need to edit it — the defaults work out of the box.

**Step 3 — Start all services**

```
docker compose up -d
```

This downloads and starts 4 services:
- **ChromaDB** — stores the medical documents
- **MLflow** — tracks experiment results
- **Ollama** — runs the AI model (this downloads ~4.7 GB on first run, so be patient)
- **FastAPI** — the main application server

Wait 3-5 minutes on first run for the AI model to download.

**Step 4 — Check that everything is healthy**

```
docker compose ps
```

You should see all 4 containers with status `healthy`. If any show `starting`, wait another minute and check again.

You can also open these URLs in your browser:
- http://localhost:8080/health — shows the overall system health
- http://localhost:5000 — MLflow dashboard (experiment tracking)
- http://localhost:8000/api/v1/heartbeat — ChromaDB heartbeat

**Step 5 — Open the dashboard**

If you have Python installed locally with the required packages:

```
pip install streamlit plotly requests mlflow pandas
streamlit run serving/dashboard.py
```

This opens the RAGOps dashboard in your browser with 5 panels:
- **Query** — type a medical question and get an AI-generated answer
- **XAI Explainability** — see why the AI gave that answer
- **RAGAS Metrics** — track answer quality over time
- **Drift Alerts** — detect when the knowledge base is going stale
- **Experiment Comparison** — compare different system configurations

**Step 6 — Try asking a question**

In the Query panel, type something like:

> What are the first-line treatments for type 2 diabetes?

Select a retriever type (start with "dense"), chunk size (512), and click Submit.

### Stopping and Restarting

```
docker compose down       # Stop all services (keeps your data)
docker compose up -d      # Start again later
docker compose down -v    # Stop AND delete all data (full reset)
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found | Install NVIDIA drivers for your GPU |
| Ollama container keeps restarting | Your GPU may not have enough VRAM (need 8 GB+). Check: `docker logs ragops_ollama` |
| Health check shows "degraded" | One or more services aren't ready yet. Wait 2 minutes and try again |
| "Cannot connect to Docker daemon" | Open Docker Desktop and make sure it's running |
| Port already in use | Another app is using that port. Edit `.env` to change port numbers |

---

## Section 2: Technical Guide

This section assumes familiarity with Python, Docker, and CLI tools.

### Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Docker + Compose | 24+ | `docker compose version` |
| NVIDIA Driver | 525+ | `nvidia-smi` |
| NVIDIA Container Toolkit | 1.13+ | `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi` |
| Python | 3.11 | `python --version` (for local dev/tests only — not needed for Docker) |
| Git | 2.x | `git --version` |

### 1. Clone and Configure

```bash
git clone https://github.com/rishsoraganvi/ML_Ops-RAG-Based-Clinical-Decision-Support.git
cd ML_Ops-RAG-Based-Clinical-Decision-Support/ragops

cp env.example .env
# Edit .env if you need to change ports, GPU count, or thresholds
```

Key env vars you might want to change:

| Variable | Default | When to change |
|----------|---------|----------------|
| `NVIDIA_GPU_COUNT` | 1 | Multi-GPU machine — set to number of GPUs |
| `OLLAMA_MODEL` | llama3:8b | Use a smaller model for low-VRAM GPUs (e.g., `llama3:1b`) |
| `CHROMA_ALLOW_RESET` | false | Set `true` for development (allows collection deletion via API) |
| `ENVIRONMENT` | production | Set `development` for verbose logging |
| `FASTAPI_PORT` | 8080 | Port conflict |

### 2. Start the Docker Stack

```bash
docker compose up -d
```

The stack creates 3 named volumes (`chroma_data`, `mlflow_data`, `ollama_data`) that persist across restarts. Ollama auto-pulls LLaMA-3-8B on first boot via `docker/ollama/pull_model.sh` (idempotent — skips if cached).

Service dependency: FastAPI waits for all 3 upstream services to be healthy before starting.

### 3. Verify Health

```bash
# All containers should show "healthy"
docker compose ps

# Programmatic health check with timeout
python scripts/healthcheck.py --timeout 300

# Or hit the health endpoint directly
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "environment": "production",
  "dependencies": {
    "chromadb": "ok",
    "mlflow": "ok",
    "ollama": "ok"
  }
}
```

### 4. Ingest Documents into ChromaDB

```bash
# Install Python dependencies (if running locally outside Docker)
pip install -r docker/fastapi/requirements.txt

# Ingest PubMed abstracts (requires ChromaDB running)
python -m rag_pipeline.ingest --data_dir ./data/pubmed --chunk_size 512

# Verify ingestion
python -c "from rag_pipeline.vectorstore import collection_stats; print(collection_stats(512))"
```

### 5. Test the RAG Pipeline

```bash
# Smoke test — queries Ollama + ChromaDB end-to-end
python -m rag_pipeline.chain
```

### 6. Use the API

```bash
# Query the RAG pipeline
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the first-line treatments for type 2 diabetes?", "config": {"retriever_type": "dense", "chunk_size": 512, "k": 5}}'

# Run RAGAS CI evaluation (5-question smoke test)
curl -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{"mode": "ci", "run_quality_gate": true}'

# Check embedding drift (requires baseline captured first)
curl -X POST http://localhost:8080/drift/check

# Check XAI explanation consistency (requires baseline)
curl -X POST http://localhost:8080/xai/check \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 7. Launch the Dashboard

```bash
pip install streamlit plotly requests
streamlit run serving/dashboard.py
```

Opens at http://localhost:8501 with 5 panels: Query, XAI, RAGAS Metrics, Drift Alerts, Experiment Comparison.

### 8. Run Tests and Linting

```bash
# Unit tests (no Docker needed — uses SQLite fixture)
MLFLOW_TRACKING_URI=sqlite:///test_mlflow.db ENVIRONMENT=test pytest src/ mlops/ -v

# Single test file
pytest src/infra/test/test_mlflow_tracker.py -v

# Coverage (CI requires 80%)
pytest src/ mlops/ --cov=src --cov=mlops --cov-report=term-missing --cov-fail-under=80

# Lint
ruff check src/ mlops/ serving/ evaluation/ --output-format=github
ruff format src/ mlops/ serving/ evaluation/ --check

# Type check
mypy src/ mlops/ serving/ --ignore-missing-imports --strict --exclude src/infra/test
```

### 9. Service Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | http://localhost:8080 | Main API (`/health`, `/query`, `/evaluate`, `/drift/check`, `/xai/check`) |
| FastAPI Docs | http://localhost:8080/docs | Interactive Swagger UI |
| MLflow | http://localhost:5000 | Experiment tracking dashboard |
| ChromaDB | http://localhost:8000 | Vector store API |
| Ollama | http://localhost:11434 | LLM inference API |
| Streamlit | http://localhost:8501 | User-facing dashboard (run separately) |

### 10. Common Operations

```bash
# View logs for a specific service
docker logs ragops_fastapi --tail 50 -f
docker logs ragops_ollama --tail 50 -f

# Restart a single service
docker compose restart fastapi

# Rebuild FastAPI after code changes (if not using volume mounts)
docker compose build fastapi && docker compose up -d fastapi

# Full reset — deletes all volumes (ChromaDB data, MLflow runs, Ollama model cache)
docker compose down -v

# Check GPU utilization inside the Ollama container
docker exec ragops_ollama nvidia-smi
```

### Project Architecture (Quick Reference)

```
serving/api.py ──────> rag_pipeline/chain.py ──> retriever.py ──> vectorstore.py (ChromaDB)
                  |                                              > ingest.py
                  |──> evaluation/ragas_runner.py               (LangChain + Ollama)
                  |──> mlops/drift_detector.py ──> src/infra/baseline_store.py
                  |──> mlops/explanation_monitor.py ──> src/infra/baseline_store.py
                  
src/config/settings.py ──> read by mlops/, serving/, src/ (central config singleton)
rag_pipeline/ ──> uses os.environ directly (LangChain convention)
```
