# RAGOps — Production MLOps for RAG-Based Clinical Decision Support

A production-grade MLOps framework that combines a Retrieval-Augmented Generation (RAG) pipeline with automated drift detection, explainable AI, and CI/CD quality gates — built on PubMed medical literature for clinical decision support.

## What It Does

- Answers clinical questions using 3,706 PubMed heart disease abstracts via LLaMA-3-8B
- Three retrieval strategies: dense (ChromaDB), BM25 (lexical), hybrid (RRF fusion + cross-encoder reranker)
- Automated evaluation using RAGAS (faithfulness, context recall, answer relevance, context precision)
- PSI-based embedding drift detection with auto knowledge base refresh
- XAI explanation consistency monitoring via cosine similarity of SHAP vectors
- MLflow experiment tracking with quality gates for CI/CD
- 5-panel Streamlit dashboard for querying, monitoring, and comparing experiments

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
        │
        ▼
┌──────────────────────┐
│  Streamlit  :8501    │
│  5-panel dashboard   │
└──────────────────────┘
```

## Services & Ports

| Service   | Internal DNS      | Host Port | Notes                        |
|-----------|-------------------|-----------|------------------------------|
| ChromaDB  | `chromadb:8000`   | 8000      | Vector store                 |
| MLflow    | `mlflow:5000`     | 5000      | SQLite backend + artifact UI |
| Ollama    | `ollama:11434`    | 11434     | LLaMA-3-8B on NVIDIA GPU     |
| FastAPI   | `fastapi:8080`    | 8080      | RAGOps orchestration layer   |
| Streamlit | —                 | 8501      | Dashboard (run separately)   |

## Project Structure

```
ragops/
├── serving/            # FastAPI API + Streamlit dashboard
│   ├── api.py          #   /health, /query, /evaluate, /drift/check, /xai/check
│   └── dashboard.py    #   5-panel Streamlit UI
├── rag_pipeline/       # RAG chain (retrieval + generation)
│   ├── chain.py        #   query() — main entry point
│   ├── retriever.py    #   dense, BM25, hybrid + cross-encoder reranker
│   ├── vectorstore.py  #   ChromaDB client + collections
│   └── ingest.py       #   document chunking + embedding
├── evaluation/         # RAGAS evaluation + ablation study
│   ├── ragas_runner.py #   run_eval(), run_ci_eval()
│   ├── benchmarks/     #   qa_pairs.json (50 Qs), ci_benchmark.json (10 Qs)
│   └── ablations/      #   9-config runner, significance tests, error analysis
├── mlops/              # MLOps infrastructure
│   ├── mlflow_tracker.py       # RAGOpsTracker context manager
│   ├── drift_detector.py       # PSI drift detection
│   ├── explanation_monitor.py  # XAI consistency tracking
│   ├── refresh_trigger.py      # Auto KB refresh on drift
│   └── compare_runs.py         # MLflow run comparison
├── src/                # Core infrastructure
│   ├── config/settings.py      # Pydantic BaseSettings (all thresholds)
│   ├── infra/baseline_store.py # PSI/XAI baseline persistence
│   ├── chroma_interface.py     # Embedding export + incremental upsert
│   ├── fetcher.py              # PubMed NCBI API
│   ├── parser.py               # XML → JSONL
│   └── run_pipeline.py         # ETL orchestrator
├── data/               # Data pipeline
│   ├── ingest.py       #   Team interface (get_embeddings, incremental_upsert)
│   └── refresh.py      #   Weekly PubMed refresh
├── docker/
│   ├── fastapi/Dockerfile
│   └── ollama/pull_model.sh
├── docker-compose.yml
└── env.example
```

## Quick Start

### Prerequisites

- NVIDIA GPU with 8+ GB VRAM
- Docker with Compose v2
- NVIDIA Container Toolkit

```bash
# Verify GPU access
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Start All Services

```bash
# 1. Configure
cp env.example .env

# 2. Start (Ollama pulls ~4.7 GB LLaMA-3 on first boot)
docker compose up -d

# 3. Wait for healthy
python scripts/healthcheck.py --timeout 300

# 4. Verify
curl http://localhost:8080/health
```

### Ingest Documents

```bash
pip install -r docker/fastapi/requirements.txt
python -m rag_pipeline.ingest --data_dir ./data/pubmed --chunk_size 512
```

### Query the Pipeline

```bash
# Via API
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the first-line treatments for type 2 diabetes?"}'

# Via Streamlit dashboard
pip install streamlit plotly requests
streamlit run serving/dashboard.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service liveness + dependency health |
| POST | `/query` | RAG query → answer + sources + scores |
| POST | `/evaluate` | RAGAS evaluation (CI or full) with quality gate |
| POST | `/drift/check` | PSI embedding drift detection |
| POST | `/xai/check` | XAI explanation consistency check |

Interactive docs at http://localhost:8080/docs (Swagger UI).

## Dashboard

Run `streamlit run serving/dashboard.py` to open the 5-panel dashboard:

| Panel | What it shows |
|-------|--------------|
| Query | Ask medical questions, view answers + source documents |
| XAI Explainability | Explanation consistency checks (full SHAP panel pending) |
| RAGAS Metrics | Faithfulness, recall, relevance over time (from MLflow) |
| Drift Alerts | PSI traffic light + thresholds |
| Experiment Comparison | Side-by-side MLflow run metrics with deltas |

## Testing

```bash
# Unit tests (no Docker needed)
MLFLOW_TRACKING_URI=sqlite:///test_mlflow.db ENVIRONMENT=test pytest src/ mlops/ -v

# Single test
pytest src/infra/test/test_mlflow_tracker.py -v

# Coverage (CI requires 80%)
pytest src/ mlops/ --cov=src --cov=mlops --cov-report=term-missing --cov-fail-under=80

# Lint + type check
ruff check src/ mlops/ serving/ evaluation/
ruff format src/ mlops/ serving/ evaluation/ --check
mypy src/ mlops/ serving/ --ignore-missing-imports --strict --exclude src/infra/test
```

## Configuration

All thresholds configured via `.env` → `src/config/settings.py` (Pydantic BaseSettings):

| Threshold | Default | Trigger |
|-----------|---------|---------|
| PSI warning | 0.1 | Log warning to MLflow |
| PSI alert | 0.25 | Auto KB refresh |
| XAI warning | 0.75 | Log warning |
| XAI alert | 0.60 | Instability alert |
| RAGAS faithfulness | 0.80 | CI quality gate fail |
| RAGAS context recall | 0.75 | CI quality gate fail |
| RAGAS answer relevancy | 0.75 | CI quality gate fail |

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

See `env.example` for all tunables including:
- PSI drift detection thresholds (`PSI_WARNING_THRESHOLD`, `PSI_ALERT_THRESHOLD`)
- XAI consistency thresholds (`XAI_WARNING_THRESHOLD`, `XAI_ALERT_THRESHOLD`)
- RAGAS quality gate minimums
- Ollama GPU/concurrency settings (`OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_PARALLEL`)

## Branch Strategy

- `main` — protected, always deployable
- `develop` — integration branch
- Feature branches: `<prefix>/<description>` off `develop` (prefixes: `infra/`, `rag/`, `eval/`, `data/`, `refactor/`)
- Commit format: `<type>: description` (types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | LLaMA-3-8B via Ollama |
| Vector DB | ChromaDB (all-MiniLM-L6-v2, 384-dim) |
| Retrieval | Dense + BM25 + Hybrid RRF + Cross-encoder reranker |
| Evaluation | RAGAS framework |
| Drift Detection | PSI on embedding distributions |
| XAI | SHAP + explanation consistency monitoring |
| Experiment Tracking | MLflow |
| Serving | FastAPI + Streamlit |
| CI/CD | GitHub Actions |
| Language | Python 3.11 |
