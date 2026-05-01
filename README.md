# RAGOps — Production MLOps for RAG-Based Clinical Decision Support

A production-grade MLOps framework that combines a Retrieval-Augmented Generation (RAG) pipeline with automated drift detection, explainable AI, and CI/CD quality gates — built on PubMed medical [...]

Targeting research venues: **IEEE ICHI**, **ACL Clinical NLP Workshop**, and **MLSys Workshop**.

## What It Does

- Answers clinical questions using **3,706 PubMed heart disease abstracts** (English, 2021–2026) via LLaMA-3.2-3B
- Three retrieval strategies: **dense** (ChromaDB), **BM25** (lexical), **hybrid** (RRF fusion + cross-encoder reranker)
- Automated evaluation using **RAGAS** — faithfulness, context recall, answer relevance, context precision
- **PSI-based embedding drift detection** with auto knowledge-base refresh (paper contribution)
- **XAI layer** — SHAP on retrieval scores, DistilBERT saliency, BM25 term attribution, hallucination classifier
- **Explanation consistency monitoring** via cosine similarity of SHAP vectors (novel MLOps metric)
- **MLflow experiment tracking** with CI quality gates (faithfulness ≥ 0.70, context_recall ≥ 0.65)
- **5-panel Streamlit dashboard** for querying, monitoring, and comparing experiments

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
│  │   /health  /query  /explain  /evaluate                │  │
│  │   /drift/check  /xai/check                            │  │
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
| Ollama    | `ollama:11434`    | 11434     | LLaMA-3.2-3B on NVIDIA GPU   |
| FastAPI   | `fastapi:8080`    | 8080      | RAGOps orchestration layer   |
| Streamlit | —                 | 8501      | Dashboard (run separately)   |

FastAPI `depends_on` all three upstream services with `condition: service_healthy`. Ollama auto-pulls LLaMA-3.2-3B on first boot via `docker/ollama/pull_model.sh` (idempotent, 120 s start period).

## Project Structure

```
ragops/
├── serving/            # FastAPI API + Streamlit dashboard
│   ├── api.py          #   /health, /query, /explain, /evaluate, /drift/check, /xai/check
│   └── dashboard.py    #   5-panel Streamlit UI
├── rag_pipeline/       # RAG chain (retrieval + generation)
│   ├── chain.py        #   query() — main entry point
│   ├── query_processor.py #  abbrev / MeSH expansion
│   ├── retriever.py    #   dense, BM25, hybrid + cross-encoder reranker
│   ├── vectorstore.py  #   ChromaDB client + collections
│   └── ingest.py       #   document chunking + embedding
├── evaluation/         # RAGAS evaluation + ablation study + XAI
│   ├── ragas_runner.py #   run_eval(), run_ci_eval()
│   ├── explainability.py #  SHAP + DistilBERT saliency + BM25 terms + hallucination classifier
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
│   ├── refresh.py      #   Weekly PubMed refresh (heart disease only)
│   ├── quality_report.py #  → quality_report.html
│   ├── stats.py        #   Dataset statistics (Table 0)
│   └── processed/pubmed_processed.jsonl  # 3,706 records
├── tests/              # Unit + integration tests
│   ├── test_rag.py, test_explainability.py, ...
│   └── test_e2e.py     #   gated by RAGOPS_E2E=1
├── scripts/
│   ├── healthcheck.py           # poll /health until green
│   └── run_baseline_eval.py     # RAGAS + PSI + XAI baseline capture
├── docker/
│   ├── fastapi/Dockerfile
│   └── ollama/pull_model.sh
├── .github/workflows/
│   ├── pr_checks.yml   #   lint, typecheck, unit tests, docker-build, ragas-ci-eval
│   └── refresh.yml     #   Mon 06:07 UTC — auto KB refresh via drift trigger
├── docker-compose.yml
├── Makefile
└── env.example
```

## Quick Start

### Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Docker + Compose | 24+ | `docker compose version` |
| NVIDIA Driver | 525+ | `nvidia-smi` |
| NVIDIA Container Toolkit | 1.13+ | `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi` |
| Python | 3.11 | `python --version` (local dev/tests only) |
| Git | 2.x | `git --version` |

**GPU:** ~4 GB VRAM recommended for LLaMA-3.2-3B. For very low-VRAM or CPU-only machines, set `OLLAMA_MODEL=llama3.2:1b` in `.env`.

### Start All Services

```bash
# 1. Configure
cp env.example .env

# 2. Start (Ollama pulls ~2.0 GB LLaMA-3.2-3B on first boot)
docker compose up -d
# or: make up

# 3. Wait for healthy
python scripts/healthcheck.py --timeout 300
# or: make healthcheck

# 4. Verify
curl http://localhost:8080/health
docker compose ps    # all 4 containers should report "healthy"
```

Expected `/health` response:

```json
{
  "status": "healthy",
  "environment": "production",
  "dependencies": {"chromadb": "ok", "mlflow": "ok", "ollama": "ok"}
}
```

### Ingest Documents

```bash
pip install -r docker/fastapi/requirements.txt
python -m rag_pipeline.ingest --data_dir ./data/raw --chunk_size 512

# Verify
python -c "from rag_pipeline.vectorstore import collection_stats; print(collection_stats(512))"
```

### Capture Baselines

Drift detection (PSI) and XAI consistency both require a reference captured from a known-good configuration. Baselines are persisted by `FileBaselineStore` to `/app/baselines` inside the FastAPI [...]

Run this once after ingestion and again after every KB refresh:

```bash
# Full baseline — RAGAS (50 Qs) + PSI embedding baseline + XAI explanation baseline
make baseline                  # runs inside the FastAPI container, writes to the shared volume

# Or directly:
docker compose exec fastapi python scripts/run_baseline_eval.py

# Partial captures
docker compose exec fastapi python scripts/run_baseline_eval.py --skip-xai    # RAGAS + PSI only
docker compose exec fastapi python scripts/run_baseline_eval.py --skip-eval   # baselines only

# Local host run (writes to ./baselines/ — set BASELINE_DIR=./baselines in .env first)
make baseline-host
```

Artifacts written to `<baseline_dir>`:
- `embedding_baseline.npy` — PSI reference embedding matrix (shape `(n_docs, 384)`)
- `xai_baseline.npz` — XAI explanation vectors (one per benchmark question)

To switch backends in tests, set `RAGOPS_BASELINE_STORE=memory` (the unit test conftest does this automatically).

### Query the Pipeline

```bash
# Smoke-test the chain end-to-end (requires Ollama + ChromaDB)
python -m rag_pipeline.chain

# Query via API
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the first-line treatments for type 2 diabetes?",
       "config": {"retriever_type": "dense", "chunk_size": 512, "k": 5}}'

# Explain the most recent query (SHAP + saliency + BM25 terms + hallucination risk)
curl -X POST http://localhost:8080/explain \
  -H "Content-Type: application/json" \
  -d @last_query.json

# Launch the Streamlit dashboard
pip install streamlit plotly requests mlflow pandas
streamlit run serving/dashboard.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/health`       | Service liveness + dependency health |
| POST | `/query`        | RAG query → answer + sources + retrieval scores |
| POST | `/explain`      | XAI explanation for a query result (SHAP, saliency, BM25 terms, hallucination risk) |
| POST | `/evaluate`     | RAGAS evaluation (CI or full) with quality gate |
| POST | `/drift/check`  | PSI embedding drift detection (requires baseline) |
| POST | `/xai/check`    | XAI explanation consistency check (requires baseline) |

Interactive docs at <http://localhost:8080/docs> (Swagger UI).

## Dashboard

`streamlit run serving/dashboard.py` opens the 5-panel dashboard at <http://localhost:8501>:

| Panel | What it shows |
|-------|---------------|
| **Query** | Ask medical questions, view answers + source documents |
| **XAI Explainability** | SHAP per-doc bars, DistilBERT saliency highlights, BM25 term overlap, hallucination-risk meter |
| **RAGAS Metrics** | Faithfulness, recall, relevance over time (pulled from MLflow) |
| **Drift Alerts** | PSI traffic light + threshold table + last 5 drift events + last-refresh timestamp |
| **Experiment Comparison** | Side-by-side MLflow run metrics with deltas |

## Testing

Two tiers: **mock-based unit tests** (CI-safe, no Docker) and **end-to-end integration tests** (opt-in, needs docker stack). The `integration` pytest marker is registered in `pytest.ini` and gate[...]

```bash
# Unit tests — no Docker required (SQLite + stubbed Ollama/Chroma)
MLFLOW_TRACKING_URI=sqlite:///test_mlflow.db ENVIRONMENT=test \
  pytest tests/ src/ mlops/ -m "not integration" -v
# or: make test-unit

# Single test file
pytest tests/test_explainability.py -v

# Coverage (CI requires 80%)
pytest tests/ src/ mlops/ -m "not integration" \
  --cov=src --cov=mlops --cov-report=term-missing --cov-fail-under=80

# Integration tier — requires docker stack + populated ChromaDB
RAGOPS_E2E=1 pytest tests/test_e2e.py -m integration -v
# or: make test-e2e

# Lint + format
ruff check src/ mlops/ serving/ evaluation/ rag_pipeline/ tests/     # or: make lint
ruff format src/ mlops/ serving/ evaluation/ rag_pipeline/ tests/ --check

# Type check
mypy src/ mlops/ serving/ --ignore-missing-imports --strict --exclude src/infra/test
# or: make typecheck
```

## Evaluation & Ablations

```bash
# Fast CI smoke eval (5 questions)
make eval-ci

# Inspect the 9-config ablation grid without running the LLM
python evaluation/ablations/run_ablations.py --dry-run

# Full sweep — 9 retrieval configs × 50 questions = 450 evaluations
python evaluation/ablations/run_ablations.py \
    --qa-file evaluation/benchmarks/qa_pairs.json --workers 1
# or: make eval-full
```

See `evaluation/ablations/RUNBOOK.md` for the sweep used to populate Paper Table 1 and produce weak labels for the hallucination classifier.

## Configuration

All thresholds configured via `.env` → `src/config/settings.py` (Pydantic `BaseSettings` singleton). **Never hardcode thresholds or URLs.**

| Threshold | Default | Trigger |
|-----------|---------|---------|
| PSI warning | 0.10 | Log warning to MLflow |
| PSI alert | 0.25 | Auto KB refresh via `refresh_trigger` |
| XAI consistency warning | 0.75 | Log warning |
| XAI consistency alert | 0.60 | Explanation-instability alert |
| RAGAS faithfulness | 0.80 (CI: 0.70) | Quality gate fail |
| RAGAS context recall | 0.75 (CI: 0.65) | Quality gate fail |
| RAGAS answer relevancy | 0.75 | Quality gate fail |
| `PSI_NUM_BINS` | 10 | Histogram binning for PSI |
| `XAI_BENCHMARK_QUESTIONS` | 10 | Questions used for consistency check |
| `BASELINE_DIR` | `/app/baselines` | Where `FileBaselineStore` reads/writes baselines |
| `BASELINE_CHUNK_SIZE` | 512 | ChromaDB collection (`pubmed_{n}`) used for the baseline embedding matrix |
| `RAGOPS_BASELINE_STORE` | `file` | Switch baseline backend: `file` (default, persistent) \| `memory` (tests only) |
| `OLLAMA_KEEP_ALIVE` | 24h | Keeps model in GPU VRAM |
| `OLLAMA_NUM_PARALLEL` | 2 | Concurrent LLM requests |
| `CHROMA_ALLOW_RESET` | false | `true` only in dev |

See `env.example` for the full list.

## CI/CD Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `.github/workflows/pr_checks.yml` | PR to `main`/`develop`, push to `develop` | ruff + mypy + no-print guard + unit tests (80% coverage) + docker-build |
| `pr_checks.yml` — `ragas-ci-eval` job | PRs to `main`/`develop` | Brings up compose stack, pulls `llama3.2:1b`, gates on `faithfulness ≥ 0.70` and `context_recall ≥ 0.65` |
| `.github/workflows/refresh.yml` | Cron `7 6 * * 1` (Mon 06:07 UTC) + manual dispatch | Runs `mlops.refresh_trigger.trigger_refresh`; on manual runs posts before/after metric comment on the spec[...]

## Named Volumes & Logs

```bash
# Inspect persistence
docker volume ls | grep ragops

# Service logs
docker compose logs -f fastapi     # RAGOps app
docker compose logs -f ollama      # Model download / inference
docker compose logs -f mlflow      # Experiment tracking

# GPU utilization inside the Ollama container
docker exec ragops_ollama nvidia-smi

# Restart a single service
docker compose restart fastapi

# Nuclear reset — deletes ChromaDB data, MLflow runs, and Ollama model cache
docker compose down -v
```

## Makefile Targets

```bash
make setup          # install Python deps + pull llama3.2:1b
make up / make down # docker compose up/down
make healthcheck    # poll /health until green
make test-unit      # pytest (mock-based, no docker)
make test-e2e       # pytest integration tier (needs docker)
make eval-ci        # RAGAS 5-question smoke eval
make eval-full      # full 9-config ablation sweep (hours on CPU)
make baseline       # docker compose exec fastapi python scripts/run_baseline_eval.py (persists to ragops_baselines volume)
make baseline-host  # local host run — writes to $BASELINE_DIR (set in .env, e.g. ./baselines)
make drift-check    # POST /drift/check
make xai-check      # POST /xai/check
make lint / format / typecheck
make clean          # remove caches and coverage artifacts
```

> **Windows note:** Makefile targets use POSIX shell idioms. Run them from **WSL** or **Git Bash**. Plain `docker` / `python` commands work fine in PowerShell. CI runs on Ubuntu — no issue.

## Code Conventions

- Type hints and docstrings on every public function and class
- `logging` module only — `print()` in `src/`, `mlops/`, `serving/` is caught by the CI no-print guard
- Mark novel work with comment tags: `# PAPER CONTRIBUTION` (PSI drift), `# XAI CONTRIBUTION` (SHAP/consistency), `# PAPER RESULT — TABLE X`, `# ABLATION EXPERIMENT`
- Settings import: always `from src.config.settings import settings` — never read `os.environ` directly in `src/`
- MLflow tracker import: `from mlops.mlflow_tracker import RAGOpsTracker, RunTrigger`
- `rag_pipeline/` is the exception — uses direct `os.environ` reads per LangChain convention
- `BaselineStore` accessed via `get_baseline_store()` singleton factory — never instantiate directly. Backend is selected by `RAGOPS_BASELINE_STORE` (`file` default → `FileBaselineStore` pers[...]
- Tests colocated with modules (`src/infra/test/...`) or centralized in `tests/`

## Branch Strategy

- `main` — protected, always deployable. PRs only.
- `develop` — integration branch. Feature branches merge here first.
- Feature branches: `<prefix>/<description>` off `develop` (`infra/`, `rag/`, `eval/`, `data/`, `refactor/`)
- Hotfixes: branch off `main` with `fix/`, PR to both `main` and `develop`
- Commit format: `<type>: description` (`feat`, `fix`, `test`, `docs`, `refactor`, `chore`)

## Key Interface Contracts

| Import path | Signature | Returns |
|---|---|---|
| `rag_pipeline.chain.query` | `(question, config=None, mlflow_run=False)` | `{answer, source_docs, retrieval_latency_ms, llm_latency_ms, total_latency_ms, retrieval_scores, config}` |
| `data.ingest.get_embeddings` | `(chunk_size: int \| None = None)` | `np.ndarray (n_docs, 384)` — defaults to `settings.baseline_chunk_size` |
| `data.ingest.incremental_upsert` | `(new_docs)` | `int` (chunks upserted) |
| `evaluation.ragas_runner.run_eval` | `(qa_pairs, config=None, per_question_path=None)` | `{faithfulness, context_recall, answer_relevance, context_precision}` |
| `evaluation.ragas_runner.run_ci_eval` | `(qa_pairs=None, config=None)` | same as `run_eval` (5-question stub if no pairs) |
| `evaluation.explainability.explain` | `(question, source_docs, retrieval_scores)` | `{shap_values, token_attributions, term_attribution, explanation_vector, hallucination_risk, hallucination_re[...]` |
| `mlops.drift_detector.compute_psi` | `(current_embeddings)` | `float` (mean PSI across 384 dims) |
| `mlops.explanation_monitor.compute_consistency` | `(current_vectors=None, baseline_vectors=None)` | `float` (mean cosine similarity 0–1) |

## Current Status

**Done:** Docker environment, MLflow tracker with quality gates, full RAG pipeline (dense/BM25/hybrid + cross-encoder reranker), PubMed ETL (3,706 heart disease abstracts), RAGAS evaluation, 9-co[...]

**Paper deliverables in progress:** ablation `results.csv`, Tables 1–3, Figures 1–3, demo video, `DATA_CARD.md`, `v1.0.0` tag + DVC data push.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | LLaMA-3.2-3B via Ollama |
| Vector DB | ChromaDB (all-MiniLM-L6-v2, 384-dim) |
| Retrieval | Dense + BM25 + Hybrid RRF + Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) |
| Evaluation | RAGAS framework |
| Drift Detection | PSI on embedding distributions |
| XAI | SHAP + DistilBERT saliency + BM25 term attribution + hallucination classifier |
| Experiment Tracking | MLflow (SQLite backend) |
| Serving | FastAPI + Streamlit + Plotly |
| Data Versioning | DVC |
| CI/CD | GitHub Actions |
| Language | Python 3.11 |

## Further Reading

- **`startup_guide.md`** — step-by-step setup (non-technical + technical sections)
- **`Solo_plan.md`** — full phase-by-phase execution plan and paper outline
- **`CONTRIBUTING.md`** — contribution guidelines
- **`CLAUDE.md`** — guidance for Claude Code when working in this repo
- **`evaluation/ablations/RUNBOOK.md`** — ablation sweep runbook
