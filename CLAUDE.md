# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGOps is a production-grade MLOps framework for RAG-based clinical decision support using PubMed medical literature. It combines a Retrieval-Augmented Generation pipeline with automated monitoring (PSI drift detection), explainable AI (SHAP), and CI/CD quality gates (RAGAS evaluation). The project targets research venues like IEEE ICHI and ACL Clinical NLP Workshop.

## Architecture

Four Docker services communicate over `ragops_net` bridge network:

- **ChromaDB** (:8000) — vector store with `pubmed_{chunk_size}` collections (256/512/1024)
- **MLflow** (:5000) — experiment tracking with SQLite backend
- **Ollama** (:11434) — LLaMA-3-8B on NVIDIA GPU (model auto-pulled on first boot via `docker/ollama/pull_model.sh`, 120s start period)
- **FastAPI** (:8080) — orchestration layer with `/health`, `/query`, `/evaluate`, `/drift/check`, `/xai/check`

FastAPI `depends_on` all three services with `condition: service_healthy`. Source directories (`src/`, `serving/`, `mlops/`) are volume-mounted read-only for dev hot-reload.

**Six code directories and how they connect:**

```
serving/api.py ──→ rag_pipeline.chain.query()        [/query endpoint]
               ──→ evaluation.ragas_runner.run_eval() [/evaluate endpoint]
               ──→ mlops.drift_detector               [/drift/check endpoint]
               ──→ mlops.explanation_monitor            [/xai/check endpoint]

rag_pipeline/chain.py ──→ mlops.mlflow_tracker.RAGOpsTracker  [MLflow logging]
src/chroma_interface.py ──→ rag_pipeline.vectorstore + ingest  [embedding export]
mlops/drift_detector.py ──→ src.config.settings + src.infra.baseline_store
mlops/refresh_trigger.py ──→ data.ingest + evaluation.ragas_runner + mlops.drift_detector
```

- `src/` — infra layer (settings, baseline store, ETL pipeline). Uses `from src.config.settings import settings`.
- `mlops/` — MLflow tracker, PSI drift detector, KB refresh trigger, XAI consistency monitor. Uses `from mlops.mlflow_tracker import RAGOpsTracker`.
- `rag_pipeline/` — RAG chain (vectorstore, ingest, retriever, chain). Uses direct `os.environ` reads and LangChain patterns.
- `evaluation/` — RAGAS runner, ablation study, error analysis, benchmarks. Subdirs: `benchmarks/`, `ablations/`.
- `serving/` — FastAPI orchestration (`api.py`) and Streamlit dashboard (`dashboard.py`).
- `data/` — ETL pipeline interface contracts and data refresh (thin wrappers around `src/chroma_interface`).

## Key Interface Contracts

| Import path | Signature | Returns |
|---|---|---|
| `rag_pipeline.chain.query` | `(question, config=None, mlflow_run=False)` | `{answer, source_docs, retrieval_latency_ms, llm_latency_ms, total_latency_ms, retrieval_scores, config}` |
| `data.ingest.get_embeddings` | `()` | `np.ndarray (n_docs, 384)` |
| `data.ingest.incremental_upsert` | `(new_docs)` | `int` (count of chunks upserted) |
| `evaluation.ragas_runner.run_eval` | `(qa_pairs, config=None, per_question_path=None)` | `{faithfulness, context_recall, answer_relevance, context_precision}` |
| `evaluation.ragas_runner.run_ci_eval` | `(qa_pairs=None, config=None)` | same as `run_eval`, uses 5-question stub if no pairs given |
| `evaluation.explainability.explain` | `(question, source_docs, retrieval_scores, answer=None)` | `{shap_values, token_attributions, term_attribution, explanation_vector, hallucination_risk, hallucination_reason}` |
| `mlops.drift_detector.compute_psi` | `(current_embeddings)` | `float` (mean PSI across dimensions) |
| `mlops.explanation_monitor.compute_consistency` | `(current_vectors=None, baseline_vectors=None)` | `float` (mean cosine similarity 0-1) |

## Build and Run Commands

The `Makefile` is the canonical entrypoint for common workflows. It uses POSIX shell idioms — on Windows run from WSL or Git Bash. CI uses Ubuntu runners.

```bash
# Canonical Makefile targets
make setup          # install Python deps + pull llama3.2:1b
make up             # docker compose up -d
make down           # docker compose down
make healthcheck    # poll /health until green
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy --strict
make test-unit      # pytest unit tier (no docker required)
make test-e2e       # pytest integration tier (needs live stack, RAGOPS_E2E=1)
make eval-ci        # RAGAS 5-question CI smoke eval
make eval-full      # 9-config ablation sweep (hours)
make baseline       # scripts/run_baseline_eval.py — capture week-2 baseline
make drift-check    # POST /drift/check
make xai-check      # POST /xai/check
```

```bash
# Direct invocations (when Make isn't available or for one-offs)

# Start all services (NVIDIA GPU required for Ollama)
cp .env.example .env
docker compose up -d

# Wait for healthy (Ollama pulls ~4.7 GB on first boot)
python scripts/healthcheck.py --timeout 300

# Nuclear reset (deletes all data volumes)
docker compose down -v

# Run unit tests (no Docker needed — uses SQLite fixture). Matches pytest.ini testpaths.
pip install -r docker/fastapi/requirements.txt
MLFLOW_TRACKING_URI=sqlite:///test_mlflow.db ENVIRONMENT=test \
  pytest tests/ src/infra/test mlops/ -m "not integration" -v

# Run a single test file
pytest src/infra/test/test_mlflow_tracker.py -v

# Run integration tier against the live docker-compose stack
RAGOPS_E2E=1 pytest tests/test_e2e.py -m integration -v

# Run with coverage (CI requires 80%)
pytest tests/ src/infra/test mlops/ -m "not integration" \
  --cov=src --cov=mlops --cov-report=term-missing --cov-fail-under=80

# Lint and type check (matches CI)
ruff check src/ mlops/ serving/ evaluation/ --output-format=github
ruff format src/ mlops/ serving/ evaluation/ --check
mypy src/ mlops/ serving/ --ignore-missing-imports --strict --exclude src/infra/test

# Ingest documents into ChromaDB (requires ChromaDB running)
python -m rag_pipeline.ingest --data_dir ./data/pubmed --chunk_size 512

# Smoke-test RAG chain (requires Ollama + ChromaDB running)
python -m rag_pipeline.chain

# Run Streamlit dashboard (requires FastAPI running)
streamlit run serving/dashboard.py
```

## Scripts

- `scripts/healthcheck.py` — polls `/health` until green; used by `make healthcheck` and the Docker compose readiness path.
- `scripts/run_baseline_eval.py` — captures the week-2 PSI / XAI baseline; used by `make baseline`.

## Testing Layout

Tests live in two places — keep the convention when adding new ones:

- **Repo-root `tests/`** — cross-module and integration tests, plus shared fixtures: `test_e2e.py`, `test_rag.py`, `test_explainability.py`, `test_drift_detector.py`, `test_explanation_monitor.py`, `test_compare_runs.py`, `conftest.py`.
- **Module-local `test/` subdirs** — unit tests for that module (e.g., `src/infra/test/test_mlflow_tracker.py`).

`pytest.ini` declares the `integration` marker and pins `testpaths = tests src/infra/test mlops`. Run unit-only with `-m "not integration"`; run the integration tier with `RAGOPS_E2E=1` against the live docker-compose stack.

## CI Pipelines

Two GitHub Actions workflows:

**`.github/workflows/pr_checks.yml`** — runs on PRs to `main`/`develop` and pushes to `develop`. Four jobs:
1. **lint-and-type-check** — ruff check + ruff format + mypy
2. **no-print-guard** — grep-based check that `print()` isn't used in `src/`, `mlops/`, `serving/`
3. **unit-tests** — pytest with coverage (env: `MLFLOW_TRACKING_URI=sqlite:///test_mlflow.db`, `ENVIRONMENT=test`)
4. **docker-build** — validates the FastAPI Dockerfile builds

**`.github/workflows/refresh.yml`** — KB Refresh. Cron Mondays 06:07 UTC + `workflow_dispatch`. Runs `mlops.refresh_trigger.trigger_refresh`: before-eval (RAGAS CI) → PubMed pull + incremental upsert into ChromaDB → re-capture PSI baseline → after-eval (RAGAS CI) → MLflow log. Manual dispatch can post a before/after delta summary on a referenced issue.

## Configuration

All thresholds and service endpoints are configured via `.env` (see `env.example`), read through `src/config/settings.py` (Pydantic `BaseSettings` singleton). Never hardcode thresholds or URLs.

Key thresholds:
- PSI drift: warning at 0.1, alert at 0.25 (triggers KB refresh). `PSI_NUM_BINS=10`.
- XAI consistency: warning below 0.75, alert below 0.60. `XAI_BENCHMARK_QUESTIONS=10`.
- RAGAS quality gates: faithfulness >= 0.80, context_recall >= 0.75, answer_relevancy >= 0.75

Key env vars: `OLLAMA_KEEP_ALIVE=24h` (keeps model in GPU VRAM), `OLLAMA_NUM_PARALLEL=2`, `CHROMA_ALLOW_RESET=false` (true only in dev).

## Code Conventions

- Type hints and docstrings on every public function and class
- `logging` module only — `print()` in `src/`, `mlops/`, `serving/` is caught by CI lint
- Mark novel work: `# PAPER CONTRIBUTION` (PSI drift detection), `# XAI CONTRIBUTION` (SHAP/explanation monitoring), `# ABLATION EXPERIMENT`
- Import order: stdlib, third-party, internal (`from src.config.settings import settings`)
- Settings import: always `from src.config.settings import settings` — never read `os.environ` directly in `src/`
- MLflow tracker import: `from mlops.mlflow_tracker import RAGOpsTracker, RunTrigger`
- `rag_pipeline/` is the exception — it uses direct `os.environ` reads per LangChain patterns
- Tests go in `test/` subdirectory next to the module (e.g., `src/infra/test/test_mlflow_tracker.py`)
- `BaselineStore` is accessed via `get_baseline_store()` singleton factory — never instantiate directly

## Branch Strategy

- `main` — protected, always deployable. PRs only.
- `develop` — integration branch. Feature branches merge here first.
- Feature branches: `<prefix>/<description>` off `develop` (prefixes: `infra/`, `rag/`, `eval/`, `data/`, `refactor/`)
- Hotfixes: branch off `main` with `fix/` prefix, PR to both `main` and `develop`
- Commit format: `<type>: description` (types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`)

## MLflow Usage

`RAGOpsTracker` is a context manager — always use `with tracker.start_run(triggered_by=RunTrigger.CI):`. All metric names are constants on `MetricNames` class, tag names on `TagNames` class. `QualityGateResult.passed` drives CI pass/fail.

Key classes: `RAGOpsTracker`, `RunTrigger` (CI/MANUAL/DRIFT/SCHEDULED), `RAGASMetrics`, `QualityGateResult`, `PSIStatus`, `XAIStatus`, `MetricNames`, `TagNames`.

## Current Status

**Done:** Docker environment, MLflow tracker (with quality gates, PSI/XAI logging), full RAG pipeline (dense/BM25/hybrid retrievers with cross-encoder reranker, chain with score-aware retrieval, document ingestion), PubMed data pipeline (3,706 heart disease abstracts), RAGAS evaluation framework, ablation runner, 50-question benchmark, PSI drift detector, XAI explanation monitor (`evaluation/explainability.py` — SHAP retrieval explainer, attention attribution, hallucination classifier), MLflow run comparison, KB refresh trigger (with weekly `refresh.yml` workflow), FastAPI orchestration (all 5 endpoints wired), Streamlit 5-panel dashboard, Makefile with developer targets, PR-checks CI workflow.

**Not yet implemented:** GitHub Actions RAGAS evaluation quality gates (scaffolded only — the `eval-ci` Make target runs locally, but no CI job blocks PRs on RAGAS thresholds yet).
