# RAGOps — Project Progress Tracker

**Project:** Production-grade MLOps framework for RAG pipelines applied to Clinical Decision Support  
**Stack:** Python 3.11 | Docker + Docker Compose | MLflow (SQLite) | GitHub Actions | ChromaDB | Ollama LLaMA-3-8B | RAGAS | NumPy/SciPy | FastAPI  
**Last Updated:** 2026-03-23

---

## Team Interface Contracts

> These are the fixed API boundaries between members. Do not change without cross-team sync.

| Owner | Function Signature | Returns |
|---|---|---|
| Member 2 | `rag_pipeline.chain.query(question)` | `{answer, source_docs, retrieval_latency_ms}` |
| Member 3 | `evaluation.ragas_runner.run_eval(qa_pairs)` | `{faithfulness, context_recall, ...}` |
| Member 3 | `evaluation.explainability.explain(q, docs)` | `{shap_values, explanation_vector, ...}` |
| Member 4 | `data.ingest.get_embeddings()` | `np.ndarray (n_docs, 384)` |

---

## Component Status

| # | Component | Status | Session Date | Files Delivered |
|---|---|---|---|---|
| 1 | Docker Environment | ✅ Done | 2026-03-23 | See below |
| 2 | MLflow Experiment Tracking | ✅ Done | 2026-03-23 | See below |
| 3 | GitHub Actions CI/CD + RAGAS Quality Gates | 🔲 Not Started | — | — |
| 4 | PSI-based Embedding Drift Detection | 🔲 Not Started | — | — |
| 5 | Automated KB Refresh Trigger | 🔲 Not Started | — | — |
| 6 | XAI Consistency Monitoring (SHAP) | 🔲 Not Started | — | — |

---

## Component 1 — Docker Environment ✅

**Completed:** 2026-03-23  
**Configuration chosen:** NVIDIA GPU · SQLite MLflow · All volumes persisted

### Files Delivered

```
ragops/
├── docker-compose.yml                  # Orchestrates all 4 services
├── .env.example                        # All tunables documented
├── .gitignore
├── README.md                           # Architecture diagram + quick-start
├── docker/
│   ├── fastapi/
│   │   ├── Dockerfile                  # Non-root, slim Python 3.11
│   │   ├── requirements.txt            # All deps pinned to major.minor.*
│   │   └── src/
│   │       └── main.py                 # FastAPI skeleton + /health endpoint
│   └── ollama/
│       └── pull_model.sh               # Idempotent LLaMA-3 pull w/ retry
└── scripts/
    └── healthcheck.py                  # Polls all 4 services; --timeout for CI
```

### Services & Ports

| Service | Internal DNS | Host Port | Volume |
|---|---|---|---|
| ChromaDB | `chromadb:8000` | 8000 | `chroma_data` |
| MLflow | `mlflow:5000` | 5000 | `mlflow_data` |
| Ollama | `ollama:11434` | 11434 | `ollama_data` |
| FastAPI | `fastapi:8080` | 8080 | _(src mount in dev)_ |

### Key Design Decisions
- GPU passthrough uses `deploy.resources.reservations` (not deprecated `runtime: nvidia`)
- `depends_on: condition: service_healthy` — FastAPI waits for all 3 upstream health checks
- Ollama health check has `start_period: 120s` to accommodate cold LLaMA-3 download (~4.7 GB)
- All PSI/XAI thresholds injected via `.env` so CI pipeline can override without code changes

### Host Prerequisites (must verify before `docker compose up`)
- [ ] NVIDIA Container Toolkit installed (`nvidia-container-toolkit`)
- [ ] Docker Compose v2 (`docker compose` not `docker-compose`)
- [ ] `cp .env.example .env` and fill in values
- [ ] `nvidia-smi` confirms GPU is visible to host

---

## Component 2 — MLflow Experiment Tracking ✅

**Completed:** 2026-03-23  
**Configuration chosen:** One run per full eval cycle · Tags: `chroma_collection`, `triggered_by`, `ollama_model`, `environment` · Baseline storage stubbed (`InMemoryBaselineStore`)

### Files Delivered

```
src/
├── config/
│   └── settings.py                  # Centralised Pydantic settings (single import everywhere)
├── infra/
│   ├── __init__.py
│   ├── baseline_store.py            # Abstract BaselineStore + InMemoryBaselineStore stub
│   ├── mlflow_tracker.py            # Core tracker — context manager, all log_* methods
│   └── tests/
│       └── test_mlflow_tracker.py   # 18 unit tests (no Docker needed — SQLite fixture)
```

### MLflow Run Schema (implemented)

| Parameter logged via `log_params()` | Type | Notes |
|---|---|---|
| `psi_num_bins` | int | From `.env` |
| `xai_benchmark_questions` | int | From `.env` |
| `embed_model` | string | Caller supplies |

| Metric | Key constant | Source |
|---|---|---|
| Faithfulness | `MetricNames.FAITHFULNESS` | Member 3 RAGAS |
| Context recall | `MetricNames.CONTEXT_RECALL` | Member 3 RAGAS |
| Answer relevancy | `MetricNames.ANSWER_RELEVANCY` | Member 3 RAGAS |
| Retrieval latency | `MetricNames.RETRIEVAL_LATENCY_MS` | Member 2 chain |
| PSI score | `MetricNames.PSI_SCORE` | Component 4 |
| XAI consistency | `MetricNames.EXPLANATION_CONSISTENCY_SCORE` | Component 6 |
| Quality gate | `MetricNames.QUALITY_GATE_PASSED` | `1.0` / `0.0` |

| Tag | Key constant | Values |
|---|---|---|
| Triggered by | `TagNames.TRIGGERED_BY` | `ci \| manual \| drift \| scheduled` |
| Chroma collection | `TagNames.CHROMA_COLLECTION` | from `.env` |
| Ollama model | `TagNames.OLLAMA_MODEL` | from `.env` |
| Environment | `TagNames.ENVIRONMENT` | from `.env` |
| PSI status | `TagNames.PSI_STATUS` | `stable \| warning \| alert` |
| XAI status | `TagNames.XAI_STATUS` | `stable \| warning \| instability` |

### Key Design Decisions
- `RAGOpsTracker.start_run()` is a context manager — run always ends (FINISHED or FAILED) even on exception
- `_get_or_create_experiment()` handles race condition between parallel CI jobs
- `QualityGateResult` dataclass returned from `log_quality_gate_results()` — CI reads `result.passed` without re-querying MLflow
- `BaselineStore` is abstract — swap `InMemoryBaselineStore` for `MLflowBaselineStore` / `VolumeBaselineStore` in one line (factory pattern)
- All metric key names are string constants on `MetricNames` — no raw strings anywhere

### How to run tests (no Docker)
```bash
pip install mlflow pytest numpy pydantic-settings
pytest src/infra/tests/test_mlflow_tracker.py -v
```

---

## Component 3 — GitHub Actions CI/CD + RAGAS Quality Gates 🔲

**Status:** Not started  
**Depends on:** Components 1 + 2

### What needs to be built
- `.github/workflows/ragas_eval.yml`
- Spins up Docker Compose in CI (GitHub-hosted runner, CPU mode for Ollama)
- Runs `evaluation.ragas_runner.run_eval()` against benchmark QA pairs
- Reads RAGAS scores from MLflow (or direct return value)
- Fails pipeline if any metric is below threshold

### RAGAS Quality Gate Thresholds (from `.env.example`)

| Metric | Minimum |
|---|---|
| `faithfulness` | 0.80 |
| `context_recall` | 0.75 |
| `answer_relevancy` | 0.75 |

### Notes for next session
- Ollama on GitHub-hosted runners is CPU only — set `OLLAMA_MODEL=llama3:8b` and expect slow inference; consider a smaller model or mocked LLM for CI speed
- Healthcheck script (`scripts/healthcheck.py --timeout 300`) should gate the eval step
- Docker layer caching via `actions/cache` on `/var/lib/docker` will significantly cut build time

---

## Component 4 — PSI-based Embedding Drift Detection 🔲  `# PAPER CONTRIBUTION`

**Status:** Not started  
**Depends on:** Components 1 + 2  
**Input:** `data.ingest.get_embeddings()` → `np.ndarray (n_docs, 384)`

### What needs to be built
- `src/monitoring/psi_detector.py`
- Compute per-dimension histograms over embedding matrix
- Store baseline distribution after first run (in MLflow or ChromaDB metadata)
- Compute PSI score on subsequent runs
- Log `psi_score` + `psi_status` (`stable` | `warning` | `alert`) to MLflow
- Return trigger flag consumed by Component 5

### PSI Formula
```
PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)
```
across `PSI_NUM_BINS` histogram bins (default: 10, set in `.env`)

### Thresholds

| PSI Range | Status | Action |
|---|---|---|
| `< 0.1` | `stable` | Log only |
| `0.1 – 0.25` | `warning` | Log warning to MLflow |
| `> 0.25` | `alert` | Trigger KB refresh (Component 5) |

### Notes for next session
- Baseline must be persisted between CI runs — MLflow artifact or a JSON file in `mlflow_data` volume
- Handle zero-count bins carefully (add small epsilon before `ln()` to avoid `log(0)`)
- Consider averaging PSI across embedding dimensions vs. computing on PCA-reduced 1D projection — decide before building

---

## Component 5 — Automated KB Refresh Trigger 🔲

**Status:** Not started  
**Depends on:** Components 1 + 4

### What needs to be built
- `src/monitoring/refresh_trigger.py`
- Receives PSI trigger flag from Component 4
- Calls Member 4's ingest pipeline to re-embed updated documents
- Clears and repopulates the ChromaDB collection
- Logs `kb_refresh_triggered=True` + timestamp to MLflow
- Should be idempotent (safe to call multiple times)

### Notes for next session
- ChromaDB collection reset requires `CHROMA_ALLOW_RESET=true` in `.env` — only safe in dev; in prod, prefer creating a new collection and hot-swapping
- Coordinate with Member 4 on whether `get_embeddings()` also handles ingest or just returns vectors

---

## Component 6 — XAI Consistency Monitoring (SHAP) 🔲  `# XAI CONTRIBUTION`

**Status:** Not started  
**Depends on:** Components 1 + 2  
**Input:** `evaluation.explainability.explain(q, docs)` → `{shap_values, explanation_vector, ...}`

### What needs to be built
- `src/monitoring/xai_monitor.py`
- After first eval run: store baseline `explanation_vector` for each of the 10 CI benchmark questions
- On subsequent runs: compute cosine similarity between current and baseline vectors
- Aggregate to `explanation_consistency_score` = mean cosine similarity across all 10 questions
- Log score + status to MLflow
- Emit warning / alert based on thresholds

### Thresholds

| Score Range | Status | Action |
|---|---|---|
| `>= 0.75` | `stable` | Log only |
| `0.60 – 0.75` | `warning` | Log warning to MLflow |
| `< 0.60` | `alert` | Explanation instability alert |

### Cosine Similarity Formula
```
cosine_sim(a, b) = (a · b) / (||a|| × ||b||)
```
Use `scipy.spatial.distance.cosine` (returns distance; subtract from 1 for similarity)

### Notes for next session
- Baseline vectors must persist between runs — store as MLflow artifact (JSON or `.npy`)
- 10 benchmark questions should be fixed and version-controlled alongside the eval suite (coordinate with Member 3)
- `explanation_vector` dtype and shape must be confirmed with Member 3 before building

---

## Global Notes & Conventions

### Code Conventions (apply to all components)
- Type hints + docstrings on **every** function and class
- Use `logging` module — **never** `print()`
- Mark paper contributions: `# PAPER CONTRIBUTION`
- Mark XAI work: `# XAI CONTRIBUTION`

### Threshold Reference (all configurable via `.env`)

| Variable | Default | Component |
|---|---|---|
| `PSI_WARNING_THRESHOLD` | `0.1` | 4 |
| `PSI_ALERT_THRESHOLD` | `0.25` | 4 |
| `XAI_WARNING_THRESHOLD` | `0.75` | 6 |
| `XAI_ALERT_THRESHOLD` | `0.60` | 6 |
| `RAGAS_FAITHFULNESS_MIN` | `0.80` | 3 |
| `RAGAS_CONTEXT_RECALL_MIN` | `0.75` | 3 |
| `RAGAS_ANSWER_RELEVANCY_MIN` | `0.75` | 3 |

### Embedding Dimension
All embedding vectors are **384-dimensional** (`all-MiniLM-L6-v2` via `sentence-transformers`). Components 4 and 6 must both handle `(n, 384)` shaped arrays.
