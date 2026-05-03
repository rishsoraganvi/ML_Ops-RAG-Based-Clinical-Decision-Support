# RAGOps — End-to-End Pipeline Explainer

> A developer walkthrough of how the RAG-based Clinical Decision Support system is wired together — what each service does, how a request flows from the user to the answer, and where the MLOps guardrails (drift, XAI, quality gates) sit in the loop.

---

## 1. What the Project Is

RAGOps is a **production-grade MLOps framework for a RAG-based clinical decision support system** built over PubMed medical literature (3,706 heart-disease abstracts at time of writing). It couples:

- A **Retrieval-Augmented Generation pipeline** (dense / BM25 / hybrid retrieval + cross-encoder reranker + LLaMA-3.2-3B answer generation)
- **Automated monitoring** — PSI drift detection on embeddings, XAI consistency monitoring on explanations
- **CI/CD quality gates** powered by RAGAS evaluation
- **MLflow tracking** for experiment reproducibility and run comparison

Target research venues: IEEE ICHI and ACL Clinical NLP Workshop. Novel contributions are flagged in code with `# PAPER CONTRIBUTION` (PSI drift) and `# XAI CONTRIBUTION` (explanation monitoring).

---

## 2. Architecture at a Glance

### Four Docker Services on `ragops_net`

| Service  | Image                          | Port  | Role                                              |
|----------|--------------------------------|-------|---------------------------------------------------|
| ChromaDB | `chromadb/chroma:0.5.3`        | 8000  | Vector store; one collection per chunk size       |
| MLflow   | `ghcr.io/mlflow/mlflow:v2.13.0`| 5000  | Experiment tracker (SQLite backend + artifacts)   |
| Ollama   | `ollama/ollama:latest`         | 11434 | LLaMA-3.2-3B generator on NVIDIA GPU              |
| FastAPI  | custom (`docker/fastapi/`)     | 8080  | Orchestration; exposes `/health`, `/query`, `/evaluate`, `/drift/check`, `/xai/check`, `/explain` |

FastAPI has `depends_on: service_healthy` for the other three. Source dirs (`src/`, `serving/`, `mlops/`) are volume-mounted read-only for hot reload.

### Six Code Directories

```
serving/    FastAPI API + Streamlit dashboard  (orchestration layer)
rag_pipeline/  Vectorstore, ingestion, retrievers, chain  (RAG core)
evaluation/    RAGAS runner, ablation study, explainability
mlops/         MLflow tracker, PSI drift, XAI monitor, refresh trigger
src/           Infra: settings, baseline store, ETL primitives
data/          Thin wrappers around src/chroma_interface for ETL contracts
```

**Flow between them:**

```
serving/api.py ──> rag_pipeline.chain.query()        [/query]
              ──> evaluation.ragas_runner.run_eval() [/evaluate]
              ──> mlops.drift_detector               [/drift/check]
              ──> mlops.explanation_monitor          [/xai/check]
              ──> evaluation.explainability.explain  [/explain]

rag_pipeline/chain.py  ──> mlops.mlflow_tracker.RAGOpsTracker
src/chroma_interface.py ──> rag_pipeline.vectorstore + ingest
mlops/drift_detector.py ──> src.config.settings + src.infra.baseline_store
mlops/refresh_trigger.py ──> data.ingest + evaluation.ragas_runner + mlops.drift_detector
```

---

## 3. Data Ingestion Pipeline (PubMed → ChromaDB)

### Call chain

```
data/refresh.py:fetch_new_records()       # pulls new PubMed records
        │
data/ingest.py:incremental_upsert()       # thin wrapper
        │
src/chroma_interface.py:incremental_upsert()   # converts to LangChain Documents
        │
rag_pipeline/ingest.py:ingest_documents()      # chunk → embed → upsert
        │
rag_pipeline/vectorstore.py:get_or_create_collection()
        │
ChromaDB collection `pubmed_{chunk_size}`
```

### Key mechanics

- **PubMed fetch** (`data/refresh.py:29`): weekly refresh across 4 queries (`"clinical practice guidelines"`, `"drug drug interactions"`, `"randomized controlled trial treatment"`, `"differential diagnosis"`), date-filtered over a 7-day window.
- **Chunking** (`rag_pipeline/ingest.py:33`): `RecursiveCharacterTextSplitter` with 10% overlap. Chunk size is specified in tokens (`256 | 512 | 1024`) and converted to characters (`chunk_size * 4`). Separators `["\n\n", "\n", ". ", " ", ""]` reflect medical-text hierarchy.
- **Deterministic IDs** (`rag_pipeline/ingest.py:92`): SHA-256 of `source_doc_id::chunk_index::text` → upsert is idempotent.
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), wrapped as a cached singleton (`rag_pipeline/vectorstore.py:39`).
- **Collections**: one per chunk size — `pubmed_256`, `pubmed_512`, `pubmed_1024` — each configured with `{"hnsw:space": "cosine"}`.
- **Export for drift** (`src/chroma_interface.py:26`): `get_embeddings()` returns an `np.ndarray` of shape `(n_docs, 384)` used by PSI.

---

## 4. Query Flow (User → Answer)

### Call chain

```
POST /query  (serving/api.py:150)
      │
asyncio.to_thread(rag_pipeline.chain.query, …)
      │
rag_pipeline/chain.py:query()
      ├── query_processor.preprocess_query()   # expand medical abbreviations
      ├── retriever.retrieve_with_scores()     # dense | bm25 | hybrid
      ├── ChatOllama(llama3.2:3b, temperature=0.0)  # generation
      └── _log_to_mlflow()  (optional)
```

### Step-by-step

1. **Query preprocessing** (`rag_pipeline/query_processor.py:54`): expands 16 medical abbreviations (`MI → myocardial infarction`, `HF → heart failure`, `T2DM → type 2 diabetes mellitus`, …) case-insensitively and idempotently. Example: `"T2DM treatment" → "T2DM treatment (type 2 diabetes mellitus)"`.

2. **Retrieval** (`rag_pipeline/retriever.py:429 retrieve_with_scores`) — three modes, all scored:
   - **Dense** (line 480): embed question → `collection.query(...)` → cosine similarity via `score = 1.0 - (distance / 2.0)`.
   - **BM25** (line 460): cached BM25Okapi index per chunk size (`_get_bm25_components` is `@lru_cache(maxsize=8)`) → unnormalized term-weighted score.
   - **Hybrid** (`hybrid_retrieve`, line 253): fetch `k*4=20` candidates from each path → **weighted Reciprocal Rank Fusion** (dense 0.6, BM25 0.4) → **cross-encoder rerank** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) → top-k.

3. **Prompt assembly** (`rag_pipeline/chain.py:34`): `CLINICAL_PROMPT` is a system message instructing the LLM to *use ONLY the provided context*, followed by the joined `page_content` of retrieved docs.

4. **Generation**: `ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)` — where `OLLAMA_MODEL` defaults to `llama3.2:3b`; deterministic for ablations; latency measured via `time.perf_counter()`.

5. **Response** (`QueryResponse`):
   ```json
   {
     "answer": "...",
     "source_docs": [{"page_content": "...", "metadata": {...}}],
     "retrieval_latency_ms": 45.2,
     "llm_latency_ms": 1234.5,
     "total_latency_ms": 1279.7,
     "retrieval_scores": [0.92, 0.81, ...],
     "config": {"retriever_type": "dense", "chunk_size": 256, "k": 5, ...}
   }
   ```

6. **Optional MLflow logging**: if `log_to_mlflow=true`, the run is recorded with params + latencies + Q/A.

---

## 5. Evaluation Pipeline (RAGAS + Quality Gates)

### Call chain

```
POST /evaluate  (serving/api.py:261)
      │
_run_eval_sync()
      ├── evaluation.ragas_runner.run_ci_eval()   # 5-question smoke (CI)
      │   or run_eval()                           # full eval
      └── RAGOpsTracker.log_quality_gate_results()
```

### Metrics and gates

`evaluation/ragas_runner.py` wraps the **RAGAS** library with four metrics:

| Metric              | Threshold | Source                                          |
|---------------------|-----------|-------------------------------------------------|
| `faithfulness`      | ≥ 0.80    | Answer grounded in retrieved context            |
| `context_recall`    | ≥ 0.75    | Retrieved context contains the ground truth     |
| `answer_relevance`  | ≥ 0.75    | Answer matches the question                     |
| `context_precision` | —         | Ranked order of relevant vs irrelevant chunks   |

- **Three-tier judge deployment** (`OLLAMA_MODEL` env): `phi3:mini` for the CI smoke gate (lightweight, JSON-friendly); `llama3:8b` for the 9-cell ablation sweep, since `faithfulness` and `context_recall` decompose answers into atomic statements + run per-statement NLI and produce malformed JSON on smaller judges; `llama3.2:3b` for production `/query`. The embedding-side metrics use `nomic-embed-text` (`OLLAMA_EMBED_MODEL`).
- `run_ci_eval` caps at **3 questions** and 90 s wall-clock (`CI_SAMPLE_SIZE=3`, `CI_TIMEOUT_SECS=90`).
- **Hardening** (`evaluation/ragas_runner.py`): RAGAS runs sequentially (`RunConfig(max_workers=1, timeout=600, max_retries=3)`); the Ollama judge is forced to `format="json"` with `temperature=0`, `num_predict=2048`, `num_ctx=2048`; the runner captures RAGAS / LangChain warnings during `evaluate()` and emits a structured warning on detected parse failures (`OutputParserException`, `Failed to parse`, `JSONDecodeError`, `invalid json`). Per-metric scores use `nanmean`; if a metric column is missing or entirely NaN the runner logs a warning naming the metric and the model and falls back to NaN — a single flaky judge call no longer discards the rest of a 25-min ablation cell. NaN scores are stripped before being sent to MLflow (it rejects NaN floats), and the quality gate (`mlops/mlflow_tracker.log_quality_gate_results`) marks any missing/NaN metric as a failure rather than raising — so CI/operators see exactly which metric the judge failed on.
- Every run logs to MLflow with tags: `run_tag`, `config_hash` (8-char SHA1 of config JSON), `ollama_model`, `n_questions`.
- **QualityGateResult** (`mlops/mlflow_tracker.py`) is a dataclass with `passed: bool` + `failures: dict[str, str]` that drives CI pass/fail.

---

## 6. Drift Detection — PSI (the "paper contribution")

### Call chain

```
POST /drift/check  (serving/api.py:291)
      │
data.ingest.get_embeddings()                     # current (n, 384) matrix
mlops.drift_detector.compute_psi(current)        # per-dim PSI, mean
mlops.drift_detector.alert(psi_score)            # stable / warning / alert
```

### How PSI works here

`mlops/drift_detector.py`:

1. **Baseline capture** (`capture_baseline`, line 33): after initial ingestion, the full embedding matrix is saved via `get_baseline_store().save_psi_baseline(embeddings)`.
2. **Per-dimension PSI** (`_psi_per_dimension`, line 56): for each of 384 dims, histogram both baseline & current into `PSI_NUM_BINS=10` bins of the baseline's edges, then
   ```
   PSI_d = Σ (actual − expected) · ln(actual / expected)
   ```
   (with `epsilon = 1e-6` to avoid `log(0)`).
3. **Mean PSI** (`compute_psi`, line 93): averages across all 384 dims.
4. **Alert thresholds** (`alert`, line 143):
   - `≥ 0.25` → `"alert"` → action `"trigger_refresh"`
   - `≥ 0.10` → `"warning"` → action `"monitor"`
   - `< 0.10` → `"stable"` → action `"none"`

### Baseline storage

`src/infra/baseline_store.py` defines an abstract `BaselineStore` with PSI and XAI baseline slots. The current implementation is `InMemoryBaselineStore` (non-persistent — logs a WARNING on startup). The factory `get_baseline_store()` is the only swap point needed to move to an MLflow- or volume-backed store.

---

## 7. XAI — Explanation Consistency Monitor

### Call chain

```
POST /xai/check  (serving/api.py:332)
      │
mlops.explanation_monitor.compute_consistency(current_vectors)
```

### How it works

`mlops/explanation_monitor.py`:

- An **explanation vector** (dim = 20) is produced per query by `evaluation/explainability.py:build_explanation_vector` — top-5 SHAP values over source docs + 5 sentences × 3 attention spans.
- `compute_consistency(current_vectors)` loads the baseline set from `BaselineStore.load_xai_baseline()` and returns the **mean pairwise cosine similarity** between current and baseline vectors.
- Thresholds (from `src/config/settings.py`):
  - `≥ 0.75` → `"stable"`
  - `≥ 0.60` → `"warning"`
  - `< 0.60` → `"instability"` (WARNING logged)

### The `/explain` endpoint (detailed XAI)

`evaluation/explainability.py:explain()` bundles five outputs:

| Key                     | What it is                                                    |
|-------------------------|---------------------------------------------------------------|
| `shap_values`           | Per-doc SHAP attribution (KernelExplainer, 50 samples)        |
| `token_attributions`    | 5 answer-sentence × 3-span attention saliency over context    |
| `term_attribution`      | BM25 term scores (global + per-doc)                           |
| `explanation_vector`    | 20-dim fused summary (top SHAP + attention)                   |
| `hallucination_risk`    | Logistic-regression probability (with rule-based fallback)    |
| `hallucination_reason`  | SHAP or rule-based string explanation                         |

The hallucination classifier loads from `mlops/artifacts/hallucination_clf.joblib` if present; otherwise falls back to a rule-based score over `(mean_retrieval_score, min_retrieval_score, shap_entropy, num_source_docs, answer_len_tokens, top_attention_score)`.

---

## 8. MLflow Tracking Layer

`mlops/mlflow_tracker.py` centralizes tracking:

- **`RAGOpsTracker`** — context-manager wrapper around MLflow runs.
  ```python
  with tracker.start_run(triggered_by=RunTrigger.CI) as run:
      tracker.log_params(config)
      tracker.log_ragas_metrics(metrics)
      result = tracker.log_quality_gate_results(metrics)
  ```
- **`RunTrigger`** enum: `CI`, `MANUAL`, `DRIFT`, `SCHEDULED`.
- **`MetricNames` / `TagNames`** — string constants so metric names never drift across call sites (`faithfulness`, `context_recall`, `answer_relevancy`, `psi_score`, `explanation_consistency_score`, `quality_gate_passed`, …).
- **`PSIStatus` / `XAIStatus`** — string enums logged as tags.
- **`QualityGateResult`** — `{passed, failures, run_id}`.

All code must import via `from mlops.mlflow_tracker import RAGOpsTracker, RunTrigger` — no direct `mlflow.start_run()` calls.

---

## 9. KB Refresh Trigger (closes the loop)

When PSI hits the `alert` threshold, `mlops/refresh_trigger.py:trigger_refresh()` runs a five-step cycle:

1. **Before metrics** — `run_ci_eval()` captures baseline quality.
2. **Fetch new records** — `data.refresh.fetch_new_records(days_back=7)`.
3. **Upsert** — `data.ingest.incremental_upsert(new_records)` adds them to ChromaDB.
4. **Re-baseline** — pulls fresh embeddings and calls `capture_baseline(...)` to reset PSI.
5. **After metrics** — second `run_ci_eval()` to quantify the refresh's impact.

Everything is wrapped in an MLflow run tagged `triggered_by=RunTrigger.DRIFT`, with the drift report logged as params.

---

## 10. CI/CD (`.github/workflows/pr_checks.yml`)

Five jobs run on PRs to `main` / `develop`:

1. **lint-and-type-check** — `ruff check`, `ruff format --check`, `mypy --strict` over `src/ mlops/ serving/ evaluation/`.
2. **no-print-guard** — grep-based reject of `print()` in `src/`, `mlops/`, `serving/`. Use the `logging` module instead.
3. **unit-tests** — `pytest` with coverage gate `--cov-fail-under=80` (env: `MLFLOW_TRACKING_URI=sqlite:///test_mlflow.db`, `ENVIRONMENT=test`).
4. **docker-build** — validates the FastAPI Dockerfile builds.
5. **ragas-quality-gate** (main/develop only) — spins up the full compose stack; `docker/ollama/pull_model.sh` pulls `phi3:mini` and `nomic-embed-text` *inside* the Ollama container (driven by `OLLAMA_MODEL` / `OLLAMA_EMBED_MODEL` set in `docker-compose.ci.yml`); runs `run_ci_eval()` (3 questions, sequential, strict JSON, NaN-fatal) with looser PR thresholds (`faithfulness ≥ 0.70`, `context_recall ≥ 0.65`) for velocity.

---

## 11. Streamlit Dashboard (`serving/dashboard.py`)

Five panels, all backed by the FastAPI `/…` endpoints and MLflow client:

1. **Query** — issue a question, pick retriever / chunk size / k / MLflow-log toggle; shows answer + latencies + source docs.
2. **XAI Explainability** — explains the last query (SHAP bars, term attribution, hallucination risk, attention heatmap).
3. **Live RAGAS Metrics** — plots recent MLflow runs' faithfulness / recall / relevance over time (Plotly).
4. **Drift Alert Panel** — traffic lights on PSI and XAI consistency.
5. **Experiment Comparison** — diff two MLflow runs side-by-side.

Env targets: `API_BASE=http://localhost:8080`, `MLFLOW_TRACKING_URI=http://localhost:5000`.

---

## 12. Configuration & Thresholds (`src/config/settings.py`)

All thresholds and service URLs live in a pydantic `BaseSettings` singleton. Never hardcode them in code.

| Setting                         | Default                    | Used by                  |
|---------------------------------|----------------------------|--------------------------|
| `chroma_host` / `chroma_port`   | `chromadb` / `8000`        | All of `rag_pipeline/`   |
| `mlflow_tracking_uri`           | `http://mlflow:5000`       | `RAGOpsTracker`, runner  |
| `ollama_base_url` / `ollama_model` | `http://ollama:11434` / `llama3.2:3b` | Chain + RAGAS judge |
| `psi_warning_threshold`         | `0.10`                     | `drift_detector.alert`   |
| `psi_alert_threshold`           | `0.25`                     | triggers KB refresh      |
| `psi_num_bins`                  | `10`                       | PSI histograms           |
| `xai_warning_threshold`         | `0.75`                     | consistency stable       |
| `xai_alert_threshold`           | `0.60`                     | instability warning      |
| `xai_benchmark_questions`       | `10`                       | XAI benchmark sweep      |
| `ragas_faithfulness_min`        | `0.80`                     | quality gate             |
| `ragas_context_recall_min`      | `0.75`                     | quality gate             |
| `ragas_answer_relevancy_min`    | `0.75`                     | quality gate             |

Ollama keeps the model resident in GPU VRAM via `OLLAMA_KEEP_ALIVE=24h`, with `OLLAMA_NUM_PARALLEL=2`.

---

## 13. Endpoint Reference (FastAPI)

| Method | Path           | Purpose                                                |
|--------|----------------|--------------------------------------------------------|
| GET    | `/health`      | Liveness + dependency status (chromadb/mlflow/ollama)  |
| POST   | `/query`       | RAG question → answer with source docs + latencies    |
| POST   | `/evaluate`    | RAGAS scoring (CI 5-q stub or full) + quality gate    |
| POST   | `/drift/check` | PSI over current ChromaDB embeddings + alert status   |
| POST   | `/xai/check`   | Explanation-consistency score vs XAI baseline         |
| POST   | `/explain`     | Full explainability payload for a query result         |

Request/response shapes are in `serving/api.py` via pydantic models — `QueryRequest`, `QueryResponse`, `EvalRequest`, `EvalResponse`, `DriftResponse`, `XAIRequest`, `XAIResponse`, `ExplainRequest`, `ExplainResponse`.

---

## 14. End-to-End: A Day in the Life of a Query

Putting it together for a single user interaction:

1. Clinician asks the dashboard: *"First-line treatment for T2DM?"*
2. Dashboard `POST /query` → FastAPI.
3. `preprocess_query` expands to *"First-line treatment for T2DM (type 2 diabetes mellitus)?"*.
4. Hybrid retrieval fires: 20 dense candidates + 20 BM25 candidates → RRF fuse → cross-encoder rerank → top-5.
5. Top-5 chunks become the context; `llama3.2:3b` generates the answer at `temperature=0.0`.
6. Response returned with `retrieval_scores` and latencies; optionally logged to MLflow.
7. Clinician clicks **Explain** → `/explain` returns SHAP / attention / term attribution + hallucination risk.
8. In the background, a scheduled `/drift/check` computes mean PSI. If `≥ 0.25`, `refresh_trigger.trigger_refresh()` pulls the last week's PubMed abstracts, upserts them, re-baselines PSI, and logs an MLflow run tagged `DRIFT`.
9. On every PR, CI runs `run_ci_eval()` against 5 canary questions; if quality gates break, the PR is blocked.

---

## 15. Where to Look for What

| If you want to…                              | Start here                                              |
|---------------------------------------------|---------------------------------------------------------|
| Change how docs are chunked                 | `rag_pipeline/ingest.py:33` (`get_text_splitter`)       |
| Add a new retriever                         | `rag_pipeline/retriever.py:364` (`get_retriever`)       |
| Tweak the answer prompt                     | `rag_pipeline/chain.py:34` (`CLINICAL_PROMPT`)          |
| Add a RAGAS metric                          | `evaluation/ragas_runner.py:122` (`_METRICS` list)      |
| Change PSI thresholds                       | `src/config/settings.py` + `docker-compose.yml` env     |
| Persist baselines across restarts           | Implement `BaselineStore` and swap in `get_baseline_store` (`src/infra/baseline_store.py:178`) |
| Add an MLflow metric/tag                    | `mlops/mlflow_tracker.py:87` (`MetricNames`, `TagNames`)|
| Add a dashboard panel                       | `serving/dashboard.py`                                  |
| Gate a new CI check                         | `.github/workflows/pr_checks.yml`                       |

---

## 16. Current Status (from `CLAUDE.md`)

**Done**: Docker environment, MLflow tracker with quality gates + PSI/XAI logging, full RAG pipeline (dense/BM25/hybrid + reranker), PubMed data pipeline (3,706 heart-disease abstracts), RAGAS evaluation + ablation runner + 50-question benchmark, CI scaffold, PSI drift detector, XAI explanation monitor, MLflow run comparison, KB refresh trigger, FastAPI orchestration (all endpoints wired), Streamlit 5-panel dashboard.

**Not yet complete**: `evaluation/explainability.py` SHAP + attention layer is implemented but integration with baseline XAI monitoring needs end-to-end wiring; GitHub Actions RAGAS quality gate is scaffolded only; Makefile not present.

---

*For API schemas and exact parameter types, read the pydantic models in `serving/api.py`. For exact interface contracts (return shapes), see the "Key Interface Contracts" table in `CLAUDE.md`.*
