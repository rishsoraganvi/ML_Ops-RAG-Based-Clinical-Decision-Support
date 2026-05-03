# RAGOps: Solo Execution Plan & Guide
### Project: Production-Grade MLOps for RAG-Based Clinical Decision Support
> **Note:** This is a restructured solo version of the original team plan. All four roles have been consolidated into a single execution timeline for one person.

---

## Quick Project Summary

**What you're building:** A production MLOps system around a Retrieval-Augmented Generation (RAG) pipeline that answers clinical/medical questions from PubMed literature. The core novelty ("RAGOps") is automated monitoring, drift detection, CI/CD, and Explainable AI (XAI) specifically designed for RAG systems in clinical settings.

| Component | Technology |
|-----------|------------|
| Dataset | PubMed abstracts (NCBI API, open access) |
| LLM | LLaMA-3-8B via Ollama (runs locally — free) |
| Vector DB | ChromaDB |
| Evaluation | RAGAS framework |
| XAI | SHAP + attention weight attribution + explanation consistency |
| Tracking | MLflow |
| CI/CD | GitHub Actions |
| Serving | FastAPI + Streamlit |
| Language | Python 3.11 throughout |

---

## Repo Structure

```
ragops/
├── data/
│   ├── ingest.py                  # Team interface (get_embeddings + upsert)
│   ├── refresh.py                 # Weekly PubMed data refresh
│   ├── quality_report.py          # Data quality checks → quality_report.html
│   ├── stats.py                   # Dataset statistics (Table 0)
│   ├── processed/
│   │   └── pubmed_processed.jsonl # 3,706 records
│   └── chroma_store/              # Local ChromaDB index
├── src/
│   ├── config/
│   │   └── settings.py            # Pydantic BaseSettings singleton
│   ├── infra/
│   │   ├── baseline_store.py      # PSI/XAI baseline persistence
│   │   └── test/
│   │       └── test_mlflow_tracker.py
│   ├── fetcher.py                 # PubMed NCBI E-utilities (esearch + efetch)
│   ├── parser.py                  # XML → JSONL parser
│   ├── chroma_interface.py        # ChromaDB interface (embeddings + upsert)
│   └── run_pipeline.py            # Main ETL orchestrator
├── rag_pipeline/
│   ├── vectorstore.py             # ChromaDB setup + embedding function
│   ├── ingest.py                  # Document chunking + embedding
│   ├── retriever.py               # Dense, BM25, hybrid retriever factory
│   └── chain.py                   # Unified query() API
├── evaluation/
│   ├── ragas_runner.py            # RAGAS evaluation (run_eval + run_ci_eval)
│   ├── benchmarks/
│   │   ├── qa_pairs.json          # 50-question benchmark
│   │   └── ci_benchmark.json      # 10-question CI subset
│   ├── ablations/
│   │   ├── run_ablations.py       # 9-config ablation runner
│   │   ├── error_analysis.py      # Error categorization
│   │   └── significance_test.py   # Statistical testing
│   └── explainability.py          # XAI layer (Phase 3)
├── mlops/
│   ├── mlflow_tracker.py          # RAGOpsTracker context manager
│   ├── drift_detector.py          # PSI drift detection
│   ├── refresh_trigger.py         # Auto KB refresh on drift
│   ├── explanation_monitor.py     # XAI consistency tracking
│   └── compare_runs.py            # MLflow run comparison
├── serving/
│   ├── api.py                     # FastAPI orchestration layer
│   └── dashboard.py               # 5-panel Streamlit dashboard
├── docker/
│   ├── fastapi/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── ollama/
│       └── pull_model.sh
├── scripts/
│   └── healthcheck.py
├── .github/workflows/
│   └── pr_checks.yml
├── docker-compose.yml
├── env.example
└── tests/                         # (planned: test_rag.py, test_e2e.py)
```

---
---

# ⚡ SOLO EXECUTION GUIDE

> This guide tells you **what to build, in what order, and why** — with explicit dependencies called out so you never block yourself.

---

## Phase Overview

```
Phase 1 (Week 1): Foundation — Infra + Data Pipeline           ✅ COMPLETE
Phase 2 (Week 2): Core Pipeline — RAG Chain + Evaluation        ✅ COMPLETE
Phase 3 (Week 3): Science — Experiments + XAI Layer + Drift     🔄 IN PROGRESS
Phase 4 (Week 4): Polish — Integration + Dashboard + Paper      🔄 IN PROGRESS
```

Each phase depends on the previous one. Do **not** skip ahead.

---

## Progress Summary (as of 12 April 2026)

| Milestone | Status | Notes |
|-----------|--------|-------|
| Repo + Docker environment | ✅ Done | All 4 services configured |
| PubMed ETL pipeline | ✅ Done | 3,706 heart disease records (English, 2021–2026) |
| Data quality + stats | ✅ Done | `quality_report.html` generated, 0 duplicates |
| ChromaDB ingestion | ✅ Done | 3,706 docs indexed, embeddings (3706, 384) |
| Interface contracts (data layer) | ✅ Done | `get_embeddings()` + `incremental_upsert()` |
| Weekly data refresh | ✅ Done | `refresh.py` for heart disease records |
| MLflow tracker | ✅ Done | `RAGOpsTracker` with quality gates |
| RAG chain (`query()`) | ✅ Done | `rag_pipeline/chain.py` |
| Dense retriever | ✅ Done | `rag_pipeline/retriever.py` |
| BM25 / Hybrid / Reranker | ✅ Done | `retriever.py` — BM25Okapi, RRF hybrid fusion, cross-encoder reranker |
| RAGAS evaluation framework | ✅ Done | `evaluation/ragas_runner.py` — `run_eval()` + `run_ci_eval()` |
| CI/CD with RAGAS gates | 🔜 Pending | Scaffolded only, RAGAS eval not wired to GH Actions yet |
| Ablation study (9 configs) | 🔜 Pending | Runner exists (`run_ablations.py`), results not yet generated |
| XAI / Explainability layer | 🔜 Pending | `evaluation/explainability.py` not yet created |
| PSI drift detection | ✅ Done | `mlops/drift_detector.py` — `capture_baseline()`, `compute_psi()`, `alert()`, `monitor_generation()` |
| XAI consistency monitor | ✅ Done | `mlops/explanation_monitor.py` — `compute_consistency()` |
| KB refresh trigger | ✅ Done | `mlops/refresh_trigger.py` — `trigger_refresh()` |
| MLflow run comparison | ✅ Done | `mlops/compare_runs.py` — `compare_runs()` |
| FastAPI integration | ✅ Done | `serving/api.py` — all 5 endpoints wired (`/health`, `/query`, `/evaluate`, `/drift/check`, `/xai/check`) |
| Streamlit dashboard | ✅ Done | `serving/dashboard.py` — 5-panel (Query, XAI, RAGAS, Drift, Comparison) |
| `chain.py` → RAGOpsTracker | ✅ Done | Replaced raw `mlflow` with `RAGOpsTracker` in `_log_to_mlflow()` |
| `chroma_interface.py` stubs | ✅ Done | `get_embeddings()` + `incremental_upsert()` now backed by `rag_pipeline` |
| Baseline store singleton fix | ✅ Done | `get_baseline_store()` now returns singleton |

---

## Phase 1 — Foundation (Week 1) ✅ COMPLETE

**Goal:** Repo live, Docker running, PubMed records in JSONL, RAGAS runnable on a mini test.

> **Status (01 April 2026):** Phase 1 fully complete. 3,706 heart-disease-focused PubMed records fetched (English only, 2021–2026), processed, and indexed into ChromaDB. All interface contracts delivered.

### Step 1.1 — Repo & Docker (Day 1) ✅ COMPLETE

**Do this first. Everything else depends on the environment.**

| Task | Output file | Status |
|------|------------|--------|
| Create GitHub repo with branch strategy (`main/dev/feature/*`) | `README.md`, `.gitignore` | ✅ Done |
| Write `docker-compose.yml` with: ChromaDB, MLflow, Ollama (LLaMA-3), FastAPI | `docker-compose.yml` | ✅ Done |
| Configure MLflow with SQLite backend + local artifact store | `mlops/mlflow_tracker.py` | ✅ Done |
| Pin all dependencies | `requirements.txt` | ✅ Done |
| Configure `.env` for NCBI and service endpoints | `.env` | ✅ Done |

**Key decisions locked in:**
- MLflow run naming: `{component}-{date}` e.g., `drift-2024-01-15`
- All thresholds in `src/config/settings.py` (Pydantic `BaseSettings`) — never hardcode

---

### Step 1.2 — PubMed ETL Pipeline (Days 2–4) ✅ COMPLETE

**This is the critical path. No data = no RAG = nothing else works.**

| Task | Output file | Status |
|------|------------|--------|
| NCBI E-utilities integration (`esearch` + `efetch`) | `src/fetcher.py` | ✅ Done |
| Fetch PubMed abstracts (heart disease focus) | `data/processed/pubmed_processed.jsonl` | ✅ Done — 3,706 records |
| Text preprocessing — extract pmid, title, abstract, pub_date, mesh_terms | `src/parser.py` | ✅ Done — all 7 fields present |
| DVC setup — track `data/processed/` | `.dvc/` | ✅ Done |
| Data quality report | `data/quality_report.py` → `data/quality_report.html` | ✅ Done |
| Dataset statistics | `data/stats.py` | ✅ Done — Table 0 for paper |
| ChromaDB upsert | `src/chroma_interface.py` | ✅ Done — 3,706 docs indexed |
| `get_embeddings()` | `data/ingest.py` | ✅ Done — returns `np.ndarray (3706, 384)` |
| `incremental_upsert()` | `data/ingest.py` | ✅ Done — deduplication + upsert working |
| Weekly data refresh | `data/refresh.py` | ✅ Done — heart disease records only |

**Data refinement (applied per team feedback):**
- **Domain filter:** Heart disease only — 4 focused MeSH queries
- **Date filter:** 2021–2026 only — no old records
- **Language filter:** English only — non-English records excluded
- **Final count:** 3,706 clean, filtered records
- **Unknown dates:** 13 records (0.3%) — retained (PubMed XML limitation)
- **Years present:** 2024, 2025, 2026

**Output format (verified):**
```json
{"pmid": "12345678", "title": "...", "abstract": "...", "text": "title. abstract", "pub_date": "2023-05", "mesh_terms": ["Hypertension"], "word_count": 245}
```
File: `data/processed/pubmed_processed.jsonl`

**Quality report results:**

| Metric | Value |
|--------|-------|
| Total records | 3,706 |
| Duplicate PMIDs | 0 |
| Average word count | 243.2 words |
| MeSH term coverage | 57.9% |
| Unknown dates | 13 |
| Embedding matrix shape | (3706, 384) |

---

### Step 1.3 — CI Scaffold (Day 5) ⏳ PARTIAL

| Task | Output file | Status |
|------|------------|--------|
| GitHub Actions workflow scaffold | `.github/workflow/pr_checks.yml` | ✅ Scaffolded (placeholders for RAGAS eval) |
| PR template | `.github/pull_request_template.md` | ✅ Done |

> RAGAS eval integration will be completed in Phase 2 when benchmark questions and `run_ci_eval()` are ready.

---

## Phase 2 — RAG Pipeline + Evaluation (Week 2) ✅ COMPLETE

**Goal:** End-to-end RAG chain answering medical questions. RAGAS running. 50-question benchmark created.

**Dependency:** Phase 1 complete. `pubmed_processed.jsonl` must exist. ✅ Met.

### Step 2.1 — ChromaDB Ingestion (Days 1–2) ✅ COMPLETE

| Task | Output file | Status |
|------|------------|--------|
| ChromaDB collection setup with `all-MiniLM-L6-v2` embeddings | `rag_pipeline/vectorstore.py` | ✅ Done — collection `pubmed_abstracts` |
| Chunking + embedding + upsert | `rag_pipeline/ingest.py` | ✅ Done |
| Expose `get_embeddings() -> np.ndarray` of shape `(n_docs, 384)` | `data/ingest.py::get_embeddings()` | ✅ Done — returns (3706, 384) |
| Expose `incremental_upsert(new_docs)` | `data/ingest.py::incremental_upsert()` | ✅ Done — dedup working |

**Verified:** 3,706 documents indexed and retrievable from ChromaDB.

---

### Step 2.2 — Retrieval Strategies (Days 2–3) ✅ COMPLETE

Build all three retrievers. Use a **factory pattern** so they're swappable by config string.

| Retriever | Implementation | Status |
|-----------|---------------|--------|
| Dense | `Chroma.as_retriever(search_kwargs={"k": 5})` | ✅ Done — `rag_pipeline/retriever.py::dense_retriever()` |
| BM25 | `rank_bm25` BM25Okapi, cached index via `_get_bm25_components()` | ✅ Done — `rag_pipeline/retriever.py::bm25_retriever()` |
| Hybrid | 0.6 × dense + 0.4 × BM25 weighted RRF fusion, top-20 candidates per side, rerank to top-k | ✅ Done — `rag_pipeline/retriever.py::hybrid_retrieve()` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` as second stage on hybrid | ✅ Done — `rag_pipeline/retriever.py::rerank()` |
| BM25 term scores | Per-term BM25 scores for XAI attribution | ✅ Done — `rag_pipeline/chain.py::bm25_term_scores()` |

---

### Step 2.3 — LangChain Chain (Day 3) ✅ COMPLETE

Build the unified `query()` function — this is the API everything else calls. **Implemented in `rag_pipeline/chain.py`.**

```python
def query(question: str, config: dict = None) -> dict:
    # config = {retriever: "hybrid", chunk_size: 512, k: 5, reranker: True}
    # returns: {
    #   answer: str,
    #   source_docs: list,           # [{pmid, text, score}]
    #   retrieval_latency_ms: float,
    #   retrieval_scores: List[float],  # raw scores — XAI needs these
    #   config: dict
    # }
```

**Prompt template (use exactly this):**
```
You are a clinical assistant. Use the context to answer precisely.
If no relevant context is found, say: "Insufficient context in knowledge base."
Context: {context}
Question: {question}
```

**Key rules:**
- Accept config dict — never hardcode retriever type or chunk size
- Log every config to MLflow using `mlops/mlflow_tracker.py`
- If no relevant context found, say "Insufficient context in knowledge base" — not a hallucination
- Performance target: < 3s per query

---

### Step 2.4 — RAGAS Evaluation Framework (Days 4–5) ✅ COMPLETE

| Task | Output file | Status |
|------|------------|--------|
| RAGAS setup with 4 metrics | `evaluation/ragas_runner.py` | ✅ Done — faithfulness, context_recall, answer_relevance, context_precision |
| 50-question medical benchmark | `evaluation/benchmarks/qa_pairs.json` | ✅ Done |
| 10-question CI subset | `evaluation/benchmarks/ci_benchmark.json` | ✅ Done |
| `run_eval()` function | `ragas_runner.py::run_eval(qa_pairs, config)` | ✅ Done — returns `{faithfulness, context_recall, answer_relevance, context_precision}` |
| `run_ci_eval()` function | `ragas_runner.py::run_ci_eval(config)` | ✅ Done — uses 5-question stub if no pairs given, < 90s |
| Baseline run | MLflow run: `baseline-eval-week2` | 🔜 Pending — eval code ready, run not yet executed |

**RAGAS metrics — what each measures:**
- **Faithfulness (0–1):** Is the answer supported by retrieved context? Catches hallucinations.
- **Context Recall (0–1):** Did retrieval fetch the relevant documents?
- **Answer Relevance (0–1):** Does the answer address the question?
- **Context Precision (0–1):** Are retrieved docs ranked with most relevant first?

**CI quality gate thresholds (add to `config.yaml`):**
- `faithfulness < 0.70` → fail the PR
- `context_recall < 0.65` → fail the PR

**Evaluation coding rules:**
1. Use RAGAS's Dataset format: columns must be `question, answer, contexts, ground_truth`
2. Handle async evaluation properly — RAGAS can run metrics in parallel
3. Fix all random seeds for reproducibility
4. Always include confidence intervals on reported means
5. Use paired tests (same 50 questions across all configs), not unpaired
6. Report Cohen's d as effect size alongside p-value

---

## Phase 3 — Experiments + XAI + Drift Detection (Week 3) 🔄 IN PROGRESS

**Goal:** Run all 9 ablation configs. Build the full XAI layer. Build drift detection. This is where all the novel paper contributions live.

**Dependency:** Phase 2 complete. `query()` and `run_eval()` must be working. ✅ Met.

---

### Step 3.1 — Ablation Study (Days 1–2)

Run all 9 retrieval configurations × 50 questions = 450 evaluations.

**9 configurations:**

| Config | Retriever | Chunk Size | Reranker |
|--------|-----------|------------|----------|
| 1 | dense | 256 | OFF |
| 2 | dense | 512 | OFF |
| 3 | dense | 1024 | OFF |
| 4 | bm25 | 256 | OFF |
| 5 | bm25 | 512 | OFF |
| 6 | bm25 | 1024 | OFF |
| 7 | hybrid | 256 | ON |
| 8 | hybrid | 512 | ON |
| 9 | hybrid | 1024 | ON |

| Task | Output file |
|------|------------|
| Automated runner looping all 9 configs | `evaluation/ablations/run_ablations.py` |
| Results aggregation (mean ± std per metric per config) | `evaluation/ablations/results.csv` |
| p50/p95/p99 latency per config | Added to `results.csv` |
| Paired t-test between best and second-best config on faithfulness | `evaluation/ablations/significance_test.py` |
| Error analysis for 10 worst-performing questions | `evaluation/ablations/error_analysis.md` |
| 3 prompt template variants (direct, CoT, step-back) on best config | 3 additional MLflow runs |

**Hypotheses to verify for the paper:**
1. Hybrid retrieval should outperform dense or BM25 alone on medical text
2. Optimal chunk size for medical abstracts is likely 512 tokens
3. Faithfulness is the most critical metric for clinical safety

---

### Step 3.2 — XAI Layer (Days 2–4)

This is the paper's novel XAI contribution. Build these 5 functions inside `evaluation/explainability.py`:

#### Function 1: SHAP on Retrieval Scores
```python
def shap_retrieval_explain(question, source_docs, retrieval_scores) -> List[float]:
    # Use shap.Explainer with retrieval_scores as features, faithfulness as target
    # Output: per-document SHAP value for every query
    # SHAP value > 0 = this doc helped faithfulness; < 0 = hurt it
```
Use `shap.LinearExplainer` for the hallucination classifier; `shap.KernelExplainer` when the model is a black box.

#### Function 2: Attention Attribution
```python
def attention_attribution(question, context, answer) -> List[dict]:
    # Extract attention weights from LLaMA-3's last decoder layer
    # Average across heads → (answer_len, context_len) matrix
    # Return top-3 context spans per answer sentence
```

#### Function 3: Explanation Vector (feeds drift monitor)
```python
def build_explanation_vector() -> np.ndarray:
    # Concatenate: SHAP values (k-dim) + top attention span positions (fixed-dim)
    # Must be FIXED dimension regardless of document count
    # Use top-k=5 SHAP values padded with zeros
    # Must be deterministic — set all random seeds
```

#### Function 4: Term-Level Retrieval Explanation
```python
def term_attribution(question, source_docs) -> dict:
    # Use bm25_term_scores() to produce human-readable output:
    # "Source 2 retrieved because: metformin (0.82), HbA1c (0.71), glycemic (0.55)"
    # Returns {term: score}
```

#### Function 5: Hallucination Classifier with XAI
```python
# Train logistic regression on features:
# [faithfulness, SHAP_max_score, attention_entropy, retrieval_score_variance]
# Binary label: hallucinated / not
# Then run SHAP on the classifier itself to explain WHY a response was flagged
# Hypothesis to test: retrieval_score_variance is the strongest predictor
```

**Unified `explain()` function — this is what the API and MLOps monitor calls:**
```python
def explain(question: str, source_docs: list, retrieval_scores: list) -> dict:
    return {
        "shap_values": List[float],          # per-document SHAP values
        "token_attributions": List[dict],    # top-3 context spans per answer token
        "term_attribution": dict,            # {term: score}
        "explanation_vector": np.ndarray,   # for consistency monitor
        "hallucination_risk": float,         # 0.0–1.0
        "hallucination_reason": str          # SHAP-derived explanation
    }
```

**XAI coding rules:**
1. `explanation_vector` must be fixed-dimension regardless of document count
2. Always test that `explanation_vector` is deterministic given the same input
3. Flag all XAI code with: `# XAI CONTRIBUTION`

---

### Step 3.3 — PSI Drift Detection (Days 3–4) ✅ COMPLETE

This is the paper's core MLOps novelty. Built inside `mlops/drift_detector.py`:

| Function | What it does | Status |
|----------|-------------|--------|
| `capture_baseline()` | Store embedding distribution via `BaselineStore` singleton | ✅ Done |
| `compute_psi()` | Per-dimension PSI via histogram binning (`settings.psi_num_bins`), return mean across 384 dims | ✅ Done |
| `alert()` | Compare against `settings` thresholds, return `{status, psi_score, timestamp, action, thresholds}` | ✅ Done |
| `monitor_generation()` | Query MLflow for faithfulness metrics, compare recent vs baseline average, flag >15% drop | ✅ Done |

**PSI thresholds:**
- PSI < 0.1 → no drift (green)
- PSI 0.1–0.25 → moderate drift, log warning (orange)
- PSI > 0.25 → significant drift, trigger refresh (red)

**Auto-refresh trigger** (`mlops/refresh_trigger.py` + `.github/workflows/refresh.yml`):
On drift alert → pull latest PubMed batch → re-run ETL → re-index ChromaDB → re-run RAGAS eval → post diff report

**Paper angle:** Log PSI scores over 4 weeks of simulated data as Figure 2. Show the trigger firing when you inject synthetic drift.

---

### Step 3.4 — XAI Consistency Monitor (Day 4) ✅ COMPLETE

Built inside `mlops/explanation_monitor.py`:
- `save_baseline(vectors)` — stores XAI baseline via `BaselineStore` singleton
- `compute_consistency(current_vectors, baseline_vectors)` — mean cosine similarity across vector pairs
- Handles vector count mismatch gracefully (computes overlap)
- Logs status based on `settings.xai_warning_threshold` / `settings.xai_alert_threshold`

**Thresholds (add to `config.yaml` under `xai.*`):**
- `< 0.75` → log warning
- `< 0.60` → escalate as "explanation instability" alert
- CI gate: if `explanation_consistency_score < 0.70` → post warning comment on PR (non-blocking)

**Paper note:** No existing RAGOps work tracks explanation stability over time — this is a novel metric.

---

### Step 3.5 — Docker Hardening & Integration Tests (Day 5)

| Task | Output |
|------|--------|
| Add health checks to all Docker services | Updated `docker-compose.yml` |
| Makefile targets: `make setup`, `make run`, `make eval`, `make drift-check` | `Makefile` |
| End-to-end test: ingest 100 docs → query 10 Qs → RAGAS eval → MLflow log → drift check → all pass | `tests/test_e2e.py` |
| Unit tests for each retriever + chain with mock ChromaDB | `tests/test_rag.py` |
| MLflow experiment comparison script (two runs side-by-side) | `mlops/compare_runs.py` | ✅ Done |

---

## Phase 4 — Dashboard + Paper (Week 4) 🔄 IN PROGRESS

**Goal:** Streamlit dashboard live. Paper sections drafted. Demo recorded.

**Dependency:** Phase 3 complete. All MLflow runs logged. `drift_report.json` being generated. `explain()` function working.

> **Status (12 April 2026):** FastAPI serving layer and Streamlit dashboard are complete. Paper writing not yet started.

---

### Step 4.1 — FastAPI Serving Layer (Days 1–2) ✅ COMPLETE

> **Implemented (12 April 2026):** All 5 endpoints wired in `serving/api.py`. Uses central `src.config.settings` (removed duplicate local Settings class). All endpoints use `asyncio.to_thread()` for sync backends. Quality gate integration via `RAGOpsTracker`.

Original spec (3 endpoints) expanded to 5 endpoints:

```
POST /query
  Body: {"question": str, "config": optional dict}
  Response: {"answer": str, "sources": [{pmid, snippet}], "latency_ms": float}

POST /explain
  Body: {"question": str, "source_docs": list, "retrieval_scores": list}
  Internally calls: evaluation.explainability.explain(...)
  Response: {shap_values, token_attributions, term_attribution,
             explanation_vector, hallucination_risk, hallucination_reason}
  Latency target: < 1.5s added latency. Run synchronously after /query.

GET /health
  Response: {"status": "ok", "chroma_connected": bool,
             "ollama_connected": bool, "explainability_ready": bool}
```

**FastAPI coding rules:**
1. Use `async def` for all route handlers
2. Add request logging middleware
3. Validate inputs with Pydantic models
4. Import XAI as: `from evaluation.explainability import explain`

---

### Step 4.2 — Streamlit Dashboard (Days 2–3) ✅ COMPLETE

> **Implemented (12 April 2026):** All 5 panels built in `serving/dashboard.py`. XAI panel shows placeholder until `evaluation/explainability.py` is implemented. Run with `streamlit run serving/dashboard.py`.

Built `serving/dashboard.py` with 5 tabs using `st.tabs`:

**Panel 1 — Query Interface:**
- Text input → `POST /query` → display answer
- Expandable "Sources" section with PubMed snippet + clickable PMID links
- After answer renders: auto-call `POST /explain` with spinner "Generating explanation..."

**Panel 2 — XAI Explainability:**
- Highlighted context: `st.markdown` with colour-coded `<span>` tags. Intensity maps to attention weight (yellow=low, orange=medium, red=high)
- SHAP bar chart (Plotly): x=PubMed IDs, y=SHAP value. Blue bars=positive, red=negative
- Term attribution ranked list: "Source 2 retrieved because: metformin (0.82) · HbA1c (0.71)"
- Hallucination risk badge: `st.metric`. Green if < 0.3, red if > 0.6. Show `hallucination_reason` below.

**Panel 3 — Live RAGAS Metrics:**
- Plotly line charts: faithfulness, context_recall, answer_relevance over MLflow runs
- Auto-refresh every 60s using `st.rerun()`
- Pull via `mlflow.search_runs()` Python API

**Panel 4 — Drift Alert Panel:**
- Read `drift_report.json`
- PSI score as traffic light metric: green/orange/red
- Table of last 5 drift events with timestamps
- `explanation_consistency_score` from MLflow alongside PSI
- "Last knowledge base refresh" timestamp

**Panel 5 — Experiment Comparison:**
- Two dropdowns: MLflow run names
- Side-by-side Plotly bar chart of RAGAS metrics + `explanation_consistency_score`

**Streamlit coding rules:**
1. Use `st.cache_data` with TTL to avoid hammering MLflow/FastAPI
2. All charts: `template="plotly_dark"`
3. Error states must be explicit — especially the XAI panel
4. `st.spinner()` for any operation > 1 second
5. XAI panel must render gracefully if `/explain` errors — fall back to "Explanation unavailable"
6. Dashboard must work even if some services are down — show "Service unavailable", never crash

**HTML for attention highlighting:**
```python
html = f'<span style="background-color: rgba(255,165,0,{weight})">{text}</span>'
st.markdown(html, unsafe_allow_html=True)
```

---

### Step 4.3 — Paper Writing (Days 3–5)

| Section | Content |
|---------|---------|
| Section 2: Related Work | RAG (Lewis et al. 2020, Guu et al. REALM), hybrid retrieval, XAI in clinical NLP (LIME/SHAP on transformers) |
| Section 3: System Architecture | Full architecture diagram + component descriptions including XAI layer |
| Section 3.1: Data Pipeline | PubMed ETL design, DVC versioning, dataset statistics |
| Section 4.1: Drift Detection Methodology | PSI formulation, threshold derivation, alarm mechanism |
| Section 4.2: RAG Pipeline Design | Chunking strategies, retriever types, reranker, prompt engineering |
| Section 4.3: Evaluation Methodology | RAGAS metrics, benchmark construction, ablation design |
| Section 4.4: Explanation Consistency Monitoring | How you track SHAP vector stability as an MLOps metric |
| Section 4.5: Explainable AI Layer | SHAP on retrieval scores, attention attribution, hallucination classifier |
| Section 5: Experimental Results | Tables 1–3, Figure 2 (temporal PSI), statistical significance |
| Section 6: Discussion | Why hybrid retrieval wins, clinical trustworthiness via XAI, failure cases |
| Limitations | Attention extraction limits, benchmark bias, SHAP approximation caveats |
| Section 7: Reproducibility | Docker setup, GitHub repo, replication instructions |
| Appendix A | CI/CD workflow diagrams including XAI quality gate |

**Figures and Tables:**

| Output | What it shows |
|--------|--------------|
| Figure 1 | Streamlit dashboard screenshot |
| Figure 2 | PSI drift scores over 4 weeks (with synthetic drift injection) |
| Figure 3 | RAGAS scores over time across 4 knowledge base snapshots |
| Table 0 | Dataset statistics (counts, date range, categories, avg length) |
| Table 1 | 9-config ablation results (RAGAS scores + latency) |
| Table 2 | Category-wise results (diagnosis / treatment / drug interactions) |
| Table 3 | XAI results: SHAP feature importances, hallucination classifier AUC, top attributed terms |

**Other tasks:**
- Record a 3-minute demo video of dashboard + CI pipeline running
- Write `CONTRIBUTING.md` and `DATA_CARD.md`
- Final DVC push so anyone can reproduce
- Tag `v1.0.0` release on GitHub

**Target venues:** IEEE ICHI (Health Informatics), ACL Clinical NLP Workshop, or MLSys Workshop

---
---

# ✅ Complete Deliverables Checklist

## Data Pipeline
- [x] `src/fetcher.py` — NCBI API integration (esearch + efetch)
- [x] `src/parser.py` — XML to JSONL parsing + cleaning
- [x] `src/chroma_interface.py` — ChromaDB interface (`get_embeddings()` + `incremental_upsert()`)
- [x] `src/run_pipeline.py` — Main pipeline orchestrator
- [x] `data/refresh.py` — weekly incremental fetch (heart disease records)
- [x] `data/quality_report.py` → `data/quality_report.html`
- [x] `data/stats.py` — dataset statistics (Table 0 for paper)
- [x] `data/ingest.py` — team interface (`get_embeddings()` + `incremental_upsert()`)
- [x] `data/processed/pubmed_processed.jsonl` — 3,706 records (heart disease, English, 2021–2026)
- [x] `.dvc/` — DVC initialized, `data/processed` tracked

## RAG Pipeline
- [x] `rag_pipeline/vectorstore.py` — ChromaDB setup
- [x] `rag_pipeline/ingest.py` — chunking + embedding + upsert
- [x] `rag_pipeline/retriever.py` — dense, BM25, hybrid (RRF), cross-encoder reranker — all complete
- [x] `rag_pipeline/chain.py` — unified `query()` API with `retrieval_scores`
- [ ] `rag_pipeline/query_processor.py` — medical query preprocessing

## Evaluation
- [x] `evaluation/ragas_runner.py` — full eval + `run_ci_eval()`
- [x] `evaluation/benchmarks/qa_pairs.json` — 50 medical Q&A pairs
- [x] `evaluation/benchmarks/ci_benchmark.json` — 10-question CI subset
- [x] `evaluation/ablations/run_ablations.py` — automated 9-config runner
- [ ] `evaluation/ablations/results.csv` — all metrics across all configs
- [x] `evaluation/ablations/significance_test.py`
- [x] `evaluation/ablations/error_analysis.py` — error categorization
- [ ] `evaluation/explainability.py` — SHAP, attention, explanation vector, term attribution, hallucination classifier

## MLOps Infrastructure
- [x] `docker-compose.yml` — all services with one command + health checks
- [x] `mlops/mlflow_tracker.py` — `RAGOpsTracker` context manager with quality gates, PSI/XAI logging
- [x] `src/config/settings.py` — Pydantic `BaseSettings` singleton (all thresholds)
- [x] `src/infra/baseline_store.py` — baseline storage for drift detection (singleton factory fixed)
- [x] `mlops/drift_detector.py` — PSI computation (`compute_psi`), baseline capture, alert, generation drift monitoring
- [x] `mlops/refresh_trigger.py` — auto KB refresh on drift (`trigger_refresh` orchestrates eval → upsert → baseline → eval)
- [x] `mlops/explanation_monitor.py` — XAI consistency via cosine similarity (`compute_consistency`, `save_baseline`)
- [x] `mlops/compare_runs.py` — MLflow run comparison with deltas + improved/regressed classification
- [x] `.github/workflows/pr_checks.yml` — PR quality gate (scaffolded, RAGAS eval pending)
- [ ] `.github/workflows/refresh.yml` — automated refresh pipeline
- [ ] `Makefile` — `setup`, `run`, `eval`, `drift-check` targets
- [x] `src/infra/test/test_mlflow_tracker.py` — MLflow tracker unit tests
- [ ] `tests/test_e2e.py` — end-to-end integration test
- [ ] `tests/test_rag.py` — unit tests for retrievers + chain

## Serving
- [x] `serving/api.py` — FastAPI with all 5 endpoints wired: `/health`, `/query`, `/evaluate`, `/drift/check`, `/xai/check`
- [x] `serving/dashboard.py` — 5-panel Streamlit app (Query, XAI, RAGAS Metrics, Drift Alerts, Experiment Comparison)

## Paper
- [ ] Paper Sections 2, 3, 3.1, 4.1–4.5, 5, 6, Limitations, Reproducibility, Appendix A
- [ ] Tables 0, 1, 2, 3
- [ ] Figures 1, 2, 3
- [ ] 3-minute demo video
- [x] `CONTRIBUTING.md` — done
- [ ] `DATA_CARD.md`, `SETUP.md`
- [ ] `v1.0.0` GitHub release + DVC data push

---
---

# 🤖 Your Claude Prompt (Paste at the Start of Every Session)

```
You are a senior ML engineer helping me build a solo research project called RAGOps — a production-grade MLOps framework for RAG-based clinical decision support using PubMed medical literature. I am building this entirely alone, so I own all roles: data pipeline, RAG pipeline, evaluation, MLOps infrastructure, XAI, and the Streamlit dashboard.

## Tech Stack
- Python 3.11
- LangChain (RetrievalQA, splitters, retrievers)
- ChromaDB (persistent, Docker service on localhost:8000)
- sentence-transformers: all-MiniLM-L6-v2
- rank_bm25 for BM25 retrieval
- cross-encoder/ms-marco-MiniLM-L-6-v2 for reranking
- Ollama running LLaMA-3-8B at http://localhost:11434
- RAGAS (RAG evaluation framework)
- shap (SHAP explainability)
- transformers (HuggingFace, attention extraction fallback)
- MLflow (SQLite backend, local artifact store)
- GitHub Actions (CI/CD)
- FastAPI + Uvicorn
- Streamlit + Plotly
- DVC (data versioning)
- NCBI E-utilities API

## Repo Structure
ragops/
├── data/               # ETL pipeline, DVC versioning
├── rag_pipeline/       # ChromaDB, retrievers, LangChain chain
├── evaluation/         # RAGAS, benchmarks, ablations, explainability
├── mlops/              # drift detection, MLflow tracker, XAI monitor
├── serving/            # FastAPI + Streamlit dashboard
├── tests/
├── .github/workflows/
├── docker-compose.yml
└── Makefile

## Key Design Decisions
- PSI (Population Stability Index) on embedding vectors for drift detection. PSI > 0.25 = trigger refresh.
- RAGAS faithfulness rolling average drop > 15% = generation drift alert.
- CI gate: faithfulness < 0.70 OR context_recall < 0.65 → fail PR.
- explanation_vector: fixed-dim np.ndarray from SHAP values (k=5, zero-padded) + attention span positions.
- explanation_consistency_score: mean cosine similarity of explanation vectors across runs. < 0.75 = warning, < 0.60 = alert.
- All thresholds live in `src/config/settings.py` (Pydantic BaseSettings). Never hardcode.
- MLflow run naming: {component}-{date} e.g., drift-2024-01-15

## Code Standards (always follow these)
1. Production-quality Python with type hints and docstrings
2. Python logging module — never use print for important messages
3. Error handling and retries (exponential backoff for external APIs)
4. Flag novel MLOps work with: # PAPER CONTRIBUTION
5. Flag explainability work with: # XAI CONTRIBUTION
6. Flag result-generating code with: # PAPER RESULT — TABLE/FIGURE X
7. Fixed random seeds everywhere for reproducibility
8. Functions must be independently testable

## Interface Contracts (internal APIs I must maintain)
- data.ingest.get_embeddings() → np.ndarray (n_docs, 384)
- data.ingest.incremental_upsert(new_docs) → int
- rag_pipeline.chain.query(question, config) → {answer, source_docs, retrieval_scores, retrieval_latency_ms, config}
- rag_pipeline.retriever.bm25_term_scores(query, doc) → {term: score}
- evaluation.ragas_runner.run_eval(qa_pairs, config) → {faithfulness, context_recall, answer_relevance, context_precision, latency_p50, latency_p95}
- evaluation.ragas_runner.run_ci_eval(config) → same structure, < 90 seconds
- evaluation.explainability.explain(question, source_docs, retrieval_scores) → {shap_values, token_attributions, term_attribution, explanation_vector, hallucination_risk, hallucination_reason}

Start each session by asking me: which phase/component am I working on today, and what have I already completed?
```

---

## Code Comment Conventions

| Comment | Use it when |
|---------|------------|
| `# PAPER CONTRIBUTION` | Novel MLOps work (PSI drift detection, auto-refresh trigger) |
| `# XAI CONTRIBUTION` | Explainability code (SHAP, attention, explanation vector) |
| `# PAPER RESULT — TABLE X` | Code that generates a result that goes into the paper |
| `# ABLATION EXPERIMENT` | Code that touches the 9-config experiment matrix |

---

## Config Reference (`config.yaml` schema)

```yaml
ragas:
  faithfulness_threshold: 0.70
  context_recall_threshold: 0.65

drift:
  psi_warning: 0.10
  psi_alert: 0.25
  faithfulness_drop_threshold: 0.15  # rolling 7-day % drop

xai:
  hallucination_risk_threshold: 0.60
  explanation_consistency_min: 0.70
  explanation_consistency_warning: 0.75

retrieval:
  best_config: {}  # fill in after ablation study

mlflow:
  run_naming: "{component}-{date}"
```
