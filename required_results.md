Here's a comprehensive breakdown of all the results you need to generate for your paper, and exactly how to produce each one.

---

## Results You Need to Generate

### 1. Ablation Study — Retrieval Strategy × Chunk Size
**What it is:** The core empirical result comparing all retriever types (Dense, BM25, Hybrid) across all chunk sizes (256, 512, 1024 tokens) on RAGAS metrics.

**How to generate:**
- Hit `POST /evaluate` for every combination: 3 retrievers × 3 chunk sizes = **9 runs**
- In each request body, set `retriever_type`, `chunk_size`, and use the full benchmark (`run_ci_eval=false`) with your 50-question set
- Each run auto-logs to MLflow — use the **Experiment Comparison panel** (Dashboard Section 5) to diff runs side-by-side
- Export the table: `faithfulness`, `context_recall`, `answer_relevancy`, `context_precision` for each of the 9 configs

**Key script:** `evaluation/ragas_runner.py` → `run_eval()`

---

### 2. RAGAS Quality Gate Pass/Fail Summary
**What it is:** Shows that your system meets clinical-grade thresholds (faithfulness ≥ 0.80, context_recall ≥ 0.75, answer_relevancy ≥ 0.75).

**How to generate:**
- The `QualityGateResult` (passed/failures) is already logged per MLflow run
- Query MLflow for runs where `quality_gate_passed = true/false` and summarize
- Your best config (likely Hybrid + chunk_size=512 or 256) should be highlighted as the production config

---

### 3. PSI Drift Detection Results *(paper contribution)*
**What it is:** Demonstrates your novel PSI-over-embeddings drift monitor. This is flagged `# PAPER CONTRIBUTION` in your code.

**How to generate:**
- Call `POST /drift/check` at different ingestion states (e.g., after initial 3,706 docs, after adding synthetic drift, after KB refresh)
- Collect `psi_score` and `status` (stable / warning / alert) at each stage
- Plot **mean PSI over time / ingestion batches** — a line chart crossing the 0.10 (warning) and 0.25 (alert) thresholds visually tells the story
- Show one full refresh cycle: PSI alert → `trigger_refresh()` → PSI drops back to stable — this is your closed-loop contribution

---

### 4. XAI Explanation Consistency Results *(XAI contribution)*
**What it is:** Shows that your explanation monitor (`# XAI CONTRIBUTION`) detects instability when the model's explanations shift.

**How to generate:**
- Call `POST /xai/check` with your 10 benchmark questions (`xai_benchmark_questions=10` in settings)
- Record `explanation_consistency_score` and `status` (stable / warning / instability)
- Run this before and after a KB refresh to show the score changes meaningfully
- Optionally show a sample `/explain` output for a single query (SHAP values, hallucination risk) as a qualitative figure

---

### 5. End-to-End Latency Benchmarks
**What it is:** Latency breakdown showing retrieval vs. LLM time — important for a clinical system paper.

**How to generate:**
- `POST /query` returns `retrieval_latency_ms`, `llm_latency_ms`, `total_latency_ms` for every call
- Run ~20 queries per retriever type and chunk size config
- Report **mean ± std** for each. Hybrid will be higher than Dense/BM25 due to cross-encoder reranking — this trade-off needs to be discussed
- A grouped bar chart (retriever type on x-axis, ms on y-axis, stacked retrieval+LLM) works well

---

### 6. Hallucination Risk Distribution
**What it is:** Shows your logistic-regression hallucination classifier is calibrated and useful.

**How to generate:**
- Call `POST /explain` for all 50 benchmark questions
- Collect `hallucination_risk` scores (0–1 probabilities)
- Plot a histogram of risk scores; ideally most clinical answers cluster at low risk
- Cross-reference: do low-`faithfulness` RAGAS answers correspond to high `hallucination_risk`? That correlation validates both metrics

---

### 7. Dataset Statistics (descriptive)
**What it is:** Characterises your PubMed corpus — required for any clinical NLP paper.

**How to generate:**
- From your ingestion pipeline: 3,706 abstracts, 4 query types (`clinical practice guidelines`, `drug drug interactions`, `randomized controlled trial treatment`, `differential diagnosis`)
- Run chunking across all 3 sizes and report: number of chunks per collection (`pubmed_256`, `pubmed_512`, `pubmed_1024`), average chunk character length, total embeddings stored

---

## Suggested Table/Figure Structure for the Paper

| # | Type | Title |
|---|------|-------|
| Table 1 | Ablation | RAGAS metrics across retriever × chunk size (9 configs) |
| Table 2 | Quality Gates | Pass/fail summary with threshold values |
| Figure 1 | Line chart | PSI drift score over ingestion batches with threshold lines |
| Figure 2 | Bar chart | End-to-end latency by retriever type |
| Figure 3 | Histogram | Hallucination risk distribution across 50 benchmark questions |
| Figure 4 | Example | Single query: SHAP bar chart from `/explain` output |

---

## Quickest Path to All Results

1. **Spin up the stack** (`docker compose up`)
2. **Run the ablation loop** — 9 `POST /evaluate` calls, log all to MLflow
3. **Run PSI in 3 stages** — baseline, post-drift injection, post-refresh
4. **Run XAI check** before and after refresh
5. **Run `/explain` on your 50-question benchmark** — collect hallucination scores
6. **Export everything from MLflow UI** at `localhost:5000` as CSV → build your tables/figures

The Streamlit dashboard at `localhost:8080` already visualises RAGAS over time (Section 3) and drift alerts (Section 4), so you can screenshot those directly for the paper too.