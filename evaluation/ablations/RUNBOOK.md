# Ablation Study — Runbook

This runbook walks you through executing the full Phase 3 ablation sweep
(9 configs × 50 questions = 450 RAGAS evaluations), post-processing the
results, and training the hallucination classifier on the output. The
sweep produces the source table for Paper TABLE 1 plus the weak labels
used by `evaluation.explainability.HallucinationClassifier`.

Plan on several hours of wall-clock time on CPU with `llama3.2:1b`.
GPU + LLaMA-3.2-3B cuts that materially but still expect a long job —
run it overnight.

---

## 1. Preflight

All services must be healthy before kicking off the sweep; the driver
process crashes hard if Ollama or ChromaDB go away mid-run.

```bash
make setup           # installs FastAPI deps + pulls llama3.2:1b
make up              # docker compose up -d (chromadb + mlflow + ollama + fastapi)
make healthcheck     # polls /health until all 3 deps report "ok"
```

Confirm the knowledge base is populated (3,706 PubMed heart-disease
chunks at `chunk_size=512` by default):

```bash
python -c "from data.ingest import get_embeddings; print(get_embeddings().shape)"
# -> (3706, 384)   (or similar — any non-zero first dim is fine)
```

If the collection is empty, run ingestion first:

```bash
python -m rag_pipeline.ingest --data_dir ./data/raw --chunk_size 512
```

## 2. Dry-run the grid

Always inspect the grid before the live run. `--dry-run` prints the 9
configs (C01–C09) and exits without calling the LLM:

```bash
python evaluation/ablations/run_ablations.py --dry-run
```

You should see 9 rows covering the full 3×3 product:

```
chunk_size ∈ {256, 512, 1024}
retriever  ∈ {bm25, dense, hybrid}
top_k      = 5   (held constant)
```

## 3. Full execution

### 3a. Recommended — single worker, SQLite MLflow

The driver warns at `evaluation/ablations/run_ablations.py:329` that
parallel workers + SQLite MLflow will deadlock. Use **one worker** and
stick with the default SQLite backend for reproducibility:

```bash
python evaluation/ablations/run_ablations.py \
    --qa-file evaluation/benchmarks/qa_pairs.json \
    --workers 1
```

### 3b. Alternative — parallel workers, file backend

If you want to burn more cores, switch MLflow to a file backend (no
SQLite lock contention) and raise `--workers`:

```bash
MLFLOW_TRACKING_URI=file:./mlruns \
python evaluation/ablations/run_ablations.py \
    --qa-file evaluation/benchmarks/qa_pairs.json \
    --workers 4
```

Note: the file backend is fine for the ablation itself, but the live
dashboard and the CI gate expect the SQLite DB. Move the runs back
(or re-point `MLFLOW_TRACKING_URI`) before opening the UI.

## 4. Checkpoint / resume

Long runs crash occasionally (GPU OOM, Ollama model eviction, network
blip). Every completed config is checkpointed — resume with `--resume`,
which reads the checkpoint via `load_checkpoint` at
`evaluation/ablations/run_ablations.py:479` and skips anything already
logged to MLflow:

```bash
python evaluation/ablations/run_ablations.py \
    --qa-file evaluation/benchmarks/qa_pairs.json \
    --workers 1 \
    --resume
```

## 5. Outputs

After a clean run you will have:

| Artifact                              | Purpose                                   |
| ------------------------------------- | ----------------------------------------- |
| `ablation_results.csv`                | Paper TABLE 1 source (9 rows × 4 metrics) |
| `ablation_results.json`               | Same data + MLflow run IDs                |
| `mlruns.db` (or `mlruns/`)            | One MLflow run per config, all tagged    |
| `ragops-ragas-eval` MLflow experiment | Live dashboard source                    |

## 6. Post-processing

Three follow-up scripts consume the sweep output:

### 6a. Statistical significance

```bash
python evaluation/ablations/significance_test.py \
    --results ablation_results.csv
```

Produces pairwise significance tables between retriever families and
chunk sizes (paired-t on per-question faithfulness).

### 6b. Error analysis

```bash
python evaluation/ablations/error_analysis.py \
    --results ablation_results.csv
```

Surfaces the questions that fail across all 9 configs — these are
the hard cases worth writing up in the paper's error-analysis section.

### 6c. Train the hallucination classifier

The ablation per-question output is the weak-label source for
`evaluation.explainability.HallucinationClassifier`. Treat any answer
with `faithfulness < 0.5` as a positive (hallucinating) example:

```python
import pandas as pd
from evaluation.explainability import HallucinationClassifier, _FEATURE_ORDER

df = pd.read_csv("ablation_per_question.csv")   # from run_ablations.py

X = df[list(_FEATURE_ORDER)].to_numpy()
y = (df["faithfulness"] < 0.5).astype(int).to_numpy()

clf = HallucinationClassifier()
clf.fit(X, y)                                   # persists to mlops/artifacts/hallucination_clf.joblib
```

On next FastAPI boot the classifier will load the fitted artifact
automatically — the rule-based fallback in
`HallucinationClassifier._rule_based_risk` stays in place until this
step completes.

## 7. Verification

Visually confirm the sweep landed in MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
# → open http://localhost:5000
# → experiment: ragops-ragas-eval
# → filter: tags.mlflow.runName LIKE "ablation-C%"
# → you should see exactly 9 runs (C01 … C09)
```

Each run must have the four RAGAS metrics populated
(`faithfulness`, `context_recall`, `answer_relevance`,
`context_precision`) plus config params (`retriever_type`,
`chunk_size`, `k`).

## 8. Troubleshooting

| Symptom                                        | Fix                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| `database is locked` during parallel run       | Drop to `--workers 1` OR switch to `MLFLOW_TRACKING_URI=file:…`     |
| Ollama returns empty answers mid-sweep         | Model was evicted from VRAM. Set `OLLAMA_KEEP_ALIVE=24h` in `.env`. |
| Sweep aborts after N configs                   | Rerun with `--resume`; checkpoint resumes from the last completed   |
| RAGAS `faithfulness=0` for every question      | Check `OLLAMA_BASE_URL` / model name — judge LLM is failing silently |
| `ChromaDB` collection missing                  | Run `python -m rag_pipeline.ingest --chunk_size {256,512,1024}` for each size the grid needs |
