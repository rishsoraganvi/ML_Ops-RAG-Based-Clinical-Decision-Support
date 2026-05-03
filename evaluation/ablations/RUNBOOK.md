# Ablation Study — Runbook

This runbook walks you through executing the full Phase 3 ablation sweep
(9 configs × 50 questions = 450 RAGAS evaluations), post-processing the
results, and training the hallucination classifier on the output. The
sweep produces the source table for Paper TABLE 1 plus the weak labels
used by `evaluation.explainability.HallucinationClassifier`.

Plan on several hours of wall-clock time. The sweep judge is **`llama3:8b`**
(set via `OLLAMA_MODEL` — see §3). It is materially slower than the
production judge `llama3.2:3b`, but the slow RAGAS metrics (`faithfulness`
decomposes answers into atomic statements + per-statement NLI;
`context_recall` matches ground-truth segments) produce malformed JSON
on smaller judges, which cost you the entire 25-minute cell. The
NaN-tolerant aggregator added to `evaluation/ragas_runner.py` no longer
crashes on a single bad metric column, but you still want the most
reliable judge you can afford here. GPU is strongly recommended; on
CPU expect to run it overnight.

---

## TL;DR — fast paths

The full sweep (§3a) takes hours. Two faster paths exist for paper-figure work:

- **`make eval-fast`** — 9 configs × 20 stratified questions × `phi3:mini` judge.
  ~2.5 hours instead of 24+. Output goes to `ablation_outputs/phi3_stratified20/`
  so it doesn't clobber the canonical full-sweep results. Subset is in
  `evaluation/benchmarks/qa_pairs_stratified20.json` (5 per category × 1 easy
  + 2 medium + 2 hard, deterministic by id sort). Use this for iteration.

- **`make eval-retrieval`** — judge-free retrieval-only ablation. Loops over
  the 9 configs, computes cosine recall@k, precision@k, nDCG@k, MRR against
  the ground-truth `contexts` field in `qa_pairs.json` using the same
  embedding model ChromaDB uses (`nomic-embed-text`). Finishes in ~30 minutes
  on the full 50-question benchmark, no LLM judge involved. Output:
  `ablation_outputs/retrieval_metrics.csv`.

- **`make eval-latency`** — per-config end-to-end latency profile. Calls
  `rag_pipeline.chain.query()` per config and aggregates retrieval/LLM/total
  latency to P50/P95/P99 + mean/std. Independent of RAGAS scores. Output:
  `ablation_outputs/latency_profile.csv` and `latency_profile_per_question.csv`.

These three targets are the recommended way to populate Paper Tables 1, 1b, and
the latency figure when full-sweep judge time is prohibitive.

---

## 1. Preflight

All services must be healthy before kicking off the sweep; the driver
process crashes hard if Ollama or ChromaDB go away mid-run.

```bash
make setup           # installs FastAPI deps + pulls phi3:mini + llama3:8b + nomic-embed-text
make up              # docker compose up -d (chromadb + mlflow + ollama + fastapi)
make healthcheck     # polls /health until all 3 deps report "ok"
```

`make setup` pulls all three judge/embedding models **into the host Ollama
daemon**. The docker-compose stack runs Ollama in a separate container
(`ragops_ollama`) at `localhost:11434`, and the eval scripts hit *that*
endpoint — so a model pulled to the host daemon is invisible to the sweep
and produces 404s like `ResponseError(model 'llama3:8b' not found)`.

If you already have the Ollama container running and just need the ablation
judge inside it:

```bash
docker exec ragops_ollama ollama pull llama3:8b
docker exec ragops_ollama ollama list   # verify
curl -s http://localhost:11434/api/tags # also verifies via HTTP
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

### 3a. Recommended — Makefile target (one worker, llama3:8b judge)

`make eval-full` is the canonical entry point. It exports
`OLLAMA_MODEL=llama3:8b` and runs with `--workers 1`:

```bash
make eval-full
# equivalent to:
#   OLLAMA_MODEL=llama3:8b python evaluation/ablations/run_ablations.py \
#       --qa-file evaluation/benchmarks/qa_pairs.json --workers 1
```

To swap the judge (e.g. for a quick sanity sweep on `llama3.2:3b`,
trading reliability for speed):

```bash
make eval-full ABLATION_JUDGE=llama3.2:3b
```

If you want to invoke the script directly (e.g. with `--resume`),
remember to set `OLLAMA_MODEL` yourself — the script picks up whatever
is in your shell:

```bash
OLLAMA_MODEL=llama3:8b python evaluation/ablations/run_ablations.py \
    --qa-file evaluation/benchmarks/qa_pairs.json \
    --workers 1
```

The driver warns at `evaluation/ablations/run_ablations.py:338` that
parallel workers + SQLite MLflow will deadlock — keep `--workers 1`
unless you have a non-SQLite backend.

### 3b. Alternative — parallel workers, file backend

If you want to burn more cores, switch MLflow to a file backend (no
SQLite lock contention) and raise `--workers`:

```bash
OLLAMA_MODEL=llama3:8b MLFLOW_TRACKING_URI=file:./mlruns \
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

Each run should have the four RAGAS metrics populated
(`faithfulness`, `context_recall`, `answer_relevance`,
`context_precision`) plus config params (`retriever_type`,
`chunk_size`, `k`).

Under the NaN-tolerant aggregator added to `evaluation/ragas_runner.py`,
a metric column may legitimately come back as NaN if the judge produced
unparseable JSON for every question on that metric — those NaN scores
are **stripped before MLflow logging** (MLflow rejects NaN floats), so
the corresponding metric will simply be absent from that run. Check the
runner logs: `RAGAS result missing column '<metric>'` or `Skipping
NaN/None RAGAS metrics for MLflow log: [...]` tells you exactly which
metric the judge failed on. The corresponding row in
`ablation_results.csv` will have that column empty (and `null` in the
JSON), with `status=success` — the rest of the cell is still valid.
If that happens, retry that single config with a stronger judge before
including it in Paper TABLE 1.

## 8. Troubleshooting

| Symptom                                        | Fix                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| `database is locked` during parallel run       | Drop to `--workers 1` OR switch to `MLFLOW_TRACKING_URI=file:…`     |
| Ollama returns empty answers mid-sweep         | Model was evicted from VRAM. Set `OLLAMA_KEEP_ALIVE=24h` in `.env`. |
| Sweep aborts after N configs                   | Rerun with `--resume`; checkpoint resumes from the last completed   |
| RAGAS `faithfulness=0` for every question      | Check `OLLAMA_BASE_URL` / model name — judge LLM is failing silently |
| `404 Not Found` / `ResponseError(model '...' not found)` for every job | Model exists on host Ollama but not in the `ragops_ollama` container. Pull into the container: `docker exec ragops_ollama ollama pull <model>` |
| `KeyError: 'faithfulness'` at end of cell      | Pre-fix bug — `_METRICS` was out of sync with the score dict. Pull latest; the runner now warns + emits NaN instead. |
| One metric column empty in CSV, `status=success` | Judge returned malformed JSON for every question on that metric. Re-run with `OLLAMA_MODEL=llama3:8b … --resume` after deleting that row from `ablation_results.csv`. |
| `ChromaDB` collection missing                  | Run `python -m rag_pipeline.ingest --chunk_size {256,512,1024}` for each size the grid needs |

## 9. Companion ablations (judge-free)

Three companion scripts produce paper-quality results without paying the
full RAGAS judge cost. They share the same 9-config grid as the sweep:

| Target / script                        | Output                                                               | Runtime  |
| -------------------------------------- | -------------------------------------------------------------------- | -------- |
| `make eval-retrieval`                  | `ablation_outputs/retrieval_metrics.csv` (recall/precision/nDCG/MRR) | ~30 min  |
| `make eval-latency`                    | `ablation_outputs/latency_profile.csv` (P50/P95/P99)                 | ~30 min  |
| `make eval-fast`                       | `ablation_outputs/phi3_stratified20/ablation_results.csv`            | ~2.5 h   |

## 10. PSI drift baseline

The PSI drift detector (`mlops.drift_detector`, `# PAPER CONTRIBUTION`)
needs a reference embedding distribution to compare against. Capture it once
after ingestion (and after every KB refresh):

```bash
docker compose exec -T fastapi sh -c 'cd /app && python -c "
from mlops.drift_detector import capture_baseline
from data.ingest import get_embeddings
emb = get_embeddings(chunk_size=512)
capture_baseline(emb)
print(\"shape:\", emb.shape)
"'
```

Equivalent CLI: `docker compose exec fastapi python scripts/run_baseline_eval.py --skip-eval --skip-xai`.

The host-side `make baseline-host` will fail unless `onnxruntime` is
installed locally — easier to run inside the container.
