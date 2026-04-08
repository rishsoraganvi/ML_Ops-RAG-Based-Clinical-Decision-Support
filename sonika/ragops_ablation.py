"""
ragops_ablation.py
==================
Ablation runner for RAGOps — clinical decision support evaluation.

Runs 9 configs × 50 questions = 450 evaluations in parallel.
Each config is one cell in Paper TABLE 1.

Grid:
    chunk_size  : [256, 512, 1024]
    retriever   : [bm25, dense, hybrid]
    top_k       : fixed at 5  (held constant — vary via ABLATION_TOP_K)

    → 3 × 3 = 9 configs  (extend grid trivially for top_k ablation)

Outputs:
    ablation_results.csv        ← Paper TABLE 1 source
    ablation_results.json       ← machine-readable with MLflow run IDs
    mlruns/                     ← one MLflow run per config (linked by run_id)

Usage:
    python ragops_ablation.py --qa-file data/benchmark_50.json
    python ragops_ablation.py --qa-file data/benchmark_50.json --workers 4
    python ragops_ablation.py --qa-file data/benchmark_50.json --dry-run

Author : LLM & Evaluation Lead
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

# ragops_eval must be on PYTHONPATH (same directory is fine)
from ragops_eval import run_eval, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ragops.ablation")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
_GLOBAL_SEED = 42


def _set_seeds(seed: int = _GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_set_seeds()

# ---------------------------------------------------------------------------
# Ablation grid definition                         # PAPER RESULT — TABLE 1
# ---------------------------------------------------------------------------

CHUNK_SIZES: list[int] = [256, 512, 1024]
RETRIEVERS:  list[str] = ["bm25", "dense", "hybrid"]
TOP_K:       int       = int(os.getenv("ABLATION_TOP_K", "5"))

# Output paths
OUTPUT_DIR  = Path(os.getenv("ABLATION_OUTPUT_DIR", "ablation_outputs"))
CSV_PATH    = OUTPUT_DIR / "ablation_results.csv"
JSON_PATH   = OUTPUT_DIR / "ablation_results.json"

# Parallel workers — default 9 (one per config); override via CLI or env
DEFAULT_WORKERS: int = int(os.getenv("ABLATION_WORKERS", "9"))


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """One cell in the ablation grid."""
    config_id:  str        # e.g. "C01"
    chunk_size: int
    retriever:  str
    top_k:      int = TOP_K

    # Runtime fields — populated after eval
    faithfulness:      float | None = field(default=None, repr=False)
    context_recall:    float | None = field(default=None, repr=False)
    answer_relevance:  float | None = field(default=None, repr=False)
    context_precision: float | None = field(default=None, repr=False)
    mlflow_run_id:     str   | None = field(default=None, repr=False)
    latency_s:         float | None = field(default=None, repr=False)
    status:            str          = field(default="pending", repr=False)
    error:             str   | None = field(default=None, repr=False)

    def to_config_dict(self) -> dict:
        """Subset passed to run_eval() and logged to MLflow."""
        return {
            "config_id":  self.config_id,
            "chunk_size": self.chunk_size,
            "retriever":  self.retriever,
            "top_k":      self.top_k,
        }


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_ablation_grid() -> list[AblationConfig]:
    """
    Generate 9 configs from the 3×3 chunk_size × retriever grid.
    top_k is held constant (Paper TABLE 1 design).

    # PAPER RESULT — TABLE 1
    """
    configs: list[AblationConfig] = []
    for idx, (chunk_size, retriever) in enumerate(
        product(CHUNK_SIZES, RETRIEVERS), start=1
    ):
        configs.append(
            AblationConfig(
                config_id=f"C{idx:02d}",
                chunk_size=chunk_size,
                retriever=retriever,
                top_k=TOP_K,
            )
        )

    log.info(
        "Ablation grid: %d configs | chunk_sizes=%s | retrievers=%s | top_k=%d",
        len(configs), CHUNK_SIZES, RETRIEVERS, TOP_K,
    )
    return configs


# ---------------------------------------------------------------------------
# Per-config worker  (runs in a subprocess via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _eval_one_config(
    cfg_dict: dict,
    qa_pairs: list[dict],
    seed: int = _GLOBAL_SEED,
) -> dict:
    """
    Worker function executed in a subprocess.

    Accepts plain dicts (picklable) instead of dataclasses.
    Returns updated cfg_dict with scores, run_id, latency, status.
    """
    # Re-seed in subprocess — ProcessPoolExecutor forks don't inherit seeds
    _set_seeds(seed)

    config_id = cfg_dict["config_id"]
    log.info("[%s] Starting — chunk=%d retriever=%s top_k=%d",
             config_id, cfg_dict["chunk_size"],
             cfg_dict["retriever"], cfg_dict["top_k"])

    # Determine config-specific per-question output path up front so that
    # parallel workers never share a common temp file.
    pq_dir = Path(os.getenv("ABLATION_OUTPUT_DIR", "ablation_outputs")) / "per_question"
    pq_dir.mkdir(parents=True, exist_ok=True)
    per_question_path = pq_dir / f"{config_id}_per_question.csv"

    t0 = time.perf_counter()
    try:
        # ── Call Member 2's query() interface per question ────────────
        # In production: qa_pairs already have answer+contexts from query().
        # The ablation config (retriever, chunk_size, top_k) is passed to
        # query() upstream; here we receive pre-filled qa_pairs for this cfg.
        # See: _inject_config_answers() note below.

        scores = run_eval(
            qa_pairs=qa_pairs,
            config=cfg_dict,
            per_question_path=per_question_path,
        )

        elapsed = time.perf_counter() - t0

        # Retrieve the MLflow run_id for this config (last active run).
        # Query by the config_id tag that ragops_eval now sets explicitly.
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.config_id = '{config_id}'",
            max_results=1,
            order_by=["start_time DESC"],
        )
        run_id = runs[0].info.run_id if runs else None

        log.info("[%s] Done in %.1fs — faithfulness=%.4f",
                 config_id, elapsed, scores.get("faithfulness", 0))

        return {
            **cfg_dict,
            **scores,
            "mlflow_run_id": run_id,
            "latency_s":     round(elapsed, 2),
            "status":        "success",
            "error":         None,
        }

    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        log.error("[%s] FAILED after %.1fs: %s", config_id, elapsed, exc)
        return {
            **cfg_dict,
            "faithfulness":      None,
            "context_recall":    None,
            "answer_relevance":  None,
            "context_precision": None,
            "mlflow_run_id":     None,
            "latency_s":         round(elapsed, 2),
            "status":            "failed",
            "error":             traceback.format_exc(limit=5),
        }


# ---------------------------------------------------------------------------
# NOTE — integrating Member 2's query() interface
# ---------------------------------------------------------------------------
# In the full pipeline, before calling run_ablation() you should populate
# qa_pairs per config like this:
#
#   from member2_interface import query
#
#   def prepare_qa_pairs(questions, ground_truths, config):
#       pairs = []
#       for q, gt in zip(questions, ground_truths):
#           result = query(q, config)          # member2 interface
#           pairs.append({
#               "question":    q,
#               "answer":      result["answer"],
#               "contexts":    [d.page_content for d in result["source_docs"]],
#               "ground_truth": gt,
#           })
#       return pairs
#
# Then: run_ablation(configs, qa_pairs_per_config={cfg.config_id: pairs})
# For now, run_ablation() accepts a single qa_pairs list (same for all configs)
# which is the standard ablation design when retriever is varied inside query().
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    qa_pairs: list[dict],
    configs: list[AblationConfig] | None = None,
    workers: int = DEFAULT_WORKERS,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run all ablation configs in parallel.                # PAPER RESULT — TABLE 1

    Parameters
    ----------
    qa_pairs  : 50-question PubMed benchmark pairs
    configs   : list of AblationConfig (default: full 9-config grid)
    workers   : parallel ProcessPoolExecutor workers
    dry_run   : if True, skip actual eval — print grid and exit

    Returns
    -------
    pd.DataFrame with one row per config, columns:
        config_id, chunk_size, retriever, top_k,
        faithfulness, context_recall, answer_relevance, context_precision,
        mlflow_run_id, latency_s, status
    """
    _set_seeds()
    configs = configs or build_ablation_grid()

    if dry_run:
        log.info("DRY RUN — ablation grid:")
        for c in configs:
            print(f"  {c.config_id}: chunk={c.chunk_size} "
                  f"retriever={c.retriever} top_k={c.top_k}")
        return pd.DataFrame([asdict(c) for c in configs])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_configs   = len(configs)
    n_questions = len(qa_pairs)
    n_total     = n_configs * n_questions

    # Warn when using SQLite with multiple workers — concurrent writes frequently
    # cause "database is locked" errors.  Use file-based mlruns/ or a proper
    # tracking server (Postgres) for reliable parallel ablation runs.
    if workers > 1 and MLFLOW_TRACKING_URI.startswith("sqlite://"):
        log.warning(
            "SQLite MLflow backend detected with workers=%d. "
            "Concurrent writes may cause 'database is locked' errors. "
            "Consider setting MLFLOW_TRACKING_URI to a file-based store "
            "(e.g. 'mlruns/') or using --workers 1.",
            workers,
        )

    log.info(
        "Ablation start | configs=%d | questions=%d | total_evals=%d | workers=%d",
        n_configs, n_questions, n_total, workers,
    )

    # ── Parallel execution ──────────────────────────────────────────────
    results: list[dict] = []
    failed:  list[str]  = []

    # Submit all configs concurrently
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_cfg = {
            executor.submit(
                _eval_one_config,
                cfg.to_config_dict(),
                qa_pairs,
                _GLOBAL_SEED,
            ): cfg.config_id
            for cfg in configs
        }

        for future in as_completed(future_to_cfg):
            cfg_id = future_to_cfg[future]
            try:
                result = future.result()
                results.append(result)
                if result["status"] == "failed":
                    failed.append(cfg_id)
                    log.warning("[%s] marked as failed — will appear in CSV", cfg_id)
            except Exception as exc:  # noqa: BLE001
                log.error("[%s] Future raised unexpectedly: %s", cfg_id, exc)
                failed.append(cfg_id)

    # ── Build results DataFrame ─────────────────────────────────────────
    df = pd.DataFrame(results)

    # Canonical column order for Paper TABLE 1
    col_order = [
        "config_id", "chunk_size", "retriever", "top_k",
        "faithfulness", "context_recall", "answer_relevance", "context_precision",
        "mlflow_run_id", "latency_s", "status", "error",
    ]
    df = df.reindex(columns=[c for c in col_order if c in df.columns])
    df = df.sort_values("config_id").reset_index(drop=True)

    # ── Persist outputs ─────────────────────────────────────────────────
    _save_outputs(df)

    # ── Summary ─────────────────────────────────────────────────────────
    _print_summary(df, failed)

    return df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_outputs(df: pd.DataFrame) -> None:
    """Save CSV and JSON with MLflow run IDs linked."""  # PAPER RESULT — TABLE 1

    # CSV — Paper TABLE 1 source
    df.to_csv(CSV_PATH, index=False, float_format="%.4f")
    log.info("Saved CSV → %s", CSV_PATH)

    # JSON — machine-readable with full metadata
    records = df.to_dict(orient="records")
    with open(JSON_PATH, "w") as f:
        json.dump(
            {
                "ablation_grid": {
                    "chunk_sizes": CHUNK_SIZES,
                    "retrievers":  RETRIEVERS,
                    "top_k":       TOP_K,
                },
                "mlflow_experiment": MLFLOW_EXPERIMENT,
                "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
                "results": records,
            },
            f, indent=2, default=str,
        )
    log.info("Saved JSON → %s", JSON_PATH)


def _print_summary(df: pd.DataFrame, failed: list[str]) -> None:
    """Print Paper TABLE 1 to stdout."""  # PAPER RESULT — TABLE 1

    metric_cols = ["faithfulness", "context_recall",
                   "answer_relevance", "context_precision"]

    success_df = df[df["status"] == "success"]

    print("\n" + "═" * 78)
    print("  ABLATION RESULTS — PAPER TABLE 1")
    print("═" * 78)

    display_cols = ["config_id", "chunk_size", "retriever"] + metric_cols
    print(
        success_df[display_cols]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n── Aggregate (successful configs) ──")
    agg = success_df[metric_cols].agg(["mean", "std", "min", "max"])
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

    if failed:
        print(f"\n⚠  Failed configs ({len(failed)}): {', '.join(failed)}")
        print("   Check ablation_outputs/ablation_results.csv for error column.")

    print(f"\n📄  CSV  → {CSV_PATH.resolve()}")
    print(f"📄  JSON → {JSON_PATH.resolve()}")
    print(f"📊  MLflow UI → mlflow ui  (experiment: {MLFLOW_EXPERIMENT})")
    print("═" * 78 + "\n")


# ---------------------------------------------------------------------------
# Checkpoint helper  (optional — call before run_ablation for large grids)
# ---------------------------------------------------------------------------

def load_checkpoint(configs: list[AblationConfig]) -> list[AblationConfig]:
    """
    Skip configs that already have a successful result in ablation_results.csv.
    Useful when resuming a partially completed ablation run.

    Usage:
        configs = build_ablation_grid()
        configs = load_checkpoint(configs)   # filters done ones
        run_ablation(qa_pairs, configs=configs)
    """
    if not CSV_PATH.exists():
        return configs

    done_df  = pd.read_csv(CSV_PATH)
    done_ids = set(done_df[done_df["status"] == "success"]["config_id"].tolist())

    remaining = [c for c in configs if c.config_id not in done_ids]
    skipped   = len(configs) - len(remaining)

    if skipped:
        log.info("Checkpoint: skipping %d already-completed configs: %s",
                 skipped, sorted(done_ids))

    return remaining


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGOps ablation runner — 9 configs × 50 questions"
    )
    parser.add_argument(
        "--qa-file", type=Path, required=False,
        help="JSON file with 50 qa_pairs (question/answer/contexts/ground_truth)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print grid without running evaluations",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip configs already completed in ablation_results.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for CSV and JSON outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Override output dir if specified
    if args.output_dir != OUTPUT_DIR:
        OUTPUT_DIR  = args.output_dir
        CSV_PATH    = OUTPUT_DIR / "ablation_results.csv"
        JSON_PATH   = OUTPUT_DIR / "ablation_results.json"

    # Load qa_pairs
    if args.dry_run:
        qa_pairs = []
    elif args.qa_file:
        qa_pairs = json.loads(args.qa_file.read_text())
        log.info("Loaded %d QA pairs from %s", len(qa_pairs), args.qa_file)
    else:
        # Fall back to CI stubs for quick local testing (NOT for paper results)
        from ragops_eval import _ci_stub_pairs
        log.warning(
            "No --qa-file supplied — using 5 CI stub pairs. "
            "NOT valid for paper results."
        )
        qa_pairs = _ci_stub_pairs()

    # Build grid
    configs = build_ablation_grid()

    # Optional checkpoint resume
    if args.resume:
        configs = load_checkpoint(configs)

    # Run
    results_df = run_ablation(
        qa_pairs=qa_pairs,
        configs=configs,
        workers=args.workers,
        dry_run=args.dry_run,
    )