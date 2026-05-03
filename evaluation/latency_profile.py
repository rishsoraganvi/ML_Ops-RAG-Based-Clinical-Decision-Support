"""
latency_profile.py — per-config latency aggregation for the paper.

For each of the 9 ablation configs, run query() over the benchmark set and
record retrieval_latency_ms, llm_latency_ms, total_latency_ms per question.
Aggregate to P50 / P95 / P99 + mean / std per config.

The chain returns these three latency fields natively (rag_pipeline/chain.py:190-192);
this script just loops over (chunk_size x retriever) and percentiles them.

Usage
-----
    python evaluation/latency_profile.py --qa-file evaluation/benchmarks/qa_pairs_stratified20.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from rag_pipeline.chain import query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ragops.latency_profile")

CHUNK_SIZES = [256, 512, 1024]
RETRIEVERS = ["bm25", "dense", "hybrid"]
TOP_K = 5

LATENCY_FIELDS = ("retrieval_latency_ms", "llm_latency_ms", "total_latency_ms")


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _profile_config(
    chunk_size: int,
    retriever_type: str,
    qa_pairs: list[dict],
    top_k: int = TOP_K,
) -> tuple[dict, list[dict]]:
    config_id = f"{retriever_type}_chunk{chunk_size}"
    cfg = {"chunk_size": chunk_size, "retriever_type": retriever_type, "k": top_k}
    log.info("[%s] start — n=%d", config_id, len(qa_pairs))

    per_q: list[dict] = []
    t0 = time.perf_counter()
    for i, qa in enumerate(qa_pairs, start=1):
        try:
            result = query(qa["question"], config=cfg, mlflow_run=False)
        except Exception as exc:
            log.warning("[%s] q=%d query failed: %s", config_id, i, exc)
            continue
        per_q.append(
            {
                "config_id": config_id,
                "question_id": qa.get("id", f"Q{i:03d}"),
                "retrieval_latency_ms": result["retrieval_latency_ms"],
                "llm_latency_ms": result["llm_latency_ms"],
                "total_latency_ms": result["total_latency_ms"],
                "answer_len_chars": len(result["answer"]),
            }
        )
    elapsed = time.perf_counter() - t0

    summary: dict = {
        "config_id": config_id,
        "chunk_size": chunk_size,
        "retriever": retriever_type,
        "top_k": top_k,
        "n_questions": len(per_q),
        "wall_time_s": round(elapsed, 2),
    }
    for field in LATENCY_FIELDS:
        stats = _percentiles([row[field] for row in per_q])
        for stat_name, val in stats.items():
            summary[f"{field}_{stat_name}"] = round(val, 2)

    log.info(
        "[%s] done in %.1fs | total_p50=%.0fms total_p95=%.0fms",
        config_id,
        elapsed,
        summary["total_latency_ms_p50"],
        summary["total_latency_ms_p95"],
    )
    return summary, per_q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_outputs"))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    qa_pairs = json.loads(args.qa_file.read_text())
    log.info("Loaded %d QA pairs from %s", len(qa_pairs), args.qa_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    all_per_q: list[dict] = []
    for chunk_size, retriever_type in product(CHUNK_SIZES, RETRIEVERS):
        summary, per_q = _profile_config(
            chunk_size=chunk_size,
            retriever_type=retriever_type,
            qa_pairs=qa_pairs,
            top_k=args.top_k,
        )
        rows.append(summary)
        all_per_q.extend(per_q)

    summary_path = args.output_dir / "latency_profile.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    log.info("Wrote %s (%d rows)", summary_path, len(rows))

    perq_path = args.output_dir / "latency_profile_per_question.csv"
    pd.DataFrame(all_per_q).to_csv(perq_path, index=False)
    log.info("Wrote %s (%d rows)", perq_path, len(all_per_q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
