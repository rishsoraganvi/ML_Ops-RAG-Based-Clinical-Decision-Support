"""
run_baseline_eval.py — capture the Week-2 production baseline.

Performs three linked actions:

    1. Full RAGAS evaluation on the 50-question PubMed benchmark using
       the current best-guess config (hybrid retrieval, chunk_size=512,
       reranker ON). Logged to MLflow as ``baseline-eval-week2``.
    2. PSI drift baseline capture via ``mlops.drift_detector.capture_baseline``
       using the full embedding matrix from ``data.ingest.get_embeddings``.
    3. XAI consistency baseline capture for the first
       ``settings.xai_benchmark_questions`` questions — runs the RAG chain,
       builds an ``explanation_vector`` per question, and persists via
       ``mlops.explanation_monitor.save_baseline``.

Run once after initial ingestion, and again after every knowledge-base
refresh so PSI/XAI monitoring always compares against a current reference
distribution.

Usage:
    python scripts/run_baseline_eval.py
    python scripts/run_baseline_eval.py --skip-xai       # PSI + RAGAS only
    python scripts/run_baseline_eval.py --skip-eval      # baselines only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("ragops.baseline")


BASELINE_CONFIG = {
    "retriever_type": "hybrid",
    "chunk_size": 512,
    "k": 5,
    "reranker": True,
    "dense_weight": 0.6,
    "preprocess_query": True,
    "config_id": "baseline-week2",
}


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_qa = repo_root / "evaluation" / "benchmarks" / "qa_pairs.json"

    parser = argparse.ArgumentParser(description="Capture RAGOps Week-2 baseline.")
    parser.add_argument("--qa-file", type=Path, default=default_qa)
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip RAGAS evaluation"
    )
    parser.add_argument(
        "--skip-psi", action="store_true", help="Skip PSI baseline capture"
    )
    parser.add_argument(
        "--skip-xai", action="store_true", help="Skip XAI baseline capture"
    )
    return parser.parse_args()


def _run_ragas(qa_file: Path) -> dict:
    from evaluation.ragas_runner import run_eval

    qa_pairs = json.loads(qa_file.read_text(encoding="utf-8"))
    logger.info(
        "Running RAGAS on %d questions (config=%s)", len(qa_pairs), BASELINE_CONFIG
    )
    scores = run_eval(qa_pairs, config=BASELINE_CONFIG)
    logger.info("RAGAS scores: %s", scores)
    return scores


def _capture_psi() -> None:
    from data.ingest import get_embeddings
    from mlops.drift_detector import capture_baseline

    logger.info("Fetching embedding matrix for PSI baseline...")
    embeddings = get_embeddings(chunk_size=BASELINE_CONFIG["chunk_size"])
    logger.info("Capturing PSI baseline (shape=%s)", embeddings.shape)
    capture_baseline(embeddings)


def _capture_xai(qa_file: Path) -> int:
    from mlops.explanation_monitor import save_baseline
    from rag_pipeline.chain import query as rag_query
    from src.config.settings import settings
    from evaluation.explainability import explain

    qa_pairs = json.loads(qa_file.read_text(encoding="utf-8"))
    n = settings.xai_benchmark_questions
    subset = qa_pairs[:n]
    logger.info("Generating %d explanation vectors for XAI baseline...", len(subset))

    vectors = []
    for i, item in enumerate(subset, start=1):
        q = item["question"]
        try:
            result = rag_query(q, config=BASELINE_CONFIG)
            exp = explain(
                q,
                result["source_docs"],
                result["retrieval_scores"],
                answer=result["answer"],
            )
            vectors.append(exp["explanation_vector"])
            logger.info("  [%d/%d] %s", i, len(subset), q[:60])
        except Exception as exc:
            logger.warning("  [%d/%d] failed: %s", i, len(subset), exc)

    if not vectors:
        logger.error("No XAI vectors captured — baseline NOT saved")
        return 0

    save_baseline(vectors)
    logger.info("XAI baseline saved with %d vectors", len(vectors))
    return len(vectors)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = _parse_args()

    if not args.qa_file.exists():
        logger.error("QA file not found: %s", args.qa_file)
        return 1

    if not args.skip_eval:
        scores = _run_ragas(args.qa_file)
        print(json.dumps({"ragas_scores": scores}, indent=2))
    else:
        logger.info("Skipping RAGAS evaluation")

    if not args.skip_psi:
        _capture_psi()
    else:
        logger.info("Skipping PSI baseline")

    n_vectors = 0
    if not args.skip_xai:
        n_vectors = _capture_xai(args.qa_file)
    else:
        logger.info("Skipping XAI baseline")

    print(
        json.dumps(
            {
                "baseline_run": "baseline-week2",
                "xai_vectors_saved": n_vectors,
                "psi_captured": not args.skip_psi,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
