"""
retrieval_only.py — judge-free retrieval ablation for the paper.

For each of the 9 ablation configs (chunk_size x retriever), retrieve top_k
chunks per question and score them against the ground-truth `contexts` field
in the benchmark file using cosine similarity over the same embedding model
that drives ChromaDB. No LLM judge is involved, so a full 50-question sweep
finishes in roughly 30 minutes.

Outputs
-------
ablation_outputs/retrieval_metrics.csv
    One row per (config_id, chunk_size, retriever): mean recall@k, precision@k,
    nDCG@k, MRR across all questions, plus per-question latency.

Usage
-----
    python evaluation/retrieval_only.py --qa-file evaluation/benchmarks/qa_pairs.json
    python evaluation/retrieval_only.py --qa-file evaluation/benchmarks/qa_pairs_stratified20.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

from rag_pipeline.retriever import get_retriever
from rag_pipeline.vectorstore import get_embedding_function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ragops.retrieval_only")

CHUNK_SIZES = [256, 512, 1024]
RETRIEVERS = ["bm25", "dense", "hybrid"]
TOP_K = 5
SIM_THRESHOLD = 0.70


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def _score_one_question(
    retrieved_texts: list[str],
    truth_texts: list[str],
    embed_fn,
    threshold: float = SIM_THRESHOLD,
) -> dict:
    """recall@k / precision@k / nDCG@k / MRR for a single question."""
    if not retrieved_texts or not truth_texts:
        return {"recall": 0.0, "precision": 0.0, "ndcg": 0.0, "mrr": 0.0}

    retrieved_emb = np.array(embed_fn.embed_documents(retrieved_texts))
    truth_emb = np.array(embed_fn.embed_documents(truth_texts))
    sim = _cosine_matrix(retrieved_emb, truth_emb)  # (k, n_truth)

    matched_truth = sim.max(axis=0) >= threshold
    recall = float(matched_truth.mean())

    retrieved_is_relevant = sim.max(axis=1) >= threshold
    precision = float(retrieved_is_relevant.mean())

    rel_at_rank = retrieved_is_relevant.astype(float)
    dcg = float(sum(r / math.log2(i + 2) for i, r in enumerate(rel_at_rank)))
    ideal_rel = sorted(rel_at_rank, reverse=True)
    idcg = float(sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rel))) or 1.0
    ndcg = dcg / idcg

    mrr = 0.0
    for i, r in enumerate(rel_at_rank):
        if r > 0:
            mrr = 1.0 / (i + 1)
            break

    return {"recall": recall, "precision": precision, "ndcg": ndcg, "mrr": mrr}


def _run_config(
    chunk_size: int,
    retriever_type: str,
    qa_pairs: list[dict],
    embed_fn,
    top_k: int = TOP_K,
) -> dict:
    config_id = f"{retriever_type}_chunk{chunk_size}"
    log.info("[%s] start — n=%d", config_id, len(qa_pairs))

    retriever = get_retriever(
        retriever_type=retriever_type, chunk_size=chunk_size, k=top_k
    )

    per_q_metrics: list[dict] = []
    t0 = time.perf_counter()
    for i, qa in enumerate(qa_pairs, start=1):
        q = qa["question"]
        truth = qa.get("contexts") or []
        try:
            docs = retriever.invoke(q)
        except Exception as exc:
            log.warning("[%s] q=%d retrieval failed: %s", config_id, i, exc)
            docs = []
        retrieved = [d.page_content for d in docs[:top_k]]
        scored = _score_one_question(retrieved, truth, embed_fn)
        scored["question_id"] = qa.get("id", f"Q{i:03d}")
        per_q_metrics.append(scored)
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(per_q_metrics)
    summary = {
        "config_id": config_id,
        "chunk_size": chunk_size,
        "retriever": retriever_type,
        "top_k": top_k,
        "n_questions": len(qa_pairs),
        "recall_at_k": float(df["recall"].mean()),
        "precision_at_k": float(df["precision"].mean()),
        "ndcg_at_k": float(df["ndcg"].mean()),
        "mrr": float(df["mrr"].mean()),
        "latency_s_total": round(elapsed, 2),
        "latency_s_per_q": round(elapsed / max(len(qa_pairs), 1), 3),
    }
    log.info(
        "[%s] done in %.1fs | recall=%.3f precision=%.3f ndcg=%.3f mrr=%.3f",
        config_id,
        elapsed,
        summary["recall_at_k"],
        summary["precision_at_k"],
        summary["ndcg_at_k"],
        summary["mrr"],
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-file",
        type=Path,
        required=True,
        help="JSON file with question/contexts entries",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ablation_outputs"),
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    qa_pairs = json.loads(args.qa_file.read_text())
    log.info("Loaded %d QA pairs from %s", len(qa_pairs), args.qa_file)

    embed_fn = get_embedding_function()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for chunk_size, retriever_type in product(CHUNK_SIZES, RETRIEVERS):
        summary = _run_config(
            chunk_size=chunk_size,
            retriever_type=retriever_type,
            qa_pairs=qa_pairs,
            embed_fn=embed_fn,
            top_k=args.top_k,
        )
        rows.append(summary)

    out_path = args.output_dir / "retrieval_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
