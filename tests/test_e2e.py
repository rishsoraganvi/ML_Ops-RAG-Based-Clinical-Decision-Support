"""
End-to-end integration tests for the full RAG pipeline.

Gated by ``RAGOPS_E2E=1`` so CI (which runs without a live docker stack
for the unit-tests job) does not accidentally execute them. The dedicated
``ragas-ci-eval`` CI job brings the stack up itself and exercises the
same surface.

Prerequisites:
    - docker compose up -d chromadb mlflow ollama fastapi
    - python scripts/healthcheck.py   (all services healthy)
    - ChromaDB populated at chunk_size=512 (run the ingest command once)

Run:
    RAGOPS_E2E=1 pytest tests/test_e2e.py -m integration -v
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RAGOPS_E2E") != "1",
        reason="End-to-end test — set RAGOPS_E2E=1 to run against a live stack.",
    ),
]


def test_full_pipeline() -> None:
    """Ingest → query → evaluate — verifies the live stack round-trips."""
    # ── 1. Healthcheck ─────────────────────────────────────────────────────
    from scripts import healthcheck

    assert healthcheck.run_once(healthcheck.SERVICES), (
        "healthcheck failed — ensure docker compose is up and services report "
        "healthy before running e2e tests."
    )

    # ── 2. Ingest a handful of sample docs ─────────────────────────────────
    from langchain.schema import Document

    from rag_pipeline.ingest import ingest_documents

    sample_docs = [
        Document(
            page_content=(
                f"Sample PubMed abstract {i}: Metformin is first-line pharmacotherapy "
                f"for type 2 diabetes mellitus in adults without contraindications."
            ),
            metadata={"source": f"e2e-sample-{i}", "pmid": f"e2e{i}"},
        )
        for i in range(10)
    ]
    summary = ingest_documents(sample_docs, chunk_size=512)
    assert (
        summary.get("chunks_added", 0) >= 1 or summary.get("total_chunks", 0) >= 1
    ), f"ingest_documents returned unexpected summary: {summary}"

    # ── 3. Run three queries through the chain ─────────────────────────────
    from rag_pipeline.chain import query as rag_query

    questions = [
        "What is the first-line treatment for type 2 diabetes?",
        "When is metformin contraindicated?",
        "Which medication class is preferred for newly-diagnosed adults?",
    ]
    for q in questions:
        result = rag_query(q)
        assert result["answer"], f"empty answer for question: {q}"
        assert (
            len(result["retrieval_scores"]) >= 1
        ), f"expected at least one retrieval score for: {q}"
        assert len(result["source_docs"]) >= 1

    # ── 4. RAGAS CI evaluation — all four metrics must be non-negative ────
    from evaluation.ragas_runner import run_ci_eval

    t_before = time.time()
    scores = run_ci_eval()
    for metric in (
        "faithfulness",
        "context_recall",
        "answer_relevance",
        "context_precision",
    ):
        assert metric in scores, f"missing RAGAS metric: {metric}"
        assert scores[metric] >= 0.0, f"RAGAS {metric} was negative: {scores[metric]}"

    # ── 5. MLflow must have logged at least one run in the last 60 s ──────
    import mlflow

    recent_cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=max(120, int(time.time() - t_before) + 60)
    )
    runs = mlflow.search_runs(
        experiment_names=["ragops-ragas-eval"],
        filter_string=f"attributes.start_time >= {int(recent_cutoff.timestamp() * 1000)}",
        output_format="list",
    )
    assert len(runs) >= 1, (
        "No MLflow run logged in the ragops-ragas-eval experiment within the last minute — "
        "check the tracker wiring."
    )
