"""
Main ETL pipeline orchestrator.

Runs the full data pipeline: fetch PubMed records -> parse XML ->
preprocess -> write JSONL -> upsert to ChromaDB.

TODO: Saumya to commit actual implementation from local.
"""

from __future__ import annotations


def run_pipeline() -> None:
    """Execute the full PubMed ETL pipeline."""
    raise NotImplementedError("Awaiting commit from local ETL pipeline")


if __name__ == "__main__":
    run_pipeline()
