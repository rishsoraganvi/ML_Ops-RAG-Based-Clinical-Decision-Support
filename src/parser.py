"""
PubMed XML parser.

Converts raw NCBI efetch XML into structured JSONL records
with fields: pmid, title, abstract, text, pub_date, mesh_terms, word_count.

TODO: Saumya to commit actual implementation from local.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_xml_to_records(xml: str) -> list[dict[str, Any]]:
    """Parse PubMed XML into a list of record dicts."""
    raise NotImplementedError("Awaiting commit from local ETL pipeline")


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Append records to a JSONL file. Returns count of records written."""
    raise NotImplementedError("Awaiting commit from local ETL pipeline")
