"""
PubMed NCBI E-utilities integration.

Provides esearch + efetch wrappers for fetching PubMed abstracts
with rate limiting and exponential backoff.

TODO: Saumya to commit actual implementation from local.
"""

from __future__ import annotations


def search_pmids(query: str, max_results: int = 100) -> list[str]:
    """Search PubMed and return matching PMIDs."""
    raise NotImplementedError("Awaiting commit from local ETL pipeline")


def fetch_records_xml(pmids: list[str]) -> str:
    """Fetch PubMed XML records for the given PMIDs."""
    raise NotImplementedError("Awaiting commit from local ETL pipeline")
