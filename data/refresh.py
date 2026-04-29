# data/refresh.py
import argparse
from datetime import datetime, timedelta
from typing import Any

from src.fetcher import search_pmids, fetch_records_xml
from src.parser import parse_xml_to_records, write_jsonl
from src.chroma_interface import incremental_upsert

JSONL_PATH = "data/processed/pubmed_processed.jsonl"

QUERIES = [
    "clinical practice guidelines",
    "drug drug interactions",
    "randomized controlled trial treatment",
    "differential diagnosis",
]


def build_date_query(query: str, days_back: int) -> str:
    end = datetime.today()
    start = end - timedelta(days=days_back)
    date_filter = f"{start.strftime('%Y/%m/%d')}:{end.strftime('%Y/%m/%d')}[dp]"
    return f"({query}) AND {date_filter}"


def fetch_new_records(
    days_back: int = 7, max_per_query: int = 100
) -> list[dict[str, Any]]:
    """Fetch the latest PubMed records across all configured QUERIES.

    Used by mlops.refresh_trigger to pull new documents during KB refresh.

    Args:
        days_back:     Days of history to pull from PubMed.
        max_per_query: Max records per query in QUERIES.

    Returns:
        Combined list of parsed record dicts suitable for incremental_upsert().
    """
    all_records: list[dict[str, Any]] = []
    for query in QUERIES:
        dated_query = build_date_query(query, days_back)
        pmids = search_pmids(dated_query, max_per_query)
        if not pmids:
            continue
        xml = fetch_records_xml(pmids)
        records = parse_xml_to_records(xml)
        if records:
            all_records.extend(records)
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh PubMed data")
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="How many days back to fetch (default: 7)",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=100,
        help="Max new records per query (default: 100)",
    )
    args = parser.parse_args()

    print(f"Fetching new PubMed abstracts from last {args.days_back} days...\n")
    total_new = 0

    for query in QUERIES:
        dated_query = build_date_query(query, args.days_back)
        print(f"Query: {query}")
        pmids = search_pmids(dated_query, args.max_per_query)

        if not pmids:
            print("  No new records found.\n")
            continue

        xml = fetch_records_xml(pmids)
        records = parse_xml_to_records(xml)

        if not records:
            print("  No valid records parsed.\n")
            continue

        written = write_jsonl(records, JSONL_PATH)
        incremental_upsert(records)
        total_new += written
        print(f"  Added {written} new records.\n")

    print(f"Refresh complete. Total new records added: {total_new}")


if __name__ == "__main__":
    main()
