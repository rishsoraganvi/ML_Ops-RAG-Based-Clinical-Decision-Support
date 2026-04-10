# data/refresh.py
import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fetcher import search_pmids, fetch_records_xml
from parser import parse_xml_to_records, write_jsonl
from chroma_interface import incremental_upsert

JSONL_PATH = "data/processed/pubmed_processed.jsonl"

QUERIES = [
    "clinical practice guidelines",
    "drug drug interactions",
    "randomized controlled trial treatment",
    "differential diagnosis",
]


def build_date_query(query: str, days_back: int) -> str:
    end   = datetime.today()
    start = end - timedelta(days=days_back)
    date_filter = (
        f"{start.strftime('%Y/%m/%d')}:{end.strftime('%Y/%m/%d')}[dp]"
    )
    return f"({query}) AND {date_filter}"


def main():
    parser = argparse.ArgumentParser(description="Refresh PubMed data")
    parser.add_argument(
        "--days-back", type=int, default=7,
        help="How many days back to fetch (default: 7)"
    )
    parser.add_argument(
        "--max-per-query", type=int, default=100,
        help="Max new records per query (default: 100)"
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

        xml     = fetch_records_xml(pmids)
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