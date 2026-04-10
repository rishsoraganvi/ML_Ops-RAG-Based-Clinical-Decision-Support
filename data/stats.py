# data/stats.py
import json
from collections import Counter
from pathlib import Path

JSONL_PATH = "data/processed/pubmed_processed.jsonl"


def load_records(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_stats(records: list):
    word_counts  = [r["word_count"] for r in records]
    pub_dates    = [r["pub_date"] for r in records if r["pub_date"] != "unknown"]
    all_mesh     = [t for r in records for t in r.get("mesh_terms", [])]
    years        = sorted(set(d[:4] for d in pub_dates if len(d) >= 4))

    print("=" * 55)
    print("  RAGOps — Dataset Statistics (Table 0 for paper)")
    print("=" * 55)
    print(f"  Total abstracts:       {len(records)}")
    print(f"  Date range:            {years[0]} – {years[-1]}" if years else "  Date range: unknown")
    print(f"  Avg abstract length:   {round(sum(word_counts)/len(word_counts), 1)} words")
    print(f"  Min abstract length:   {min(word_counts)} words")
    print(f"  Max abstract length:   {max(word_counts)} words")
    print(f"  Unique MeSH terms:     {len(set(all_mesh))}")
    print(f"  MeSH coverage:         {round(sum(1 for r in records if r.get('mesh_terms'))/len(records)*100, 1)}%")
    print()
    print("  Top 5 medical categories (MeSH):")
    for term, count in Counter(all_mesh).most_common(5):
        print(f"    {term:<40} {count}")
    print("=" * 55)


if __name__ == "__main__":
    records = load_records(JSONL_PATH)
    print_stats(records)