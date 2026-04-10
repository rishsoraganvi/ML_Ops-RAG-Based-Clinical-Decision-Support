# data/quality_report.py
import json
import os
from collections import Counter
from pathlib import Path

JSONL_PATH = "data/processed/pubmed_processed.jsonl"
OUTPUT_HTML = "data/quality_report.html"


def load_records(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_quality_checks(records: list) -> dict:
    pmids = [r["pmid"] for r in records]
    word_counts = [r["word_count"] for r in records]
    pub_dates = [r["pub_date"] for r in records if r["pub_date"] != "unknown"]
    mesh_terms = [term for r in records for term in r.get("mesh_terms", [])]
    records_with_mesh = sum(1 for r in records if r.get("mesh_terms"))

    duplicate_pmids = [p for p, c in Counter(pmids).items() if c > 1]
    date_counts = Counter(d[:4] for d in pub_dates)

    return {
        "total_records":        len(records),
        "duplicate_pmids":      len(duplicate_pmids),
        "duplicate_list":       duplicate_pmids[:10],
        "min_word_count":       min(word_counts),
        "max_word_count":       max(word_counts),
        "avg_word_count":       round(sum(word_counts) / len(word_counts), 1),
        "records_under_50":     sum(1 for w in word_counts if w < 50),
        "records_with_mesh":    records_with_mesh,
        "mesh_coverage_pct":    round(records_with_mesh / len(records) * 100, 1),
        "top_mesh_terms":       Counter(mesh_terms).most_common(10),
        "date_distribution":    sorted(date_counts.items()),
        "unknown_dates":        sum(1 for r in records if r["pub_date"] == "unknown"),
        "missing_abstract":     sum(1 for r in records if not r.get("abstract")),
        "missing_title":        sum(1 for r in records if not r.get("title")),
    }


def generate_html(stats: dict, output_path: str):
    rows_date = "".join(
        f"<tr><td>{yr}</td><td>{cnt}</td></tr>"
        for yr, cnt in stats["date_distribution"]
    )
    rows_mesh = "".join(
        f"<tr><td>{term}</td><td>{cnt}</td></tr>"
        for term, cnt in stats["top_mesh_terms"]
    )
    dup_warning = (
        f'<p style="color:red">Found {stats["duplicate_pmids"]} duplicate PMIDs!</p>'
        if stats["duplicate_pmids"] > 0 else
        '<p style="color:green">No duplicate PMIDs found.</p>'
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>RAGOps Data Quality Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
    h1 {{ color: #2c3e50; }}
    h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 600px; margin-bottom: 30px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #f4f4f4; font-weight: bold; }}
    .stat {{ background: #eaf4fb; border-left: 4px solid #3498db;
             padding: 10px 16px; margin: 8px 0; border-radius: 4px; }}
    .good {{ border-color: green; background: #eaffea; }}
    .warn {{ border-color: orange; background: #fff8e1; }}
    .bad  {{ border-color: red;   background: #ffeaea; }}
  </style>
</head>
<body>
  <h1>RAGOps — Data Quality Report</h1>
  <p>Generated from: <code>{JSONL_PATH}</code></p>

  <h2>Overview</h2>
  <div class="stat good">Total records: <strong>{stats["total_records"]}</strong></div>
  <div class="stat {'bad' if stats['duplicate_pmids'] > 0 else 'good'}">
    Duplicate PMIDs: <strong>{stats["duplicate_pmids"]}</strong>
  </div>
  <div class="stat {'warn' if stats['missing_abstract'] > 0 else 'good'}">
    Missing abstracts: <strong>{stats["missing_abstract"]}</strong>
  </div>
  <div class="stat {'warn' if stats['missing_title'] > 0 else 'good'}">
    Missing titles: <strong>{stats["missing_title"]}</strong>
  </div>

  <h2>Abstract length distribution</h2>
  <div class="stat">Min word count: <strong>{stats["min_word_count"]}</strong></div>
  <div class="stat">Max word count: <strong>{stats["max_word_count"]}</strong></div>
  <div class="stat">Average word count: <strong>{stats["avg_word_count"]}</strong></div>
  <div class="stat {'warn' if stats['records_under_50'] > 0 else 'good'}">
    Records under 50 words (too short): <strong>{stats["records_under_50"]}</strong>
  </div>

  <h2>MeSH term coverage</h2>
  <div class="stat">Records with MeSH terms: <strong>{stats["records_with_mesh"]}</strong>
    ({stats["mesh_coverage_pct"]}%)</div>
  <table>
    <tr><th>MeSH term</th><th>Count</th></tr>
    {rows_mesh}
  </table>

  <h2>Publication date distribution</h2>
  <div class="stat warn">Unknown dates: <strong>{stats["unknown_dates"]}</strong></div>
  <table>
    <tr><th>Year</th><th>Records</th></tr>
    {rows_date}
  </table>

  {dup_warning}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Quality report saved to: {output_path}")


if __name__ == "__main__":
    print("Loading records...")
    records = load_records(JSONL_PATH)
    print(f"Loaded {len(records)} records. Running checks...")
    stats = run_quality_checks(records)
    generate_html(stats, OUTPUT_HTML)
    print("\nSummary:")
    print(f"  Total records:     {stats['total_records']}")
    print(f"  Duplicates:        {stats['duplicate_pmids']}")
    print(f"  Avg word count:    {stats['avg_word_count']}")
    print(f"  MeSH coverage:     {stats['mesh_coverage_pct']}%")
    print(f"  Unknown dates:     {stats['unknown_dates']}")