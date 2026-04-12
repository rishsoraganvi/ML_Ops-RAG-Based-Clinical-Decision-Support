# RAGOps — Progress Report
**Name:** Saumya Priyadarshinee
**GitHub:** saumyaapriyadarshinee
**Date:** 01 April 2026

---

## Phase 1 — PubMed ETL Pipeline ✅ COMPLETE

| Task | Status | Details |
|------|--------|---------|
| Project setup | ✅ Done | Folders, requirements.txt, .env configured |
| PubMed API Integration | ✅ Done | NCBI E-utilities esearch + efetch |
| Fetch Volume | ✅ Done | 3,706 records fetched |
| Text Preprocessing | ✅ Done | All 7 fields present |
| Output Format (JSONL) | ✅ Done | Schema verified |
| ChromaDB Upsert | ✅ Done | 3,706 docs indexed |
| `get_embeddings()` | ✅ Done | Returns `np.ndarray (3706, 384)` |
| `incremental_upsert()` | ✅ Done | Deduplication + upsert working |
| DVC Setup | ✅ Done | `data/processed` tracked via DVC |

---

## Phase 1 — Output Contract Delivered

- **File:** `data/processed/pubmed_processed.jsonl`
- **Total Records:** 3,706
- **Schema:** `{pmid, title, abstract, text, pub_date, mesh_terms, word_count}`
- **ChromaDB Collection:** `pubmed_abstracts`
- **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions)

---

## Phase 1 — Project Structure

```
ragops/
├── src/
│   ├── fetcher.py            ✅ PubMed API — esearch + efetch
│   ├── parser.py             ✅ XML to JSONL parser
│   ├── chroma_interface.py   ✅ get_embeddings() + incremental_upsert()
│   └── run_pipeline.py       ✅ Main pipeline orchestrator
├── data/
│   ├── processed/
│   │   └── pubmed_processed.jsonl   ✅ 3,706 records
│   └── chroma_store/                ✅ ChromaDB index
├── .dvc/                            ✅ DVC initialized
├── .env                             ✅ NCBI config
└── requirements.txt                 ✅ All deps installed
```

---

## Week 2 — Data Quality & Embedding Export ✅ COMPLETE

| Task | Status | Details |
|------|--------|---------|
| Data quality checks | ✅ Done | `quality_report.py` → `quality_report.html` |
| Dataset statistics | ✅ Done | `stats.py` → Table 0 for paper |
| Embedding export | ✅ Done | `ingest.py::get_embeddings()` → shape (3706, 384) |
| Incremental ingestion | ✅ Done | `ingest.py::incremental_upsert()` working |
| Weekly data refresh | ✅ Done | `refresh.py` → heart disease records only |
| Git/DVC commit | ⏳ Waiting | Waiting for Rishabh to add to team repo |

---

## Week 2 — Data Refinement (per Rishabh's feedback) ✅ DONE

| Change | Details |
|--------|---------|
| Disease filter | Heart disease only — 4 focused MeSH queries |
| Date filter | 2021–2026 only — no old records |
| Language filter | English only — non-English records excluded |
| Records after re-fetch | 3,706 (clean, filtered dataset) |
| Unknown dates | 13 records (0.3%) — retained, PubMed XML limitation |
| Years present | 2024, 2025, 2026 |

---

## Week 2 — Quality Report Results

| Metric | Value |
|--------|-------|
| Total records | 3,706 |
| Duplicate PMIDs | 0 |
| Average word count | 243.2 words |
| MeSH term coverage | 57.9% |
| Unknown dates | 13 |
| Embedding matrix shape | (3706, 384) |

---

## Week 2 — Project Structure

```
ragops/
├── src/
│   ├── fetcher.py            ✅ PubMed API
│   ├── parser.py             ✅ XML parser
│   ├── chroma_interface.py   ✅ ChromaDB interface
│   └── run_pipeline.py       ✅ Pipeline runner (updated: heart disease queries)
├── data/
│   ├── processed/
│   │   └── pubmed_processed.jsonl   ✅ 3,706 records (heart disease, English, 2021–2026)
│   ├── chroma_store/                ✅ ChromaDB index
│   ├── quality_report.py            ✅ Quality checks script
│   ├── quality_report.html          ✅ Generated HTML report
│   ├── stats.py                     ✅ Dataset statistics
│   ├── ingest.py                    ✅ Team interface (get_embeddings + upsert)
│   └── refresh.py                   ✅ Weekly refresh script
├── .dvc/                            ✅ DVC initialized
├── .env                             ✅ Config
└── requirements.txt                 ✅ Dependencies
```

---

## Pending

- [ ] Rishabh to add GitHub usernames to team repo
- [ ] `git remote add origin <URL>`
- [ ] Push to `saumya/etl-pipeline` branch
- [ ] Raise PR to `dev` branch

---

## Phase 2 — Upcoming (Weeks 3–4)

| Panel | Description | Status |
|-------|-------------|--------|
| Panel 1 | ETL Metrics Dashboard | 🔜 Pending |
| Panel 2 | RAG Performance | 🔜 Pending |
| Panel 3 | Vector Store Stats | 🔜 Pending |
| Panel 4 | MLflow Experiments | 🔜 Pending |
| Panel 5 | XAI Panel | 🔜 Pending |

---

*Generated: 01 April 2026*
