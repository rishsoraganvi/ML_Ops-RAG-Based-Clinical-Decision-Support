import json
import os

os.makedirs("data/raw", exist_ok=True)
with open("data/processed/pubmed_processed.jsonl", encoding="utf-8") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    doc = json.loads(line)
    text = doc.get("text") or doc.get("abstract") or doc.get("content") or str(doc)
    with open(f"data/raw/doc_{i}.txt", "w", encoding="utf-8") as out:
        out.write(text)
print("Done -", len(lines), "files written")
