"""
ingest.py — Document chunking and embedding into ChromaDB.

Supports three chunk sizes (256 / 512 / 1024 tokens) with 10% overlap.
Designed for PubMed abstracts / medical text; works with any plain-text corpus.

Usage (CLI):
    python -m rag_pipeline.ingest --data_dir ./data/pubmed --chunk_size 512

Usage (Python):
    from rag_pipeline.ingest import ingest_documents
    ingest_documents(docs, chunk_size=512)
"""

import argparse
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .vectorstore import get_chroma_client, get_or_create_collection, VALID_CHUNK_SIZES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def get_text_splitter(chunk_size: int) -> RecursiveCharacterTextSplitter:
    """
    Return a RecursiveCharacterTextSplitter for the given chunk_size.
    Overlap = 10% of chunk_size (e.g. 512 → 51 tokens ≈ 50).
    chunk_size here is in *characters* approximated from tokens
    (1 token ≈ 4 chars for English biomedical text).
    """
    if chunk_size not in VALID_CHUNK_SIZES:
        raise ValueError(f"chunk_size must be one of {VALID_CHUNK_SIZES}, got {chunk_size}")

    char_size    = chunk_size * 4          # token → character approximation
    char_overlap = max(1, int(char_size * 0.10))   # 10% overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],   # medical text hierarchy
        length_function=len,
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int,
) -> List[Document]:
    """
    Split a list of LangChain Documents into smaller chunks.

    Metadata is propagated and enriched with:
        chunk_size, chunk_index, source_doc_id
    """
    splitter = get_text_splitter(chunk_size)
    chunks: List[Document] = []

    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata.update(
                {
                    "chunk_size":     chunk_size,
                    "chunk_index":    idx,
                    "source_doc_id":  doc.metadata.get("doc_id", "unknown"),
                }
            )
        chunks.extend(doc_chunks)

    logger.info(
        "Chunked %d documents → %d chunks (chunk_size=%d).",
        len(documents),
        len(chunks),
        chunk_size,
    )
    return chunks


# ---------------------------------------------------------------------------
# Stable ID generation
# ---------------------------------------------------------------------------

def _chunk_id(text: str, metadata: dict) -> str:
    """
    Deterministic SHA-256 ID for a chunk so re-ingestion is idempotent.
    Combines text content + source_doc_id + chunk_index.
    """
    raw = f"{metadata.get('source_doc_id', '')}::{metadata.get('chunk_index', 0)}::{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Ingestion entry point
# ---------------------------------------------------------------------------

def ingest_documents(
    documents: List[Document],
    chunk_size: int = 512,
    batch_size: int = 256,
    client=None,
    reset_collection: bool = False,
) -> dict:
    """
    Chunk and embed documents into the ChromaDB collection for chunk_size.

    Args:
        documents:         LangChain Documents (page_content + metadata).
        chunk_size:        Token chunk size — one of 256 / 512 / 1024.
        batch_size:        Upsert batch size (avoids ChromaDB request limits).
        client:            Optional pre-built ChromaDB client.
        reset_collection:  If True, drop the collection before ingesting.

    Returns:
        Ingestion summary dict.
    """
    client     = client or get_chroma_client()
    collection = get_or_create_collection(chunk_size, client)

    if reset_collection:
        from .vectorstore import delete_collection
        delete_collection(chunk_size, client)
        collection = get_or_create_collection(chunk_size, client)
        logger.warning("Collection reset — starting fresh.")

    # ── 1. Chunk ──────────────────────────────────────────────────────────
    chunks = chunk_documents(documents, chunk_size)

    # ── 2. Prepare upsert payloads ─────────────────────────────────────────
    ids        = [_chunk_id(c.page_content, c.metadata) for c in chunks]
    texts      = [c.page_content for c in chunks]
    metadatas  = [c.metadata for c in chunks]

    # ── 3. Batch upsert ───────────────────────────────────────────────────
    t0          = time.perf_counter()
    total_upserted = 0

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        total_upserted += len(ids[start:end])
        logger.debug("Upserted batch %d–%d.", start, end)

    elapsed = (time.perf_counter() - t0) * 1000

    summary = {
        "collection":       collection.name,
        "docs_ingested":    len(documents),
        "chunks_upserted":  total_upserted,
        "chunk_size":       chunk_size,
        "ingest_time_ms":   round(elapsed, 2),
    }
    logger.info("Ingestion complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# File-based loader helpers
# ---------------------------------------------------------------------------

def load_text_files(data_dir: str, glob: str = "**/*.txt") -> List[Document]:
    """
    Load plain-text files from a directory into LangChain Documents.
    Each file becomes one Document; filename used as doc_id / source.
    """
    root  = Path(data_dir)
    files = list(root.glob(glob))
    docs  = []

    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source":   str(fp),
                    "doc_id":   fp.stem,
                    "filename": fp.name,
                },
            )
        )

    logger.info("Loaded %d documents from '%s'.", len(docs), data_dir)
    return docs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--data_dir",   required=True,  help="Directory of .txt files")
    parser.add_argument("--chunk_size", type=int, default=512, choices=list(VALID_CHUNK_SIZES))
    parser.add_argument("--reset",      action="store_true", help="Drop collection before ingesting")
    args = parser.parse_args()

    documents = load_text_files(args.data_dir)
    summary   = ingest_documents(
        documents,
        chunk_size=args.chunk_size,
        reset_collection=args.reset,
    )
    print(summary)
