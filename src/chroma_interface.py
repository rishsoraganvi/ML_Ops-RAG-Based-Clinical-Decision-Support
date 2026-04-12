"""
ChromaDB interface for the data pipeline.

Provides embedding export and incremental upsert for the MLOps layer
(drift detection baseline, weekly data refresh).

Uses rag_pipeline.vectorstore for ChromaDB connectivity and
rag_pipeline.ingest for document chunking + upsert.
"""

from __future__ import annotations

import logging

import numpy as np
from langchain.schema import Document

from rag_pipeline.vectorstore import get_or_create_collection
from rag_pipeline.ingest import ingest_documents

logger = logging.getLogger("ragops.chroma_interface")


def get_embeddings(chunk_size: int = 512) -> np.ndarray:
    """Return embedding matrix of shape (n_docs, 384) from ChromaDB.

    Retrieves all document embeddings from the specified collection.
    Used by the MLOps layer for PSI drift detection baseline.

    Args:
        chunk_size: Which pubmed_{chunk_size} collection to query.

    Returns:
        np.ndarray of shape (n_docs, 384).

    Raises:
        RuntimeError: If the collection is empty.
    """
    collection = get_or_create_collection(chunk_size)
    total = collection.count()

    if total == 0:
        raise RuntimeError(
            f"Collection pubmed_{chunk_size} is empty. Run ingest first."
        )

    results = collection.get(
        limit=total,
        include=["embeddings"],
    )

    matrix = np.array(results["embeddings"], dtype=np.float32)
    logger.info(
        "Retrieved embeddings from pubmed_%d — shape=%s",
        chunk_size,
        matrix.shape,
    )
    return matrix


def incremental_upsert(new_docs: list[dict], chunk_size: int = 512) -> int:
    """Add new documents to ChromaDB without re-embedding existing ones.

    Converts raw PubMed records to LangChain Documents, then delegates
    to rag_pipeline.ingest for chunking and upsert.

    Args:
        new_docs: list of dicts with keys: pmid, text, title, pub_date,
                  mesh_terms, word_count.
        chunk_size: Target collection chunk size.

    Returns:
        Count of chunks upserted.
    """
    if not new_docs:
        logger.info("No new documents to upsert.")
        return 0

    documents = []
    for doc in new_docs:
        text = doc.get("text", "")
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": doc.get("pmid", "unknown"),
                    "doc_id": doc.get("pmid", "unknown"),
                    "title": doc.get("title", ""),
                    "pub_date": doc.get("pub_date", ""),
                    "mesh_terms": str(doc.get("mesh_terms", [])),
                    "word_count": doc.get("word_count", 0),
                },
            )
        )

    summary = ingest_documents(documents, chunk_size=chunk_size)
    logger.info(
        "Incremental upsert complete — %d docs → %d chunks",
        len(documents),
        summary["chunks_upserted"],
    )
    return summary["chunks_upserted"]
