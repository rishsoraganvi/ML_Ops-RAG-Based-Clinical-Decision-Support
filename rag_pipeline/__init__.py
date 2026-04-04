"""
rag_pipeline — RAGOps retrieval pipeline.

Public API (Week 1):
    from rag_pipeline import query, bm25_term_scores
    from rag_pipeline import ingest_documents, load_text_files
    from rag_pipeline import get_retriever
"""

from .chain import query, bm25_term_scores, run_ablation, ABLATION_CONFIGS
from .ingest import ingest_documents, load_text_files, chunk_documents
from .retriever import get_retriever, dense_retriever, retrieve_with_scores
from .vectorstore import (
    get_chroma_client,
    get_or_create_collection,
    collection_stats,
)

__all__ = [
    # Primary API
    "query",
    "bm25_term_scores",
    "run_ablation",
    "ABLATION_CONFIGS",
    # Ingestion
    "ingest_documents",
    "load_text_files",
    "chunk_documents",
    # Retrieval
    "get_retriever",
    "dense_retriever",
    "retrieve_with_scores",
    # Vector store
    "get_chroma_client",
    "get_or_create_collection",
    "collection_stats",
]
