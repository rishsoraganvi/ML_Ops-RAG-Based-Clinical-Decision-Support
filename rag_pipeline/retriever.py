"""
retriever.py — Retriever factory for Week 1 (dense only).

Weeks 2–3 will add bm25_retriever() and hybrid_retriever() to this file.
The factory pattern keeps retriever types independently swappable.

Exposed functions
-----------------
dense_retriever(chunk_size, k)          → LangChain BaseRetriever
get_retriever(retriever_type, ...)      → BaseRetriever   (factory)
retrieve_with_scores(question, ...)     → (docs, scores)  ← XAI hook
"""

import logging
import time
from typing import List, Tuple, Optional

from langchain.schema import BaseRetriever, Document
from langchain_community.vectorstores import Chroma

from .vectorstore import (
    get_chroma_client,
    get_or_create_collection,
    get_embedding_function,
    VALID_CHUNK_SIZES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_K          = 5
VALID_RETRIEVERS   = ("dense",)   # bm25 | hybrid added in Week 2


# ---------------------------------------------------------------------------
# Internal: LangChain Chroma vectorstore wrapper
# ---------------------------------------------------------------------------

def _chroma_vectorstore(chunk_size: int) -> Chroma:
    """
    Build a LangChain Chroma vectorstore backed by the persistent HttpClient.
    We pass the collection object directly so LangChain does not spin up
    its own ephemeral client.
    """
    collection = get_or_create_collection(chunk_size)
    ef         = get_embedding_function()

    # LangChain wraps the chromadb.Collection directly
    vectorstore = Chroma(
        client=get_chroma_client(),
        collection_name=collection.name,
        embedding_function=ef,     # LangChain-compatible wrapper
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

def dense_retriever(
    chunk_size: int = 512,
    k: int = DEFAULT_K,
) -> BaseRetriever:
    """
    Return a semantic (dense) retriever backed by ChromaDB.

    Args:
        chunk_size: Which pubmed_{chunk_size} collection to query.
        k:          Number of documents to retrieve.

    Returns:
        LangChain BaseRetriever — drop-in for RetrievalQA.
    """
    if chunk_size not in VALID_CHUNK_SIZES:
        raise ValueError(f"chunk_size must be one of {VALID_CHUNK_SIZES}")

    vectorstore = _chroma_vectorstore(chunk_size)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    logger.info(
        "Dense retriever ready — collection=pubmed_%d, k=%d", chunk_size, k
    )
    return retriever


# ---------------------------------------------------------------------------
# Factory — extended in Weeks 2 & 3
# ---------------------------------------------------------------------------

def get_retriever(
    retriever_type: str = "dense",
    chunk_size: int = 512,
    k: int = DEFAULT_K,
    # Week 2 kwargs (accepted but ignored in Week 1):
    bm25_corpus: Optional[List[str]] = None,
    dense_weight: float = 0.5,
) -> BaseRetriever:
    """
    Factory that returns the requested retriever type.

    retriever_type:
        'dense'  — semantic similarity via ChromaDB   (Week 1)
        'bm25'   — BM25 lexical retrieval             (Week 2)
        'hybrid' — dense + BM25 + cross-encoder rerank (Week 2)

    Args:
        retriever_type: One of 'dense' | 'bm25' | 'hybrid'.
        chunk_size:     Token chunk size for collection selection.
        k:              Number of docs to retrieve.
        bm25_corpus:    Pre-tokenised corpus for BM25 (Week 2).
        dense_weight:   Weight in [0,1] for dense side of hybrid (Week 2).

    Returns:
        Configured LangChain BaseRetriever.
    """
    retriever_type = retriever_type.lower()

    if retriever_type == "dense":
        return dense_retriever(chunk_size=chunk_size, k=k)

    # Stubs — implemented in Week 2
    if retriever_type == "bm25":
        raise NotImplementedError(
            "BM25 retriever is implemented in Week 2 (retriever.py::bm25_retriever)."
        )
    if retriever_type == "hybrid":
        raise NotImplementedError(
            "Hybrid retriever is implemented in Week 2 (retriever.py::hybrid_retriever)."
        )

    raise ValueError(
        f"Unknown retriever_type '{retriever_type}'. "
        f"Choose from: dense | bm25 | hybrid"
    )


# ---------------------------------------------------------------------------
# Score-aware retrieval — hooks for XAI layer (Member 3, Week 3)
# ---------------------------------------------------------------------------

def retrieve_with_scores(
    question: str,
    retriever_type: str = "dense",
    chunk_size: int = 512,
    k: int = DEFAULT_K,
) -> Tuple[List[Document], List[float], float]:
    """
    Retrieve documents *and* their similarity scores.

    Bypasses the LangChain Retriever abstraction and queries ChromaDB
    directly so that raw distance scores are available for the XAI layer.

    Args:
        question:       Natural-language query.
        retriever_type: Currently only 'dense' in Week 1.
        chunk_size:     Collection to query.
        k:              Top-k results.

    Returns:
        (docs, scores, latency_ms)
        - docs:     List[Document] with page_content + metadata
        - scores:   List[float] — cosine similarities in [0, 1]
                    (ChromaDB returns L2/cosine distances; we convert to similarity)
        - latency_ms: Wall-clock retrieval time in ms
    """
    if retriever_type != "dense":
        raise NotImplementedError(
            "Score-aware retrieval for bm25/hybrid added in Week 2."
        )

    collection = get_or_create_collection(chunk_size)
    ef         = get_embedding_function()

    # Embed the query
    t0         = time.perf_counter()
    query_emb  = ef([question])          # returns List[List[float]]

    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw_docs    = results["documents"][0]     # List[str]
    raw_metas   = results["metadatas"][0]     # List[dict]
    raw_dists   = results["distances"][0]     # List[float] — cosine distance [0,2]

    # Convert cosine distance → similarity score in [0, 1]
    scores = [round(1.0 - (d / 2.0), 6) for d in raw_dists]

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw_docs, raw_metas)
    ]

    logger.debug(
        "retrieve_with_scores: q='%s...', k=%d, latency=%.1f ms, "
        "top_score=%.4f",
        question[:60], k, latency_ms, scores[0] if scores else 0.0,
    )

    return docs, scores, round(latency_ms, 2)
