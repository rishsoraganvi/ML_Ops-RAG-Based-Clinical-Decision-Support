"""
retriever.py — Retriever factory for Week 1 + Week 2.

Week 1: dense_retriever()
Week 2: bm25_retriever(), hybrid_retrieve(), rerank()

Exposed functions
-----------------
dense_retriever(chunk_size, k)             → LangChain BaseRetriever
bm25_retriever(corpus_docs, k)            → BM25Retriever
hybrid_retrieve(question, chunk_size, k)  → (docs, scores, latency_ms)
rerank(question, docs, top_n)             → (docs, scores)
get_retriever(retriever_type, ...)        → BaseRetriever   (factory)
retrieve_with_scores(question, ...)       → (docs, scores, latency_ms)  ← XAI hook
"""

import functools
import hashlib
import logging
import re
import time
from typing import Any, List, Tuple, Optional

from langchain.schema import BaseRetriever, Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

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

DEFAULT_K        = 5
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VALID_RETRIEVERS = ("dense", "bm25", "hybrid")

# Hybrid: retrieve k * HYBRID_FETCH_MULT candidates from each side
# before reranking down to k.
# 4 x 5 = 20 candidates — matches Week 2 spec "re-rank top-20, return top-k"
HYBRID_FETCH_MULT = 4                          # ABLATION EXPERIMENT

# Hybrid dense/BM25 weights — spec: 0.6 x dense + 0.4 x BM25
DEFAULT_DENSE_WEIGHT = 0.6                     # ABLATION EXPERIMENT

# Cache the cross-encoder so it loads only once per process
_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _tokenise(text: str) -> List[str]:
    """Shared BM25 tokeniser — lowercase alphabetic words only.

    Used by both bm25_retriever() and retrieve_with_scores() so that
    document ranking is consistent across retrieval paths.
    """
    return re.findall(r"[a-z]+", text.lower())


# ---------------------------------------------------------------------------
# Internal: LangChain Chroma vectorstore wrapper
# ---------------------------------------------------------------------------

def _chroma_vectorstore(chunk_size: int) -> Chroma:
    """Build a LangChain Chroma vectorstore backed by the persistent HttpClient."""
    collection = get_or_create_collection(chunk_size)
    ef         = get_embedding_function()
    vectorstore = Chroma(
        client=get_chroma_client(),
        collection_name=collection.name,
        embedding_function=ef,
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Dense retriever (Week 1)
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
    logger.info("Dense retriever ready — collection=pubmed_%d, k=%d", chunk_size, k)
    return retriever


# ---------------------------------------------------------------------------
# BM25 retriever (Week 2)                                # ABLATION EXPERIMENT
# ---------------------------------------------------------------------------

def bm25_retriever(
    corpus_docs: List[Document],
    k: int = DEFAULT_K,
) -> LangChainBM25Retriever:
    """
    Build a BM25 lexical retriever from an in-memory document corpus.

    The corpus is fetched from ChromaDB at call time so BM25 operates
    over the same documents as the dense retriever.

    Args:
        corpus_docs: List of LangChain Documents (from ChromaDB collection).
        k:           Number of documents to retrieve.

    Returns:
        LangChain BM25Retriever.
    """
    # ABLATION EXPERIMENT
    retriever = LangChainBM25Retriever.from_documents(
        corpus_docs, k=k, preprocess_func=_tokenise
    )
    logger.info("BM25 retriever ready — corpus=%d docs, k=%d", len(corpus_docs), k)
    return retriever


def fetch_corpus(chunk_size: int) -> List[Document]:
    """
    Pull all documents from a ChromaDB collection as LangChain Documents.
    Used to build the BM25 in-memory corpus.

    Args:
        chunk_size: Collection to pull from (pubmed_{chunk_size}).

    Returns:
        List[Document]
    """
    collection = get_or_create_collection(chunk_size)
    total      = collection.count()

    if total == 0:
        raise RuntimeError(
            f"Collection pubmed_{chunk_size} is empty. Run ingest.py first."
        )

    results = collection.get(
        limit=total,
        include=["documents", "metadatas"],
    )
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(results["documents"], results["metadatas"])
    ]
    logger.info("Fetched %d docs from pubmed_%d for BM25 corpus.", total, chunk_size)
    return docs


@functools.lru_cache(maxsize=8)
def _get_bm25_components(chunk_size: int) -> Tuple[List[Document], Any]:
    """
    Fetch the corpus and build a BM25Okapi index once per chunk_size.

    Cached with lru_cache keyed solely on chunk_size (an int), so repeated
    calls within the same process reuse the already-built index — avoids
    per-query corpus fetch + tokenisation.
    Uses the shared _tokenise() function so ranking is consistent with
    bm25_retriever() (which also uses _tokenise via preprocess_func).

    Returns:
        (corpus_docs, bm25_index)
    """
    corpus = fetch_corpus(chunk_size)
    tokenised = [_tokenise(d.page_content) for d in corpus]
    bm25_index = BM25Okapi(tokenised)
    logger.info(
        "Built BM25Okapi index for chunk_size=%d (%d docs).", chunk_size, len(corpus)
    )
    return corpus, bm25_index


# ---------------------------------------------------------------------------
# Cross-encoder reranker (Week 2)
# ---------------------------------------------------------------------------

def rerank(
    question: str,
    docs: List[Document],
    top_n: int = DEFAULT_K,
) -> Tuple[List[Document], List[float]]:
    """
    Rerank a list of candidate documents with a cross-encoder.

    Used as second stage in hybrid retrieval.
    Spec: cross-encoder/ms-marco-MiniLM-L-6-v2

    Args:
        question: Original query string.
        docs:     Candidate documents from first-stage retrieval (top-20).
        top_n:    How many to keep after reranking (k=5).

    Returns:
        (reranked_docs, rerank_scores) — top_n docs sorted by cross-encoder score.
    """
    reranker = _get_reranker()
    pairs    = [[question, d.page_content] for d in docs]
    scores   = reranker.predict(pairs).tolist()

    ranked     = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs   = [d for d, _ in ranked[:top_n]]
    top_scores = [round(s, 6) for _, s in ranked[:top_n]]

    logger.debug(
        "Reranker: %d candidates -> %d kept | top_score=%.4f",
        len(docs), top_n, top_scores[0] if top_scores else 0.0,
    )
    return top_docs, top_scores


# ---------------------------------------------------------------------------
# Hybrid retrieval (Week 2)                              # ABLATION EXPERIMENT
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    question: str,
    chunk_size: int = 256,
    k: int = DEFAULT_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
) -> Tuple[List[Document], List[float], float]:
    """
    Hybrid retrieval: dense + BM25 fusion -> cross-encoder rerank.

    Spec: 0.6 x dense + 0.4 x BM25, re-rank top-20, return top-k (5).

    Strategy:
        1. Retrieve k * 4 = 20 candidates from dense (ChromaDB).
        2. Retrieve k * 4 = 20 candidates from BM25.
        3. Merge with weighted Reciprocal Rank Fusion (RRF).
        4. Rerank merged candidates with cross-encoder; return top k.

    Args:
        question:     Query string.
        chunk_size:   Collection to use.
        k:            Final number of docs to return (default 5).
        dense_weight: RRF weight for dense side (default 0.6).
                      BM25 weight = 1 - dense_weight = 0.4.

    Returns:
        (docs, rerank_scores, latency_ms)
    """
    # ABLATION EXPERIMENT
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError(
            f"dense_weight must be in [0, 1], got {dense_weight!r}. "
            "BM25 weight is computed as 1 - dense_weight."
        )
    fetch_k = k * HYBRID_FETCH_MULT    # 5 * 4 = 20 candidates per side
    t0      = time.perf_counter()

    # 1. Dense candidates (top-20)
    collection = get_or_create_collection(chunk_size)
    ef         = get_embedding_function()
    query_emb  = ef([question])

    dense_results = collection.query(
        query_embeddings=query_emb,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    dense_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(
            dense_results["documents"][0],
            dense_results["metadatas"][0],
        )
    ]

    # 2. BM25 candidates (top-20) — reuse cached corpus and index
    corpus, bm25_index = _get_bm25_components(chunk_size)
    query_tokens = _tokenise(question)
    bm25_raw     = bm25_index.get_scores(query_tokens)
    bm25_ranked  = sorted(
        zip(corpus, bm25_raw), key=lambda x: x[1], reverse=True
    )[:fetch_k]
    bm25_docs = [d for d, _ in bm25_ranked]

    # 3. Weighted RRF fusion (0.6 dense + 0.4 BM25)
    merged = _reciprocal_rank_fusion(
        [dense_docs, bm25_docs],
        weights=[dense_weight, 1.0 - dense_weight],
    )

    # 4. Cross-encoder rerank -> top-k (5)
    reranked_docs, rerank_scores = rerank(question, merged, top_n=k)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "Hybrid retrieve — chunk=%d, k=%d, fetch_k=%d, "
        "dense_weight=%.1f, latency=%.1f ms, top_score=%.4f",
        chunk_size, k, fetch_k, dense_weight,
        latency_ms, rerank_scores[0] if rerank_scores else 0.0,
    )
    return reranked_docs, rerank_scores, latency_ms


def _reciprocal_rank_fusion(
    doc_lists: List[List[Document]],
    weights: List[float],
    rrf_k: int = 60,
) -> List[Document]:
    """
    Merge multiple ranked lists with weighted Reciprocal Rank Fusion.

    RRF score for doc d = sum_i( weight_i / (rrf_k + rank_i(d)) )
    Deduplication is by page_content hash.
    """
    scores:  dict = {}
    doc_map: dict = {}

    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list, start=1):
            # Use SHA-256 of page_content for stable, collision-resistant dedup
            key = hashlib.sha256(doc.page_content.encode()).hexdigest()
            scores[key]  = scores.get(key, 0.0) + weight / (rrf_k + rank)
            doc_map[key] = doc

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in ranked_keys]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_retriever(
    retriever_type: str = "dense",
    chunk_size: int = 256,
    k: int = DEFAULT_K,
    bm25_corpus: Optional[List[Document]] = None,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
) -> BaseRetriever:
    """
    Factory that returns the requested retriever type.

    retriever_type:
        'dense'  — semantic similarity via ChromaDB
        'bm25'   — BM25 lexical retrieval
        'hybrid' — dense + BM25 (0.6/0.4) + cross-encoder rerank

    Note: For hybrid, prefer calling hybrid_retrieve() directly to get scores.
    """
    retriever_type = retriever_type.lower()

    if retriever_type == "dense":
        return dense_retriever(chunk_size=chunk_size, k=k)

    if retriever_type == "bm25":
        corpus = bm25_corpus or fetch_corpus(chunk_size)
        return bm25_retriever(corpus, k=k)

    if retriever_type == "hybrid":
        return _HybridRetrieverWrapper(
            chunk_size=chunk_size, k=k, dense_weight=dense_weight
        )

    raise ValueError(
        f"Unknown retriever_type '{retriever_type}'. "
        f"Choose from: dense | bm25 | hybrid"
    )


class _HybridRetrieverWrapper(BaseRetriever):
    """
    Thin LangChain BaseRetriever wrapper around hybrid_retrieve()
    so it can be dropped into RetrievalQA without changes.
    Scores accessible via retrieve_with_scores() for XAI.
    """

    chunk_size:   int   = 256
    k:            int   = DEFAULT_K
    dense_weight: float = DEFAULT_DENSE_WEIGHT

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        docs, _, _ = hybrid_retrieve(
            query,
            chunk_size=self.chunk_size,
            k=self.k,
            dense_weight=self.dense_weight,
        )
        return docs

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return self._get_relevant_documents(query)


# ---------------------------------------------------------------------------
# Score-aware retrieval — XAI hook (all retriever types)
# ---------------------------------------------------------------------------

def retrieve_with_scores(
    question: str,
    retriever_type: str = "dense",
    chunk_size: int = 256,
    k: int = DEFAULT_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
) -> Tuple[List[Document], List[float], float]:
    """
    Retrieve documents AND their scores for ALL retriever types.
    Used by chain.py to populate retrieval_scores in query() output.

    Returns:
        (docs, scores, latency_ms)
        - scores for dense:  cosine similarity in [0, 1]
        - scores for bm25:   BM25 relevance scores (unnormalised)
        - scores for hybrid: cross-encoder scores (unnormalised logits)
    """
    retriever_type = retriever_type.lower()
    if retriever_type not in VALID_RETRIEVERS:
        raise ValueError(
            f"Unknown retriever_type '{retriever_type}'. "
            f"Choose from: {' | '.join(VALID_RETRIEVERS)}"
        )

    # Hybrid
    if retriever_type == "hybrid":
        return hybrid_retrieve(
            question, chunk_size=chunk_size, k=k, dense_weight=dense_weight
        )

    # BM25 — reuse cached corpus and index; same tokeniser as bm25_retriever()
    if retriever_type == "bm25":
        t0 = time.perf_counter()
        corpus, bm25_index = _get_bm25_components(chunk_size)

        query_tokens = _tokenise(question)
        raw_scores   = bm25_index.get_scores(query_tokens).tolist()

        ranked = sorted(
            zip(corpus, raw_scores), key=lambda x: x[1], reverse=True
        )[:k]
        docs   = [d for d, _ in ranked]
        scores = [round(s, 6) for _, s in ranked]

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            "BM25 retrieve_with_scores: k=%d, latency=%.1f ms, top=%.4f",
            k, latency_ms, scores[0] if scores else 0.0,
        )
        return docs, scores, latency_ms

    # Dense (default)
    collection = get_or_create_collection(chunk_size)
    ef         = get_embedding_function()

    t0        = time.perf_counter()
    query_emb = ef([question])

    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    raw_docs  = results["documents"][0]
    raw_metas = results["metadatas"][0]
    raw_dists = results["distances"][0]

    scores = [round(1.0 - (d / 2.0), 6) for d in raw_dists]
    docs   = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw_docs, raw_metas)
    ]

    logger.debug(
        "Dense retrieve_with_scores: k=%d, latency=%.1f ms, top=%.4f",
        k, latency_ms, scores[0] if scores else 0.0,
    )
    return docs, scores, latency_ms
