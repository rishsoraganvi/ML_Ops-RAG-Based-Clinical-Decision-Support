"""
chain.py — LangChain RetrievalQA chain connecting retrieval to LLaMA-3 via Ollama.

Exposes two public APIs required by the project spec:

    query(question, config) -> dict
        answer | source_docs | retrieval_latency_ms | retrieval_scores | config

    bm25_term_scores(query, doc) -> dict
        {term: score}  ← token-level XAI attribution stub (fully implemented Week 3)
"""

import logging
import time
from typing import Optional

import mlflow
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .retriever import get_retriever, retrieve_with_scores, DEFAULT_K
from .vectorstore import VALID_CHUNK_SIZES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "llama3"               # ollama pull llama3
OLLAMA_TEMP      = 0.0                    # deterministic for ablations
MLFLOW_EXP_NAME  = "ragops_ablation"

CLINICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a clinical assistant. "
        "Use ONLY the provided context to answer the question. "
        "If the context does not contain enough information, "
        "respond with exactly: 'Insufficient context'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "retriever_type": "dense",   # dense | bm25 | hybrid
    "chunk_size":     512,       # 256 | 512 | 1024   ← ABLATION EXPERIMENT
    "k":              DEFAULT_K,
    "reranker":       False,     # True only for hybrid (Week 2)
}


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMP,
    )


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_chain(config: dict) -> RetrievalQA:
    """
    Assemble a RetrievalQA chain for the given config.

    Args:
        config: Must contain retriever_type, chunk_size, k.

    Returns:
        LangChain RetrievalQA chain (stuff document combination).
    """
    retriever = get_retriever(
        retriever_type=config["retriever_type"],
        chunk_size=config["chunk_size"],
        k=config["k"],
    )
    llm = _get_llm()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CLINICAL_PROMPT},
    )
    logger.info(
        "Chain built — retriever=%s, chunk_size=%d, k=%d",
        config["retriever_type"],
        config["chunk_size"],
        config["k"],
    )
    return chain


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------

def query(
    question: str,
    config: Optional[dict] = None,
    mlflow_run: bool = False,
) -> dict:
    """
    Run an end-to-end RAG query and return a structured result dict.

    Args:
        question:    Natural-language medical question.
        config:      Override any key in DEFAULT_CONFIG.
                     e.g. {"retriever_type": "dense", "chunk_size": 256}
        mlflow_run:  If True, log latency + config to MLflow.

    Returns:
        {
            "answer":               str,
            "source_docs":          List[dict],   # page_content + metadata
            "retrieval_latency_ms": float,
            "llm_latency_ms":       float,
            "total_latency_ms":     float,
            "retrieval_scores":     List[float],  # ← XAI hook for Member 3
            "config":               dict,
        }
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── 1. Score-aware retrieval (bypasses LangChain for raw scores) ──────
    docs, scores, retrieval_ms = retrieve_with_scores(
        question=question,
        retriever_type=cfg["retriever_type"],
        chunk_size=cfg["chunk_size"],
        k=cfg["k"],
    )

    # ── 2. Build context string from retrieved docs ───────────────────────
    context = "\n\n".join(d.page_content for d in docs)

    # ── 3. LLM call ───────────────────────────────────────────────────────
    llm    = _get_llm()
    t_llm  = time.perf_counter()

    prompt_text = CLINICAL_PROMPT.format(context=context, question=question)
    llm_resp    = llm.invoke(prompt_text)
    answer      = llm_resp.content.strip()

    llm_ms      = round((time.perf_counter() - t_llm) * 1000, 2)
    total_ms    = round(retrieval_ms + llm_ms, 2)

    # ── 4. Package source docs ────────────────────────────────────────────
    source_docs = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]

    result = {
        "answer":               answer,
        "source_docs":          source_docs,
        "retrieval_latency_ms": retrieval_ms,
        "llm_latency_ms":       llm_ms,
        "total_latency_ms":     total_ms,
        "retrieval_scores":     scores,        # List[float] — XAI hook
        "config":               cfg,
    }

    # ── 5. Optional MLflow logging ─────────────────────────────────────────
    if mlflow_run:
        _log_to_mlflow(question, result, cfg)

    logger.info(
        "query() done | retriever=%s | chunk=%d | retrieval=%.1f ms | "
        "llm=%.1f ms | top_score=%.4f",
        cfg["retriever_type"],
        cfg["chunk_size"],
        retrieval_ms,
        llm_ms,
        scores[0] if scores else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# XAI support — token-level BM25 attribution
# ---------------------------------------------------------------------------

def bm25_term_scores(query_text: str, doc: str) -> dict:
    """
    Return per-term BM25 scores for a query against a single document.

    This is the token-level XAI attribution hook used by Member 3.
    Full BM25 integration (rank_bm25 corpus) lands in Week 2/3;
    this Week-1 stub provides correct term-frequency weighting
    against the single document so the XAI layer can develop in parallel.

    Args:
        query_text: Raw query string.
        doc:        Document text to score against.

    Returns:
        {term: idf_tf_score}  — only query terms that appear in doc.
        Example: {"diabetes": 0.847, "insulin": 0.631}
    """
    import math
    from collections import Counter

    # Tokenise (lowercase, alpha only)
    def tokenise(text: str):
        import re
        return re.findall(r"[a-z]+", text.lower())

    query_terms = set(tokenise(query_text))
    doc_tokens  = tokenise(doc)
    doc_len     = len(doc_tokens)

    if doc_len == 0:
        return {}

    tf_counts = Counter(doc_tokens)

    # BM25 hyperparameters (Okapi defaults)
    k1, b, avgdl = 1.5, 0.75, 150.0

    # Single-document IDF approximation (log(1 + 1/tf_norm))
    scores = {}
    for term in query_terms:
        tf = tf_counts.get(term, 0)
        if tf == 0:
            continue
        tf_norm   = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
        idf_approx = math.log(1.0 + 1.0 / (tf / doc_len))
        scores[term] = round(tf_norm * idf_approx, 6)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# MLflow helper
# ---------------------------------------------------------------------------

def _log_to_mlflow(question: str, result: dict, cfg: dict) -> None:
    """Log one RAG query as an MLflow run."""
    mlflow.set_experiment(MLFLOW_EXP_NAME)
    with mlflow.start_run(run_name=f"{cfg['retriever_type']}_{cfg['chunk_size']}"):
        mlflow.log_params(cfg)
        mlflow.log_metrics(
            {
                "retrieval_latency_ms": result["retrieval_latency_ms"],
                "llm_latency_ms":       result["llm_latency_ms"],
                "total_latency_ms":     result["total_latency_ms"],
                "top_retrieval_score":  result["retrieval_scores"][0]
                                        if result["retrieval_scores"] else 0.0,
            }
        )
        mlflow.log_text(question,          "question.txt")
        mlflow.log_text(result["answer"],  "answer.txt")


# ---------------------------------------------------------------------------
# ABLATION EXPERIMENT — convenience runner for Table 1 configs
# ---------------------------------------------------------------------------

# ABLATION EXPERIMENT
ABLATION_CONFIGS = [
    {"retriever_type": rt, "chunk_size": cs}
    for rt in ("dense",)                   # bm25 | hybrid added Week 2
    for cs in (256, 512, 1024)             # ABLATION EXPERIMENT
]


def run_ablation(question: str, mlflow_run: bool = True) -> list:
    """
    Run question across all ablation configs and return results list.
    Maps directly to Paper Table 1 rows.

    # ABLATION EXPERIMENT
    """
    results = []
    for cfg in ABLATION_CONFIGS:
        logger.info("ABLATION: running config %s", cfg)
        try:
            r = query(question, config=cfg, mlflow_run=mlflow_run)
            results.append(r)
        except Exception as exc:
            logger.error("ABLATION config %s failed: %s", cfg, exc)
            results.append({"config": cfg, "error": str(exc)})
    return results


# ---------------------------------------------------------------------------
# Quick smoke-test (run with: python -m rag_pipeline.chain)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    test_q = "What are the first-line treatments for type 2 diabetes?"
    print(f"\nQuestion: {test_q}\n")

    out = query(test_q, config={"retriever_type": "dense", "chunk_size": 512})
    print("Answer:", out["answer"])
    print("Retrieval latency:", out["retrieval_latency_ms"], "ms")
    print("Top retrieval scores:", out["retrieval_scores"])
    print("Sources:", [s["metadata"].get("source", "?") for s in out["source_docs"]])

    print("\n--- BM25 term scores (XAI stub) ---")
    if out["source_docs"]:
        term_scores = bm25_term_scores(test_q, out["source_docs"][0]["page_content"])
        print(term_scores)
