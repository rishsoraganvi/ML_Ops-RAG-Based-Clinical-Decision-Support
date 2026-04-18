"""
explainability.py — Unified XAI layer for the RAGOps pipeline.

# XAI CONTRIBUTION

Provides five building blocks plus one unified ``explain()`` entrypoint:

    shap_retrieval_explain()   → per-document SHAP values
    attention_attribution()    → token/span saliency per answer sentence
    term_attribution()         → BM25 term scores aggregated across docs
    build_explanation_vector() → fixed-dimension (20,) float64 vector
    HallucinationClassifier    → logistic regression with rule-based fallback

    explain()                  → combines all of the above into one dict

Determinism
-----------
- Random seeds are fixed at module scope (``_SEED = 42``); both NumPy and
  torch are seeded immediately before any sampling step.
- ``build_explanation_vector()`` is pure (no randomness, stable sort) — two
  calls on the same inputs pass ``np.array_equal``.
- ``explanation_vector`` is ALWAYS shape ``(EXPLANATION_VECTOR_DIM,)`` =
  ``(20,)`` regardless of how many source docs or answer sentences there are.

Graceful degradation
--------------------
- If ``transformers``/``torch`` cannot load, ``attention_attribution()``
  returns a padded empty list and the explanation_vector zero-fills the
  attention slots. ``explain()`` still returns a shape-stable dict.
- ``HallucinationClassifier`` ships unfitted; it emits a deterministic
  rule-based score until a persisted joblib model exists.
"""

from __future__ import annotations

import functools
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Module constants (locked — these define the explanation_vector shape)
# ---------------------------------------------------------------------------

TOP_K_SHAP: int = 5
ATTENTION_FIXED_SENTENCES: int = 5
ATTENTION_SPANS_PER_SENTENCE: int = 3
EXPLANATION_VECTOR_DIM: int = (
    TOP_K_SHAP + ATTENTION_FIXED_SENTENCES * ATTENTION_SPANS_PER_SENTENCE
)  # 20

HF_MODEL_NAME: str = "distilbert-base-uncased"
CLF_PATH: Path = Path("mlops/artifacts/hallucination_clf.joblib")
_SEED: int = 42
_SHAP_NSAMPLES: int = 50

# Feature order used by ``HallucinationClassifier`` — DO NOT REORDER;
# persisted joblib models depend on this ordering.
_FEATURE_ORDER: Tuple[str, ...] = (
    "mean_retrieval_score",
    "min_retrieval_score",
    "shap_entropy",
    "num_source_docs",
    "answer_len_tokens",
    "top_attention_score",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy HuggingFace loader (distilbert) — CPU, cached
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_hf_model() -> Optional[Tuple[Any, Any]]:
    """
    Lazy-load ``distilbert-base-uncased`` + tokenizer. Returns None on
    failure so callers can degrade gracefully.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.manual_seed(_SEED)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModel.from_pretrained(HF_MODEL_NAME)
        model.eval()
        logger.info("Loaded HF model for XAI: %s", HF_MODEL_NAME)
        return tokenizer, model
    except Exception as exc:  # pragma: no cover — env-dependent
        logger.warning(
            "Could not load HF model %s for attention attribution: %s",
            HF_MODEL_NAME,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_embedder() -> Any:
    """Return the shared all-MiniLM embedder (same one the retriever uses)."""
    from rag_pipeline.vectorstore import get_embedding_function

    return get_embedding_function()


def _embed(texts: Sequence[str]) -> np.ndarray:
    """Embed texts via the shared sentence-transformers function."""
    ef = _get_embedder()
    try:
        vectors = ef(list(texts))
    except TypeError:
        # LangChain-style embedder
        vectors = ef.embed_documents(list(texts))
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _split_sentences(text: str) -> List[str]:
    """Conservative sentence splitter — no NLTK download required."""
    if not text or not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# 1. SHAP on retrieval scores
# ---------------------------------------------------------------------------


def shap_retrieval_explain(  # XAI CONTRIBUTION
    question: str,
    source_docs: List[dict],
    retrieval_scores: List[float],
    answer: Optional[str] = None,
) -> List[float]:
    """
    Per-document SHAP values for the current query.

    Approach: treat each source doc as a binary feature (present/absent);
    the target is cosine similarity between the mask-weighted context
    embedding and the answer embedding (falls back to the mean retrieval
    score if no answer is supplied). Returns one float per source doc.
    """
    n = len(source_docs)
    if n == 0:
        return []

    # Cache context + answer embeddings once.
    contexts = [d.get("page_content", "") for d in source_docs]
    try:
        ctx_vecs = _embed(contexts)
        if answer:
            ans_vec = _embed([answer])[0]
        else:
            ans_vec = None
    except Exception as exc:
        logger.warning("SHAP embedding failed (%s); returning zeros", exc)
        return [0.0] * n

    scores_arr = np.asarray(retrieval_scores, dtype=np.float64)
    if scores_arr.shape[0] != n:
        # Pad/truncate defensively.
        scores_arr = np.resize(scores_arr, n)

    def f(mask_batch: np.ndarray) -> np.ndarray:
        out = np.zeros(mask_batch.shape[0], dtype=np.float64)
        for i, mask in enumerate(mask_batch):
            weights = mask * scores_arr
            total = weights.sum()
            if total <= 0:
                out[i] = 0.0
                continue
            fused = (weights[:, None] * ctx_vecs).sum(axis=0) / total
            if ans_vec is not None:
                out[i] = _cosine(fused, ans_vec)
            else:
                out[i] = float(weights.mean())
        return out

    try:
        import shap

        background = np.zeros((1, n), dtype=np.float64)
        explainer = shap.KernelExplainer(f, background, silent=True)
        np.random.seed(_SEED)
        x = np.ones((1, n), dtype=np.float64)
        values = explainer.shap_values(x, nsamples=_SHAP_NSAMPLES, silent=True)
        values = np.asarray(values).reshape(-1)
        if values.shape[0] != n:
            values = np.resize(values, n)
        return [float(v) for v in values]
    except Exception as exc:
        logger.warning("SHAP KernelExplainer failed (%s); using score proxy", exc)
        # Fallback: normalized retrieval scores centred on the mean.
        centred = scores_arr - scores_arr.mean()
        return [float(v) for v in centred]


# ---------------------------------------------------------------------------
# 2. Attention attribution via distilbert gradient saliency
# ---------------------------------------------------------------------------


def _saliency_spans(
    tokenizer: Any,
    model: Any,
    context: str,
    sentence: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    """Compute gradient saliency of context tokens w.r.t. the sentence."""
    import torch

    pair = tokenizer(
        sentence,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        return_offsets_mapping=True,
    )
    offsets = pair.pop("offset_mapping")[0].tolist()
    input_ids = pair["input_ids"]

    embeds = model.get_input_embeddings()(input_ids).clone().detach()
    embeds.requires_grad_(True)

    outputs = model(inputs_embeds=embeds, attention_mask=pair["attention_mask"])
    pooled = outputs.last_hidden_state.mean(dim=1).sum()
    pooled.backward()

    saliency = embeds.grad.abs().sum(dim=-1).squeeze(0).detach().numpy()

    # Only keep context-side tokens (token_type_ids == 1 for BERT/distilbert pair).
    tok_types = pair.get("token_type_ids")
    if tok_types is not None:
        mask = tok_types.squeeze(0).numpy().astype(bool)
    else:
        # distilbert does not expose token_type_ids by default — fall back
        # to locating the [SEP] between the two segments.
        sep_id = tokenizer.sep_token_id
        ids = input_ids.squeeze(0).tolist()
        try:
            first_sep = ids.index(sep_id)
        except ValueError:
            first_sep = len(ids) // 2
        mask = np.zeros(len(ids), dtype=bool)
        mask[first_sep + 1 :] = True

    spans: List[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(offsets):
        if not mask[idx]:
            continue
        if start == 0 and end == 0:
            continue
        text = context[start:end]
        if not text.strip():
            continue
        spans.append(
            {
                "text": text,
                "score": float(saliency[idx]),
                "char_start": int(start),
                "char_end": int(end),
            }
        )

    spans.sort(key=lambda s: s["score"], reverse=True)
    return spans[:top_n]


def attention_attribution(  # XAI CONTRIBUTION
    question: str,
    context: str,
    answer: str,
) -> List[dict]:
    """
    Gradient-saliency attribution per answer sentence.

    Returns EXACTLY ``ATTENTION_FIXED_SENTENCES`` entries (empty-padded or
    truncated) so the downstream explanation vector is shape-stable.
    """
    sentences = _split_sentences(answer)
    sentences = sentences[:ATTENTION_FIXED_SENTENCES]

    loaded = _get_hf_model()
    results: List[dict] = []

    if loaded is None or not context:
        logger.info("Attention attribution unavailable; returning empty padded slots")
    else:
        tokenizer, model = loaded
        for idx, sent in enumerate(sentences):
            try:
                spans = _saliency_spans(
                    tokenizer,
                    model,
                    context,
                    sent,
                    ATTENTION_SPANS_PER_SENTENCE,
                )
            except Exception as exc:
                logger.warning("Saliency failed for sentence %d (%s)", idx, exc)
                spans = []
            results.append({"sentence_idx": idx, "sentence": sent, "spans": spans})

    # Pad to fixed length.
    while len(results) < ATTENTION_FIXED_SENTENCES:
        results.append(
            {"sentence_idx": len(results), "sentence": "", "spans": []}
        )
    return results


# ---------------------------------------------------------------------------
# 3. Fixed-dim explanation vector
# ---------------------------------------------------------------------------


def build_explanation_vector(  # XAI CONTRIBUTION
    shap_values: Sequence[float],
    attention_spans: Sequence[dict],
) -> np.ndarray:
    """
    Concatenate top-K SHAP values and flattened top-N attention span
    scores into a fixed-dim (20,) float64 vector. Deterministic — two
    calls with the same inputs pass ``np.array_equal``.
    """
    vec = np.zeros(EXPLANATION_VECTOR_DIM, dtype=np.float64)

    # ─ SHAP slice ────────────────────────────────────────────────────────
    shap_arr = np.asarray(list(shap_values), dtype=np.float64)
    if shap_arr.size:
        # Stable sort: by (-abs(value), original_index)
        order = sorted(range(shap_arr.size), key=lambda i: (-abs(shap_arr[i]), i))
        topk = order[:TOP_K_SHAP]
        for slot, idx in enumerate(topk):
            vec[slot] = float(shap_arr[idx])
    # (remaining SHAP slots stay as zeros)

    # ─ Attention slice ───────────────────────────────────────────────────
    for sent_slot in range(ATTENTION_FIXED_SENTENCES):
        if sent_slot < len(attention_spans):
            spans = attention_spans[sent_slot].get("spans", [])
        else:
            spans = []
        for span_slot in range(ATTENTION_SPANS_PER_SENTENCE):
            offset = TOP_K_SHAP + sent_slot * ATTENTION_SPANS_PER_SENTENCE + span_slot
            if span_slot < len(spans):
                vec[offset] = float(spans[span_slot].get("score", 0.0))
            # else: already 0.0

    return vec


# ---------------------------------------------------------------------------
# 4. Term-level attribution (wraps the BM25 helper in chain.py)
# ---------------------------------------------------------------------------


def term_attribution(  # XAI CONTRIBUTION
    question: str,
    source_docs: List[dict],
) -> Dict[str, Any]:
    """
    Aggregate BM25 term scores across all source documents.

    Returns
    -------
    {
        "global":  {term: max_score_across_docs},
        "per_doc": [{"doc_idx": int, "scores": {term: score}}, ...],
    }
    """
    from rag_pipeline.chain import bm25_term_scores

    global_scores: Dict[str, float] = {}
    per_doc: List[Dict[str, Any]] = []

    for idx, doc in enumerate(source_docs):
        content = doc.get("page_content", "")
        scores = bm25_term_scores(question, content)
        per_doc.append({"doc_idx": idx, "scores": scores})
        for term, score in scores.items():
            global_scores[term] = max(global_scores.get(term, 0.0), score)

    sorted_global = dict(
        sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
    )
    return {"global": sorted_global, "per_doc": per_doc}


# ---------------------------------------------------------------------------
# 5. Hallucination classifier (rule-based fallback when unfitted)
# ---------------------------------------------------------------------------


def _clip01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


class HallucinationClassifier:  # XAI CONTRIBUTION
    """
    Logistic-regression classifier over ``_FEATURE_ORDER``.

    Ships unfitted; ``predict_proba`` uses a deterministic rule-based
    score until the model has been trained (typically after the
    ablation run — see ``evaluation/ablations/RUNBOOK.md``).
    """

    def __init__(self, model_path: Path = CLF_PATH) -> None:
        self.model_path = Path(model_path)
        self._model: Any = None
        self._fitted: bool = False
        if self.model_path.exists():
            try:
                import joblib

                self._model = joblib.load(self.model_path)
                self._fitted = True
                logger.info("Loaded hallucination classifier from %s", self.model_path)
            except Exception as exc:  # pragma: no cover — env-dependent
                logger.info(
                    "Could not load hallucination classifier from %s (%s); "
                    "using rule-based fallback",
                    self.model_path,
                    exc,
                )
        else:
            logger.info(
                "Hallucination classifier not found at %s; using rule-based fallback",
                self.model_path,
            )

    # ---- public API -------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        import joblib

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if X.ndim != 2 or X.shape[1] != len(_FEATURE_ORDER):
            raise ValueError(
                f"X must be shape (n, {len(_FEATURE_ORDER)}); got {X.shape}"
            )

        model = LogisticRegression(max_iter=1000, random_state=_SEED)
        model.fit(X, y)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        self._model = model
        self._fitted = True
        logger.info("Saved hallucination classifier to %s", self.model_path)

    def predict_proba(self, features: Any) -> float:
        vec = self._features_to_vector(features)
        if self._fitted and self._model is not None:
            probs = self._model.predict_proba(vec.reshape(1, -1))[0]
            # Class 1 = hallucination. Guard against single-class fits.
            if len(probs) == 2:
                return float(probs[1])
            return float(probs[-1])
        return self._rule_based_risk(vec)

    def explain(self, features: Any) -> str:
        vec = self._features_to_vector(features)
        if self._fitted and self._model is not None:
            try:
                import shap

                background = np.zeros((1, vec.size), dtype=np.float64)
                explainer = shap.LinearExplainer(self._model, background)
                sv = np.asarray(explainer.shap_values(vec.reshape(1, -1))).reshape(-1)
                order = sorted(
                    range(sv.size), key=lambda i: -abs(sv[i])
                )[:2]
                parts = [
                    f"{_FEATURE_ORDER[i]} ({sv[i]:+.2f})" for i in order
                ]
                return "SHAP-explained: " + ", ".join(parts)
            except Exception as exc:  # pragma: no cover
                logger.info("LinearExplainer failed (%s); using rule-based reason", exc)
        # rule-based
        mean_rs, min_rs, shap_ent, *_ = vec
        dominant = max(
            (
                ("low mean_retrieval_score", 1.0 - mean_rs),
                ("low min_retrieval_score", 1.0 - min_rs),
                ("high shap_entropy", shap_ent),
            ),
            key=lambda kv: kv[1],
        )
        return f"rule-based: dominant factor = {dominant[0]} ({dominant[1]:+.2f})"

    # ---- internals --------------------------------------------------

    def _features_to_vector(self, features: Any) -> np.ndarray:
        if isinstance(features, dict):
            return np.asarray(
                [float(features.get(name, 0.0)) for name in _FEATURE_ORDER],
                dtype=np.float64,
            )
        arr = np.asarray(features, dtype=np.float64).reshape(-1)
        if arr.size != len(_FEATURE_ORDER):
            raise ValueError(
                f"features vector must have length {len(_FEATURE_ORDER)}"
            )
        return arr

    @staticmethod
    def _rule_based_risk(vec: np.ndarray) -> float:
        # Map: retrieval confidence up → risk down; shap entropy up → risk up.
        mean_rs = _clip01(float(vec[0]))
        min_rs = _clip01(float(vec[1]))
        shap_ent = _clip01(float(vec[2]))
        risk = 0.5 * (1.0 - mean_rs) + 0.3 * (1.0 - min_rs) + 0.2 * shap_ent
        return _clip01(risk)


# ---------------------------------------------------------------------------
# 6. Unified entrypoint
# ---------------------------------------------------------------------------


def _shap_entropy(shap_values: Sequence[float]) -> float:
    arr = np.abs(np.asarray(list(shap_values), dtype=np.float64))
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    # Normalized Shannon entropy.
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(p * np.log(p + 1e-12))
    if arr.size <= 1:
        return 0.0
    return float(ent / math.log(arr.size))


def _top_attention_score(attention_spans: Sequence[dict]) -> float:
    best = 0.0
    for entry in attention_spans:
        for span in entry.get("spans", []):
            best = max(best, float(span.get("score", 0.0)))
    return best


def explain(  # XAI CONTRIBUTION
    question: str,
    source_docs: List[dict],
    retrieval_scores: List[float],
    answer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified XAI entrypoint. Returns the six-key payload consumed by the
    FastAPI ``/explain`` endpoint and the Streamlit XAI panel.
    """
    shap_values = shap_retrieval_explain(
        question, source_docs, retrieval_scores, answer=answer
    )

    # Combine all retrieved contexts for attention attribution.
    context_joined = "\n\n".join(d.get("page_content", "") for d in source_docs)
    token_attributions = attention_attribution(
        question, context_joined, answer or ""
    )

    term_scores = term_attribution(question, source_docs)
    explanation_vector = build_explanation_vector(shap_values, token_attributions)

    features = {
        "mean_retrieval_score": float(np.mean(retrieval_scores)) if retrieval_scores else 0.0,
        "min_retrieval_score": float(np.min(retrieval_scores)) if retrieval_scores else 0.0,
        "shap_entropy": _shap_entropy(shap_values),
        "num_source_docs": float(len(source_docs)),
        "answer_len_tokens": float(len((answer or "").split())),
        "top_attention_score": _top_attention_score(token_attributions),
    }

    classifier = HallucinationClassifier()
    risk = classifier.predict_proba(features)
    reason = classifier.explain(features)

    return {
        "shap_values": shap_values,
        "token_attributions": token_attributions,
        "term_attribution": term_scores,
        "explanation_vector": explanation_vector,
        "hallucination_risk": risk,
        "hallucination_reason": reason,
    }
