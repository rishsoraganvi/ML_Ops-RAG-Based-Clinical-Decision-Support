"""
Unit tests for ``evaluation/explainability.py``.

These tests lock the determinism + shape contracts required by the
Streamlit XAI panel, the FastAPI ``/explain`` endpoint, and the
``mlops.explanation_monitor.compute_consistency`` consumer.

Heavy dependencies (distilbert, shap.KernelExplainer) are monkeypatched
so the unit-tests CI job does not need torch/transformers downloads.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from evaluation import explainability as xai
from evaluation.explainability import (
    ATTENTION_FIXED_SENTENCES,
    ATTENTION_SPANS_PER_SENTENCE,
    EXPLANATION_VECTOR_DIM,
    HallucinationClassifier,
    TOP_K_SHAP,
    _FEATURE_ORDER,
    build_explanation_vector,
    term_attribution,
)


# ---------------------------------------------------------------------------
# build_explanation_vector
# ---------------------------------------------------------------------------


def _stub_attention(n_sentences: int = 3) -> List[dict]:
    """Helper: build an attention_spans list of the given length."""
    out = []
    for i in range(n_sentences):
        out.append(
            {
                "sentence_idx": i,
                "sentence": f"s{i}",
                "spans": [
                    {
                        "text": "a",
                        "score": 0.9 - i * 0.1,
                        "char_start": 0,
                        "char_end": 1,
                    },
                    {
                        "text": "b",
                        "score": 0.5 - i * 0.1,
                        "char_start": 2,
                        "char_end": 3,
                    },
                    {
                        "text": "c",
                        "score": 0.2 - i * 0.05,
                        "char_start": 4,
                        "char_end": 5,
                    },
                ],
            }
        )
    return out


class TestBuildExplanationVector:
    def test_shape_and_dtype(self) -> None:
        vec = build_explanation_vector([0.5, -0.3, 0.1], _stub_attention(2))
        assert vec.shape == (EXPLANATION_VECTOR_DIM,)
        assert vec.shape == (20,)
        assert vec.dtype == np.float64

    def test_determinism(self) -> None:
        shap_values = [0.2, -0.5, 0.1, 0.4, -0.33, 0.02]
        spans = _stub_attention(4)
        a = build_explanation_vector(shap_values, spans)
        b = build_explanation_vector(list(shap_values), list(spans))
        assert np.array_equal(a, b)

    def test_empty_inputs_produce_zero_vector(self) -> None:
        vec = build_explanation_vector([], [])
        assert vec.shape == (EXPLANATION_VECTOR_DIM,)
        assert np.all(vec == 0.0)

    def test_padding_when_fewer_sentences(self) -> None:
        vec = build_explanation_vector([1.0], _stub_attention(1))
        # Only the first SHAP slot and the first-sentence attention slots are set.
        assert vec[0] == pytest.approx(1.0)
        assert np.all(vec[1:TOP_K_SHAP] == 0.0)
        # Slots for sentence index >= 1 stay zero.
        start = TOP_K_SHAP + 1 * ATTENTION_SPANS_PER_SENTENCE
        assert np.all(vec[start:] == 0.0)

    def test_shap_topk_orders_by_abs_value(self) -> None:
        # |value| ordering: -0.9, 0.8, 0.7, -0.4, 0.3
        shap_values = [0.3, -0.4, 0.7, 0.8, -0.9]
        vec = build_explanation_vector(shap_values, [])
        assert vec[0] == pytest.approx(-0.9)
        assert vec[1] == pytest.approx(0.8)
        assert vec[2] == pytest.approx(0.7)
        assert vec[3] == pytest.approx(-0.4)
        assert vec[4] == pytest.approx(0.3)

    def test_truncates_excess_sentences(self) -> None:
        # 7 sentences > ATTENTION_FIXED_SENTENCES (5) — extras ignored.
        vec = build_explanation_vector([], _stub_attention(7))
        assert vec.shape == (EXPLANATION_VECTOR_DIM,)


# ---------------------------------------------------------------------------
# term_attribution
# ---------------------------------------------------------------------------


class TestTermAttribution:
    def test_wraps_bm25_term_scores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: List[tuple] = []

        def fake_scores(q: str, doc: str) -> dict:
            calls.append((q, doc))
            return (
                {"metformin": 0.8, "diabetes": 0.5}
                if "metformin" in doc
                else {"diabetes": 0.3}
            )

        # Patch where it's imported inside term_attribution.
        monkeypatch.setattr("rag_pipeline.chain.bm25_term_scores", fake_scores)

        source_docs = [
            {"page_content": "metformin is first-line", "metadata": {}},
            {"page_content": "diabetes affects millions", "metadata": {}},
        ]
        result = term_attribution("metformin for diabetes", source_docs)

        assert len(calls) == 2
        assert "global" in result
        assert "per_doc" in result
        assert result["global"]["metformin"] == pytest.approx(0.8)
        # max across docs for overlapping terms
        assert result["global"]["diabetes"] == pytest.approx(0.5)
        assert len(result["per_doc"]) == 2

    def test_empty_source_docs(self) -> None:
        result = term_attribution("anything", [])
        assert result == {"global": {}, "per_doc": []}


# ---------------------------------------------------------------------------
# HallucinationClassifier
# ---------------------------------------------------------------------------


class TestHallucinationClassifier:
    def test_rule_based_fallback_when_unfitted(self, tmp_path) -> None:
        clf = HallucinationClassifier(model_path=tmp_path / "none.joblib")
        risk = clf.predict_proba(
            {"mean_retrieval_score": 0.9, "min_retrieval_score": 0.8}
        )
        assert 0.0 <= risk <= 1.0
        reason = clf.explain({"mean_retrieval_score": 0.9, "min_retrieval_score": 0.8})
        assert isinstance(reason, str) and reason

    def test_rule_based_risk_increases_when_retrieval_is_poor(self, tmp_path) -> None:
        clf = HallucinationClassifier(model_path=tmp_path / "none.joblib")
        healthy = clf.predict_proba(
            {
                "mean_retrieval_score": 0.95,
                "min_retrieval_score": 0.9,
                "shap_entropy": 0.1,
            }
        )
        risky = clf.predict_proba(
            {
                "mean_retrieval_score": 0.2,
                "min_retrieval_score": 0.05,
                "shap_entropy": 0.95,
            }
        )
        assert risky > healthy

    def test_features_to_vector_respects_order(self, tmp_path) -> None:
        clf = HallucinationClassifier(model_path=tmp_path / "none.joblib")
        vec = clf._features_to_vector(
            {name: float(i) for i, name in enumerate(_FEATURE_ORDER)}
        )
        for i, _ in enumerate(_FEATURE_ORDER):
            assert vec[i] == pytest.approx(float(i))

    def test_fit_persists_and_predicts(self, tmp_path) -> None:
        pytest.importorskip("sklearn")
        pytest.importorskip("joblib")

        model_path = tmp_path / "clf.joblib"
        clf = HallucinationClassifier(model_path=model_path)

        rng = np.random.default_rng(0)
        # Healthy rows — high retrieval, low entropy.
        healthy = rng.uniform(0.8, 1.0, size=(15, len(_FEATURE_ORDER)))
        healthy[:, 2] = rng.uniform(0.0, 0.2, size=15)
        # Risky rows — low retrieval, high entropy.
        risky = rng.uniform(0.0, 0.2, size=(15, len(_FEATURE_ORDER)))
        risky[:, 2] = rng.uniform(0.7, 1.0, size=15)

        X = np.vstack([healthy, risky])
        y = np.array([0] * 15 + [1] * 15)

        clf.fit(X, y)
        assert model_path.exists(), "joblib artifact should be written"

        healthy_risk = clf.predict_proba(
            {name: 0.9 if name != "shap_entropy" else 0.1 for name in _FEATURE_ORDER}
        )
        risky_risk = clf.predict_proba(
            {name: 0.1 if name != "shap_entropy" else 0.9 for name in _FEATURE_ORDER}
        )
        assert risky_risk > healthy_risk


# ---------------------------------------------------------------------------
# explain() end-to-end smoke (with monkey-patched heavy deps)
# ---------------------------------------------------------------------------


class TestExplainEndToEnd:
    def test_returns_six_key_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stub SHAP so no real KernelExplainer runs.
        monkeypatch.setattr(
            xai,
            "shap_retrieval_explain",
            lambda q, sd, rs, answer=None: [0.4, -0.2, 0.1],
        )
        # Stub attention so transformers isn't loaded.
        monkeypatch.setattr(
            xai,
            "attention_attribution",
            lambda q, ctx, ans: [
                {"sentence_idx": i, "sentence": f"s{i}", "spans": []}
                for i in range(ATTENTION_FIXED_SENTENCES)
            ],
        )
        # Stub bm25 so we don't hit real chain internals.
        monkeypatch.setattr(
            "rag_pipeline.chain.bm25_term_scores",
            lambda q, d: {"diabetes": 0.5},
        )

        source_docs = [
            {"page_content": "diabetes treatment", "metadata": {}},
            {"page_content": "metformin is first-line", "metadata": {}},
            {"page_content": "lifestyle changes help", "metadata": {}},
        ]
        result = xai.explain(
            "What treats diabetes?",
            source_docs,
            [0.9, 0.8, 0.7],
            answer="Metformin is first-line.",
        )

        expected = {
            "shap_values",
            "token_attributions",
            "term_attribution",
            "explanation_vector",
            "hallucination_risk",
            "hallucination_reason",
        }
        assert set(result.keys()) == expected

        assert isinstance(result["shap_values"], list)
        assert isinstance(result["explanation_vector"], np.ndarray)
        assert result["explanation_vector"].shape == (EXPLANATION_VECTOR_DIM,)
        assert 0.0 <= result["hallucination_risk"] <= 1.0
        assert result["hallucination_reason"]

    def test_graceful_degradation_when_docs_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            xai,
            "attention_attribution",
            lambda q, ctx, ans: [
                {"sentence_idx": i, "sentence": "", "spans": []}
                for i in range(ATTENTION_FIXED_SENTENCES)
            ],
        )
        result = xai.explain("q", [], [], answer=None)
        assert result["shap_values"] == []
        assert result["explanation_vector"].shape == (EXPLANATION_VECTOR_DIM,)
        assert np.all(result["explanation_vector"] == 0.0)
