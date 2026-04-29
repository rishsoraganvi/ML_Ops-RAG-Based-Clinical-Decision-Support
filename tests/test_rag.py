"""
Unit tests for the RAG pipeline (retrievers, chain, query preprocessing).

All external services (ChromaDB, Ollama) are mocked via ``conftest.py``
fixtures so these tests run inside the lint/unit-tests CI job.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import pytest

from rag_pipeline import chain as chain_mod
from rag_pipeline.chain import bm25_term_scores
from rag_pipeline.query_processor import (
    MEDICAL_ABBREVIATIONS,
    MESH_SYNONYMS,
    preprocess_query,
)


# ---------------------------------------------------------------------------
# bm25_term_scores (existing chain.py helper used by term_attribution)
# ---------------------------------------------------------------------------


class TestBm25TermScores:
    def test_nonempty_for_overlapping_query(self) -> None:
        doc = (
            "Metformin is the preferred initial treatment for type 2 diabetes "
            "and reduces HbA1c effectively."
        )
        scores = bm25_term_scores("What is the first line treatment for diabetes?", doc)
        assert scores, "expected at least one overlapping term"
        assert all(isinstance(v, float) for v in scores.values())

    def test_sorted_descending(self) -> None:
        doc = "diabetes diabetes diabetes metformin metformin insulin"
        scores = bm25_term_scores("diabetes metformin insulin", doc)
        values = list(scores.values())
        assert values == sorted(values, reverse=True)

    def test_empty_on_no_overlap(self) -> None:
        doc = "completely unrelated text about astrophysics and quasars"
        scores = bm25_term_scores("hypertension treatment", doc)
        assert scores == {}

    def test_empty_doc_returns_empty(self) -> None:
        assert bm25_term_scores("anything", "") == {}


# ---------------------------------------------------------------------------
# query_processor.preprocess_query
# ---------------------------------------------------------------------------


class TestPreprocessQuery:
    def test_expands_known_abbreviation(self) -> None:
        out = preprocess_query("What is first-line for MI?")
        assert "myocardial infarction" in out
        # Original token preserved.
        assert "MI" in out

    def test_no_expansion_when_flag_off(self) -> None:
        out = preprocess_query("What is first-line for MI?", expand_abbreviations=False)
        assert "myocardial infarction" not in out

    def test_mesh_off_by_default(self) -> None:
        out = preprocess_query("How is diabetes diagnosed?")
        # Default: abbreviations only, no MeSH expansion.
        assert "diabetes mellitus" not in out

    def test_mesh_opt_in(self) -> None:
        out = preprocess_query("How is diabetes diagnosed?", expand_mesh=True)
        assert "diabetes mellitus" in out

    def test_idempotent(self) -> None:
        q = "CAD and HTN management"
        once = preprocess_query(q)
        twice = preprocess_query(once)
        assert once == twice, "preprocess_query must be idempotent"

    def test_unknown_abbrev_is_noop(self) -> None:
        q = "what is XYZQ disease?"
        assert preprocess_query(q) == q

    def test_empty_input(self) -> None:
        assert preprocess_query("") == ""
        assert preprocess_query("   ") == "   "

    def test_determinism(self) -> None:
        q = "MI in CAD with CKD and HTN"
        outs = {preprocess_query(q) for _ in range(50)}
        assert len(outs) == 1, "same input must yield identical output every time"

    def test_dictionaries_nonempty(self) -> None:
        assert "MI" in MEDICAL_ABBREVIATIONS
        assert "diabetes" in MESH_SYNONYMS


# ---------------------------------------------------------------------------
# query() — schema + preprocess flag wiring
# ---------------------------------------------------------------------------


def _fake_docs() -> List[SimpleNamespace]:
    return [
        SimpleNamespace(
            page_content="Metformin is first-line for T2DM.",
            metadata={"pmid": "1"},
        ),
        SimpleNamespace(
            page_content="Lifestyle changes support glycaemic control.",
            metadata={"pmid": "2"},
        ),
    ]


class TestQuery:
    def test_query_returns_expected_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Assert the 7-key schema contract from chain.py:169."""
        monkeypatch.setattr(
            chain_mod,
            "retrieve_with_scores",
            lambda **kw: (_fake_docs(), [0.91, 0.82], 12.5),
        )
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="mock answer")
        monkeypatch.setattr(chain_mod, "_get_llm", lambda: llm)

        out = chain_mod.query(
            "What is first-line for MI?",
            config={
                "retriever_type": "dense",
                "chunk_size": 256,
                "preprocess_query": False,
            },
        )

        expected = {
            "answer",
            "source_docs",
            "retrieval_latency_ms",
            "llm_latency_ms",
            "total_latency_ms",
            "retrieval_scores",
            "config",
        }
        assert set(out.keys()) == expected
        assert out["answer"] == "mock answer"
        assert out["retrieval_scores"] == [0.91, 0.82]
        assert len(out["source_docs"]) == 2
        assert out["source_docs"][0]["page_content"].startswith("Metformin")

    def test_preprocess_flag_enabled_expands_question(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        def fake_retrieve(**kwargs: object) -> tuple:
            captured["question"] = kwargs["question"]
            return (_fake_docs(), [0.9, 0.8], 10.0)

        monkeypatch.setattr(chain_mod, "retrieve_with_scores", fake_retrieve)
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="x")
        monkeypatch.setattr(chain_mod, "_get_llm", lambda: llm)

        chain_mod.query(
            "treatment for MI?",
            config={
                "retriever_type": "dense",
                "chunk_size": 256,
                "preprocess_query": True,
            },
        )
        assert "myocardial infarction" in captured["question"]

    def test_preprocess_flag_disabled_passes_raw_question(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        def fake_retrieve(**kwargs: object) -> tuple:
            captured["question"] = kwargs["question"]
            return (_fake_docs(), [0.9, 0.8], 10.0)

        monkeypatch.setattr(chain_mod, "retrieve_with_scores", fake_retrieve)
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="x")
        monkeypatch.setattr(chain_mod, "_get_llm", lambda: llm)

        chain_mod.query(
            "treatment for MI?",
            config={
                "retriever_type": "dense",
                "chunk_size": 256,
                "preprocess_query": False,
            },
        )
        assert captured["question"] == "treatment for MI?"


# ---------------------------------------------------------------------------
# retriever factory validation (lightweight — avoid loading Chroma)
# ---------------------------------------------------------------------------


class TestRetrieverFactory:
    def test_invalid_retriever_type_raises(self) -> None:
        from rag_pipeline.retriever import get_retriever

        with pytest.raises((ValueError, KeyError, AssertionError)):
            get_retriever(retriever_type="bogus", chunk_size=256, k=5)
