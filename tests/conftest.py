"""
Shared pytest fixtures for RAGOps unit and integration tests.

Unit tests mock ChromaDB, Ollama and the embedding function so they can
run inside the ``unit-tests`` CI job without any live services.
Integration tests opt in via the ``integration`` marker and require
``RAGOPS_E2E=1`` plus the full Docker stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# MLflow — isolated SQLite store per test
# ---------------------------------------------------------------------------


@pytest.fixture
def mlflow_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Fresh SQLite MLflow store scoped to a single test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("ENVIRONMENT", "test")
    return uri


# ---------------------------------------------------------------------------
# ChromaDB — mocked client + collection
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_chroma_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``rag_pipeline.vectorstore.get_chroma_client`` with a MagicMock."""
    client = MagicMock(name="mock_chroma_client")
    collection = MagicMock(name="mock_chroma_collection")
    collection.count.return_value = 3
    collection.query.return_value = {
        "ids": [["doc1", "doc2", "doc3"]],
        "distances": [[0.08, 0.19, 0.26]],
        "documents": [["ctx one", "ctx two", "ctx three"]],
        "metadatas": [[{"pmid": "1"}, {"pmid": "2"}, {"pmid": "3"}]],
    }
    client.list_collections.return_value = []
    client.get_or_create_collection.return_value = collection

    import rag_pipeline.vectorstore as vs

    monkeypatch.setattr(vs, "get_chroma_client", lambda: client)
    monkeypatch.setattr(vs, "get_or_create_collection", lambda *a, **k: collection)
    return client


# ---------------------------------------------------------------------------
# Ollama — deterministic chat + embedding stubs
# ---------------------------------------------------------------------------


class _StubChatResponse:
    def __init__(self, content: str = "[mocked answer]") -> None:
        self.content = content


class _StubChatOllama:
    """Stand-in for ``langchain_community.chat_models.ChatOllama``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def invoke(self, prompt: str) -> _StubChatResponse:
        return _StubChatResponse(f"[mocked answer for: {prompt[:40]}]")


class _StubEmbeddings:
    """Deterministic embedder — returns arange/n so tests are reproducible."""

    dim = 16

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    def _vec(self, text: str) -> List[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.random(self.dim)
        return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def mock_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch both chat and embedding Ollama clients with deterministic stubs."""
    import rag_pipeline.chain as chain_mod

    monkeypatch.setattr(chain_mod, "ChatOllama", _StubChatOllama)
    monkeypatch.setattr(chain_mod, "_get_llm", lambda: _StubChatOllama())

    # Patch the embedding function factory used by the retriever.
    import rag_pipeline.vectorstore as vs

    monkeypatch.setattr(vs, "get_embedding_function", lambda: _StubEmbeddings())


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_source_docs() -> List[dict]:
    """Three fake source docs matching ``chain.query()`` output shape."""
    return [
        {
            "page_content": (
                "Metformin remains the preferred initial pharmacological agent for "
                "type 2 diabetes due to its efficacy, safety profile, and low cost."
            ),
            "metadata": {"pmid": "1001", "source": "PubMed"},
        },
        {
            "page_content": (
                "Lifestyle modification including diet, weight loss, and exercise "
                "is recommended alongside pharmacotherapy for type 2 diabetes."
            ),
            "metadata": {"pmid": "1002", "source": "PubMed"},
        },
        {
            "page_content": (
                "HbA1c below 7% is a reasonable glycaemic target for most adults "
                "with type 2 diabetes, balancing efficacy and hypoglycaemia risk."
            ),
            "metadata": {"pmid": "1003", "source": "PubMed"},
        },
    ]


@pytest.fixture
def fake_retrieval_scores() -> List[float]:
    return [0.92, 0.81, 0.74]


@pytest.fixture
def fake_qa_pairs() -> List[dict]:
    """
    Small QA set for unit tests. Prefers the CI benchmark if available,
    otherwise falls back to the runner's built-in stub pairs.
    """
    repo_root = Path(__file__).resolve().parents[1]
    ci_path = repo_root / "evaluation" / "benchmarks" / "ci_benchmark.json"
    if ci_path.exists():
        with ci_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list) and data:
            return data[:5]

    from evaluation.ragas_runner import _ci_stub_pairs

    return _ci_stub_pairs()


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_shared_caches() -> Iterator[None]:
    """Reset explainability caches between tests to keep determinism."""
    yield
    try:
        from evaluation import explainability as _xai
    except Exception:
        return
    if hasattr(_xai, "_get_hf_model") and hasattr(_xai._get_hf_model, "cache_clear"):
        _xai._get_hf_model.cache_clear()
