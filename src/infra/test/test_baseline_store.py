"""Tests for src.infra.baseline_store — FileBaselineStore + factory switch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.infra.baseline_store as bs
from src.infra.baseline_store import (
    FileBaselineStore,
    InMemoryBaselineStore,
    get_baseline_store,
)


@pytest.fixture(autouse=True)
def _reset_store_singleton() -> None:
    """Ensure factory state doesn't leak between cases in this file."""
    bs._store = None
    yield
    bs._store = None


# ---------------------------------------------------------------------------
# FileBaselineStore — PSI
# ---------------------------------------------------------------------------


class TestFilePsi:
    def test_roundtrip_preserves_matrix(self, tmp_path: Path) -> None:
        store = FileBaselineStore(tmp_path)
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((50, 384)).astype(np.float32)

        store.save_psi_baseline(embeddings)

        assert store.has_psi_baseline() is True
        loaded = store.load_psi_baseline()
        assert loaded is not None
        np.testing.assert_array_equal(loaded, embeddings)

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        store = FileBaselineStore(tmp_path)
        assert store.has_psi_baseline() is False
        assert store.load_psi_baseline() is None

    def test_rejects_1d_array(self, tmp_path: Path) -> None:
        store = FileBaselineStore(tmp_path)
        with pytest.raises(ValueError, match="2-D embedding matrix"):
            store.save_psi_baseline(np.array([1.0, 2.0, 3.0]))

    def test_load_returns_none_on_corrupt_file(self, tmp_path: Path) -> None:
        store = FileBaselineStore(tmp_path)
        store._psi_path.write_bytes(b"not a real npy file")
        assert store.load_psi_baseline() is None


# ---------------------------------------------------------------------------
# FileBaselineStore — XAI
# ---------------------------------------------------------------------------


class TestFileXai:
    def test_roundtrip_preserves_order_and_shapes(self, tmp_path: Path) -> None:
        store = FileBaselineStore(tmp_path)
        rng = np.random.default_rng(1)
        vectors = [
            rng.standard_normal(10),
            rng.standard_normal(15),
            rng.standard_normal(8),
            rng.standard_normal(12),
            rng.standard_normal(20),
        ]

        store.save_xai_baseline(vectors)

        assert store.has_xai_baseline() is True
        loaded = store.load_xai_baseline()
        assert loaded is not None
        assert len(loaded) == len(vectors)
        for got, want in zip(loaded, vectors):
            np.testing.assert_array_equal(got, want)

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        store = FileBaselineStore(tmp_path)
        assert store.has_xai_baseline() is False
        assert store.load_xai_baseline() is None


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "subdir" / "baselines"
        assert not target.exists()
        FileBaselineStore(target)
        assert target.is_dir()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_uses_file_store_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RAGOPS_BASELINE_STORE", raising=False)
        from src.config.settings import settings

        monkeypatch.setattr(settings, "baseline_dir", tmp_path)

        store = get_baseline_store()
        assert isinstance(store, FileBaselineStore)

    def test_uses_memory_store_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAGOPS_BASELINE_STORE", "memory")

        store = get_baseline_store()
        assert isinstance(store, InMemoryBaselineStore)
