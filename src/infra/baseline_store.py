"""
RAGOps — Baseline Store
========================
Abstract interface for persisting PSI embedding distribution baselines
and XAI explanation vector baselines between eval runs.

Currently stubbed with an in-memory implementation so Components 2, 4,
and 6 can be developed and tested before the persistence backend is
decided (MLflow artifact vs. volume-mounted JSON).

Swap in a concrete implementation (e.g. ``MLflowBaselineStore``) by
changing the single ``get_baseline_store()`` factory at the bottom of
this module — callers are unaffected.

Usage::

    from src.infra.baseline_store import get_baseline_store

    store = get_baseline_store()
    store.save_psi_baseline(embeddings)
    baseline = store.load_psi_baseline()   # None on first run
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger("ragops.infra.baseline_store")


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaselineStore(ABC):
    """
    Abstract persistence contract for baseline distributions.

    Both PSI (embedding histograms) and XAI (explanation vectors)
    baselines flow through this interface so the storage backend is
    decoupled from detection logic.
    """

    # ── PSI baseline ─────────────────────────────────────────────────────────

    @abstractmethod
    def save_psi_baseline(self, embeddings: np.ndarray) -> None:
        """
        Persist the reference embedding matrix for future PSI comparisons.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n_docs, 384)`` — full embedding matrix from
            ``data.ingest.get_embeddings()``.  # PAPER CONTRIBUTION
        """

    @abstractmethod
    def load_psi_baseline(self) -> np.ndarray | None:
        """
        Retrieve the stored PSI baseline embedding matrix.

        Returns
        -------
        np.ndarray | None
            Baseline matrix, or ``None`` if no baseline has been saved yet
            (first run — caller should save current embeddings as baseline).
            # PAPER CONTRIBUTION
        """

    # ── XAI baseline ─────────────────────────────────────────────────────────

    @abstractmethod
    def save_xai_baseline(self, vectors: list[np.ndarray]) -> None:
        """
        Persist the reference explanation vectors for all benchmark questions.

        Parameters
        ----------
        vectors : list[np.ndarray]
            One explanation vector per benchmark question, each of shape
            ``(n_features,)``, sourced from
            ``evaluation.explainability.explain()``.  # XAI CONTRIBUTION
        """

    @abstractmethod
    def load_xai_baseline(self) -> list[np.ndarray] | None:
        """
        Retrieve the stored XAI baseline explanation vectors.

        Returns
        -------
        list[np.ndarray] | None
            List of baseline vectors, or ``None`` on first run.
            # XAI CONTRIBUTION
        """

    @abstractmethod
    def has_psi_baseline(self) -> bool:
        """Return True if a PSI baseline has been saved."""  # PAPER CONTRIBUTION

    @abstractmethod
    def has_xai_baseline(self) -> bool:
        """Return True if an XAI baseline has been saved."""  # XAI CONTRIBUTION


# ---------------------------------------------------------------------------
# In-memory stub (active implementation)
# ---------------------------------------------------------------------------


class InMemoryBaselineStore(BaselineStore):
    """
    Non-persistent in-memory baseline store.

    Baselines survive for the lifetime of the process only.
    Suitable for unit tests and single-run CI jobs.

    Replace with ``MLflowBaselineStore`` or ``VolumeBaselineStore``
    once the persistence backend is decided (Component 4 / 6 build session).
    """

    def __init__(self) -> None:
        self._psi_baseline: np.ndarray | None = None
        self._xai_baseline: list[np.ndarray] | None = None
        logger.warning(
            "InMemoryBaselineStore active — baselines will NOT persist "
            "across restarts. Replace with a durable store before production."
        )

    # ── PSI ──────────────────────────────────────────────────────────────────

    def save_psi_baseline(self, embeddings: np.ndarray) -> None:  # PAPER CONTRIBUTION
        """Store embedding matrix in memory."""
        logger.info(
            "PSI baseline saved in memory — shape=%s dtype=%s",
            embeddings.shape,
            embeddings.dtype,
        )
        self._psi_baseline = embeddings.copy()

    def load_psi_baseline(self) -> np.ndarray | None:  # PAPER CONTRIBUTION
        """Return in-memory PSI baseline, or None if not yet set."""
        return self._psi_baseline

    def has_psi_baseline(self) -> bool:  # PAPER CONTRIBUTION
        return self._psi_baseline is not None

    # ── XAI ──────────────────────────────────────────────────────────────────

    def save_xai_baseline(self, vectors: list[np.ndarray]) -> None:  # XAI CONTRIBUTION
        """Store explanation vectors in memory."""
        logger.info(
            "XAI baseline saved in memory — %d vectors, first shape=%s",
            len(vectors),
            vectors[0].shape if vectors else "n/a",
        )
        self._xai_baseline = [v.copy() for v in vectors]

    def load_xai_baseline(self) -> list[np.ndarray] | None:  # XAI CONTRIBUTION
        """Return in-memory XAI baseline vectors, or None if not yet set."""
        return self._xai_baseline

    def has_xai_baseline(self) -> bool:  # XAI CONTRIBUTION
        return self._xai_baseline is not None


# ---------------------------------------------------------------------------
# File-backed store — durable across process / container restarts
# ---------------------------------------------------------------------------


class FileBaselineStore(BaselineStore):
    """
    File-backed baseline store using NumPy ``.npy`` / ``.npz`` files.

    Persists to ``<baseline_dir>/embedding_baseline.npy`` (PSI) and
    ``<baseline_dir>/xai_baseline.npz`` (XAI), so baselines survive
    container restarts when the directory is on a Docker volume.

    The XAI baseline is stored as a single ``.npz`` archive with keys
    ``vec_0000``, ``vec_0001``, … so vectors of varying shapes can
    coexist without ``allow_pickle``.
    """

    def __init__(self, baseline_dir: Path | str) -> None:
        self._dir = Path(baseline_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._psi_path = self._dir / "embedding_baseline.npy"
        self._xai_path = self._dir / "xai_baseline.npz"
        logger.info("FileBaselineStore active — persisting baselines to %s", self._dir)

    # ── PSI ──────────────────────────────────────────────────────────────────

    def save_psi_baseline(self, embeddings: np.ndarray) -> None:  # PAPER CONTRIBUTION
        """Persist embedding matrix to ``embedding_baseline.npy``."""
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2-D embedding matrix, got shape {embeddings.shape}"
            )
        np.save(self._psi_path, embeddings)
        logger.info(
            "PSI baseline saved to %s — shape=%s dtype=%s",
            self._psi_path,
            embeddings.shape,
            embeddings.dtype,
        )

    def load_psi_baseline(self) -> np.ndarray | None:  # PAPER CONTRIBUTION
        """Load PSI baseline from disk, or ``None`` if missing/corrupt."""
        if not self._psi_path.is_file():
            return None
        try:
            return np.load(self._psi_path)
        except (OSError, ValueError) as exc:
            logger.error("Failed to load PSI baseline from %s: %s", self._psi_path, exc)
            return None

    def has_psi_baseline(self) -> bool:  # PAPER CONTRIBUTION
        return self._psi_path.is_file()

    # ── XAI ──────────────────────────────────────────────────────────────────

    def save_xai_baseline(self, vectors: list[np.ndarray]) -> None:  # XAI CONTRIBUTION
        """Persist explanation vectors to ``xai_baseline.npz``."""
        payload = {f"vec_{i:04d}": v for i, v in enumerate(vectors)}
        np.savez(self._xai_path, **payload)
        logger.info(
            "XAI baseline saved to %s — %d vectors, first shape=%s",
            self._xai_path,
            len(vectors),
            vectors[0].shape if vectors else "n/a",
        )

    def load_xai_baseline(self) -> list[np.ndarray] | None:  # XAI CONTRIBUTION
        """Load XAI baseline vectors from disk, or ``None`` if missing/corrupt."""
        if not self._xai_path.is_file():
            return None
        try:
            with np.load(self._xai_path) as data:
                keys = sorted(data.files, key=lambda k: int(k.split("_")[1]))
                return [np.array(data[k]) for k in keys]
        except (OSError, ValueError, KeyError, IndexError) as exc:
            logger.error("Failed to load XAI baseline from %s: %s", self._xai_path, exc)
            return None

    def has_xai_baseline(self) -> bool:  # XAI CONTRIBUTION
        return self._xai_path.is_file()


# ---------------------------------------------------------------------------
# Factory — single swap point for production store
# ---------------------------------------------------------------------------


_store: BaselineStore | None = None


def get_baseline_store() -> BaselineStore:
    """
    Return the active baseline store implementation (singleton).

    Selects backend via the ``RAGOPS_BASELINE_STORE`` environment variable:

    * ``"file"`` (default) — :class:`FileBaselineStore` rooted at
      ``settings.baseline_dir``. Required for cross-process persistence
      between the baseline script and the FastAPI container.
    * ``"memory"`` — :class:`InMemoryBaselineStore`. Used by the unit
      test suite to avoid filesystem side effects.

    Returns
    -------
    BaselineStore
        Active store instance.
    """
    global _store
    if _store is None:
        mode = os.environ.get("RAGOPS_BASELINE_STORE", "file").lower()
        if mode == "memory":
            _store = InMemoryBaselineStore()
        else:
            # Deferred import — keeps src.infra leaf-free of settings cycles.
            from src.config.settings import settings

            _store = FileBaselineStore(settings.baseline_dir)
    return _store
