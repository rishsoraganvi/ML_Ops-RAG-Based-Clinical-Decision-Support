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
from abc import ABC, abstractmethod

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
# Factory — single swap point for production store
# ---------------------------------------------------------------------------

def get_baseline_store() -> BaselineStore:
    """
    Return the active baseline store implementation.

    To switch backends, replace ``InMemoryBaselineStore()`` here with
    the desired concrete class. All callers automatically use the new store.

    Returns
    -------
    BaselineStore
        Active store instance.
    """
    return InMemoryBaselineStore()
