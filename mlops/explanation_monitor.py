"""
XAI Explanation Consistency Monitor.
# XAI CONTRIBUTION

Tracks explanation vector stability over time using cosine similarity.
Consumes explanation_vector from evaluation.explainability.explain().

Thresholds (from settings):
    >= 0.75 -> stable
    [0.60, 0.75) -> log warning
    < 0.60 -> escalate as "explanation instability" alert

Novel metric: no existing RAGOps work tracks explanation stability over time.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config.settings import settings
from src.infra.baseline_store import get_baseline_store

FloatArray = NDArray[np.floating[Any]]

logger = logging.getLogger("ragops.explanation_monitor")


def _cosine_similarity(a: FloatArray, b: FloatArray) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# XAI CONTRIBUTION
def save_baseline(vectors: list[FloatArray]) -> None:
    """Store explanation vectors as the XAI baseline for future comparisons.

    Should be called after the first successful evaluation run with
    the benchmark questions.

    Args:
        vectors: One explanation vector per benchmark question.
    """
    store = get_baseline_store()
    store.save_xai_baseline(vectors)
    logger.info("XAI baseline saved — %d vectors", len(vectors))


# XAI CONTRIBUTION
def compute_consistency(
    current_vectors: list[FloatArray] | None = None,
    baseline_vectors: list[FloatArray] | None = None,
) -> float:
    """Compute mean cosine similarity between current and baseline explanation vectors.

    Args:
        current_vectors: Explanation vectors for benchmark questions from current run.
                         Each vector is np.ndarray of shape (n_features,).
                         If None, raises RuntimeError.
        baseline_vectors: Stored baseline vectors. If None, loads from baseline store.

    Returns:
        explanation_consistency_score (0.0 - 1.0).
        Mean cosine similarity across all benchmark question pairs.

    Raises:
        RuntimeError: If no baseline exists and baseline_vectors is None.
        ValueError: If current_vectors is None or empty.
    """
    # XAI CONTRIBUTION
    if current_vectors is None or len(current_vectors) == 0:
        raise ValueError("current_vectors must be provided and non-empty")

    if baseline_vectors is None:
        store = get_baseline_store()
        baseline_vectors = store.load_xai_baseline()

    if baseline_vectors is None:
        raise RuntimeError(
            "No XAI baseline found. Call save_baseline() first "
            "after initial evaluation with benchmark questions."
        )

    # Compute similarity for overlapping pairs
    n_pairs = min(len(current_vectors), len(baseline_vectors))
    if n_pairs < len(current_vectors) or n_pairs < len(baseline_vectors):
        logger.warning(
            "Vector count mismatch: current=%d, baseline=%d. "
            "Computing similarity for %d overlapping pairs.",
            len(current_vectors),
            len(baseline_vectors),
            n_pairs,
        )

    similarities = [
        _cosine_similarity(current_vectors[i], baseline_vectors[i])
        for i in range(n_pairs)
    ]

    score = float(np.mean(similarities))

    # Log status based on thresholds
    if score >= settings.xai_warning_threshold:
        logger.info("XAI consistency STABLE: %.4f", score)
    elif score >= settings.xai_alert_threshold:
        logger.warning(
            "XAI consistency WARNING: %.4f < %.2f",
            score,
            settings.xai_warning_threshold,
        )
    else:
        logger.warning(
            "XAI INSTABILITY ALERT: %.4f < %.2f",
            score,
            settings.xai_alert_threshold,
        )

    return score
