"""
PSI Drift Detection for RAG Embedding Distributions.
# PAPER CONTRIBUTION

Monitors embedding distribution shift using Population Stability Index (PSI).
Triggers knowledge base refresh when PSI > alert threshold.

Thresholds (from settings):
    PSI < 0.1  -> no drift (green)
    PSI 0.1-0.25 -> moderate drift, log warning (orange)
    PSI > 0.25 -> significant drift, trigger refresh (red)
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import mlflow
import numpy as np

from src.config.settings import settings
from src.infra.baseline_store import get_baseline_store

logger = logging.getLogger("ragops.drift_detector")

# Small constant to avoid log(0) in PSI calculation
_EPSILON = 1e-6


# PAPER CONTRIBUTION
def capture_baseline(embeddings: np.ndarray) -> None:
    """Compute and store embedding distribution baseline after initial ingestion.

    Called once after initial document ingestion, and again after each
    knowledge base refresh to reset the reference distribution.

    Args:
        embeddings: np.ndarray of shape (n_docs, 384).
    """
    # PAPER CONTRIBUTION
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError(f"Expected 2-D embedding matrix, got shape {embeddings.shape}")

    store = get_baseline_store()
    store.save_psi_baseline(embeddings)
    logger.info(
        "PSI baseline captured — shape=%s, %d documents",
        embeddings.shape,
        embeddings.shape[0],
    )


# PAPER CONTRIBUTION
def _psi_per_dimension(
    baseline: np.ndarray,
    current: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """Compute PSI for each embedding dimension.

    Args:
        baseline: 2-D array of shape (n_docs, n_dims) — reference matrix.
        current:  2-D array of shape (n_docs, n_dims) — current matrix.
        num_bins: Number of histogram bins (per dimension).

    Returns:
        1-D array of length n_dims with per-dimension PSI scores.
    """
    n_dims = baseline.shape[1]
    psi_scores = np.zeros(n_dims)

    for dim in range(n_dims):
        # Determine bin edges from baseline distribution
        _, bin_edges = np.histogram(baseline[:, dim], bins=num_bins)

        # Compute histograms for baseline and current using same edges
        baseline_counts, _ = np.histogram(baseline[:, dim], bins=bin_edges)
        current_counts, _ = np.histogram(current[:, dim], bins=bin_edges)

        # Normalise to proportions, add epsilon to avoid division by zero
        expected = (baseline_counts / baseline_counts.sum()) + _EPSILON
        actual = (current_counts / current_counts.sum()) + _EPSILON

        # PSI = sum((actual - expected) * ln(actual / expected))
        psi_scores[dim] = float(np.sum((actual - expected) * np.log(actual / expected)))

    return psi_scores


# PAPER CONTRIBUTION
def compute_psi(current_embeddings: np.ndarray) -> float:
    """Compute PSI between current embeddings and stored baseline.

    Computes PSI per embedding dimension using histogram binning,
    then returns the mean across all dimensions.

    PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))

    Args:
        current_embeddings: np.ndarray of shape (n_docs, dim).

    Returns:
        Mean PSI score across all dimensions (float >= 0).

    Raises:
        RuntimeError: If no baseline has been captured yet.
    """
    # PAPER CONTRIBUTION
    store = get_baseline_store()
    baseline = store.load_psi_baseline()

    if baseline is None:
        raise RuntimeError(
            "No PSI baseline found. Call capture_baseline() first "
            "after initial document ingestion."
        )

    if current_embeddings.shape[1] != baseline.shape[1]:
        raise ValueError(
            f"Dimension mismatch: baseline has {baseline.shape[1]} dims, "
            f"current has {current_embeddings.shape[1]} dims"
        )

    psi_scores = _psi_per_dimension(
        baseline, current_embeddings, num_bins=settings.psi_num_bins
    )
    mean_psi = float(np.mean(psi_scores))

    logger.info(
        "PSI computed — mean=%.4f, max=%.4f, min=%.4f, baseline=%d docs, current=%d docs",
        mean_psi,
        float(np.max(psi_scores)),
        float(np.min(psi_scores)),
        baseline.shape[0],
        current_embeddings.shape[0],
    )
    return mean_psi


# PAPER CONTRIBUTION
def alert(psi_score: float) -> dict[str, Any]:
    """Evaluate PSI score against thresholds and generate drift report.

    Args:
        psi_score: Mean PSI from compute_psi().

    Returns:
        Drift report dict::

            {
                "status": "stable" | "warning" | "alert",
                "psi_score": float,
                "timestamp": str (ISO 8601),
                "action": "none" | "monitor" | "trigger_refresh",
                "thresholds": {"warning": float, "alert": float},
            }
    """
    # PAPER CONTRIBUTION
    if psi_score >= settings.psi_alert_threshold:
        status = "alert"
        action = "trigger_refresh"
        logger.warning(
            "PSI ALERT: %.4f >= %.2f — triggering knowledge base refresh",
            psi_score,
            settings.psi_alert_threshold,
        )
    elif psi_score >= settings.psi_warning_threshold:
        status = "warning"
        action = "monitor"
        logger.warning(
            "PSI WARNING: %.4f >= %.2f — moderate drift detected",
            psi_score,
            settings.psi_warning_threshold,
        )
    else:
        status = "stable"
        action = "none"
        logger.info(
            "PSI STABLE: %.4f < %.2f", psi_score, settings.psi_warning_threshold
        )

    return {
        "status": status,
        "psi_score": round(psi_score, 6),
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "action": action,
        "thresholds": {
            "warning": settings.psi_warning_threshold,
            "alert": settings.psi_alert_threshold,
        },
    }


# PAPER CONTRIBUTION
def monitor_generation() -> dict[str, Any]:
    """Track RAGAS faithfulness rolling 7-day average in MLflow.

    Queries MLflow for recent evaluation runs, computes a rolling
    average of the faithfulness metric, and flags generation drift
    if the drop exceeds 15% from the earliest recorded average.

    Returns:
        {
            "current_avg": float,
            "baseline_avg": float,
            "drop_pct": float,
            "generation_drift": bool,
        }
    """
    # PAPER CONTRIBUTION
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        logger.warning(
            "MLflow experiment '%s' not found.", settings.mlflow_experiment_name
        )
        return {
            "current_avg": 0.0,
            "baseline_avg": 0.0,
            "drop_pct": 0.0,
            "generation_drift": False,
        }

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.faithfulness > 0",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )

    if len(runs) < 2:
        logger.info("Not enough runs for generation drift monitoring (%d).", len(runs))
        return {
            "current_avg": runs[0].data.metrics.get("faithfulness", 0.0)
            if runs
            else 0.0,
            "baseline_avg": 0.0,
            "drop_pct": 0.0,
            "generation_drift": False,
        }

    faithfulness_scores = [
        r.data.metrics["faithfulness"] for r in runs if "faithfulness" in r.data.metrics
    ]

    # Split: recent half vs older half for comparison
    midpoint = len(faithfulness_scores) // 2
    current_avg = float(np.mean(faithfulness_scores[:midpoint]))
    baseline_avg = float(np.mean(faithfulness_scores[midpoint:]))

    if baseline_avg > 0:
        drop_pct = (baseline_avg - current_avg) / baseline_avg * 100
    else:
        drop_pct = 0.0

    generation_drift = drop_pct > 15.0

    if generation_drift:
        logger.warning(
            "GENERATION DRIFT: faithfulness dropped %.1f%% "
            "(baseline=%.3f, current=%.3f)",
            drop_pct,
            baseline_avg,
            current_avg,
        )

    return {
        "current_avg": round(current_avg, 4),
        "baseline_avg": round(baseline_avg, 4),
        "drop_pct": round(drop_pct, 2),
        "generation_drift": generation_drift,
    }
