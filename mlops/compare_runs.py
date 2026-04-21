"""
MLflow Experiment Comparison.

Side-by-side comparison of two MLflow runs with RAGAS metrics
and explanation_consistency_score.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow

from src.config.settings import settings

logger = logging.getLogger("ragops.compare_runs")

# Metrics where lower values are better
_LOWER_IS_BETTER = {
    "retrieval_latency_ms",
    "llm_latency_ms",
    "total_latency_ms",
    "psi_score",
    "latency_p50",
    "latency_p95",
}


def compare_runs(run_id_a: str, run_id_b: str) -> dict[str, Any]:
    """Compare two MLflow runs side-by-side.

    Args:
        run_id_a: First MLflow run ID.
        run_id_b: Second MLflow run ID.

    Returns:
        {
            "run_a": {"run_id": str, "metrics": dict, "params": dict, "tags": dict},
            "run_b": {"run_id": str, "metrics": dict, "params": dict, "tags": dict},
            "deltas": {metric_name: float (b - a), ...},
            "improved": [metric names where run_b is better],
            "regressed": [metric names where run_b is worse],
            "unchanged": [metric names with no change],
        }
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    run_a = client.get_run(run_id_a)
    run_b = client.get_run(run_id_b)

    def _extract(run: mlflow.entities.Run) -> dict[str, Any]:
        return {
            "run_id": run.info.run_id,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }

    data_a = _extract(run_a)
    data_b = _extract(run_b)

    # Compute deltas for metrics present in both runs
    shared_metrics = set(data_a["metrics"]) & set(data_b["metrics"])
    deltas: dict[str, float] = {}
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []

    for metric in sorted(shared_metrics):
        val_a = data_a["metrics"][metric]
        val_b = data_b["metrics"][metric]
        delta = val_b - val_a
        deltas[metric] = round(delta, 6)

        if abs(delta) < 1e-6:
            unchanged.append(metric)
        elif metric in _LOWER_IS_BETTER:
            (improved if delta < 0 else regressed).append(metric)
        else:
            (improved if delta > 0 else regressed).append(metric)

    logger.info(
        "Compared runs %s vs %s — %d improved, %d regressed, %d unchanged",
        run_id_a[:8],
        run_id_b[:8],
        len(improved),
        len(regressed),
        len(unchanged),
    )

    return {
        "run_a": data_a,
        "run_b": data_b,
        "deltas": deltas,
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
    }
