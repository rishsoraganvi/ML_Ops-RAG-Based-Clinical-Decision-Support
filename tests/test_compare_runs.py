"""Tests for mlops.compare_runs — MLflow run side-by-side comparison."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mlops.compare_runs import compare_runs


def _make_run(
    run_id: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a fake mlflow.entities.Run for MlflowClient.get_run to return."""
    run = MagicMock(name=f"run_{run_id}")
    run.info.run_id = run_id
    run.data.metrics = metrics
    run.data.params = params or {}
    run.data.tags = tags or {}
    return run


class TestCompareRuns:
    def test_all_metrics_improved_higher_is_better(
        self, mock_mlflow_client: MagicMock
    ) -> None:
        mock_mlflow_client.get_run.side_effect = [
            _make_run("a" * 8, {"faithfulness": 0.70, "context_recall": 0.60}),
            _make_run("b" * 8, {"faithfulness": 0.85, "context_recall": 0.80}),
        ]

        result = compare_runs("a" * 8, "b" * 8)

        assert set(result["improved"]) == {"faithfulness", "context_recall"}
        assert result["regressed"] == []
        assert result["unchanged"] == []
        assert result["deltas"]["faithfulness"] == pytest.approx(0.15)
        assert result["run_a"]["run_id"] == "a" * 8
        assert result["run_b"]["run_id"] == "b" * 8

    def test_latency_lower_is_better(self, mock_mlflow_client: MagicMock) -> None:
        mock_mlflow_client.get_run.side_effect = [
            _make_run("run1", {"total_latency_ms": 1200.0, "psi_score": 0.15}),
            _make_run("run2", {"total_latency_ms": 900.0, "psi_score": 0.05}),
        ]

        result = compare_runs("run1", "run2")

        # Both metrics decreased, and both are in _LOWER_IS_BETTER.
        assert set(result["improved"]) == {"total_latency_ms", "psi_score"}
        assert result["regressed"] == []

    def test_identical_metrics_all_unchanged(
        self, mock_mlflow_client: MagicMock
    ) -> None:
        metrics = {"faithfulness": 0.80, "context_recall": 0.75}
        mock_mlflow_client.get_run.side_effect = [
            _make_run("x", metrics),
            _make_run("y", metrics),
        ]

        result = compare_runs("x", "y")

        assert set(result["unchanged"]) == {"faithfulness", "context_recall"}
        assert result["improved"] == []
        assert result["regressed"] == []
        assert all(abs(d) < 1e-6 for d in result["deltas"].values())

    def test_mixed_improved_and_regressed_plus_shape(
        self, mock_mlflow_client: MagicMock
    ) -> None:
        mock_mlflow_client.get_run.side_effect = [
            _make_run(
                "a",
                metrics={"faithfulness": 0.80, "total_latency_ms": 900.0},
                params={"retriever": "hybrid"},
                tags={"env": "test"},
            ),
            _make_run(
                "b",
                metrics={"faithfulness": 0.70, "total_latency_ms": 800.0},
                params={"retriever": "dense"},
                tags={"env": "test"},
            ),
        ]

        result = compare_runs("a", "b")

        assert "faithfulness" in result["regressed"]
        assert "total_latency_ms" in result["improved"]
        # Deltas are rounded to 6 decimals.
        assert result["deltas"]["faithfulness"] == pytest.approx(-0.10)
        # run_a / run_b carry metrics, params, tags dicts.
        assert result["run_a"]["params"] == {"retriever": "hybrid"}
        assert result["run_b"]["tags"] == {"env": "test"}
