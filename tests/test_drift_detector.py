"""Tests for mlops.drift_detector — PSI drift detection + generation monitoring."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mlops.drift_detector import (
    _psi_per_dimension,
    alert,
    capture_baseline,
    compute_psi,
    monitor_generation,
)
from src.config.settings import settings
from src.infra.baseline_store import get_baseline_store


# ---------------------------------------------------------------------------
# capture_baseline
# ---------------------------------------------------------------------------


class TestCaptureBaseline:
    def test_rejects_1d_array(self) -> None:
        with pytest.raises(ValueError, match="2-D embedding matrix"):
            capture_baseline(np.array([1.0, 2.0, 3.0]))

    def test_rejects_zero_dim_matrix(self) -> None:
        with pytest.raises(ValueError, match="2-D embedding matrix"):
            capture_baseline(np.zeros((5, 0)))

    def test_persists_through_store(self) -> None:
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((20, 8))
        capture_baseline(embeddings)

        stored = get_baseline_store().load_psi_baseline()
        assert stored is not None
        np.testing.assert_array_equal(stored, embeddings)


# ---------------------------------------------------------------------------
# compute_psi
# ---------------------------------------------------------------------------


class TestComputePsi:
    def test_raises_without_baseline(self) -> None:
        with pytest.raises(RuntimeError, match="No PSI baseline found"):
            compute_psi(np.zeros((10, 8)))

    def test_raises_on_dimension_mismatch(self) -> None:
        rng = np.random.default_rng(0)
        capture_baseline(rng.standard_normal((20, 8)))

        with pytest.raises(ValueError, match="Dimension mismatch"):
            compute_psi(rng.standard_normal((10, 16)))

    def test_identical_distributions_yield_low_psi(self) -> None:
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((200, 4))
        capture_baseline(embeddings)

        psi = compute_psi(embeddings)
        # Same distribution on both sides — PSI should be essentially zero.
        assert psi < 0.01

    def test_shifted_distribution_yields_high_psi(self) -> None:
        rng = np.random.default_rng(123)
        baseline = rng.standard_normal((300, 4))
        capture_baseline(baseline)

        # Shift each dimension by 3 std devs — produces strong drift.
        shifted = rng.standard_normal((300, 4)) + 3.0
        psi = compute_psi(shifted)
        assert psi > settings.psi_warning_threshold


class TestPsiPerDimension:
    def test_returns_one_score_per_dimension(self) -> None:
        rng = np.random.default_rng(7)
        baseline = rng.standard_normal((100, 5))
        current = rng.standard_normal((100, 5))

        scores = _psi_per_dimension(baseline, current, num_bins=10)
        assert scores.shape == (5,)
        assert np.all(scores >= 0.0)


# ---------------------------------------------------------------------------
# alert
# ---------------------------------------------------------------------------


class TestAlert:
    def test_stable_below_warning(self) -> None:
        report = alert(0.05)
        assert report["status"] == "stable"
        assert report["action"] == "none"
        assert report["psi_score"] == 0.05
        assert report["thresholds"]["warning"] == settings.psi_warning_threshold
        assert report["thresholds"]["alert"] == settings.psi_alert_threshold
        assert "timestamp" in report

    def test_warning_between_thresholds(self) -> None:
        report = alert(0.15)
        assert report["status"] == "warning"
        assert report["action"] == "monitor"

    def test_alert_above_alert_threshold(self) -> None:
        report = alert(0.40)
        assert report["status"] == "alert"
        assert report["action"] == "trigger_refresh"


# ---------------------------------------------------------------------------
# monitor_generation
# ---------------------------------------------------------------------------


def _fake_run(faithfulness: float) -> MagicMock:
    run = MagicMock()
    run.data.metrics = {"faithfulness": faithfulness}
    return run


class TestMonitorGeneration:
    def test_missing_experiment_returns_empty(
        self, mock_mlflow_client: MagicMock
    ) -> None:
        mock_mlflow_client.get_experiment_by_name.return_value = None

        result = monitor_generation()

        assert result == {
            "current_avg": 0.0,
            "baseline_avg": 0.0,
            "drop_pct": 0.0,
            "generation_drift": False,
        }

    def test_insufficient_runs_returns_early(
        self, mock_mlflow_client: MagicMock
    ) -> None:
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"
        mock_mlflow_client.get_experiment_by_name.return_value = experiment
        mock_mlflow_client.search_runs.return_value = [_fake_run(0.82)]

        result = monitor_generation()

        assert result["generation_drift"] is False
        assert result["current_avg"] == 0.82
        assert result["baseline_avg"] == 0.0

    def test_detects_drift_on_large_drop(self, mock_mlflow_client: MagicMock) -> None:
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"
        mock_mlflow_client.get_experiment_by_name.return_value = experiment

        # search_runs is ordered DESC by start_time, so the first half is "current".
        # Recent runs average ~0.40, older runs average ~0.90 — ≈56% drop.
        recent = [_fake_run(0.40) for _ in range(5)]
        older = [_fake_run(0.90) for _ in range(5)]
        mock_mlflow_client.search_runs.return_value = recent + older

        result = monitor_generation()

        assert result["generation_drift"] is True
        assert result["current_avg"] == pytest.approx(0.40)
        assert result["baseline_avg"] == pytest.approx(0.90)
        assert result["drop_pct"] > 15.0

    def test_stable_when_faithfulness_flat(self, mock_mlflow_client: MagicMock) -> None:
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"
        mock_mlflow_client.get_experiment_by_name.return_value = experiment
        mock_mlflow_client.search_runs.return_value = [
            _fake_run(0.80) for _ in range(6)
        ]

        result = monitor_generation()

        assert result["generation_drift"] is False
        assert result["drop_pct"] == pytest.approx(0.0)
