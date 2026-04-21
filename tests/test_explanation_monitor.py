"""Tests for mlops.explanation_monitor — XAI consistency monitoring."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlops.explanation_monitor import (
    _cosine_similarity,
    compute_consistency,
    save_baseline,
)
from src.infra.baseline_store import get_baseline_store


class TestCosineSimilarity:
    def test_parallel_vectors_returns_one(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        assert _cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_norm_returns_zero(self) -> None:
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(a, b) == 0.0
        assert _cosine_similarity(b, a) == 0.0


class TestSaveBaseline:
    def test_persists_vectors_through_store(self) -> None:
        vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        save_baseline(vectors)

        loaded = get_baseline_store().load_xai_baseline()
        assert loaded is not None
        assert len(loaded) == 2
        np.testing.assert_array_equal(loaded[0], vectors[0])
        np.testing.assert_array_equal(loaded[1], vectors[1])


class TestComputeConsistency:
    def test_raises_on_empty_current(self) -> None:
        with pytest.raises(ValueError, match="must be provided and non-empty"):
            compute_consistency(current_vectors=[], baseline_vectors=[np.array([1.0])])

    def test_raises_on_none_current(self) -> None:
        with pytest.raises(ValueError, match="must be provided and non-empty"):
            compute_consistency(current_vectors=None)

    def test_raises_when_no_baseline_exists(self) -> None:
        # Autouse fixture clears _store; load_xai_baseline will return None.
        with pytest.raises(RuntimeError, match="No XAI baseline found"):
            compute_consistency(current_vectors=[np.array([1.0, 0.0])])

    def test_stable_score_for_identical_vectors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        vectors = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        with caplog.at_level(logging.INFO, logger="ragops.explanation_monitor"):
            score = compute_consistency(current_vectors=vectors, baseline_vectors=vectors)
        assert score == pytest.approx(1.0)
        assert any("STABLE" in msg for msg in caplog.messages)

    def test_warning_score_in_mid_band(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Build a pair whose cosine similarity lands in [0.60, 0.75).
        baseline = [np.array([1.0, 0.0])]
        current = [np.array([0.7, 0.7])]  # cos ≈ 0.707
        with caplog.at_level(logging.WARNING, logger="ragops.explanation_monitor"):
            score = compute_consistency(
                current_vectors=current, baseline_vectors=baseline
            )
        assert 0.60 <= score < 0.75
        assert any("WARNING" in msg for msg in caplog.messages)

    def test_alert_score_for_near_orthogonal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        baseline = [np.array([1.0, 0.0])]
        current = [np.array([0.0, 1.0])]
        with caplog.at_level(logging.WARNING, logger="ragops.explanation_monitor"):
            score = compute_consistency(
                current_vectors=current, baseline_vectors=baseline
            )
        assert score < 0.60
        assert any("INSTABILITY ALERT" in msg for msg in caplog.messages)

    def test_length_mismatch_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        baseline = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        current = [np.array([1.0, 0.0])]  # one fewer vector
        with caplog.at_level(logging.WARNING, logger="ragops.explanation_monitor"):
            score = compute_consistency(
                current_vectors=current, baseline_vectors=baseline
            )
        assert score == pytest.approx(1.0)
        assert any("Vector count mismatch" in msg for msg in caplog.messages)

    def test_loads_baseline_from_store_when_not_provided(self) -> None:
        saved = [np.array([1.0, 0.0])]
        save_baseline(saved)

        current = [np.array([1.0, 0.0])]
        score = compute_consistency(current_vectors=current)
        assert score == pytest.approx(1.0)
