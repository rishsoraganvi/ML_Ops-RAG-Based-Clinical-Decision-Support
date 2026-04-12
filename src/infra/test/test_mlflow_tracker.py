"""
RAGOps — MLflow Tracker Unit Tests
=====================================
Tests run against a local SQLite MLflow instance spun up in a tmp dir —
no Docker required. Each test gets a fresh experiment to avoid
cross-test state pollution.

Run::

    pytest src/infra/tests/test_mlflow_tracker.py -v
"""

import pytest
import mlflow
import numpy as np

from mlops.mlflow_tracker import (
    RAGOpsTracker,
    RAGASMetrics,
    MetricNames,
    TagNames,
    RunTrigger,
    PSIStatus,
    XAIStatus,
)
from src.infra.baseline_store import InMemoryBaselineStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path):
    """
    Point MLflow at a fresh SQLite DB in a temp dir for each test.
    Restores the original tracking URI on teardown.
    """
    original_uri = mlflow.get_tracking_uri()
    db_path = tmp_path / "mlflow_test.db"
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    yield
    mlflow.set_tracking_uri(original_uri)


@pytest.fixture()
def tracker(isolated_mlflow):
    """Return a fresh RAGOpsTracker pointed at the test MLflow instance."""
    return RAGOpsTracker(
        tracking_uri=mlflow.get_tracking_uri(),
        experiment_name="test_ragops",
    )


@pytest.fixture()
def good_ragas_metrics():
    """RAGAS metrics that pass all quality gates."""
    return RAGASMetrics(
        faithfulness=0.90,
        context_recall=0.85,
        answer_relevancy=0.82,
    )


@pytest.fixture()
def failing_ragas_metrics():
    """RAGAS metrics that fail faithfulness and context_recall gates."""
    return RAGASMetrics(
        faithfulness=0.70,  # below 0.80 threshold
        context_recall=0.60,  # below 0.75 threshold
        answer_relevancy=0.80,
    )


# ---------------------------------------------------------------------------
# Experiment creation
# ---------------------------------------------------------------------------


class TestExperimentSetup:
    def test_creates_experiment_on_first_init(self, tracker):
        """Tracker should create the experiment if it does not exist."""
        exp = mlflow.get_experiment_by_name("test_ragops")
        assert exp is not None

    def test_reuses_existing_experiment(self, isolated_mlflow):
        """Second tracker init with same name should not raise."""
        t1 = RAGOpsTracker(
            tracking_uri=mlflow.get_tracking_uri(), experiment_name="shared_exp"
        )
        t2 = RAGOpsTracker(
            tracking_uri=mlflow.get_tracking_uri(), experiment_name="shared_exp"
        )
        assert t1._experiment_id == t2._experiment_id


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_run_starts_and_finishes(self, tracker):
        """Context manager should open and close a run cleanly."""
        with tracker.start_run(triggered_by=RunTrigger.CI):
            run_id = tracker.active_run_id
            assert run_id is not None

        assert tracker.active_run_id is None
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"

    def test_run_marked_failed_on_exception(self, tracker):
        """An exception inside start_run should mark the run FAILED."""
        run_id = None
        with pytest.raises(ValueError):
            with tracker.start_run():
                run_id = tracker.active_run_id
                raise ValueError("simulated failure")

        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        assert run.info.status == "FAILED"

    def test_standard_tags_logged(self, tracker):
        """Every run must carry the four mandatory tags."""
        with tracker.start_run(triggered_by=RunTrigger.DRIFT):
            run_id = tracker.active_run_id

        client = mlflow.MlflowClient()
        tags = client.get_run(run_id).data.tags
        assert tags[TagNames.TRIGGERED_BY] == RunTrigger.DRIFT.value
        assert TagNames.CHROMA_COLLECTION in tags
        assert TagNames.OLLAMA_MODEL in tags
        assert TagNames.ENVIRONMENT in tags

    def test_assert_run_active_raises_outside_context(self, tracker):
        """Calling log methods outside start_run should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="No active MLflow run"):
            tracker.log_retrieval_latency(100.0)


# ---------------------------------------------------------------------------
# RAGAS metric logging
# ---------------------------------------------------------------------------


class TestRAGASLogging:
    def test_core_metrics_logged(self, tracker, good_ragas_metrics):
        """Core RAGAS fields should appear in the run's metrics dict."""
        with tracker.start_run():
            tracker.log_ragas_metrics(good_ragas_metrics)
            run_id = tracker.active_run_id

        client = mlflow.MlflowClient()
        metrics = client.get_run(run_id).data.metrics
        assert metrics[MetricNames.FAITHFULNESS] == pytest.approx(0.90)
        assert metrics[MetricNames.CONTEXT_RECALL] == pytest.approx(0.85)
        assert metrics[MetricNames.ANSWER_RELEVANCY] == pytest.approx(0.82)

    def test_extra_metrics_logged(self, tracker):
        """Extra RAGAS fields passed in ``extra`` should also be logged."""
        metrics = RAGASMetrics(
            faithfulness=0.88,
            context_recall=0.80,
            answer_relevancy=0.79,
            extra={"context_precision": 0.77},
        )
        with tracker.start_run():
            tracker.log_ragas_metrics(metrics)
            run_id = tracker.active_run_id

        client = mlflow.MlflowClient()
        logged = client.get_run(run_id).data.metrics
        assert "context_precision" in logged


# ---------------------------------------------------------------------------
# PSI drift logging  (PAPER CONTRIBUTION)
# ---------------------------------------------------------------------------


class TestPSILogging:
    def test_stable_psi_tag(self, tracker):
        """PSI below warning threshold should get 'stable' tag."""
        with tracker.start_run():
            tracker.log_psi_score(0.05)
            run_id = tracker.active_run_id

        tags = mlflow.MlflowClient().get_run(run_id).data.tags
        assert tags[TagNames.PSI_STATUS] == PSIStatus.STABLE.value

    def test_warning_psi_tag(self, tracker):
        """PSI in [0.1, 0.25) should get 'warning' tag."""
        with tracker.start_run():
            tracker.log_psi_score(0.15)
            run_id = tracker.active_run_id

        tags = mlflow.MlflowClient().get_run(run_id).data.tags
        assert tags[TagNames.PSI_STATUS] == PSIStatus.WARNING.value

    def test_alert_psi_tag(self, tracker):
        """PSI >= 0.25 should get 'alert' tag."""
        with tracker.start_run():
            tracker.log_psi_score(0.30)
            run_id = tracker.active_run_id

        tags = mlflow.MlflowClient().get_run(run_id).data.tags
        assert tags[TagNames.PSI_STATUS] == PSIStatus.ALERT.value

    def test_psi_score_value_logged(self, tracker):
        """PSI numeric value must be stored in metrics."""
        with tracker.start_run():
            tracker.log_psi_score(0.18)
            run_id = tracker.active_run_id

        metrics = mlflow.MlflowClient().get_run(run_id).data.metrics
        assert metrics[MetricNames.PSI_SCORE] == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# XAI consistency logging  (XAI CONTRIBUTION)
# ---------------------------------------------------------------------------


class TestXAILogging:
    def test_stable_xai_tag(self, tracker):
        """Score >= 0.75 should get 'stable' tag."""
        with tracker.start_run():
            tracker.log_xai_consistency(0.90)
            run_id = tracker.active_run_id

        tags = mlflow.MlflowClient().get_run(run_id).data.tags
        assert tags[TagNames.XAI_STATUS] == XAIStatus.STABLE.value

    def test_warning_xai_tag(self, tracker):
        """Score in [0.60, 0.75) should get 'warning' tag."""
        with tracker.start_run():
            tracker.log_xai_consistency(0.68)
            run_id = tracker.active_run_id

        tags = mlflow.MlflowClient().get_run(run_id).data.tags
        assert tags[TagNames.XAI_STATUS] == XAIStatus.WARNING.value

    def test_instability_xai_tag(self, tracker):
        """Score < 0.60 should get 'instability' tag."""
        with tracker.start_run():
            tracker.log_xai_consistency(0.45)
            run_id = tracker.active_run_id

        tags = mlflow.MlflowClient().get_run(run_id).data.tags
        assert tags[TagNames.XAI_STATUS] == XAIStatus.INSTABILITY.value


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


class TestQualityGate:
    def test_gate_passes_with_good_metrics(self, tracker, good_ragas_metrics):
        """All metrics above threshold → gate passes, metric logged as 1.0."""
        with tracker.start_run():
            tracker.log_ragas_metrics(good_ragas_metrics)
            result = tracker.log_quality_gate_results()
            run_id = tracker.active_run_id

        assert result.passed is True
        assert result.failures == {}
        metrics = mlflow.MlflowClient().get_run(run_id).data.metrics
        assert metrics[MetricNames.QUALITY_GATE_PASSED] == pytest.approx(1.0)

    def test_gate_fails_with_bad_metrics(self, tracker, failing_ragas_metrics):
        """Metrics below threshold → gate fails, failures dict populated."""
        with tracker.start_run():
            tracker.log_ragas_metrics(failing_ragas_metrics)
            result = tracker.log_quality_gate_results()

        assert result.passed is False
        assert MetricNames.FAITHFULNESS in result.failures
        assert MetricNames.CONTEXT_RECALL in result.failures
        assert MetricNames.ANSWER_RELEVANCY not in result.failures

    def test_gate_raises_if_ragas_not_logged(self, tracker):
        """Calling gate before RAGAS metrics are logged should raise."""
        with tracker.start_run():
            with pytest.raises(RuntimeError, match="not found in run"):
                tracker.log_quality_gate_results()


# ---------------------------------------------------------------------------
# Baseline store
# ---------------------------------------------------------------------------


class TestBaselineStore:
    def test_psi_baseline_roundtrip(self):
        """Save and load PSI baseline should return identical array."""
        store = InMemoryBaselineStore()
        embeddings = np.random.rand(50, 384).astype(np.float32)
        assert store.has_psi_baseline() is False
        store.save_psi_baseline(embeddings)
        assert store.has_psi_baseline() is True
        loaded = store.load_psi_baseline()
        np.testing.assert_array_equal(embeddings, loaded)

    def test_xai_baseline_roundtrip(self):
        """Save and load XAI baseline should return identical vectors."""
        store = InMemoryBaselineStore()
        vectors = [np.random.rand(384).astype(np.float32) for _ in range(10)]
        assert store.has_xai_baseline() is False
        store.save_xai_baseline(vectors)
        assert store.has_xai_baseline() is True
        loaded = store.load_xai_baseline()
        for orig, reloaded in zip(vectors, loaded):
            np.testing.assert_array_equal(orig, reloaded)

    def test_psi_baseline_is_copy(self):
        """Mutating original array after save should not affect stored baseline."""
        store = InMemoryBaselineStore()
        embeddings = np.ones((10, 384), dtype=np.float32)
        store.save_psi_baseline(embeddings)
        embeddings[:] = 0.0  # mutate original
        loaded = store.load_psi_baseline()
        assert loaded[0, 0] == pytest.approx(1.0)
