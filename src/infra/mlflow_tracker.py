"""
RAGOps — MLflow Experiment Tracker
====================================
Central interface for logging all RAGOps eval-cycle data to MLflow.

One MLflow run is created per full eval cycle and captures:
  - RAGAS quality metrics        (from Member 3)
  - Retrieval latency            (from Member 2)
  - PSI embedding drift score    (PAPER CONTRIBUTION — from Component 4)
  - XAI consistency score        (XAI CONTRIBUTION  — from Component 6)
  - Run parameters and tags

Design principles
-----------------
* ``RAGOpsTracker`` is a context manager — wrap each eval cycle in a
  ``with tracker.start_run(...):`` block; the run is always cleanly
  ended even if an exception is raised mid-cycle.
* All metric names are string constants on ``MetricNames`` so callers
  never type raw strings.
* ``log_quality_gate_results`` returns a ``QualityGateResult`` dataclass
  that the CI step can inspect to decide pass/fail without re-querying
  MLflow.

Usage (typical eval cycle)::

    from src.infra.mlflow_tracker import RAGOpsTracker, RunTrigger

    tracker = RAGOpsTracker()

    with tracker.start_run(triggered_by=RunTrigger.CI):
        tracker.log_params({...})
        tracker.log_ragas_metrics(ragas_output)
        tracker.log_retrieval_latency(latency_ms)
        tracker.log_psi_score(psi_score)           # PAPER CONTRIBUTION
        tracker.log_xai_consistency(xai_score)     # XAI CONTRIBUTION
        gate = tracker.log_quality_gate_results()

    if not gate.passed:
        sys.exit(1)
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator

import mlflow
from mlflow.entities import Run
from mlflow.exceptions import MlflowException

from src.config.settings import settings

logger = logging.getLogger("ragops.infra.mlflow_tracker")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class RunTrigger(str, Enum):
    """Controlled vocabulary for the ``triggered_by`` run tag."""
    CI = "ci"
    MANUAL = "manual"
    DRIFT = "drift"       # PAPER CONTRIBUTION — auto-triggered by PSI alert
    SCHEDULED = "scheduled"


class PSIStatus(str, Enum):
    """PSI drift severity levels.  # PAPER CONTRIBUTION"""
    STABLE = "stable"
    WARNING = "warning"
    ALERT = "alert"


class XAIStatus(str, Enum):
    """XAI explanation consistency levels.  # XAI CONTRIBUTION"""
    STABLE = "stable"
    WARNING = "warning"
    INSTABILITY = "instability"


class MetricNames:
    """
    Canonical MLflow metric key names.

    Using a constants class (rather than inline strings) ensures
    consistency across tracker, CI quality-gate checks, and dashboards.
    """
    # RAGAS metrics
    FAITHFULNESS = "faithfulness"
    CONTEXT_RECALL = "context_recall"
    ANSWER_RELEVANCY = "answer_relevancy"

    # Retrieval
    RETRIEVAL_LATENCY_MS = "retrieval_latency_ms"

    # PSI drift  (PAPER CONTRIBUTION)
    PSI_SCORE = "psi_score"

    # XAI consistency  (XAI CONTRIBUTION)
    EXPLANATION_CONSISTENCY_SCORE = "explanation_consistency_score"

    # Quality gate summary
    QUALITY_GATE_PASSED = "quality_gate_passed"   # logged as 1.0 / 0.0


class TagNames:
    """Canonical MLflow tag key names."""
    TRIGGERED_BY = "triggered_by"
    CHROMA_COLLECTION = "chroma_collection"
    OLLAMA_MODEL = "ollama_model"
    ENVIRONMENT = "environment"
    PSI_STATUS = "psi_status"           # PAPER CONTRIBUTION
    XAI_STATUS = "xai_status"           # XAI CONTRIBUTION


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RAGASMetrics:
    """
    Output shape returned by ``evaluation.ragas_runner.run_eval()``.
    Mirrors Member 3's interface contract.
    """
    faithfulness: float
    context_recall: float
    answer_relevancy: float
    # Additional RAGAS metrics are accepted as kwargs and stored in ``extra``
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class QualityGateResult:
    """
    Outcome of the RAGAS quality gate check for a single eval run.

    Returned by ``log_quality_gate_results()`` so the CI step can
    determine pass/fail without re-querying MLflow.
    """
    passed: bool
    failures: dict[str, str]   # metric_name -> "actual=X < threshold=Y"
    run_id: str


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class RAGOpsTracker:
    """
    MLflow experiment tracker for the RAGOps eval cycle.

    One instance per application lifetime; one MLflow run per eval cycle.
    Thread-safety: MLflow's Python client is not thread-safe — use one
    tracker instance per thread/process.

    Parameters
    ----------
    tracking_uri : str | None
        Override the MLflow tracking URI. Defaults to
        ``settings.mlflow_tracking_uri``.
    experiment_name : str | None
        Override the experiment name. Defaults to
        ``settings.mlflow_experiment_name``.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        self._tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self._experiment_name = experiment_name or settings.mlflow_experiment_name
        self._active_run: Run | None = None

        mlflow.set_tracking_uri(self._tracking_uri)
        self._experiment_id = self._get_or_create_experiment()
        logger.info(
            "RAGOpsTracker initialised — uri=%s experiment=%s (id=%s)",
            self._tracking_uri,
            self._experiment_name,
            self._experiment_id,
        )

    # ── Experiment setup ─────────────────────────────────────────────────────

    def _get_or_create_experiment(self) -> str:
        """
        Return the experiment ID, creating the experiment if it does not exist.

        MLflow raises ``MlflowException`` if you attempt to create a duplicate;
        this method handles that race-condition gracefully.

        Returns
        -------
        str
            MLflow experiment ID.
        """
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is not None:
            logger.debug("Reusing existing experiment '%s'", self._experiment_name)
            return experiment.experiment_id

        try:
            experiment_id = mlflow.create_experiment(
                name=self._experiment_name,
                artifact_location=None,   # use server default (/mlflow/artifacts)
            )
            logger.info("Created MLflow experiment '%s' (id=%s)", self._experiment_name, experiment_id)
            return experiment_id
        except MlflowException as exc:
            # Race condition: another process created it between get and create
            logger.warning("Experiment creation race — falling back to get: %s", exc)
            experiment = mlflow.get_experiment_by_name(self._experiment_name)
            if experiment is None:
                raise RuntimeError(
                    f"Cannot resolve MLflow experiment '{self._experiment_name}'"
                ) from exc
            return experiment.experiment_id

    # ── Run lifecycle ─────────────────────────────────────────────────────────

    @contextmanager
    def start_run(
        self,
        triggered_by: RunTrigger = RunTrigger.MANUAL,
        run_name: str | None = None,
    ) -> Generator["RAGOpsTracker", None, None]:
        """
        Context manager that wraps a full eval cycle in a single MLflow run.

        Always logs the standard set of tags (chroma_collection,
        triggered_by, ollama_model, environment) on entry.  The run is
        ended — even if an exception propagates — so MLflow never has a
        dangling RUNNING run.

        Parameters
        ----------
        triggered_by : RunTrigger
            What initiated this eval cycle.
        run_name : str | None
            Optional human-readable run name shown in the MLflow UI.

        Yields
        ------
        RAGOpsTracker
            Self, so callers can chain: ``with tracker.start_run() as t:``.

        Example
        -------
        ::

            with tracker.start_run(triggered_by=RunTrigger.CI) as t:
                t.log_ragas_metrics(metrics)
        """
        run = mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
        )
        self._active_run = run
        logger.info("MLflow run started — run_id=%s triggered_by=%s", run.info.run_id, triggered_by.value)

        # Standard tags on every run
        mlflow.set_tags({
            TagNames.TRIGGERED_BY: triggered_by.value,
            TagNames.CHROMA_COLLECTION: settings.chroma_collection_name,
            TagNames.OLLAMA_MODEL: settings.ollama_model,
            TagNames.ENVIRONMENT: settings.environment,
        })

        try:
            yield self
        except Exception:
            logger.exception("Exception inside MLflow run %s — run will be ended with FAILED status", run.info.run_id)
            mlflow.end_run(status="FAILED")
            self._active_run = None
            raise
        else:
            mlflow.end_run(status="FINISHED")
            logger.info("MLflow run finished — run_id=%s", run.info.run_id)
            self._active_run = None

    @property
    def active_run_id(self) -> str | None:
        """Return the active MLflow run ID, or None if no run is open."""
        return self._active_run.info.run_id if self._active_run else None

    def _assert_run_active(self) -> None:
        """Raise RuntimeError if called outside a ``start_run`` context."""
        if self._active_run is None:
            raise RuntimeError(
                "No active MLflow run. Wrap calls inside 'with tracker.start_run():'."
            )

    # ── Logging methods ───────────────────────────────────────────────────────

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log arbitrary run parameters to MLflow.

        Parameters
        ----------
        params : dict[str, Any]
            Key-value pairs. Values are coerced to strings by MLflow.
            Typical keys: ``psi_num_bins``, ``xai_benchmark_questions``,
            ``embed_model``.
        """
        self._assert_run_active()
        mlflow.log_params(params)
        logger.debug("Logged params: %s", list(params.keys()))

    def log_ragas_metrics(self, metrics: RAGASMetrics) -> None:
        """
        Log RAGAS evaluation output to the active MLflow run.

        Parameters
        ----------
        metrics : RAGASMetrics
            Output from ``evaluation.ragas_runner.run_eval(qa_pairs)``.
            Core fields (faithfulness, context_recall, answer_relevancy)
            are always logged; any extra fields in ``metrics.extra`` are
            logged under their original keys.
        """
        self._assert_run_active()
        core = {
            MetricNames.FAITHFULNESS: metrics.faithfulness,
            MetricNames.CONTEXT_RECALL: metrics.context_recall,
            MetricNames.ANSWER_RELEVANCY: metrics.answer_relevancy,
        }
        mlflow.log_metrics(core)
        if metrics.extra:
            mlflow.log_metrics(metrics.extra)
        logger.info(
            "RAGAS metrics logged — faithfulness=%.3f context_recall=%.3f answer_relevancy=%.3f",
            metrics.faithfulness,
            metrics.context_recall,
            metrics.answer_relevancy,
        )

    def log_retrieval_latency(self, latency_ms: float) -> None:
        """
        Log retrieval latency from Member 2's chain output.

        Parameters
        ----------
        latency_ms : float
            Value of ``retrieval_latency_ms`` from
            ``rag_pipeline.chain.query(question)``.
        """
        self._assert_run_active()
        mlflow.log_metric(MetricNames.RETRIEVAL_LATENCY_MS, latency_ms)
        logger.info("Retrieval latency logged — %.1f ms", latency_ms)

    def log_psi_score(self, psi_score: float) -> None:
        """
        Log PSI embedding drift score and derive its status tag.

        PSI threshold logic (PAPER CONTRIBUTION):
          - ``< PSI_WARNING_THRESHOLD``  → stable
          - ``[WARNING, ALERT)``         → warning
          - ``>= PSI_ALERT_THRESHOLD``   → alert  (triggers KB refresh)

        Parameters
        ----------
        psi_score : float
            Population Stability Index computed by Component 4.
            # PAPER CONTRIBUTION
        """
        self._assert_run_active()
        mlflow.log_metric(MetricNames.PSI_SCORE, psi_score)

        if psi_score >= settings.psi_alert_threshold:
            status = PSIStatus.ALERT
            logger.warning(
                "PSI ALERT — score=%.4f >= alert_threshold=%.2f — KB refresh will be triggered",  # PAPER CONTRIBUTION
                psi_score, settings.psi_alert_threshold,
            )
        elif psi_score >= settings.psi_warning_threshold:
            status = PSIStatus.WARNING
            logger.warning(
                "PSI WARNING — score=%.4f >= warning_threshold=%.2f",  # PAPER CONTRIBUTION
                psi_score, settings.psi_warning_threshold,
            )
        else:
            status = PSIStatus.STABLE
            logger.info("PSI stable — score=%.4f", psi_score)  # PAPER CONTRIBUTION

        mlflow.set_tag(TagNames.PSI_STATUS, status.value)

    def log_xai_consistency(self, consistency_score: float) -> None:
        """
        Log XAI explanation consistency score and derive its status tag.

        Cosine-similarity-based threshold logic (XAI CONTRIBUTION):
          - ``>= XAI_WARNING_THRESHOLD``            → stable
          - ``[XAI_ALERT, XAI_WARNING)``            → warning
          - ``< XAI_ALERT_THRESHOLD``               → instability alert

        Parameters
        ----------
        consistency_score : float
            Mean cosine similarity between current and baseline SHAP
            explanation vectors, computed by Component 6.
            # XAI CONTRIBUTION
        """
        self._assert_run_active()
        mlflow.log_metric(MetricNames.EXPLANATION_CONSISTENCY_SCORE, consistency_score)

        if consistency_score < settings.xai_alert_threshold:
            status = XAIStatus.INSTABILITY
            logger.warning(
                "XAI INSTABILITY — score=%.4f < alert_threshold=%.2f",  # XAI CONTRIBUTION
                consistency_score, settings.xai_alert_threshold,
            )
        elif consistency_score < settings.xai_warning_threshold:
            status = XAIStatus.WARNING
            logger.warning(
                "XAI WARNING — score=%.4f < warning_threshold=%.2f",  # XAI CONTRIBUTION
                consistency_score, settings.xai_warning_threshold,
            )
        else:
            status = XAIStatus.STABLE
            logger.info("XAI consistent — score=%.4f", consistency_score)  # XAI CONTRIBUTION

        mlflow.set_tag(TagNames.XAI_STATUS, status.value)

    def log_quality_gate_results(self) -> QualityGateResult:
        """
        Evaluate RAGAS metrics against configured minimum thresholds and
        log a binary ``quality_gate_passed`` metric to MLflow.

        Reads the metrics already logged in the active run via the MLflow
        client (avoids re-passing values through the call stack).

        Returns
        -------
        QualityGateResult
            Contains ``passed`` bool and a ``failures`` dict describing
            any metrics that fell below their threshold.

        Raises
        ------
        RuntimeError
            If RAGAS metrics have not been logged before calling this method.
        """
        self._assert_run_active()
        run_id = self.active_run_id

        client = mlflow.MlflowClient()
        run_data = client.get_run(run_id).data.metrics

        thresholds: dict[str, float] = {
            MetricNames.FAITHFULNESS: settings.ragas_faithfulness_min,
            MetricNames.CONTEXT_RECALL: settings.ragas_context_recall_min,
            MetricNames.ANSWER_RELEVANCY: settings.ragas_answer_relevancy_min,
        }

        failures: dict[str, str] = {}
        for metric, threshold in thresholds.items():
            actual = run_data.get(metric)
            if actual is None:
                raise RuntimeError(
                    f"Quality gate check: metric '{metric}' not found in run {run_id}. "
                    "Ensure log_ragas_metrics() is called before log_quality_gate_results()."
                )
            if actual < threshold:
                failures[metric] = f"actual={actual:.4f} < threshold={threshold:.4f}"
                logger.warning("Quality gate FAIL — %s: %s", metric, failures[metric])

        passed = len(failures) == 0
        mlflow.log_metric(MetricNames.QUALITY_GATE_PASSED, 1.0 if passed else 0.0)

        if passed:
            logger.info("Quality gate PASSED — all RAGAS metrics above thresholds")
        else:
            logger.error("Quality gate FAILED — %d metric(s) below threshold: %s", len(failures), list(failures.keys()))

        return QualityGateResult(passed=passed, failures=failures, run_id=run_id)

    def log_kb_refresh_triggered(self, reason: str = "psi_alert") -> None:
        """
        Record that an automated knowledge base refresh was triggered.

        Called by Component 5 (KB refresh trigger) after it detects a
        PSI alert and initiates re-ingestion.

        Parameters
        ----------
        reason : str
            Human-readable reason string logged as an MLflow tag.
            # PAPER CONTRIBUTION
        """
        self._assert_run_active()
        mlflow.set_tag("kb_refresh_triggered", "true")
        mlflow.set_tag("kb_refresh_reason", reason)
        logger.info("KB refresh trigger logged — reason=%s", reason)  # PAPER CONTRIBUTION
