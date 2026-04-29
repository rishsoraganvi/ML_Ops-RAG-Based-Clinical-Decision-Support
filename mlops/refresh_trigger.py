"""
Auto Knowledge Base Refresh Trigger.
# PAPER CONTRIBUTION

On drift alert: pull latest PubMed batch -> re-run ETL -> re-index ChromaDB
-> re-run RAGAS eval -> post diff report.

Triggered by drift_detector.alert() when PSI > 0.25 or via GitHub Actions
repository dispatch.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("ragops.refresh_trigger")


# PAPER CONTRIBUTION
def trigger_refresh(drift_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Trigger a full knowledge base refresh pipeline.

    Pipeline steps:
        1. Run RAGAS CI eval to capture "before" metrics.
        2. Fetch latest PubMed articles and upsert into ChromaDB.
        3. Re-capture PSI baseline with new embeddings.
        4. Run RAGAS CI eval again for "after" metrics.
        5. Log the refresh event to MLflow via RAGOpsTracker.

    Args:
        drift_report: Optional drift report from drift_detector.alert().

    Returns:
        {
            "status": "completed" | "failed",
            "docs_added": int,
            "before_metrics": dict | None,
            "after_metrics": dict | None,
            "drift_report": dict | None,
        }
    """
    # PAPER CONTRIBUTION
    from data.ingest import get_embeddings, incremental_upsert
    from evaluation.ragas_runner import run_ci_eval
    from mlops.drift_detector import capture_baseline
    from mlops.mlflow_tracker import RAGOpsTracker, RunTrigger

    result: dict[str, Any] = {
        "status": "failed",
        "docs_added": 0,
        "before_metrics": None,
        "after_metrics": None,
        "drift_report": drift_report,
    }

    tracker = RAGOpsTracker()
    try:
        with tracker.start_run(
            triggered_by=RunTrigger.DRIFT,
            run_name="kb-refresh",
        ):
            # Step 1: "before" metrics
            logger.info("KB refresh: capturing before-metrics via CI eval...")
            try:
                result["before_metrics"] = run_ci_eval()
            except Exception as exc:
                logger.warning("Before-metrics eval failed (continuing): %s", exc)

            # Step 2: Fetch + upsert new documents
            logger.info("KB refresh: fetching and upserting new documents...")
            try:
                from data.refresh import fetch_new_records

                new_records = fetch_new_records()
                docs_added = incremental_upsert(new_records)
                result["docs_added"] = docs_added
                logger.info("KB refresh: upserted %d new chunks", docs_added)
            except ImportError:
                logger.warning(
                    "data.refresh.fetch_new_records not available. "
                    "Skipping data fetch — only re-baselining."
                )
            except Exception as exc:
                logger.error("KB refresh: upsert failed: %s", exc)
                raise

            # Step 3: Re-capture PSI baseline
            logger.info("KB refresh: re-capturing PSI baseline...")
            try:
                from src.config.settings import settings

                embeddings = get_embeddings(
                    chunk_size=settings.baseline_chunk_size
                )
                capture_baseline(embeddings)
            except Exception as exc:
                logger.warning(
                    "KB refresh: baseline re-capture failed (continuing): %s",
                    exc,
                )

            # Step 4: "after" metrics
            logger.info("KB refresh: capturing after-metrics via CI eval...")
            try:
                result["after_metrics"] = run_ci_eval()
            except Exception as exc:
                logger.warning("After-metrics eval failed (continuing): %s", exc)

            # Step 5: Log refresh event
            reason = "psi_alert"
            if drift_report:
                reason = f"psi_{drift_report.get('status', 'alert')}"
                tracker.log_params({"psi_score": drift_report.get("psi_score", 0.0)})
            tracker.log_kb_refresh_triggered(reason=reason)

            result["status"] = "completed"
            logger.info("KB refresh completed successfully.")

    except Exception as exc:
        logger.error("KB refresh pipeline failed: %s", exc)
        result["status"] = "failed"

    return result
