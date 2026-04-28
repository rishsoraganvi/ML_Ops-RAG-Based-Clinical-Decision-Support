"""
RAGOps — Centralised Settings
==============================
Single source of truth for every configurable value used across
the infra layer. All values are resolved from environment variables
(populated by Docker Compose from .env) with safe defaults.

Import pattern (use everywhere, never re-read os.environ directly):

    from src.config.settings import settings
    uri = settings.mlflow_tracking_uri
"""

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("ragops.config")


class RAGOpsSettings(BaseSettings):
    """
    Validated application settings loaded from environment variables.

    Pydantic-settings coerces types and raises at startup if a required
    variable is missing, which surfaces misconfigurations before any
    service call is attempted.
    """

    # ── Service endpoints ─────────────────────────────────────────────────────
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    mlflow_tracking_uri: str = "http://mlflow:5000"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"

    # ── MLflow experiment ─────────────────────────────────────────────────────
    mlflow_experiment_name: str = "ragops_clinical"
    chroma_collection_name: str = "clinical_docs"

    # ── PSI drift detection thresholds (PAPER CONTRIBUTION) ──────────────────
    psi_warning_threshold: float = 0.1
    psi_alert_threshold: float = 0.25
    psi_num_bins: int = 10

    # ── XAI consistency thresholds (XAI CONTRIBUTION) ────────────────────────
    xai_warning_threshold: float = 0.75
    xai_alert_threshold: float = 0.60
    xai_benchmark_questions: int = 10

    # ── RAGAS quality gates ───────────────────────────────────────────────────
    ragas_faithfulness_min: float = 0.80
    ragas_context_recall_min: float = 0.75
    ragas_answer_relevancy_min: float = 0.75

    # ── Application ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    environment: str = "production"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


# Module-level singleton — import this everywhere
settings = RAGOpsSettings()
