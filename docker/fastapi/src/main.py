"""
RAGOps FastAPI Application Entry Point
=======================================
Orchestration layer for the RAG clinical decision support pipeline.

Exposes:
    GET  /health          — service liveness + dependency status
    POST /query           — RAG query (delegates to Member 2's chain)
    POST /evaluate        — RAGAS evaluation run (delegates to Member 3)
    POST /drift/check     — PSI embedding drift check  [PAPER CONTRIBUTION]
    POST /xai/check       — XAI explanation consistency check  [XAI CONTRIBUTION]
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings

logger = logging.getLogger("ragops.main")

# ---------------------------------------------------------------------------
# Settings — loaded from environment / .env
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings resolved from environment variables."""

    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    mlflow_tracking_uri: str = "http://mlflow:5000"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3:8b"

    # PSI thresholds (PAPER CONTRIBUTION)
    psi_warning_threshold: float = 0.1
    psi_alert_threshold: float = 0.25

    # XAI thresholds (XAI CONTRIBUTION)
    xai_warning_threshold: float = 0.75
    xai_alert_threshold: float = 0.60

    log_level: str = "INFO"
    environment: str = "production"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared resources on startup; clean up on shutdown."""
    logger.info(
        "RAGOps starting — env=%s model=%s",
        settings.environment,
        settings.ollama_model,
    )
    # Attach settings to app state so routers can access them
    app.state.settings = settings
    yield
    logger.info("RAGOps shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAGOps Clinical Decision Support",
    description="Production RAG orchestration layer with PSI drift detection and XAI monitoring.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class DependencyStatus(BaseModel):
    chromadb: str
    mlflow: str
    ollama: str


class HealthResponse(BaseModel):
    status: str
    environment: str
    dependencies: DependencyStatus


async def _ping(url: str, timeout: float = 3.0) -> str:
    """Return 'ok' if URL responds 200, else a descriptive error string."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=timeout)
        return "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """
    Liveness + dependency health check.

    Returns upstream status for ChromaDB, MLflow, and Ollama so the
    Docker health check and CI pipeline can gate on real connectivity.
    """
    chroma_status = await _ping(
        f"http://{settings.chroma_host}:{settings.chroma_port}/api/v1/heartbeat"
    )
    mlflow_status = await _ping(f"{settings.mlflow_tracking_uri}/health")
    ollama_status = await _ping(f"{settings.ollama_base_url}/api/tags")

    all_ok = all(
        s == "ok" for s in [chroma_status, mlflow_status, ollama_status]
    )

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        environment=settings.environment,
        dependencies=DependencyStatus(
            chromadb=chroma_status,
            mlflow=mlflow_status,
            ollama=ollama_status,
        ),
    )
