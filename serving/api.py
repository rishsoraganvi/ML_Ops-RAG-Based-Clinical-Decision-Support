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

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from src.config.settings import settings

logger = logging.getLogger("ragops.main")

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

    all_ok = all(s == "ok" for s in [chroma_status, mlflow_status, ollama_status])

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        environment=settings.environment,
        dependencies=DependencyStatus(
            chromadb=chroma_status,
            mlflow=mlflow_status,
            ollama=ollama_status,
        ),
    )


# ---------------------------------------------------------------------------
# Query endpoint — delegates to rag_pipeline.chain.query()
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Input for the /query endpoint."""

    question: str
    config: dict | None = None
    log_to_mlflow: bool = False


class SourceDoc(BaseModel):
    """A single retrieved source document."""

    page_content: str
    metadata: dict


class QueryResponse(BaseModel):
    """Structured response from the RAG chain."""

    answer: str
    source_docs: list[SourceDoc]
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    retrieval_scores: list[float]
    config: dict


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Run an end-to-end RAG query against the clinical knowledge base."""
    from rag_pipeline.chain import query as rag_query

    try:
        result = await asyncio.to_thread(
            rag_query,
            req.question,
            req.config,
            mlflow_run=req.log_to_mlflow,
        )
    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"RAG pipeline unavailable: {exc}",
        ) from exc

    return QueryResponse(
        answer=result["answer"],
        source_docs=[
            SourceDoc(page_content=d["page_content"], metadata=d["metadata"])
            for d in result["source_docs"]
        ],
        retrieval_latency_ms=result["retrieval_latency_ms"],
        llm_latency_ms=result["llm_latency_ms"],
        total_latency_ms=result["total_latency_ms"],
        retrieval_scores=result["retrieval_scores"],
        config=result["config"],
    )


# ---------------------------------------------------------------------------
# Evaluate endpoint — delegates to evaluation.ragas_runner
# ---------------------------------------------------------------------------


class QAPair(BaseModel):
    """A single QA pair for RAGAS evaluation."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


class EvalRequest(BaseModel):
    """Input for the /evaluate endpoint."""

    qa_pairs: list[QAPair] | None = None
    config: dict | None = None
    mode: str = "ci"  # "ci" or "full"
    run_quality_gate: bool = True


class EvalResponse(BaseModel):
    """RAGAS evaluation results with optional quality gate outcome."""

    faithfulness: float
    context_recall: float
    answer_relevance: float
    context_precision: float
    quality_gate_passed: bool | None = None
    quality_gate_failures: dict[str, str] | None = None


def _run_eval_sync(req: EvalRequest) -> dict:
    """Synchronous wrapper for RAGAS evaluation + optional quality gate."""
    from evaluation.ragas_runner import run_ci_eval, run_eval
    from mlops.mlflow_tracker import RAGASMetrics, RAGOpsTracker, RunTrigger

    pairs = [p.model_dump() for p in req.qa_pairs] if req.qa_pairs else None

    if req.mode == "ci":
        scores = run_ci_eval(qa_pairs=pairs, config=req.config)
    else:
        if not pairs:
            raise ValueError("qa_pairs required for full evaluation mode")
        scores = run_eval(qa_pairs=pairs, config=req.config)

    result: dict = {
        "faithfulness": scores.get("faithfulness", 0.0),
        "context_recall": scores.get("context_recall", 0.0),
        "answer_relevance": scores.get("answer_relevance", 0.0),
        "context_precision": scores.get("context_precision", 0.0),
    }

    if req.run_quality_gate:
        tracker = RAGOpsTracker()
        with tracker.start_run(
            triggered_by=RunTrigger.CI,
            run_name="api-eval-quality-gate",
        ):
            tracker.log_ragas_metrics(
                RAGASMetrics(
                    faithfulness=result["faithfulness"],
                    context_recall=result["context_recall"],
                    answer_relevancy=result["answer_relevance"],
                )
            )
            if req.config:
                tracker.log_params(req.config)
            gate = tracker.log_quality_gate_results()

        result["quality_gate_passed"] = gate.passed
        result["quality_gate_failures"] = gate.failures

    return result


@app.post("/evaluate", response_model=EvalResponse, tags=["eval"])
async def evaluate_endpoint(req: EvalRequest) -> EvalResponse:
    """Run RAGAS evaluation with optional quality gate."""
    try:
        result = await asyncio.to_thread(_run_eval_sync, req)
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Evaluation unavailable: {exc}",
        ) from exc

    return EvalResponse(**result)


# ---------------------------------------------------------------------------
# Drift check endpoint — delegates to mlops.drift_detector
# ---------------------------------------------------------------------------


class DriftResponse(BaseModel):
    """PSI drift detection result."""

    status: str
    psi_score: float
    timestamp: str
    action: str
    thresholds: dict[str, float]


@app.post("/drift/check", response_model=DriftResponse, tags=["drift"])
async def drift_check() -> DriftResponse:
    """Check for embedding distribution drift via PSI. # PAPER CONTRIBUTION"""
    from data.ingest import get_embeddings
    from mlops.drift_detector import alert, compute_psi

    try:
        embeddings = await asyncio.to_thread(get_embeddings)
        psi_score = await asyncio.to_thread(compute_psi, embeddings)
        report = alert(psi_score)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Drift check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Drift check unavailable: {exc}",
        ) from exc

    return DriftResponse(**report)


# ---------------------------------------------------------------------------
# XAI consistency check endpoint — delegates to mlops.explanation_monitor
# ---------------------------------------------------------------------------


class XAIRequest(BaseModel):
    """Input for XAI consistency check."""

    current_vectors: list[list[float]] | None = None


class XAIResponse(BaseModel):
    """XAI explanation consistency result."""

    consistency_score: float
    status: str
    thresholds: dict[str, float]


@app.post("/xai/check", response_model=XAIResponse, tags=["xai"])
async def xai_check(req: XAIRequest) -> XAIResponse:
    """Check XAI explanation consistency against baseline. # XAI CONTRIBUTION"""
    import numpy as np

    from mlops.explanation_monitor import compute_consistency

    try:
        vectors = (
            [np.array(v) for v in req.current_vectors]
            if req.current_vectors
            else None
        )
        score = await asyncio.to_thread(compute_consistency, vectors)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("XAI check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"XAI check unavailable: {exc}",
        ) from exc

    if score >= settings.xai_warning_threshold:
        status = "stable"
    elif score >= settings.xai_alert_threshold:
        status = "warning"
    else:
        status = "instability"

    return XAIResponse(
        consistency_score=score,
        status=status,
        thresholds={
            "warning": settings.xai_warning_threshold,
            "alert": settings.xai_alert_threshold,
        },
    )
