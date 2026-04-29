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
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException
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
    except httpx.ConnectError as exc:
        return f"unreachable ({url}): {exc}"
    except httpx.TimeoutException as exc:
        return f"timeout after {timeout}s ({url}): {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"error[{type(exc).__name__}] ({url}): {exc}"


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
    config: dict[str, Any] | None = None
    log_to_mlflow: bool = False


class SourceDoc(BaseModel):
    """A single retrieved source document."""

    page_content: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """Structured response from the RAG chain."""

    answer: str
    source_docs: list[SourceDoc]
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    retrieval_scores: list[float]
    config: dict[str, Any]


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
    except (ConnectionError, httpx.ConnectError) as exc:
        logger.error("RAG query upstream unreachable", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=(
                f"RAG query failed [{type(exc).__name__}]: cannot reach an upstream "
                f"dependency (ChromaDB at {settings.chroma_host}:{settings.chroma_port} "
                f"or Ollama at {settings.ollama_base_url}). Underlying error: {exc}. "
                "Verify dependency health via GET /health and `docker compose ps`."
            ),
        ) from exc
    except (TimeoutError, httpx.TimeoutException) as exc:
        logger.error("RAG query timed out", exc_info=True)
        raise HTTPException(
            status_code=504,
            detail=(
                f"RAG query failed [{type(exc).__name__}]: upstream timed out. "
                f"Underlying error: {exc}. The Ollama model may be cold-loading from disk; "
                "first request after restart can exceed 60s. Check OLLAMA_KEEP_ALIVE and "
                "retry, or inspect `docker compose logs ollama`."
            ),
        ) from exc
    except (KeyError, ValueError) as exc:
        logger.error(
            "RAG query received invalid config or response shape", exc_info=True
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"RAG query failed [{type(exc).__name__}]: {exc}. "
                "Check the `config` payload (retriever_type ∈ {dense,bm25,hybrid}, "
                "chunk_size ∈ {256,512,1024}, k ∈ [1,10]) and that the requested "
                "ChromaDB collection (`pubmed_<chunk_size>`) has been ingested."
            ),
        ) from exc
    except Exception as exc:
        logger.error("RAG query failed unexpectedly", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                f"RAG query failed [{type(exc).__name__}]: {exc}. "
                "Inspect FastAPI logs (`docker compose logs fastapi`) for the full traceback."
            ),
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
    config: dict[str, Any] | None = None
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


def _run_eval_sync(req: EvalRequest) -> dict[str, Any]:
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

    result: dict[str, Any] = {
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
    except ValueError as exc:
        logger.error("Evaluation rejected invalid input", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Evaluation failed [ValueError]: {exc}. "
                "For `mode=full` you must supply `qa_pairs`; for `mode=ci` an empty list "
                "falls back to the 5-question stub. Verify each QA pair has "
                "`question`, `answer`, `contexts`, `ground_truth`."
            ),
        ) from exc
    except (ConnectionError, httpx.ConnectError) as exc:
        logger.error("Evaluation upstream unreachable", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Evaluation failed [{type(exc).__name__}]: cannot reach MLflow at "
                f"{settings.mlflow_tracking_uri} for quality-gate logging. Underlying error: "
                f"{exc}. Verify the mlflow service via GET /health and `docker compose ps mlflow`."
            ),
        ) from exc
    except ImportError as exc:
        logger.error("Evaluation missing dependency", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Evaluation failed [ImportError]: {exc}. The RAGAS / datasets stack "
                "is missing or broken in the FastAPI image. Rebuild via "
                "`docker compose build fastapi` after confirming "
                "docker/fastapi/requirements.txt pins ragas + datasets."
            ),
        ) from exc
    except Exception as exc:
        logger.error("Evaluation failed unexpectedly", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Evaluation failed [{type(exc).__name__}]: {exc}. "
                "Inspect FastAPI logs (`docker compose logs fastapi`) for the full traceback."
            ),
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
        embeddings = await asyncio.to_thread(
            get_embeddings, settings.baseline_chunk_size
        )
        psi_score = await asyncio.to_thread(compute_psi, embeddings)
        report = alert(psi_score)
    except RuntimeError as exc:
        if "baseline" in str(exc).lower():
            logger.error("Drift check missing baseline", exc_info=True)
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Drift check failed [RuntimeError]: {exc} "
                    "The PSI baseline snapshot has not been captured. Run "
                    "`make baseline` (or `python scripts/run_baseline_eval.py`) "
                    "after document ingestion to populate the baseline before "
                    "invoking /drift/check."
                ),
            ) from exc
        logger.warning("Drift check guardrail triggered: %s", exc)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Drift check rejected by guardrail [RuntimeError]: {exc}. "
                "The PSI detector refused to run \u2014 typical causes are mismatched "
                "embedding dimensionality between baseline and current snapshot, or "
                "an empty current sample."
            ),
        ) from exc
    except FileNotFoundError as exc:
        logger.error("Drift check missing baseline file", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Drift check failed [FileNotFoundError]: {exc}. The PSI baseline "
                "file is missing on disk. Run `make baseline` (or "
                "`python scripts/run_baseline_eval.py`) to populate "
                "`baselines/embedding_baseline.npy` before invoking /drift/check."
            ),
        ) from exc
    except (ConnectionError, httpx.ConnectError) as exc:
        logger.error("Drift check cannot reach ChromaDB", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Drift check failed [{type(exc).__name__}]: cannot reach ChromaDB at "
                f"{settings.chroma_host}:{settings.chroma_port} to fetch current embeddings. "
                f"Underlying error: {exc}. Verify the chromadb service via GET /health."
            ),
        ) from exc
    except ValueError as exc:
        logger.error("Drift check received malformed embeddings", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Drift check failed [ValueError]: {exc}. The current embeddings could "
                "not be aligned with the baseline \u2014 check that the embedding model "
                "(MiniLM-L6-v2 \u2192 384 dims) and the baseline file agree on shape."
            ),
        ) from exc
    except Exception as exc:
        logger.error("Drift check failed unexpectedly", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Drift check failed [{type(exc).__name__}]: {exc}. "
                "Inspect FastAPI logs (`docker compose logs fastapi`) for the full traceback."
            ),
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


def _replay_xai_benchmark() -> list[Any]:
    """Run the RAG chain on the first N benchmark questions and return their
    explanation vectors. Mirrors `scripts/run_baseline_eval.py::_capture_xai`
    so /xai/check can self-serve when the caller did not supply vectors.
    """
    import json
    from pathlib import Path

    from evaluation.explainability import explain
    from rag_pipeline.chain import query as rag_query

    qa_file = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "benchmarks"
        / "qa_pairs.json"
    )
    if not qa_file.exists():
        raise FileNotFoundError(
            f"XAI benchmark file not found at {qa_file}. "
            "Cannot generate replay vectors automatically."
        )

    qa_pairs = json.loads(qa_file.read_text(encoding="utf-8"))
    subset = qa_pairs[: settings.xai_benchmark_questions]

    vectors: list[Any] = []
    for item in subset:
        try:
            result = rag_query(item["question"])
            exp = explain(
                item["question"],
                result["source_docs"],
                result["retrieval_scores"],
                answer=result["answer"],
            )
            vectors.append(exp["explanation_vector"])
        except Exception as exc:
            logger.warning(
                "XAI replay skipped question %r: %s", item.get("question"), exc
            )

    if not vectors:
        raise RuntimeError(
            "XAI replay produced zero explanation vectors. "
            "All benchmark questions failed against the live RAG pipeline."
        )
    return vectors


@app.post("/xai/check", response_model=XAIResponse, tags=["xai"])
async def xai_check(req: XAIRequest) -> XAIResponse:
    """Check XAI explanation consistency against baseline. # XAI CONTRIBUTION"""
    import numpy as np

    from mlops.explanation_monitor import compute_consistency

    try:
        if req.current_vectors:
            vectors: list[Any] = [np.array(v) for v in req.current_vectors]
        else:
            logger.info(
                "/xai/check called without current_vectors \u2014 replaying "
                "the XAI benchmark to generate them."
            )
            vectors = await asyncio.to_thread(_replay_xai_benchmark)
        score = await asyncio.to_thread(compute_consistency, vectors)
    except RuntimeError as exc:
        if "baseline" in str(exc).lower():
            logger.error("XAI check missing baseline", exc_info=True)
            raise HTTPException(
                status_code=422,
                detail=(
                    f"XAI check failed [RuntimeError]: {exc} "
                    "Run `python scripts/run_baseline_eval.py` "
                    "(or its `--skip-eval --skip-psi` variant) to populate "
                    "`baselines/xai_baseline.npy` before invoking /xai/check."
                ),
            ) from exc
        logger.warning("XAI consistency guardrail triggered: %s", exc)
        raise HTTPException(
            status_code=409,
            detail=(
                f"XAI consistency check rejected by guardrail [RuntimeError]: {exc}. "
                "Common causes: explanation-vector dimensionality differs from baseline, "
                "or the replay benchmark produced zero usable vectors."
            ),
        ) from exc
    except FileNotFoundError as exc:
        logger.error("XAI check missing baseline", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=(
                f"XAI check failed [FileNotFoundError]: {exc}. The XAI explanation "
                "baseline has not been captured. Run `python scripts/run_baseline_eval.py` "
                "to populate `baselines/xai_baseline.npy` before invoking /xai/check."
            ),
        ) from exc
    except ValueError as exc:
        logger.error("XAI check received malformed vectors", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=(
                f"XAI check failed [ValueError]: {exc}. Ensure each entry in "
                "`current_vectors` has the same length as the baseline explanation vector."
            ),
        ) from exc
    except Exception as exc:
        logger.error("XAI check failed unexpectedly", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                f"XAI check failed [{type(exc).__name__}]: {exc}. "
                "Inspect FastAPI logs (`docker compose logs fastapi`) for the full traceback."
            ),
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


# ---------------------------------------------------------------------------
# Explain endpoint — runs the full XAI pipeline on one query result
# ---------------------------------------------------------------------------


class ExplainRequest(BaseModel):
    """Input for the /explain endpoint — the output of a prior /query call."""

    question: str
    source_docs: list[SourceDoc]
    retrieval_scores: list[float]
    answer: str | None = None


class TokenAttributionSpan(BaseModel):
    text: str
    score: float
    char_start: int
    char_end: int


class TokenAttributionSentence(BaseModel):
    sentence_idx: int
    sentence: str
    spans: list[TokenAttributionSpan]


class ExplainResponse(BaseModel):
    """Full XAI payload consumed by the Streamlit dashboard."""

    shap_values: list[float]
    token_attributions: list[TokenAttributionSentence]
    term_attribution: dict[str, Any]
    explanation_vector: list[float]
    hallucination_risk: float
    hallucination_reason: str


@app.post("/explain", response_model=ExplainResponse, tags=["xai"])
async def explain_endpoint(req: ExplainRequest) -> ExplainResponse:
    """Run the unified XAI pipeline on a prior /query result. # XAI CONTRIBUTION"""
    if not req.source_docs:
        raise HTTPException(
            status_code=400,
            detail=(
                "Explain failed [ValidationError]: `source_docs` must be non-empty. "
                "Pass the `source_docs` array returned by a prior /query call."
            ),
        )

    try:
        from evaluation.explainability import explain as xai_explain
    except ImportError as exc:
        logger.error("XAI imports failed", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Explainability unavailable [ImportError]: {exc}. The XAI stack "
                "(shap / transformers / torch) is missing or broken in the FastAPI image. "
                "Confirm these packages are pinned in docker/fastapi/requirements.txt and "
                "rebuild via `docker compose build fastapi`."
            ),
        ) from exc

    try:
        result = await asyncio.to_thread(
            xai_explain,
            req.question,
            [d.model_dump() for d in req.source_docs],
            req.retrieval_scores,
            req.answer,
        )
    except ValueError as exc:
        logger.error("Explain pipeline received malformed input", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Explain failed [ValueError]: {exc}. Verify each source_doc has "
                "non-empty `page_content` and that `retrieval_scores` length matches "
                "`source_docs`."
            ),
        ) from exc
    except RuntimeError as exc:
        logger.error(
            "Explain pipeline runtime error (likely model load)", exc_info=True
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Explainability unavailable [RuntimeError]: {exc}. The HuggingFace "
                "attention model failed to load \u2014 check the FastAPI container has GPU "
                "access (or fallback to CPU) and that the transformers cache is writable."
            ),
        ) from exc
    except Exception as exc:
        logger.error("Explain pipeline failed unexpectedly", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Explainability unavailable [{type(exc).__name__}]: {exc}. "
                "Inspect FastAPI logs (`docker compose logs fastapi`) for the full traceback."
            ),
        ) from exc

    # Convert numpy array → list for JSON serialization.
    ev = result["explanation_vector"]
    explanation_vector = ev.tolist() if hasattr(ev, "tolist") else list(ev)

    return ExplainResponse(
        shap_values=[float(v) for v in result["shap_values"]],
        token_attributions=[
            TokenAttributionSentence(
                sentence_idx=s["sentence_idx"],
                sentence=s["sentence"],
                spans=[TokenAttributionSpan(**sp) for sp in s["spans"]],
            )
            for s in result["token_attributions"]
        ],
        term_attribution=result["term_attribution"],
        explanation_vector=explanation_vector,
        hallucination_risk=float(result["hallucination_risk"]),
        hallucination_reason=result["hallucination_reason"],
    )


# ---------------------------------------------------------------------------
# MLflow proxy endpoints — only FastAPI is permitted to talk to MLflow,
# so the Streamlit dashboard goes through these instead of the tracking server.
# ---------------------------------------------------------------------------


class MLflowRunsResponse(BaseModel):
    """Recent MLflow runs flattened for the dashboard."""

    runs: list[dict[str, Any]]


class MLflowCompareRequest(BaseModel):
    """Inputs for /mlflow/compare."""

    run_a: str
    run_b: str


class MLflowCompareResponse(BaseModel):
    """Output of mlops.compare_runs.compare_runs."""

    run_a: dict[str, Any]
    run_b: dict[str, Any]
    deltas: dict[str, float]
    improved: list[str]
    regressed: list[str]
    unchanged: list[str]


def _list_mlflow_runs_sync(
    experiment_name: str, max_results: int
) -> list[dict[str, Any]]:
    """Synchronous MLflow run search; runs in a worker thread."""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )

    records: list[dict[str, Any]] = []
    for r in runs:
        row: dict[str, Any] = {"run_id": r.info.run_id, "start_time": r.info.start_time}
        row.update(r.data.metrics)
        row.update({f"param_{k}": v for k, v in r.data.params.items()})
        records.append(row)
    return records


@app.get("/mlflow/runs", response_model=MLflowRunsResponse, tags=["mlflow"])
async def mlflow_runs(
    experiment_name: str | None = None,
    max_results: int = 50,
) -> MLflowRunsResponse:
    """List recent MLflow runs for the dashboard. Returns [] if experiment is missing."""
    import mlflow

    name = experiment_name or settings.mlflow_experiment_name
    capped = max(1, min(int(max_results), 200))

    try:
        records = await asyncio.to_thread(_list_mlflow_runs_sync, name, capped)
    except (ConnectionError, httpx.ConnectError) as exc:
        logger.error("MLflow runs query cannot reach tracking server", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=(
                f"MLflow runs query failed [{type(exc).__name__}]: cannot reach MLflow at "
                f"{settings.mlflow_tracking_uri}. Underlying error: {exc}. "
                "Verify the mlflow service via GET /health and `docker compose ps mlflow`."
            ),
        ) from exc
    except mlflow.exceptions.MlflowException as exc:
        logger.error("MLflow rejected runs query", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=(
                f"MLflow runs query failed [MlflowException]: {exc}. "
                f"Tracking server at {settings.mlflow_tracking_uri} returned an error."
            ),
        ) from exc
    except Exception as exc:
        logger.error("MLflow runs query failed unexpectedly", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                f"MLflow runs query failed [{type(exc).__name__}]: {exc}. "
                "Inspect FastAPI logs (`docker compose logs fastapi`) for the full traceback."
            ),
        ) from exc

    return MLflowRunsResponse(runs=records)


@app.post("/mlflow/compare", response_model=MLflowCompareResponse, tags=["mlflow"])
async def mlflow_compare(req: MLflowCompareRequest) -> MLflowCompareResponse:
    """Compare two MLflow runs side-by-side via mlops.compare_runs."""
    import mlflow

    from mlops.compare_runs import compare_runs

    try:
        result = await asyncio.to_thread(compare_runs, req.run_a, req.run_b)
    except (ConnectionError, httpx.ConnectError) as exc:
        logger.error("MLflow compare cannot reach tracking server", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=(
                f"MLflow compare failed [{type(exc).__name__}]: cannot reach MLflow at "
                f"{settings.mlflow_tracking_uri}. Underlying error: {exc}."
            ),
        ) from exc
    except mlflow.exceptions.MlflowException as exc:
        logger.error("MLflow compare could not load run", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail=(
                f"MLflow compare failed [MlflowException]: {exc}. "
                f"One of the requested runs ({req.run_a}, {req.run_b}) is not present "
                f"on the tracking server at {settings.mlflow_tracking_uri}."
            ),
        ) from exc
    except KeyError as exc:
        logger.error("MLflow compare hit missing key", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=(
                f"MLflow compare failed [KeyError]: {exc}. The requested run is missing an "
                "expected field — confirm both runs were logged by RAGOpsTracker."
            ),
        ) from exc
    except Exception as exc:
        logger.error("MLflow compare failed unexpectedly", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                f"MLflow compare failed [{type(exc).__name__}]: {exc}. "
                "Inspect FastAPI logs (`docker compose logs fastapi`) for the full traceback."
            ),
        ) from exc

    return MLflowCompareResponse(**result)
