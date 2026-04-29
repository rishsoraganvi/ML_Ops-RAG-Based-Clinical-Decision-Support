"""
RAGOps Streamlit Dashboard — 5-Panel Clinical Decision Support UI.

Panels:
    1. Query Interface — ask medical questions, view RAG answers + sources
    2. XAI Explainability — SHAP bar chart, term attribution, hallucination risk
    3. Live RAGAS Metrics — faithfulness / recall / relevance over time
    4. Drift Alert Panel — PSI traffic light, XAI consistency
    5. Experiment Comparison — side-by-side MLflow run comparison

All MLflow access is proxied through the FastAPI service — the dashboard
never talks to the MLflow tracking server directly.

Run:
    streamlit run serving/dashboard.py
"""

from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests  # type: ignore[import-untyped]
import streamlit as st

logger = logging.getLogger("ragops.dashboard")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8080"
MLFLOW_EXPERIMENT = "ragops_clinical"


# ---------------------------------------------------------------------------
# HTTP helpers — every request goes through FastAPI
# ---------------------------------------------------------------------------


def _surface_request_error(
    endpoint: str,
    url: str,
    timeout: float,
    exc: requests.RequestException,
) -> None:
    """Common error -> st.error translator shared by GET and POST helpers."""
    if isinstance(exc, requests.ConnectionError):
        logger.error("Dashboard could not reach FastAPI at %s", url, exc_info=True)
        st.error(
            f"{endpoint} failed [ConnectionError]: cannot reach FastAPI at {API_BASE}. "
            f"Underlying error: {exc}. Verify the fastapi service is running "
            "(`docker compose ps fastapi`) and accessible on the host port."
        )
        return
    if isinstance(exc, requests.Timeout):
        logger.error(
            "Dashboard request to %s timed out after %ss", url, timeout, exc_info=True
        )
        st.error(
            f"{endpoint} failed [Timeout]: no response after {timeout}s. "
            f"Underlying error: {exc}. The Ollama model may be cold-loading "
            "(first request can exceed 60s) or the upstream RAG step is stuck \u2014 "
            "inspect `docker compose logs fastapi`."
        )
        return
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else "unknown"
        try:
            body = exc.response.json() if exc.response is not None else {}
            detail = body.get("detail", body) if isinstance(body, dict) else body
        except ValueError:
            detail = exc.response.text if exc.response is not None else str(exc)
        logger.error(
            "Dashboard got HTTP %s from %s: %s", status, url, detail, exc_info=True
        )
        st.error(f"{endpoint} failed (HTTP {status}): {detail}")
        return
    logger.error("Dashboard request to %s failed unexpectedly", url, exc_info=True)
    st.error(
        f"{endpoint} failed [{type(exc).__name__}]: {exc}. "
        "Inspect dashboard and FastAPI logs for the full traceback."
    )


def _api_post(
    endpoint: str,
    json_body: dict[str, Any] | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    """POST to FastAPI backend with error handling."""
    url = f"{API_BASE}{endpoint}"
    try:
        resp = requests.post(url, json=json_body or {}, timeout=timeout)
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())
    except requests.RequestException as exc:
        _surface_request_error(endpoint, url, timeout, exc)
        return {}


def _api_get(
    endpoint: str,
    params: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    """GET from FastAPI backend with error handling."""
    url = f"{API_BASE}{endpoint}"
    try:
        resp = requests.get(url, params=params or {}, timeout=timeout)
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())
    except requests.RequestException as exc:
        _surface_request_error(endpoint, url, timeout, exc)
        return {}


@st.cache_data(ttl=60)  # type: ignore[misc,untyped-decorator]
def _fetch_mlflow_runs(experiment_name: str, max_results: int = 50) -> pd.DataFrame:
    """Pull recent MLflow runs as a DataFrame via the FastAPI proxy."""
    payload = _api_get(
        "/mlflow/runs",
        params={"experiment_name": experiment_name, "max_results": max_results},
    )
    if not payload:
        return pd.DataFrame()

    records = payload.get("runs", [])
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
    return df


# ---------------------------------------------------------------------------
# Panel 1 — Query Interface
# ---------------------------------------------------------------------------


def _render_query_panel() -> None:
    """Interactive RAG query interface."""
    st.header("Clinical Query")

    question = st.text_area(
        "Ask a medical question:",
        placeholder="What are the first-line treatments for type 2 diabetes?",
        height=100,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        retriever = st.selectbox("Retriever", ["dense", "bm25", "hybrid"])
    with col2:
        chunk_size = st.selectbox("Chunk Size", [256, 512, 1024], index=1)
    with col3:
        top_k = st.slider("Top-K", 1, 10, 5)

    log_mlflow = st.checkbox("Log to MLflow", value=False)

    if st.button("Submit Query", type="primary") and question.strip():
        with st.spinner("Querying RAG pipeline..."):
            result = _api_post(
                "/query",
                {
                    "question": question,
                    "config": {
                        "retriever_type": retriever,
                        "chunk_size": chunk_size,
                        "k": top_k,
                    },
                    "log_to_mlflow": log_mlflow,
                },
            )

        if result:
            # Cache for the XAI panel to explain without re-running the query.
            st.session_state["last_query_result"] = result
            st.session_state["last_query_question"] = question

            st.subheader("Answer")
            st.write(result.get("answer", "No answer returned."))

            st.metric("Total Latency", f"{result.get('total_latency_ms', 0):.1f} ms")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Retrieval", f"{result.get('retrieval_latency_ms', 0):.1f} ms"
                )
            with col_b:
                st.metric("LLM", f"{result.get('llm_latency_ms', 0):.1f} ms")

            with st.expander("Source Documents"):
                for i, doc in enumerate(result.get("source_docs", [])):
                    scores = result.get("retrieval_scores", [])
                    score = scores[i] if i < len(scores) else 0
                    st.markdown(f"**Source {i + 1}** (score: {score:.4f})")
                    st.markdown(f"_{doc.get('metadata', {}).get('source', 'unknown')}_")
                    st.text(doc.get("page_content", "")[:500])
                    st.divider()


# ---------------------------------------------------------------------------
# Panel 2 — XAI Explainability
# ---------------------------------------------------------------------------


def _render_xai_panel() -> None:
    """XAI explanation visualisations powered by /explain + /xai/check."""
    st.header("Explainability (XAI)")

    last = st.session_state.get("last_query_result")
    last_q = st.session_state.get("last_query_question", "")

    if last is None:
        st.info(
            "Run a query from the **Query** tab first — the XAI panel explains "
            "the most recent answer (SHAP values, attention attribution, "
            "hallucination risk)."
        )
    else:
        st.caption(f"Explaining last query: _{last_q}_")
        if st.button("Explain Last Query", type="primary"):
            payload = {
                "question": last_q,
                "source_docs": last.get("source_docs", []),
                "retrieval_scores": last.get("retrieval_scores", []),
                "answer": last.get("answer"),
            }
            with st.spinner("Generating explanation..."):
                exp = _api_post("/explain", payload, timeout=180)

            if not exp:
                st.info(
                    "Explainability service unavailable — ensure `shap`, "
                    "`transformers`, and `torch` are installed in the FastAPI image."
                )
            else:
                _render_explanation(exp, last.get("source_docs", []))

    # Keep the consistency-monitor channel visible as a secondary section.
    st.divider()
    st.subheader("XAI Consistency Check")
    if st.button("Run XAI Consistency Check"):
        with st.spinner("Checking explanation consistency..."):
            result = _api_post("/xai/check", {"current_vectors": None})

        if result:
            score = result.get("consistency_score", 0)
            status = result.get("status", "unknown")

            if status == "stable":
                st.success(f"STABLE — consistency score: {score:.4f}")
            elif status == "warning":
                st.warning(f"WARNING — consistency score: {score:.4f}")
            else:
                st.error(f"INSTABILITY — consistency score: {score:.4f}")


def _render_explanation(exp: dict[str, Any], source_docs: list[dict[str, Any]]) -> None:
    """Render the six-key XAI payload returned by /explain."""
    # ── Hallucination risk ─────────────────────────────────────────────
    risk = float(exp.get("hallucination_risk", 0.0))
    reason = exp.get("hallucination_reason", "")
    col_risk, _ = st.columns([1, 2])
    with col_risk:
        st.metric("Hallucination Risk", f"{risk * 100:.1f}%")
    if risk < 0.3:
        st.success(f"Low risk. {reason}")
    elif risk < 0.6:
        st.warning(f"Moderate risk. {reason}")
    else:
        st.error(f"High risk. {reason}")

    # ── SHAP bar chart ─────────────────────────────────────────────────
    st.subheader("Per-Document SHAP Values")
    shap_values = exp.get("shap_values", [])
    if shap_values:
        shap_df = pd.DataFrame(
            {
                "Source": [f"Doc {i + 1}" for i in range(len(shap_values))],
                "SHAP": shap_values,
                "Direction": ["helpful" if v >= 0 else "harmful" for v in shap_values],
            }
        )
        fig = px.bar(
            shap_df,
            x="Source",
            y="SHAP",
            color="Direction",
            color_discrete_map={"helpful": "#3182bd", "harmful": "#e6550d"},
            template="plotly_dark",
            title="How each retrieved doc influenced faithfulness proxy",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No SHAP values returned.")

    # ── Attention highlights ───────────────────────────────────────────
    st.subheader("Attention-Weighted Context")
    token_attrs = exp.get("token_attributions", [])
    rendered_any = False
    for attr in token_attrs:
        sent = attr.get("sentence", "")
        spans = attr.get("spans", [])
        if not sent and not spans:
            continue
        rendered_any = True
        st.markdown(f"**Answer sentence {attr.get('sentence_idx', 0) + 1}:** {sent}")
        if not spans:
            st.caption("(no top spans)")
            continue
        max_score = max((s.get("score", 0.0) for s in spans), default=1e-6) or 1e-6
        html_parts = []
        for s in spans:
            weight = max(0.15, min(1.0, float(s.get("score", 0.0)) / max_score))
            html_parts.append(
                f'<mark style="background: rgba(255,200,0,{weight:.2f}); '
                f'padding: 0 3px; border-radius: 3px;">{s.get("text", "")}</mark>'
            )
        st.markdown(" · ".join(html_parts), unsafe_allow_html=True)
    if not rendered_any:
        st.caption("Attention attribution unavailable (HF model may not be loaded).")

    # ── Term attribution ───────────────────────────────────────────────
    st.subheader("BM25 Term Attribution (top 15)")
    term_attr = exp.get("term_attribution", {}) or {}
    global_scores = term_attr.get("global", {})
    if global_scores:
        top = list(global_scores.items())[:15]
        term_df = pd.DataFrame(top, columns=["Term", "Score"])
        st.dataframe(term_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No BM25 term overlap with source documents.")


# ---------------------------------------------------------------------------
# Panel 3 — Live RAGAS Metrics
# ---------------------------------------------------------------------------


def _render_ragas_panel() -> None:
    """RAGAS metric time series from MLflow (via FastAPI proxy)."""
    st.header("RAGAS Metrics Over Time")

    df = _fetch_mlflow_runs(MLFLOW_EXPERIMENT)

    if df.empty:
        st.warning(f"No runs found for experiment: {MLFLOW_EXPERIMENT}")
        return

    ragas_metrics = [
        "faithfulness",
        "context_recall",
        "answer_relevancy",
        "context_precision",
    ]
    available = [m for m in ragas_metrics if m in df.columns]

    if not available:
        st.warning(
            f"MLflow runs exist under `{MLFLOW_EXPERIMENT}` but none expose RAGAS "
            "metrics (faithfulness / context_recall / answer_relevancy / "
            "context_precision). Run `make eval-ci` so RAGOpsTracker logs the "
            "RAGAS metric set, or verify the experiment name."
        )
        return

    for metric in available:
        fig = px.line(
            df.sort_values("start_time"),
            x="start_time",
            y=metric,
            title=metric.replace("_", " ").title(),
            template="plotly_dark",
            markers=True,
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Panel 4 — Drift Alert Panel
# ---------------------------------------------------------------------------


def _render_drift_panel() -> None:
    """PSI drift status and alerts."""
    st.header("Drift Detection")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Embedding Drift (PSI)")
        if st.button("Run Drift Check"):
            with st.spinner("Computing PSI..."):
                result = _api_post("/drift/check")

            if result:
                status = result.get("status", "unknown")
                psi = result.get("psi_score", 0)

                if status == "stable":
                    st.success(f"STABLE — PSI: {psi:.4f}")
                elif status == "warning":
                    st.warning(f"WARNING — PSI: {psi:.4f}")
                else:
                    st.error(f"ALERT — PSI: {psi:.4f}")

                st.caption(f"Action: {result.get('action', 'none')}")
                st.caption(f"Timestamp: {result.get('timestamp', '')}")

                thresholds = result.get("thresholds", {})
                st.caption(
                    f"Thresholds — warning: {thresholds.get('warning', 0.1)}, "
                    f"alert: {thresholds.get('alert', 0.25)}"
                )

    with col2:
        st.subheader("Generation Drift")
        st.info(
            "Generation drift monitoring tracks RAGAS faithfulness "
            "rolling average. Available via mlops.drift_detector.monitor_generation()."
        )


# ---------------------------------------------------------------------------
# Panel 5 — Experiment Comparison
# ---------------------------------------------------------------------------


def _render_comparison_panel() -> None:
    """Side-by-side MLflow run comparison (via FastAPI proxy)."""
    st.header("Experiment Comparison")

    df = _fetch_mlflow_runs(MLFLOW_EXPERIMENT)

    if df.empty:
        st.warning(
            f"No MLflow runs found under experiment `{MLFLOW_EXPERIMENT}`. "
            "Run `make eval-ci` to create at least two runs before using this view."
        )
        return

    run_ids = df["run_id"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        run_a = st.selectbox("Run A", run_ids, index=0)
    with col2:
        run_b = st.selectbox("Run B", run_ids, index=min(1, len(run_ids) - 1))

    if st.button("Compare Runs") and run_a and run_b:
        with st.spinner("Comparing runs..."):
            result = _api_post(
                "/mlflow/compare",
                {"run_a": run_a, "run_b": run_b},
                timeout=60,
            )

        if not result:
            return

        st.subheader("Metric Deltas (Run B - Run A)")

        deltas = result.get("deltas", {})
        if deltas:
            delta_df = pd.DataFrame(
                [
                    {
                        "Metric": k,
                        "Run A": result["run_a"]["metrics"].get(k, 0),
                        "Run B": result["run_b"]["metrics"].get(k, 0),
                        "Delta": v,
                    }
                    for k, v in deltas.items()
                ]
            )
            st.dataframe(delta_df, use_container_width=True)

            fig = go.Figure(
                data=[
                    go.Bar(
                        name="Run A",
                        x=delta_df["Metric"],
                        y=delta_df["Run A"],
                    ),
                    go.Bar(
                        name="Run B",
                        x=delta_df["Metric"],
                        y=delta_df["Run B"],
                    ),
                ]
            )
            fig.update_layout(
                barmode="group",
                template="plotly_dark",
                title="Side-by-Side Metrics",
            )
            st.plotly_chart(fig, use_container_width=True)

        improved = result.get("improved", [])
        regressed = result.get("regressed", [])
        if improved:
            st.success(f"Improved: {', '.join(improved)}")
        if regressed:
            st.error(f"Regressed: {', '.join(regressed)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the Streamlit 5-panel dashboard."""
    st.set_page_config(
        page_title="RAGOps Dashboard",
        layout="wide",
    )
    st.title("RAGOps Clinical Decision Support")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Query",
            "XAI Explainability",
            "RAGAS Metrics",
            "Drift Alerts",
            "Experiment Comparison",
        ]
    )

    with tab1:
        _render_query_panel()
    with tab2:
        _render_xai_panel()
    with tab3:
        _render_ragas_panel()
    with tab4:
        _render_drift_panel()
    with tab5:
        _render_comparison_panel()


if __name__ == "__main__":
    main()
