"""
ragas_runner.py
==============
RAGAS evaluation pipeline for RAGOps — clinical decision support system.

Exposes:
    run_eval(qa_pairs, config)   -> dict   # full 50-question eval
    run_ci_eval(config=None)     -> dict   # smoke-test, < 90 s

LLM Judge : Ollama (default: llama3.2:1b via OLLAMA_MODEL env var)
Metrics   : faithfulness | context_recall | answer_relevance | context_precision
Tracking  : MLflow — every run logged automatically

Author : LLM & Evaluation Lead
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Memory guards — must be set BEFORE numpy/torch/ragas are imported
# Prevents OpenBLAS from allocating huge contiguous memory blocks
# ---------------------------------------------------------------------------
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from datasets import Dataset
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import OllamaEmbeddings
from ragas import evaluate

# RAGAS metrics — version-safe import
# ragas>=0.2 uses class instances; older versions use module-level objects
import ragas.metrics as _ragas_metrics_module


def _get_metric(name: str) -> Any:
    """Get metric by name, instantiating if it is a class."""
    import inspect

    obj = getattr(_ragas_metrics_module, name, None)
    if obj is None:
        # Try collections submodule (ragas>=0.2)
        try:
            import ragas.metrics.collections as _col

            obj = getattr(_col, name, None)
        except ImportError:
            pass
    if obj is None:
        raise ImportError(f"Cannot find RAGAS metric: {name}")
    # If it's a class, instantiate it; if already an instance/module obj, use as-is
    if inspect.isclass(obj):
        return obj()
    return obj


faithfulness = _get_metric("faithfulness")
context_recall = _get_metric("context_recall")
answer_relevancy = _get_metric("answer_relevancy")
context_precision = _get_metric("context_precision")
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ragops.eval")

# ---------------------------------------------------------------------------
# Reproducibility — MUST be set before any evaluation
# ---------------------------------------------------------------------------
_GLOBAL_SEED = 42


def _set_seeds(seed: int = _GLOBAL_SEED) -> None:
    """Fix all random seeds for deterministic evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_set_seeds()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "ragops-ragas-eval")
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")

CI_SAMPLE_SIZE: int = 5  # questions used in run_ci_eval
CI_TIMEOUT_SECS: int = 90  # hard SLA for run_ci_eval

# RAGAS metric objects — instantiated via _get_metric() above
_METRICS = [faithfulness, context_recall, answer_relevancy, context_precision]

# Canonical output keys — keep stable for downstream consumers
METRIC_KEYS = [
    "faithfulness",
    "context_recall",
    "answer_relevance",
    "context_precision",
]

# ---------------------------------------------------------------------------
# LLM / Embedding wrappers
# ---------------------------------------------------------------------------


def _build_ragas_llm() -> LangchainLLMWrapper:
    """Wrap the configured Ollama model as a RAGAS-compatible LLM judge."""
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,  # deterministic judge
        num_predict=512,
        # Long timeout — needed for local models on CPU. Set via env var
        # OLLAMA_TIMEOUT (not a constructor arg in current langchain-ollama).
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Wrap Ollama nomic-embed-text for RAGAS embedding-based metrics."""
    emb = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return LangchainEmbeddingsWrapper(emb)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _build_ragas_dataset(qa_pairs: list[dict[str, Any]]) -> Dataset:
    """
    Convert internal qa_pairs format → ragas HuggingFace Dataset.

    Expected qa_pairs schema (per item):
    {
        "question"       : str,
        "answer"         : str,          # model answer (from Member 2's query())
        "contexts"       : List[str],    # source_docs as plain text
        "ground_truth"   : str,          # reference answer from benchmark
    }
    """
    required = {"question", "answer", "contexts", "ground_truth"}
    for i, item in enumerate(qa_pairs):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"qa_pairs[{i}] missing keys: {missing}")
        if not isinstance(item["contexts"], list):
            raise TypeError(f"qa_pairs[{i}]['contexts'] must be a list of strings")

    return Dataset.from_list(qa_pairs)


# ---------------------------------------------------------------------------
# Config fingerprint (for MLflow run naming + reproducibility audit)
# ---------------------------------------------------------------------------


def _config_fingerprint(config: dict[str, Any] | None) -> str:
    """Stable short hash of the config dict for run naming."""
    blob = json.dumps(config or {}, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:8]  # noqa: S324 — non-crypto use


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def _run_ragas(
    dataset: Dataset,
    run_tag: str,
    config: dict[str, Any] | None,
    extra_tags: dict[str, Any] | None = None,
    per_question_path: Path | None = None,
) -> dict[str, float]:
    """
    Execute RAGAS evaluate() and log results to MLflow.

    Returns dict with keys matching METRIC_KEYS.
    """
    _set_seeds()  # re-seed right before evaluation for full determinism

    # CPU-safe concurrency — run 1 job at a time, long timeout
    # llama3.2:1b on CPU takes ~20-40s per call
    os.environ["RAGAS_MAX_WORKERS"] = "1"
    os.environ["RAGAS_TIMEOUT"] = "600"  # 10 min per job

    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()

    # Inject judge into metric instances
    for metric in _METRICS:
        if hasattr(metric, "llm"):
            metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_emb

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    cfg_hash = _config_fingerprint(config)
    config_id = (config or {}).get("config_id", "")
    run_name = f"{run_tag}_{cfg_hash}"

    log.info("Starting RAGAS evaluation | run=%s | n=%d", run_name, len(dataset))
    t0 = time.perf_counter()

    with mlflow.start_run(run_name=run_name):
        # ── Tags ──────────────────────────────────────────────────────────
        mlflow.set_tags(
            {
                "run_tag": run_tag,
                "config_hash": cfg_hash,
                "config_id": config_id,
                "ollama_model": OLLAMA_MODEL,
                "n_questions": len(dataset),
                **(extra_tags or {}),
            }
        )

        # Log full config as a JSON artifact for reproducibility
        if config:
            mlflow.log_dict(config, "config.json")  # PAPER RESULT — TABLE 1

        # ── RAGAS evaluate ─────────────────────────────────────────────
        from ragas.run_config import RunConfig

        run_cfg = RunConfig(
            timeout=600,  # 10 min per LLM call — needed for CPU inference
            max_retries=3,  # retry on transient 500s
            max_workers=1,  # sequential — prevents RAM overload on CPU
        )
        result = evaluate(
            dataset=dataset,
            metrics=_METRICS,
            run_config=run_cfg,
        )

        elapsed = time.perf_counter() - t0
        log.info("RAGAS finished in %.1f s", elapsed)

        # ── Extract scores ─────────────────────────────────────────────
        result_df: pd.DataFrame = result.to_pandas()

        scores: dict[str, float] = {
            "faithfulness": float(result_df["faithfulness"].mean()),
            "context_recall": float(result_df["context_recall"].mean()),
            "answer_relevance": float(result_df["answer_relevancy"].mean()),
            "context_precision": float(result_df["context_precision"].mean()),
        }

        # ── MLflow metrics ─────────────────────────────────────────────
        mlflow.log_metrics(scores)
        mlflow.log_metric("eval_latency_s", elapsed)
        mlflow.log_metric("n_questions", len(dataset))

        # Per-question scores as artifact (needed for paired t-tests)
        per_q = result_df[
            ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]
        ].rename(columns={"answer_relevancy": "answer_relevance"})

        import tempfile

        if per_question_path is not None:
            # Caller specified a destination — write directly (no temp file)
            Path(per_question_path).parent.mkdir(parents=True, exist_ok=True)
            per_q.to_csv(per_question_path, index=False)
            mlflow.log_artifact(str(per_question_path), "per_question")
        else:
            # Use a unique temp file to avoid collisions when running in parallel
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                prefix="per_question_scores_",
                delete=False,
            ) as _tmp_file:
                _tmp = _tmp_file.name
            try:
                per_q.to_csv(_tmp, index=False)
                mlflow.log_artifact(_tmp, "per_question")
            finally:
                if os.path.exists(_tmp):
                    os.remove(_tmp)

        log.info("Scores: %s", scores)

    return scores


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def run_eval(
    qa_pairs: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    per_question_path: Path | None = None,
) -> dict[str, float]:
    """
    Run full RAGAS evaluation over all qa_pairs.

    Parameters
    ----------
    qa_pairs : list of dicts
        Each dict: {question, answer, contexts, ground_truth}
    config   : dict | None
        Ablation config (chunk_size, retriever, top_k, …).
        Logged verbatim to MLflow for reproducibility.
    per_question_path : Path | None
        If provided, per-question scores CSV is written directly to this path
        instead of a temporary file.  Pass a config-specific path when calling
        from the ablation runner to avoid concurrent-write races.

    Returns
    -------
    dict with keys: faithfulness, context_recall, answer_relevance,
                    context_precision
    """
    if not qa_pairs:
        raise ValueError("qa_pairs must be non-empty")

    dataset = _build_ragas_dataset(qa_pairs)
    return _run_ragas(
        dataset,
        run_tag="full_eval",
        config=config,
        per_question_path=per_question_path,
    )


def run_ci_eval(
    qa_pairs: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Lightweight CI smoke-test — completes in < 90 seconds.

    Uses the first CI_SAMPLE_SIZE questions from qa_pairs (or a
    built-in stub set if qa_pairs is None).  Intended for pipeline
    health checks, not paper results.

    Parameters
    ----------
    qa_pairs : list of dicts | None
        If None, uses internal stub questions.
    config   : dict | None
        Passed through to MLflow.

    Returns
    -------
    Same schema as run_eval().
    """
    source = qa_pairs if qa_pairs else _ci_stub_pairs()

    # deterministic sample — same questions every CI run
    _set_seeds()
    sample = source[:CI_SAMPLE_SIZE]

    log.info("run_ci_eval — %d questions (SLA: %ds)", len(sample), CI_TIMEOUT_SECS)

    t0 = time.perf_counter()
    dataset = _build_ragas_dataset(sample)
    scores = _run_ragas(
        dataset,
        run_tag="ci_eval",
        config=config,
        extra_tags={"ci": "true", "ci_sample_size": str(CI_SAMPLE_SIZE)},
    )

    elapsed = time.perf_counter() - t0
    if elapsed > CI_TIMEOUT_SECS:
        log.warning("run_ci_eval exceeded SLA: %.1fs > %ds", elapsed, CI_TIMEOUT_SECS)

    return scores


# ---------------------------------------------------------------------------
# CI stub — minimal synthetic pairs so CI can run without real data
# ---------------------------------------------------------------------------


def _ci_stub_pairs() -> list[dict[str, Any]]:
    """
    5 minimal medical QA pairs for CI smoke-testing.
    NOT for paper results — use the 50-question PubMed benchmark for those.
    """
    return [
        {
            "question": "What is the first-line treatment for type 2 diabetes?",
            "answer": "Metformin is the first-line pharmacological treatment for type 2 diabetes.",
            "contexts": [
                "Metformin remains the preferred initial pharmacological agent for type 2 "
                "diabetes due to its efficacy, safety profile, and low cost.",
                "Lifestyle modification including diet and exercise is recommended alongside "
                "pharmacotherapy.",
            ],
            "ground_truth": "Metformin is the recommended first-line treatment for type 2 diabetes.",
        },
        {
            "question": "What are the major risk factors for myocardial infarction?",
            "answer": (
                "Major risk factors include hypertension, hyperlipidaemia, smoking, "
                "diabetes mellitus, obesity, and family history."
            ),
            "contexts": [
                "Cardiovascular risk factors include hypertension, dyslipidaemia, cigarette "
                "smoking, diabetes, sedentary lifestyle, and a family history of coronary artery disease.",
            ],
            "ground_truth": (
                "Hypertension, hyperlipidaemia, smoking, diabetes, obesity, and family "
                "history are major risk factors for MI."
            ),
        },
        {
            "question": "How does aspirin reduce platelet aggregation?",
            "answer": (
                "Aspirin irreversibly inhibits COX-1 and COX-2, reducing thromboxane A2 "
                "synthesis and thereby inhibiting platelet aggregation."
            ),
            "contexts": [
                "Aspirin acetylates and irreversibly inhibits cyclooxygenase (COX) enzymes, "
                "blocking the synthesis of thromboxane A2 in platelets.",
            ],
            "ground_truth": (
                "Aspirin irreversibly inhibits COX enzymes, reducing TXA2 and platelet "
                "aggregation."
            ),
        },
        {
            "question": "What is the mechanism of action of beta-blockers in heart failure?",
            "answer": (
                "Beta-blockers competitively block catecholamines at beta-adrenergic receptors, "
                "reducing heart rate and myocardial oxygen demand."
            ),
            "contexts": [
                "Beta-adrenergic receptor antagonists reduce sympathetic stimulation of the "
                "heart, decreasing heart rate, contractility, and oxygen consumption.",
                "In heart failure, beta-blockers reduce adverse cardiac remodelling and "
                "improve long-term survival.",
            ],
            "ground_truth": (
                "Beta-blockers block beta-adrenergic receptors, reducing heart rate and "
                "myocardial oxygen demand, improving outcomes in heart failure."
            ),
        },
        {
            "question": "What is the role of HbA1c in diabetes monitoring?",
            "answer": (
                "HbA1c reflects average blood glucose over the preceding 2-3 months and "
                "is used to assess long-term glycaemic control."
            ),
            "contexts": [
                "Glycated haemoglobin (HbA1c) provides an index of average plasma glucose "
                "concentration over the preceding 8-12 weeks.",
                "Current guidelines recommend HbA1c < 7% (53 mmol/mol) as a target for "
                "most patients with type 2 diabetes.",
            ],
            "ground_truth": (
                "HbA1c measures average blood glucose over 2-3 months and is the standard "
                "marker for long-term glycaemic control."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# CLI helper (not part of the paper pipeline — for local sanity checks)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGOps RAGAS eval runner")
    parser.add_argument(
        "--mode",
        choices=["ci", "full"],
        default="ci",
        help="'ci' runs the 5-question smoke test; 'full' requires --qa-file",
    )
    parser.add_argument(
        "--qa-file",
        type=Path,
        help="JSON file with qa_pairs for full eval",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="JSON file with ablation config",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config_file.read_text()) if args.config_file else None

    if args.mode == "ci":
        results = run_ci_eval(config=cfg)
    else:
        if not args.qa_file:
            parser.error("--qa-file required for full eval mode")
        pairs = json.loads(args.qa_file.read_text())
        results = run_eval(pairs, config=cfg)

    print("\n=== RAGAS Results ===")
    for k, v in results.items():
        print(f"  {k:<22} {v:.4f}")
