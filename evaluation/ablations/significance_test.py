"""
significance_test.py
====================
Statistical significance testing for RAGOps ablation results.

Reads: ablation_outputs/ablation_results.csv
       ablation_outputs/per_question/ (per-question CSVs from MLflow)

Tests:
  - Paired t-test between best and second-best config on faithfulness
  - Cohen's d effect size for all metric comparisons
  - Bonferroni correction for multiple comparisons

Output:
  ablation_outputs/significance_report.txt   ← paper-ready stats
  ablation_outputs/significance_report.json  ← machine-readable

# PAPER RESULT — TABLE 1 (statistical significance row)

Author: LLM & Evaluation Lead
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ragops.significance")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.getenv("ABLATION_OUTPUT_DIR", "ablation_outputs"))
RESULTS_CSV = OUTPUT_DIR / "ablation_results.csv"
PER_Q_DIR = OUTPUT_DIR / "per_question"
REPORT_TXT = OUTPUT_DIR / "significance_report.txt"
REPORT_JSON = OUTPUT_DIR / "significance_report.json"

METRIC_KEYS = [
    "faithfulness",
    "context_recall",
    "answer_relevance",
    "context_precision",
]
ALPHA = 0.05  # significance threshold


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute paired-samples Cohen's dz effect size between two arrays."""
    diff = a - b
    diff_std = np.std(diff, ddof=1)
    if np.isclose(diff_std, 0.0):
        return 0.0
    return float(np.mean(diff) / diff_std)


def interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Load per-question scores for a config
# ---------------------------------------------------------------------------
def _load_per_question(config_id: str) -> pd.DataFrame | None:
    """Load per-question scores CSV for a given config_id."""
    # Try per-question directory
    path = PER_Q_DIR / f"{config_id}_per_question.csv"
    if path.exists():
        return pd.read_csv(path)

    # Fallback: look for any CSV with config_id in name
    matches = list(PER_Q_DIR.glob(f"*{config_id}*.csv"))
    if matches:
        return pd.read_csv(matches[0])

    log.warning("No per-question CSV found for config %s", config_id)
    return None


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def run_significance_tests(
    results_csv: Path = RESULTS_CSV,
) -> dict:
    """
    Run paired t-tests and compute Cohen's d between top configs.
    Returns full results dict.

    # PAPER RESULT — TABLE 1
    """
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Ablation results not found at {results_csv}.\n"
            "Run ragops_ablation.py first to generate results."
        )

    df = pd.read_csv(results_csv)
    success_df = df[df["status"] == "success"].copy()

    if success_df.empty:
        raise ValueError("No successful ablation runs found in results CSV.")

    log.info("Loaded %d successful configs from %s", len(success_df), results_csv)

    # ── Rank configs by faithfulness (primary metric) ──────────────────────
    ranked = success_df.sort_values("faithfulness", ascending=False).reset_index(
        drop=True
    )

    best = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else None

    log.info(
        "Best config:   %s (faithfulness=%.4f)", best["config_id"], best["faithfulness"]
    )
    if second is not None:
        log.info(
            "Second config: %s (faithfulness=%.4f)",
            second["config_id"],
            second["faithfulness"],
        )

    # ── Results container ──────────────────────────────────────────────────
    results = {
        "best_config": best["config_id"],
        "second_config": second["config_id"] if second is not None else None,
        "n_configs": len(success_df),
        "aggregate": {},
        "pairwise": {},
        "bonferroni": {},
    }

    # ── Aggregate stats per config ─────────────────────────────────────────
    for _, row in success_df.iterrows():
        cfg_id = row["config_id"]
        results["aggregate"][cfg_id] = {
            m: round(float(row[m]), 4)
            for m in METRIC_KEYS
            if m in row and not pd.isna(row[m])
        }

    # ── Paired t-test: best vs second-best ────────────────────────────────
    if second is not None:
        best_pq = _load_per_question(best["config_id"])
        second_pq = _load_per_question(second["config_id"])

        if best_pq is not None and second_pq is not None:
            log.info("Running paired t-tests on per-question scores...")

            # Align on question index (same 50 questions, same order)
            n = min(len(best_pq), len(second_pq))
            pairwise_results = {}

            # Bonferroni correction for 4 metrics
            alpha_corrected = ALPHA / len(METRIC_KEYS)

            for metric in METRIC_KEYS:
                if metric not in best_pq.columns or metric not in second_pq.columns:
                    continue

                a = best_pq[metric].values[:n]
                b = second_pq[metric].values[:n]

                # Drop NaN pairs
                mask = ~(np.isnan(a) | np.isnan(b))
                a, b = a[mask], b[mask]

                if len(a) < 2:
                    log.warning("Not enough valid pairs for %s", metric)
                    continue

                t_stat, p_val = stats.ttest_rel(a, b)
                d = cohens_d(a, b)
                significant = bool(p_val < alpha_corrected)

                pairwise_results[metric] = {
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "cohens_d": round(d, 4),
                    "effect_size": interpret_d(d),
                    "significant": significant,
                    "alpha_bonf": round(alpha_corrected, 4),
                    "n_pairs": int(len(a)),
                    "best_mean": round(float(np.mean(a)), 4),
                    "second_mean": round(float(np.mean(b)), 4),
                    "mean_diff": round(float(np.mean(a - b)), 4),
                }

                log.info(
                    "%s: t=%.3f, p=%.4f, d=%.3f (%s) %s",
                    metric,
                    t_stat,
                    p_val,
                    d,
                    interpret_d(d),
                    "✓ SIGNIFICANT" if significant else "✗ not significant",
                )

            results["pairwise"] = pairwise_results

        else:
            log.warning(
                "Per-question CSVs not found — running aggregate-level comparison only.\n"
                "For proper paired tests, re-run ablation with per-question logging enabled."
            )
            # Fallback: single-value comparison (not a true paired test)
            results["pairwise"] = _aggregate_fallback(best, second)

    # ── All-vs-best comparison ─────────────────────────────────────────────
    results["ranking"] = _build_ranking(success_df)

    # ── Save outputs ───────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_txt_report(results)
    _save_json_report(results)

    return results


# ---------------------------------------------------------------------------
# Fallback when per-question CSVs are not available
# ---------------------------------------------------------------------------
def _aggregate_fallback(best: pd.Series, second: pd.Series) -> dict:
    """Rough comparison when per-question data is unavailable."""
    out = {}
    for metric in METRIC_KEYS:
        if metric not in best or metric not in second:
            continue
        diff = float(best[metric]) - float(second[metric])
        out[metric] = {
            "best_mean": round(float(best[metric]), 4),
            "second_mean": round(float(second[metric]), 4),
            "mean_diff": round(diff, 4),
            "note": "Aggregate comparison only — per-question CSV not available",
        }
    return out


# ---------------------------------------------------------------------------
# Build ranking table
# ---------------------------------------------------------------------------
def _build_ranking(df: pd.DataFrame) -> list[dict]:
    ranked = df.sort_values("faithfulness", ascending=False).reset_index(drop=True)
    out = []
    for i, row in ranked.iterrows():
        out.append(
            {
                "rank": int(i + 1),
                "config_id": row["config_id"],
                "chunk_size": int(row["chunk_size"]),
                "retriever": row["retriever"],
                **{
                    m: round(float(row[m]), 4)
                    for m in METRIC_KEYS
                    if m in row and not pd.isna(row[m])
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Save text report (paper-ready)
# ---------------------------------------------------------------------------
def _save_txt_report(results: dict) -> None:  # PAPER RESULT — TABLE 1
    lines = []
    lines.append("=" * 70)
    lines.append("  RAGOps — Statistical Significance Report")
    lines.append("  Primary metric: faithfulness (hallucination detection)")
    lines.append("=" * 70)

    lines.append(f"\nBest config:    {results['best_config']}")
    lines.append(f"Second config:  {results['second_config']}")
    lines.append(f"Configs tested: {results['n_configs']}")

    lines.append("\n── Config Ranking (by faithfulness) ──────────────────")

    def _fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    for r in results.get("ranking", []):
        lines.append(
            f"  #{r['rank']} {r['config_id']:4s}  chunk={r['chunk_size']:4d}  "
            f"retriever={r['retriever']:6s}  "
            f"faith={_fmt(r.get('faithfulness', 'N/A'))}  "
            f"recall={_fmt(r.get('context_recall', 'N/A'))}  "
            f"relevance={_fmt(r.get('answer_relevance', 'N/A'))}  "
            f"precision={_fmt(r.get('context_precision', 'N/A'))}"
        )

    if results.get("pairwise"):
        lines.append("\n── Paired t-tests: Best vs Second-Best ───────────────")
        lines.append(f"  α = {0.05:.2f} (Bonferroni corrected per metric)")
        lines.append("")
        for metric, r in results["pairwise"].items():
            if "t_statistic" not in r:
                lines.append(f"  {metric}: {r.get('note', 'N/A')}")
                continue
            sig = "✓ SIGNIFICANT" if r["significant"] else "✗ not significant"
            lines.append(
                f"  {metric:<22}  "
                f"t={r['t_statistic']:+.3f}  "
                f"p={r['p_value']:.4f}  "
                f"d={r['cohens_d']:+.3f} ({r['effect_size']})  "
                f"{sig}"
            )
            lines.append(
                f"  {'':22}  "
                f"best_mean={r['best_mean']:.4f}  "
                f"second_mean={r['second_mean']:.4f}  "
                f"diff={r['mean_diff']:+.4f}"
            )
            lines.append("")

    lines.append("=" * 70)

    with open(REPORT_TXT, "w") as f:
        f.write("\n".join(lines))

    log.info("Text report saved → %s", REPORT_TXT)
    print("\n".join(lines))


def _save_json_report(results: dict) -> None:
    with open(REPORT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info("JSON report saved → %s", REPORT_JSON)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAGOps statistical significance tests"
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=RESULTS_CSV,
        help="Path to ablation_results.csv",
    )
    args = parser.parse_args()

    try:
        results = run_significance_tests(results_csv=args.results_csv)
        print(f"\n✓ Report saved to {REPORT_TXT}")
    except FileNotFoundError as e:
        print(f"\n⚠  {e}")
        print("   Run ragops_ablation.py first to generate results.")
