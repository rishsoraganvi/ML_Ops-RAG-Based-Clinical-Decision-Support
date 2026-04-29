"""
query_processor.py — Medical-domain query preprocessing.

Expands common cardiology/clinical abbreviations (and, optionally, MeSH
synonyms) so that downstream retrieval sees both the user's shorthand and
the canonical form. Terms are **appended** (not substituted) so that
BM25 term attribution still sees the original tokens.

Deterministic and idempotent — safe to call twice on the same string.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain dictionaries
# ---------------------------------------------------------------------------

# Common cardiology / internal-medicine abbreviations.
# Key = short form (matched case-insensitively on word boundaries).
# Value = canonical long form appended in parentheses.
MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    "MI": "myocardial infarction",
    "HF": "heart failure",
    "HTN": "hypertension",
    "T2DM": "type 2 diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "CVD": "cardiovascular disease",
    "AFib": "atrial fibrillation",
    "CAD": "coronary artery disease",
    "COPD": "chronic obstructive pulmonary disease",
    "CKD": "chronic kidney disease",
}

# Small curated MeSH-style synonym bank. Off by default — opt in via the
# `expand_mesh=True` flag. Keys are lowercased canonical terms.
MESH_SYNONYMS: Dict[str, List[str]] = {
    "diabetes": ["diabetes mellitus", "hyperglycemia"],
    "hypertension": ["high blood pressure"],
    "heart failure": ["cardiac failure", "congestive heart failure"],
    "cancer": ["neoplasm", "malignancy"],
    "stroke": ["cerebrovascular accident"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_query(
    question: str,
    expand_abbreviations: bool = True,
    expand_mesh: bool = False,
) -> str:
    """
    Append canonical expansions of medical abbreviations / MeSH synonyms.

    The original tokens are preserved; expansions are appended in
    parentheses so BM25 term attribution, highlighting, and debug logs
    still reflect the user's original phrasing.

    Args:
        question:             Raw user question.
        expand_abbreviations: Append long forms for abbreviations in
                              ``MEDICAL_ABBREVIATIONS``.
        expand_mesh:          Append MeSH synonyms from ``MESH_SYNONYMS``.

    Returns:
        The (possibly) augmented query string. Deterministic and
        idempotent — running it twice produces the same output as
        running it once.
    """
    if not question or not question.strip():
        return question

    expansions: List[str] = []
    seen: set[str] = set()
    lowered = question.lower()

    if expand_abbreviations:
        # Sort for determinism (dict iteration order is insertion order in
        # 3.7+, but explicit sort protects against future edits).
        for abbrev in sorted(MEDICAL_ABBREVIATIONS):
            long_form = MEDICAL_ABBREVIATIONS[abbrev]
            # Word-boundary regex, case-insensitive.
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            if re.search(pattern, question, flags=re.IGNORECASE):
                # Idempotence: skip if long form already present.
                if long_form.lower() in lowered:
                    continue
                if long_form not in seen:
                    expansions.append(long_form)
                    seen.add(long_form)

    if expand_mesh:
        for term in sorted(MESH_SYNONYMS):
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, question, flags=re.IGNORECASE):
                for syn in MESH_SYNONYMS[term]:
                    if syn.lower() in lowered:
                        continue
                    if syn not in seen:
                        expansions.append(syn)
                        seen.add(syn)

    if not expansions:
        return question

    appendix = " (" + "; ".join(expansions) + ")"
    return question.rstrip() + appendix
