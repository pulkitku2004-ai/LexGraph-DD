"""
Contradiction Detector Agent — Sprint 5.

Architecture: Cypher-first, LLM-second.

Step 1 — Structural detection (cypher_queries.py):
  Two Cypher queries against the shared Neo4j graph:
  a. Value conflicts — same clause_type, both found=True, different normalized_value.
     Example: Governing Law = "Delaware" vs "New York".
  b. Absence conflicts — same clause_type, one found=True, one found=False.
     Example: Termination for Convenience present in contract-a, absent in contract-b.

Step 2 — LLM explanation (settings.llm_reasoning_model):
  For each structural contradiction found, call the reasoning model to generate
  a plain-language explanation of the legal risk. This is what a lawyer reads in
  the final brief — the explanation must be actionable, not technical.

  On LLM failure: fall back to a template explanation so the Contradiction object
  is always complete. Conservative: better to have a templated explanation than
  to drop the contradiction from the report.

Why Cypher instead of Python nested loops?
  Python comparison works for pairwise doc comparison but doesn't scale to
  graph-level queries: "find all parties with conflicting obligations across any
  clause type." Cypher handles this as a single traversal. As the document set
  grows to 50 documents (1225 unique pairs), nested Python loops become O(n²)
  per clause type while Cypher uses the composite index on (doc_id, clause_type).

Why not async?
  5 contradictions across 2 sample documents = 5 LLM calls. Sequential execution
  completes in seconds for local Ollama. asyncio adds complexity before Sprint 7
  benchmarks show a bottleneck.

Why catch all exceptions?
  A Neo4j query failure should not crash the pipeline — clause extraction and
  risk scoring have already run. On failure: log, append to state.errors, return
  empty contradictions. The report will note what failed.
"""

from __future__ import annotations

import json
import logging
import re

import litellm

from agents.contradiction_detector.cypher_queries import (
    find_absence_conflicts,
    find_value_conflicts,
)
from agents.risk_scorer.rules import MISSING_CLAUSE_RISK
from core.config import settings
from core.models import Contradiction
from core.state import GraphState
from infrastructure.neo4j_client import get_neo4j_session

# Clause types where an absence conflict is meaningful.
# Low-importance absences (e.g. "Source Code Escrow absent in doc B") are
# extraction noise, not genuine contractual differences.
_ABSENCE_CONFLICT_CATEGORIES: frozenset[str] = frozenset(
    ct for ct, level in MISSING_CLAUSE_RISK.items() if level in ("high", "medium")
)

# ── Value normalization ────────────────────────────────────────────────────────
# The Cypher query catches toLower/trim differences but not LLM extraction
# variance. "thirty days" vs "30 days" is not a real conflict.

_NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "ninety": "90",
}

_MULTIPLIERS: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000, "k": 1_000,
    "million": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "bn": 1_000_000_000,
}

# Currency/unit words to strip before number parsing
_CURRENCY_RE = re.compile(
    r"[$£€¥]|"
    r"\b(dollars?|usd|euros?|eur|pounds?|gbp|cents?)\b",
    re.IGNORECASE,
)
# "1.5 million", "500k", "2bn" — digit(s) immediately followed by multiplier
_NUM_MULT_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(hundred|thousand|k|million|m|billion|bn)$",
    re.IGNORECASE,
)


def _parse_to_canonical_number(text: str) -> str | None:
    """
    Try to parse a monetary/numeric string to a canonical integer string.
    Returns None if the text is not purely a number (e.g. "30 days" → None).

    Handles:
      "$1,000,000"           → "1000000"
      "one million dollars"  → "1000000"  (after _NUMBER_WORDS substitution)
      "$1.5 million"         → "1500000"
      "500k"                 → "500000"
      "1bn"                  → "1000000000"
    """
    t = _CURRENCY_RE.sub("", text).replace(",", "").strip()
    if not t:
        return None

    # "X.Y multiplier" or "X multiplier"
    m = _NUM_MULT_RE.fullmatch(t)
    if m:
        return str(int(float(m.group(1)) * _MULTIPLIERS[m.group(2).lower()]))

    # Plain integer or float with no non-numeric suffix
    try:
        f = float(t)
        # Reject if t still contains letters (e.g. "12 months" won't reach here,
        # but guard against edge cases like "1e6" which we don't want to flatten)
        if re.search(r"[a-zA-Z]", t):
            return None
        return str(int(f))
    except ValueError:
        return None


def _normalize_for_comparison(value: str) -> str:
    """
    Reduce LLM extraction variance before conflict comparison.

    Pass 1 — surface form cleanup:
      - Parenthesized repetitions:  "thirty (30) days" → "thirty days"
      - Number words → digits:      "thirty days" → "30 days"
      - Jurisdiction wrappers:      "State of Delaware" → "Delaware"

    Pass 2 — canonical number:
      - After number-word substitution, attempt to parse the whole string
        as a monetary/numeric value and return a canonical integer string.
      - "$1,000,000" == "one million dollars" == "$1M" → "1000000"
      - "30 days" is NOT a pure number (has "days") → unchanged
    """
    v = value.lower().strip()
    v = re.sub(r"\(\d+\)", "", v)                  # drop "(30)" style repeats
    for word, digit in _NUMBER_WORDS.items():
        v = re.sub(rf"\b{word}\b", digit, v)
    v = re.sub(r"^state of\s+", "", v)             # "State of X" → "X"
    v = re.sub(r"\s+state$", "", v)                # "X State" → "X"
    v = " ".join(v.split())

    # Pass 2: try to collapse to canonical integer (currency/large numbers)
    canonical = _parse_to_canonical_number(v)
    if canonical is not None:
        return canonical

    return v

logger = logging.getLogger(__name__)

# ── LLM explanation configuration ─────────────────────────────────────────────

_EXPLANATION_SYSTEM_PROMPT = """You are a legal due diligence analyst reviewing contract portfolios.
Your job is to assess the risk level and explain the practical legal risk when two contracts have conflicting provisions.

Rules:
- Return ONLY valid JSON. No markdown, no code blocks, no explanation outside the JSON.
- "risk_level" must be "high", "medium", or "low":
    high   — direct financial exposure, enforceability breakdown, or ownership dispute
    medium — operational friction, asymmetric obligations, or compliance uncertainty
    low    — administrative difference with minimal legal consequence
- "explanation" must be one or two sentences: practical consequence for a party involved with both contracts.
- Be specific: name the clause type and the conflicting values."""


def _build_explanation_prompt(
    clause_type: str,
    doc_a: str,
    value_a: str,
    doc_b: str,
    value_b: str,
) -> str:
    return (
        f'These two contracts have a conflicting "{clause_type}" provision:\n'
        f"  - {doc_a}: {value_a}\n"
        f"  - {doc_b}: {value_b}\n\n"
        f'Return JSON with exactly these fields: {{"risk_level": "high"|"medium"|"low", "explanation": "..."}}'
    )


def _call_explanation_llm(prompt: str) -> dict | None:
    """
    Call the reasoning model for a risk-assessed conflict explanation.

    Returns parsed dict with 'risk_level' and 'explanation', or None on failure.
    """
    try:
        response = litellm.completion(
            model=settings.llm_reasoning_model,
            messages=[
                {"role": "system", "content": _EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        logger.error("[contradiction_detector] LLM explanation call failed: %s", e)
        return None


def _generate_explanation(
    clause_type: str,
    doc_id_a: str,
    value_a: str,
    doc_id_b: str,
    value_b: str,
) -> tuple[str, str]:
    """
    Generate explanation + risk_level via LLM, falling back to a template.

    Returns (explanation, risk_level). Template fallback defaults to "medium"
    — conservative, ensures every Contradiction object is complete even when
    Ollama is unreachable.
    """
    prompt = _build_explanation_prompt(clause_type, doc_id_a, value_a, doc_id_b, value_b)
    result = _call_explanation_llm(prompt)
    if result:
        risk_level = result.get("risk_level", "medium")
        if risk_level not in ("high", "medium", "low"):
            risk_level = "medium"
        explanation = result.get("explanation", "").strip()
        if explanation:
            return explanation, risk_level

    # Template fallback
    return (
        f"Conflicting {clause_type}: {doc_id_a} specifies '{value_a}' "
        f"while {doc_id_b} specifies '{value_b}'. Manual review recommended."
    ), "medium"


# ── Core detection logic ───────────────────────────────────────────────────────

def _build_contradictions(doc_ids: list[str]) -> list[Contradiction]:
    """
    Run both Cypher queries scoped to doc_ids and generate LLM explanations.

    Scoped to doc_ids so each job only sees its own documents — the graph
    accumulates data across jobs and an unscoped query would return spurious
    cross-job contradictions.

    Opens one Neo4j session for the queries, closes it before making LLM calls
    (avoids holding the session open during potentially slow LLM calls).
    """
    with get_neo4j_session() as session:
        value_rows = find_value_conflicts(session, doc_ids)
        absence_rows = find_absence_conflicts(session, doc_ids)

    logger.info(
        "[contradiction_detector] Cypher found %d value conflict(s), %d absence conflict(s)",
        len(value_rows),
        len(absence_rows),
    )

    contradictions: list[Contradiction] = []

    # ── Value conflicts ────────────────────────────────────────────────────────
    for row in value_rows:
        clause_type = row["clause_type"]
        doc_id_a = row["doc_id_a"]
        doc_id_b = row["doc_id_b"]
        value_a = row["value_a"]
        value_b = row["value_b"]

        # Fix 2: drop false positives from extraction variance
        # ("thirty days" vs "30 days" is not a real conflict)
        if _normalize_for_comparison(value_a) == _normalize_for_comparison(value_b):
            logger.debug(
                "[contradiction_detector] VALUE skipped (normalizes equal) | %s | %r vs %r",
                clause_type, value_a, value_b,
            )
            continue

        explanation, risk_level = _generate_explanation(clause_type, doc_id_a, value_a, doc_id_b, value_b)
        logger.info(
            "[contradiction_detector] VALUE | %s | %s | %s=%r vs %s=%r",
            risk_level.upper(), clause_type, doc_id_a, value_a, doc_id_b, value_b,
        )
        contradictions.append(
            Contradiction(
                clause_type=clause_type,
                document_id_a=doc_id_a,
                document_id_b=doc_id_b,
                value_a=value_a,
                value_b=value_b,
                explanation=explanation,
                risk_level=risk_level,
            )
        )

    # ── Absence conflicts ──────────────────────────────────────────────────────
    for row in absence_rows:
        clause_type = row["clause_type"]

        # Fix 1: skip low-importance absences — extraction noise dominates there
        if clause_type not in _ABSENCE_CONFLICT_CATEGORIES:
            logger.debug(
                "[contradiction_detector] ABSENCE skipped (low importance) | %s", clause_type
            )
            continue

        doc_id_a = row["doc_id_a"]
        doc_id_b = row["doc_id_b"]
        found_a: bool = row["found_a"]
        raw_a: str | None = row.get("value_a")
        raw_b: str | None = row.get("value_b")

        if found_a:
            value_a = raw_a or "(present, value not extracted)"
            value_b = "(clause absent)"
        else:
            value_a = "(clause absent)"
            value_b = raw_b or "(present, value not extracted)"

        explanation, risk_level = _generate_explanation(clause_type, doc_id_a, value_a, doc_id_b, value_b)
        logger.info(
            "[contradiction_detector] ABSENCE | %s | %s | %s=%r vs %s=%r",
            risk_level.upper(), clause_type, doc_id_a, value_a, doc_id_b, value_b,
        )
        contradictions.append(
            Contradiction(
                clause_type=clause_type,
                document_id_a=doc_id_a,
                document_id_b=doc_id_b,
                value_a=value_a,
                value_b=value_b,
                explanation=explanation,
                risk_level=risk_level,
            )
        )

    return contradictions


# ── LangGraph node ─────────────────────────────────────────────────────────────

def contradiction_detector_node(state: GraphState) -> dict:
    """
    LangGraph node — replaces the Sprint 0 stub.

    Queries Neo4j for value conflicts and absence conflicts across all documents
    in the shared knowledge graph. Generates a plain-language LLM explanation
    for each finding.

    Returns contradictions list and updated status.
    Skips gracefully if graph_built=False (Neo4j was down during entity mapping).
    """
    logger.info(
        "[contradiction_detector] querying Neo4j for contradictions across "
        "%d document(s). Graph built: %s",
        len(state.documents),
        state.graph_built,
    )

    if not state.graph_built:
        logger.warning(
            "[contradiction_detector] graph_built=False — skipping contradiction detection"
        )
        return {"contradictions": [], "status": "contradictions_detected"}

    errors = list(state.errors)
    contradictions: list[Contradiction] = []

    doc_ids = [d.doc_id for d in state.documents]
    try:
        contradictions = _build_contradictions(doc_ids)
    except Exception as e:
        msg = f"[contradiction_detector] Neo4j query failed: {e}"
        logger.error(msg)
        errors.append(msg)
        return {
            "contradictions": [],
            "errors": errors,
            "status": "contradictions_detected",
        }

    logger.info(
        "[contradiction_detector] complete — %d contradiction(s) detected",
        len(contradictions),
    )

    return {
        "contradictions": contradictions,
        "errors": errors,
        "status": "contradictions_detected",
    }
