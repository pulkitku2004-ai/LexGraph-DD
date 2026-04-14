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

import logging
from typing import Optional

import litellm

from agents.contradiction_detector.cypher_queries import (
    find_absence_conflicts,
    find_value_conflicts,
)
from core.config import settings
from core.models import Contradiction
from core.state import GraphState
from infrastructure.neo4j_client import get_neo4j_session

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# ── LLM explanation configuration ─────────────────────────────────────────────

_EXPLANATION_SYSTEM_PROMPT = """You are a legal due diligence analyst reviewing contract portfolios.
Your job is to explain the practical legal risk created when two contracts have conflicting provisions.

Rules:
- Return ONLY plain text. No markdown, no bullet points, no headers.
- One or two sentences maximum.
- Focus on the practical consequence for a party involved with both contracts.
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
        f"Explain in one or two sentences the legal risk this conflict creates."
    )


def _call_explanation_llm(prompt: str) -> Optional[str]:
    """
    Call the reasoning model for a plain-language conflict explanation.

    Uses settings.llm_reasoning_model (default: ollama/mistral-nemo).
    temperature=0.0 — deterministic, auditable output.
    max_tokens=150 — explanation is one or two sentences.
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("[contradiction_detector] LLM explanation call failed: %s", e)
        return None


def _generate_explanation(
    clause_type: str,
    doc_id_a: str,
    value_a: str,
    doc_id_b: str,
    value_b: str,
) -> str:
    """
    Generate a plain-language explanation via LLM, falling back to a template.

    The template fallback ensures every Contradiction object has a non-empty
    explanation even when Ollama is unreachable or the LLM call fails.
    """
    prompt = _build_explanation_prompt(clause_type, doc_id_a, value_a, doc_id_b, value_b)
    result = _call_explanation_llm(prompt)
    if result:
        return result
    # Template fallback — always produces a meaningful (if generic) explanation
    return (
        f"Conflicting {clause_type}: {doc_id_a} specifies '{value_a}' "
        f"while {doc_id_b} specifies '{value_b}'. Manual review recommended."
    )


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

        explanation = _generate_explanation(clause_type, doc_id_a, value_a, doc_id_b, value_b)
        logger.info(
            "[contradiction_detector] VALUE | %s | %s=%r vs %s=%r",
            clause_type, doc_id_a, value_a, doc_id_b, value_b,
        )
        contradictions.append(
            Contradiction(
                clause_type=clause_type,
                document_id_a=doc_id_a,
                document_id_b=doc_id_b,
                value_a=value_a,
                value_b=value_b,
                explanation=explanation,
            )
        )

    # ── Absence conflicts ──────────────────────────────────────────────────────
    for row in absence_rows:
        clause_type = row["clause_type"]
        doc_id_a = row["doc_id_a"]
        doc_id_b = row["doc_id_b"]
        found_a: bool = row["found_a"]
        raw_a: Optional[str] = row.get("value_a")
        raw_b: Optional[str] = row.get("value_b")

        if found_a:
            value_a = raw_a or "(present, value not extracted)"
            value_b = "(clause absent)"
        else:
            value_a = "(clause absent)"
            value_b = raw_b or "(present, value not extracted)"

        explanation = _generate_explanation(clause_type, doc_id_a, value_a, doc_id_b, value_b)
        logger.info(
            "[contradiction_detector] ABSENCE | %s | %s=%r vs %s=%r",
            clause_type, doc_id_a, value_a, doc_id_b, value_b,
        )
        contradictions.append(
            Contradiction(
                clause_type=clause_type,
                document_id_a=doc_id_a,
                document_id_b=doc_id_b,
                value_a=value_a,
                value_b=value_b,
                explanation=explanation,
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
