"""
Report & Q&A Agent — Sprint 6.

Architecture: formatter-first, LLM-second.

Step 1 — Deterministic sections (formatter.py):
  All tables are built in pure Python from GraphState:
  - Risk flags table grouped by document, sorted HIGH → MEDIUM → LOW
  - Cross-document contradictions table
  - Missing clauses inventory
  - Processing notes / error list

Step 2 — LLM narrative (settings.llm_reasoning_model):
  One LLM call produces two narrative sections as JSON:
  - executive_summary: 2–3 sentences covering documents reviewed + key risk picture
  - recommended_actions: 3–5 specific, actionable bullets

Step 3 — Assembly (formatter.assemble_report):
  Narrative sections are inserted into the report template alongside the
  deterministic tables. The final markdown brief is stored in state.final_report.

Why formatter-first instead of one big LLM call?
  A fully LLM-generated report risks misstated risk levels, truncated table rows,
  and fabricated clause text when context is long. The deterministic tables are
  always correct. The LLM adds value only in the two narrative slots where prose
  quality matters and factual accuracy is structurally constrained by the context.

Why Sonnet / reasoning model for the narrative, not the extraction model?
  The report is the deliverable — a lawyer reads this. Haiku/fast-inference models
  produce acceptable JSON for structured extraction but their prose quality and
  legal reasoning fall short for an executive summary a partner will sign off on.
  Swap LLM_REASONING_MODEL=anthropic/claude-sonnet-4-6 in .env for production.

Why catch all exceptions?
  Report generation fails after all other agents have already run successfully.
  A crash here loses the entire pipeline output. On any failure: log, use
  template fallback narrative, assemble the report from deterministic sections.
  A report with template executive summary is far more useful than no report.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import litellm

from agents.report_qa.formatter import (
    assemble_report,
    build_narrative_prompt,
)
from core.config import settings
from core.state import GraphState

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# ── LLM narrative generation ───────────────────────────────────────────────────

_REPORT_SYSTEM_PROMPT = """You are a legal due diligence analyst preparing a brief for a senior lawyer.
Return ONLY valid JSON with these exact keys:
  "executive_summary": one string, 2–3 sentences
  "recommended_actions": array of 3–5 strings, each a specific actionable instruction

No markdown, no code blocks, no extra keys, no explanation outside the JSON object."""


def _call_narrative_llm(prompt: str) -> Optional[str]:
    """
    Call the reasoning model for executive summary + recommended actions.
    temperature=0.0 — deterministic, auditable.
    max_tokens=500 — summary (100t) + 5 actions (~80t each) = ~500t.
    """
    try:
        response = litellm.completion(
            model=settings.llm_reasoning_model,
            messages=[
                {"role": "system", "content": _REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content  # type: ignore[union-attr]
    except Exception as e:
        logger.error("[report_qa] LLM narrative call failed: %s", e)
        return None


def _parse_narrative(raw: Optional[str], state: GraphState) -> tuple[str, list[str]]:
    """
    Parse LLM JSON into (executive_summary, recommended_actions).
    Falls back to template text on any failure so the report always assembles.
    """
    if raw is None:
        return _template_narrative(state)

    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        data = json.loads(text)
        summary = str(data.get("executive_summary", "")).strip()
        actions = [str(a).strip() for a in data.get("recommended_actions", []) if a]
        if summary and actions:
            return summary, actions
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("[report_qa] JSON parse failed for narrative: %s | raw=%s", e, (raw or "")[:120])

    return _template_narrative(state)


def _template_narrative(state: GraphState) -> tuple[str, list[str]]:
    """
    Template fallback when the LLM call fails or returns unparseable output.
    Produces a factually accurate (if generic) summary from state data.
    """
    high = sum(1 for f in state.risk_flags if f.risk_level == "high")
    medium = sum(1 for f in state.risk_flags if f.risk_level == "medium")
    docs = ", ".join(d.doc_id for d in state.documents)

    summary = (
        f"This review covered {len(state.documents)} document(s): {docs}. "
        f"The analysis identified {len(state.risk_flags)} risk flag(s) "
        f"({high} high, {medium} medium) and {len(state.contradictions)} "
        f"cross-document contradiction(s). Detailed findings are in the sections below."
    )

    actions: list[str] = []
    if high > 0:
        actions.append(f"Address {high} HIGH-severity risk flag(s) before execution.")
    if state.contradictions:
        actions.append(
            f"Reconcile {len(state.contradictions)} cross-document contradiction(s) — "
            "conflicting provisions must be resolved to determine which terms govern."
        )
    missing = [f for f in state.risk_flags if f.is_missing_clause and f.risk_level == "high"]
    if missing:
        clauses = ", ".join(dict.fromkeys(f.clause_type for f in missing))
        actions.append(f"Add missing high-risk clauses: {clauses}.")
    if not actions:
        actions.append("Review medium-severity flags for completeness before signing.")

    return summary, actions


# ── LangGraph node ─────────────────────────────────────────────────────────────

def report_qa_node(state: GraphState) -> dict:
    """
    LangGraph node — replaces the Sprint 0 stub.

    Synthesizes a structured due diligence brief as markdown from all agent
    outputs accumulated in state. The brief is stored in state.final_report.
    """
    logger.info(
        "[report_qa] synthesizing brief from: %d clause(s), %d flag(s), %d contradiction(s)",
        len(state.extracted_clauses),
        len(state.risk_flags),
        len(state.contradictions),
    )

    errors = list(state.errors)

    try:
        # Step 1 — build narrative prompt (deterministic, no I/O)
        prompt = build_narrative_prompt(state)

        # Step 2 — LLM call for narrative sections
        raw = _call_narrative_llm(prompt)
        executive_summary, recommended_actions = _parse_narrative(raw, state)

        # Step 3 — assemble full markdown report
        report = assemble_report(state, executive_summary, recommended_actions)

    except Exception as e:
        msg = f"[report_qa] unexpected error during synthesis: {e}"
        logger.error(msg)
        errors.append(msg)
        # Assemble with template fallback — never return None as final_report
        summary, actions = _template_narrative(state)
        report = assemble_report(state, summary, actions)

    logger.info("[report_qa] brief assembled — %d chars", len(report))

    return {
        "final_report": report,
        "errors": errors,
        "status": "complete",
    }
