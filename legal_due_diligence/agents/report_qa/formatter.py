"""
Report formatter — deterministic section builder.

All tables and structured sections are built here in pure Python with no LLM
involvement. The LLM is responsible only for the two narrative sections:
executive summary and recommended actions.

Why split formatter from LLM?

The alternative — having the LLM generate the entire report — risks the model
reformatting tables, misstating risk levels, or truncating rows when the
context is long. Deterministic tables guarantee accuracy; LLM adds prose
quality only where prose quality actually matters.

assemble_report() combines formatter output with LLM-generated narrative into
the final markdown brief. The LLM sections are clearly bounded — if the LLM
call fails, template fallback text keeps the report complete and correct.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.models import Contradiction, RiskFlag
from core.state import GraphState

# Risk level display ordering
_RISK_ORDER = {"high": 0, "medium": 1, "low": 2}


# ── Section builders ───────────────────────────────────────────────────────────

def _format_risk_table(risk_flags: list[RiskFlag]) -> str:
    """
    Build a per-document risk flag table sorted by risk level (HIGH first).
    Returns markdown string. Returns empty-section note if no flags.
    """
    if not risk_flags:
        return "*No risk flags detected.*\n"

    # Group by document
    by_doc: dict[str, list[RiskFlag]] = {}
    for flag in risk_flags:
        by_doc.setdefault(flag.document_id, []).append(flag)

    sections: list[str] = []
    for doc_id in sorted(by_doc):
        flags = sorted(by_doc[doc_id], key=lambda f: _RISK_ORDER.get(f.risk_level, 3))
        lines = [
            f"### {doc_id} ({len(flags)} flag(s))\n",
            "| Clause | Risk | Type | Reason |",
            "|---|---|---|---|",
        ]
        for f in flags:
            flag_type = "Missing" if f.is_missing_clause else "Content"
            reason = f.reason[:120].replace("|", "\\|")
            lines.append(f"| {f.clause_type} | {f.risk_level.upper()} | {flag_type} | {reason} |")
        sections.append("\n".join(lines))

    return "\n\n".join(sections) + "\n"


def _format_contradiction_table(contradictions: list[Contradiction]) -> str:
    """
    Build a cross-document contradiction table sorted by risk level (HIGH first).
    Returns markdown string. Returns empty-section note if none.
    """
    if not contradictions:
        return "*No cross-document contradictions detected.*\n"

    lines = [
        "| Risk | Clause Type | Document A | Value A | Document B | Value B | Explanation |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in sorted(contradictions, key=lambda x: (_RISK_ORDER.get(x.risk_level, 3), x.clause_type)):
        expl = c.explanation[:120].replace("|", "\\|")
        lines.append(
            f"| {c.risk_level.upper()} | {c.clause_type} | {c.document_id_a} | {c.value_a} "
            f"| {c.document_id_b} | {c.value_b} | {expl} |"
        )
    return "\n".join(lines) + "\n"


def _format_missing_clauses(risk_flags: list[RiskFlag]) -> str:
    """
    Build a table of missing clause flags (is_missing_clause=True), sorted by risk.
    Returns markdown string. Returns empty-section note if none.
    """
    missing = [f for f in risk_flags if f.is_missing_clause]
    if not missing:
        return "*No missing clause flags.*\n"

    missing.sort(key=lambda f: (_RISK_ORDER.get(f.risk_level, 3), f.document_id, f.clause_type))
    lines = [
        "| Document | Clause | Risk Level | Reason |",
        "|---|---|---|---|",
    ]
    for f in missing:
        reason = f.reason[:100].replace("|", "\\|")
        lines.append(f"| {f.document_id} | {f.clause_type} | {f.risk_level.upper()} | {reason} |")
    return "\n".join(lines) + "\n"


def _format_processing_notes(errors: list[str]) -> str:
    if not errors:
        return "No errors."
    items = "\n".join(f"- {e}" for e in errors)
    return f"{len(errors)} error(s) encountered:\n{items}"


# ── Narrative prompt ───────────────────────────────────────────────────────────

def build_narrative_prompt(state: GraphState) -> str:
    """
    Build the LLM prompt for generating executive summary + recommended actions.

    The LLM receives a compact structured summary — not the raw state — to
    keep token count bounded and avoid the model losing focus on a 2000-token
    state dump. The LLM returns JSON with two keys.
    """
    docs = ", ".join(d.doc_id for d in state.documents)
    high = sum(1 for f in state.risk_flags if f.risk_level == "high")
    medium = sum(1 for f in state.risk_flags if f.risk_level == "medium")
    low = sum(1 for f in state.risk_flags if f.risk_level == "low")
    missing_count = sum(1 for f in state.risk_flags if f.is_missing_clause)

    # Top HIGH flags — labelled by type so the LLM doesn't conflate missing-clause
    # risks with content risks or contradictions
    high_flags = [f for f in state.risk_flags if f.risk_level == "high"][:5]
    high_summary = "\n".join(
        f"  - [{f.document_id}] {f.clause_type} "
        f"({'MISSING CLAUSE' if f.is_missing_clause else 'CONTENT RISK'}): "
        f"{f.reason[:80]}"
        for f in high_flags
    ) or "  (none)"

    # Contradiction summary — contradictions are conflicts between clauses that
    # EXIST in both documents but have different values. They are NOT missing clauses.
    contradiction_summary = "\n".join(
        f"  - [{c.risk_level.upper()}] {c.clause_type}: {c.document_id_a}={c.value_a!r} vs "
        f"{c.document_id_b}={c.value_b!r}"
        for c in sorted(state.contradictions, key=lambda x: _RISK_ORDER.get(x.risk_level, 3))
    ) or "  (none)"

    return f"""You are preparing a legal due diligence brief for a lawyer.
Based on the structured analysis below, write:
1. An executive summary (2–3 sentences): what documents were reviewed, the overall risk picture, and the most critical finding.
2. A list of 3–5 specific, actionable recommended actions based on the flags and contradictions.

Return ONLY valid JSON with these exact keys:
  "executive_summary": "...",
  "recommended_actions": ["...", "...", ...]

No markdown, no code blocks, no explanation outside the JSON.

IMPORTANT: Contradictions are conflicts between clauses that are PRESENT in both documents
but have different values (e.g. both contracts have a Governing Law clause but specify
different jurisdictions). Do NOT describe a contradiction as a missing clause.

--- ANALYSIS DATA ---

Documents reviewed: {len(state.documents)} — {docs}
Risk flags: {len(state.risk_flags)} total (HIGH={high}, MEDIUM={medium}, LOW={low})
Missing clauses: {missing_count}
Contradictions: {len(state.contradictions)}
Errors: {len(state.errors)}

Top HIGH risk flags (each labelled MISSING CLAUSE or CONTENT RISK):
{high_summary}

Cross-document contradictions (clauses present in both docs but with conflicting values):
{contradiction_summary}
"""


# ── Report assembly ────────────────────────────────────────────────────────────

def assemble_report(
    state: GraphState,
    executive_summary: str,
    recommended_actions: list[str],
) -> str:
    """
    Assemble the final markdown brief from deterministic sections + LLM narrative.

    All tables are built here in Python — the LLM output is only inserted into
    the two narrative slots. This guarantees report correctness even if the LLM
    produces unexpected prose.
    """
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc_list = ", ".join(d.doc_id for d in state.documents)
    high = sum(1 for f in state.risk_flags if f.risk_level == "high")
    medium = sum(1 for f in state.risk_flags if f.risk_level == "medium")

    actions_md = "\n".join(f"- {a}" for a in recommended_actions) or "- No specific actions recommended."

    report = f"""# Due Diligence Brief

**Job ID:** {state.job_id}
**Generated:** {generated_at}
**Documents reviewed:** {len(state.documents)} — {doc_list}
**Risk flags:** {len(state.risk_flags)} (HIGH={high}, MEDIUM={medium})
**Contradictions:** {len(state.contradictions)}{f" (HIGH={sum(1 for c in state.contradictions if c.risk_level == 'high')})" if state.contradictions else ""}

---

## Executive Summary

{executive_summary}

---

## Risk Flags by Document

{_format_risk_table(state.risk_flags)}
---

## Cross-Document Contradictions

{_format_contradiction_table(state.contradictions)}
---

## Missing Clauses Inventory

{_format_missing_clauses(state.risk_flags)}
---

## Recommended Actions

{actions_md}

---

*Processing notes:* {_format_processing_notes(state.errors)}
"""
    return report.strip()
