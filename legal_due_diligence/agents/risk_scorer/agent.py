"""
Risk Scorer Agent — Sprint 3.

Architecture: rules-first, LLM-second.

Pass 1 — Deterministic rules (rules.py):
  - found=False → missing clause risk (always high/medium for key clauses)
  - found=True, presence flag → clause presence is itself a risk signal
  - found=True, confidence < 0.4 → ambiguous clause (medium risk)
  These cases need no LLM: the rule is bright-line, and LLM cost is wasted.

Pass 2 — LLM reasoning (ollama/mistral-nemo, settings.llm_reasoning_model):
  Only for a curated set of high-value clauses where clause_text reveals
  nuance that rules cannot capture:
    - "Is this liability cap amount actually protective?"
    - "Is this indemnification clause one-sided?"
    - "Is this non-compete scope overly broad?"
  The LLM returns JSON: {"flag": bool, "risk_level": str, "reason": str}.
  If flag=False, no RiskFlag is emitted — the clause is considered standard.

Why ollama/mistral-nemo for reasoning?
  Risk flags feed the final brief that a lawyer reads. Wrong risk levels
  have real consequences. We use the local reasoning model (not the fast
  extraction model) for quality. When an Anthropic key is available,
  swap settings.llm_reasoning_model to "anthropic/claude-sonnet-4-6" —
  zero code change.

Why not async?
  Risk scoring touches each clause once. Sequential execution across ~82
  clauses (41 per doc × 2 docs) completes in seconds for local Ollama.
  asyncio would be the right upgrade if Sprint 7 benchmarks show a
  bottleneck, but it adds complexity before we know it's needed.

Why catch all exceptions?
  One LLM failure should not abort scoring for the remaining clauses.
  On exception: log the error, skip LLM flag, let deterministic rules
  stand. Conservative: we'd rather miss an LLM-detected risk than crash.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import litellm

from agents.risk_scorer.rules import (
    CONFIDENCE_THRESHOLD,
    score_low_confidence,
    score_missing_clause,
    score_presence_flag,
)
from core.config import settings
from core.models import ExtractedClause, RiskFlag
from core.state import GraphState

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# ── LLM assessment configuration ──────────────────────────────────────────────
# Clause types that warrant LLM nuance assessment (found=True, conf >= threshold).
# Kept small deliberately — each LLM call costs latency and tokens.
# Only include clauses where clause_text content changes the risk assessment.

LLM_ASSESSMENT_CATEGORIES: frozenset[str] = frozenset({
    "Limitation of Liability",
    "Liability Cap",
    "Indemnification",
    "IP Ownership Assignment",
    "Non-Compete",
    "Governing Law",
    "Termination for Convenience",
    "Confidentiality",
})

_RISK_SYSTEM_PROMPT = """You are a legal risk assessor reviewing individual contract clauses.
Your job is to identify risks in clause language and return a JSON assessment.

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no code blocks.
- "flag" must be true if there is a material risk issue, false if the clause looks standard and protective.
- "risk_level" must be "high", "medium", or "low" — only meaningful when flag=true.
- "reason" must be one concise sentence explaining the risk or why the clause is acceptable.
- Be conservative: flag ambiguous or one-sided language. Do not flag standard mutual obligations."""

# Per-category criteria injected into the assessment prompt.
# Each entry tells the LLM exactly what to look for — without this, the model
# applies generic legal intuition and produces inconsistent risk levels.
_CATEGORY_RISK_CRITERIA: dict[str, str] = {
    "Limitation of Liability": (
        "This clause is about damage TYPE exclusions, not cap amounts (that is Liability Cap). "
        "Flag HIGH if: consequential, indirect, or punitive damages are NOT excluded, leaving "
        "open-ended exposure to loss-of-profit or speculative claims. "
        "Flag MEDIUM if: exclusions are one-sided (only one party gets the protection), or "
        "standard carve-outs for gross negligence, fraud, or wilful misconduct are absent. "
        "Do NOT flag if: consequential/indirect damages are mutually excluded and standard "
        "carve-outs exist. Do NOT comment on cap amounts — that is assessed separately."
    ),
    "Liability Cap": (
        "Flag HIGH if: no specific cap amount or formula is stated, or the cap is less than "
        "3 months of fees. Flag MEDIUM if: the cap applies asymmetrically (one party capped, "
        "other is not), or the cap resets annually giving an illusion of limits. "
        "Do NOT flag if: a reasonable mutual cap (12–24 months of fees or fixed amount) is stated."
    ),
    "Indemnification": (
        "Flag HIGH if: indemnification is one-sided (only one party indemnifies), covers "
        "consequential/indirect losses without a cap, or there is no control-of-defense provision. "
        "Flag MEDIUM if: indemnification scope is unusually broad (e.g. 'any claim arising from "
        "the agreement'), or IP indemnification is absent when IP is being licensed or assigned. "
        "Do NOT flag if: indemnification is mutual, capped, and limited to third-party claims."
    ),
    "IP Ownership Assignment": (
        "Flag HIGH if: all IP created under the agreement (including pre-existing background IP) "
        "is assigned to the counterparty, or the clause is silent on background IP ownership. "
        "Flag MEDIUM if: the assignment scope is ambiguous ('related to the agreement' without "
        "defining scope), or there is no licence-back for background IP. "
        "Do NOT flag if: only foreground IP (created specifically for this contract) is assigned "
        "and background IP is explicitly retained."
    ),
    "Non-Compete": (
        "Flag HIGH if: the non-compete applies post-termination for more than 2 years, covers "
        "an entire industry rather than specific competing products/services, or has no "
        "geographic limitation. Flag MEDIUM if: the scope is broader than the actual business "
        "relationship (e.g. bans all software development when the contract is for one product), "
        "or applies during the term with no carve-out for pre-existing business. "
        "Do NOT flag if: restricted to directly competing products/services, limited duration "
        "(≤1 year post-term), and reasonable geographic scope."
    ),
    "Governing Law": (
        "Flag MEDIUM if: the governing law is a non-standard or obscure jurisdiction "
        "(outside Delaware, New York, California, England & Wales, Singapore, Hong Kong), "
        "or governing law and dispute resolution venue are specified in different jurisdictions "
        "(creates enforcement conflict). "
        "Do NOT flag if: any standard commercial jurisdiction is specified "
        "(Delaware, New York, California, England & Wales, Singapore, Hong Kong) — "
        "jurisdiction preference is not a risk. Do NOT flag solely because the jurisdiction "
        "differs from what you assume the reviewer's home state to be."
    ),
    "Termination for Convenience": (
        "Flag HIGH if: only one party has termination-for-convenience rights (asymmetric exit). "
        "Flag MEDIUM if: the notice period is unusually long (>90 days), there is a termination "
        "fee or penalty for exercising this right, or payment obligations continue past the "
        "termination effective date without a pro-rata wind-down. "
        "Do NOT flag if: both parties have mutual termination-for-convenience rights with a "
        "reasonable notice period (30–60 days) and no penalty."
    ),
    "Confidentiality": (
        "Flag HIGH if: confidentiality obligations are one-sided (only one party is bound), "
        "there is no time limit on obligations (perpetual confidentiality of operational data "
        "is unreasonable), or standard exclusions (publicly known, independently developed, "
        "received from third party) are absent. Flag MEDIUM if: the obligation term exceeds "
        "5 years for general business information, or the definition of confidential information "
        "is so broad it captures publicly available information. "
        "Do NOT flag if: obligations are mutual, duration is 2–5 years, and standard exclusions "
        "are present."
    ),
}


def _build_assessment_prompt(clause: ExtractedClause) -> str:
    criteria = _CATEGORY_RISK_CRITERIA.get(clause.clause_type, "")
    criteria_block = f"\nAssessment criteria for this clause type:\n{criteria}\n" if criteria else ""
    return f"""Assess this "{clause.clause_type}" clause for legal risk.
{criteria_block}
Document: {clause.document_id}
Clause text: {clause.clause_text or "(not available)"}
Extracted value: {clause.normalized_value or "(not available)"}
Extraction confidence: {clause.confidence:.2f}

Return JSON with exactly these fields:
{{
  "flag": true or false,
  "risk_level": "high", "medium", or "low",
  "reason": "one sentence explanation"
}}"""


def _call_reasoning_llm(prompt: str) -> Optional[str]:
    """
    Call the reasoning LLM for nuanced clause risk assessment.

    Uses settings.llm_reasoning_model (default: ollama/mistral-nemo).
    No fallback chain needed — Ollama is local with no rate limit.
    temperature=0.0 for deterministic, auditable output.
    max_tokens=150 — JSON response never exceeds 100 tokens.
    """
    try:
        response = litellm.completion(
            model=settings.llm_reasoning_model,
            messages=[
                {"role": "system", "content": _RISK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        return response.choices[0].message.content  # type: ignore[union-attr]
    except Exception as e:
        logger.error("[risk_scorer] LLM call failed for reasoning: %s", e)
        return None


def _parse_llm_assessment(
    raw: Optional[str],
    clause: ExtractedClause,
) -> Optional[RiskFlag]:
    """
    Parse LLM JSON assessment into a RiskFlag, or None if no flag warranted.

    Returns None on parse failure — conservative: skip LLM flag, let
    deterministic rules stand.
    """
    if raw is None:
        return None

    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        data = json.loads(text)
        if not data.get("flag", False):
            return None  # LLM found no risk — clause is standard

        risk_level = data.get("risk_level", "medium")
        if risk_level not in ("high", "medium", "low"):
            risk_level = "medium"

        return RiskFlag(
            document_id=clause.document_id,
            clause_type=clause.clause_type,
            risk_level=risk_level,
            reason=data.get("reason", "LLM assessment flagged this clause."),
            is_missing_clause=False,
            source_clause_id=clause.source_chunk_id,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(
            "[risk_scorer] JSON parse failed for LLM assessment %s/%s: %s | raw=%s",
            clause.document_id, clause.clause_type, e, (raw or "")[:120],
        )
        return None


def score_clause(clause: ExtractedClause) -> list[RiskFlag]:
    """
    Score one ExtractedClause through both rule and LLM passes.

    Returns a list of RiskFlags (usually 0 or 1, occasionally 2 if both
    a presence flag and an LLM flag fire for the same clause).

    Pass order matters: deterministic rules run first. If a deterministic
    rule fires AND the clause is in LLM_ASSESSMENT_CATEGORIES, the LLM
    still runs — it may surface additional nuance the rule doesn't capture
    (e.g. "missing indemnification" is high risk; but if it was found with
    very one-sided language, the LLM adds a second flag for that too).
    """
    flags: list[RiskFlag] = []

    if not clause.found:
        # ── Pass 1a: missing clause ────────────────────────────────────────
        flag = score_missing_clause(clause)
        if flag:
            flags.append(flag)
        # Missing clause → no clause_text to assess; skip LLM
        return flags

    # ── Pass 1b: presence-based deterministic flag ─────────────────────────
    presence_flag = score_presence_flag(clause)
    if presence_flag:
        flags.append(presence_flag)

    # ── Pass 1c: low confidence flag ───────────────────────────────────────
    conf_flag = score_low_confidence(clause)
    if conf_flag:
        flags.append(conf_flag)
        # Low confidence means clause_text may be unreliable; skip LLM
        return flags

    # ── Pass 2: LLM nuance assessment ─────────────────────────────────────
    if clause.clause_type in LLM_ASSESSMENT_CATEGORIES and clause.clause_text:
        prompt = _build_assessment_prompt(clause)
        raw = _call_reasoning_llm(prompt)
        llm_flag = _parse_llm_assessment(raw, clause)
        if llm_flag:
            logger.info(
                "[risk_scorer] LLM flagged %s/%s as %s",
                clause.document_id, clause.clause_type, llm_flag.risk_level,
            )
            flags.append(llm_flag)

    return flags


def risk_scorer_node(state: GraphState) -> dict:
    """
    LangGraph node — replaces the Sprint 0 stub.

    Scores every ExtractedClause in state.extracted_clauses.
    Emits one or more RiskFlags per clause that has issues.
    Clauses with no issues produce no flags (good clauses are silent).

    Returns dict with risk_flags and updated status.
    """
    clauses = state.extracted_clauses
    logger.info("[risk_scorer] scoring %d extracted clause(s)", len(clauses))

    if not clauses:
        logger.warning("[risk_scorer] no extracted clauses to score — skipping")
        return {"risk_flags": [], "status": "risks_scored"}

    all_flags: list[RiskFlag] = []
    llm_call_count = 0
    rule_flag_count = 0

    for clause in clauses:
        try:
            flags = score_clause(clause)
        except Exception as e:
            logger.error(
                "[risk_scorer] unexpected error scoring %s/%s: %s",
                clause.document_id, clause.clause_type, e,
            )
            continue

        for flag in flags:
            # Count by source for the summary log
            if flag.is_missing_clause or clause.clause_type not in LLM_ASSESSMENT_CATEGORIES:
                rule_flag_count += 1
            else:
                llm_call_count += 1

        if flags:
            for flag in flags:
                logger.info(
                    "[risk_scorer] %s | %s | %s | missing=%s | %s",
                    flag.document_id,
                    flag.clause_type,
                    flag.risk_level.upper(),
                    flag.is_missing_clause,
                    flag.reason[:80],
                )
        all_flags.extend(flags)

    high = sum(1 for f in all_flags if f.risk_level == "high")
    medium = sum(1 for f in all_flags if f.risk_level == "medium")
    low = sum(1 for f in all_flags if f.risk_level == "low")

    logger.info(
        "[risk_scorer] complete — %d flags total (HIGH=%d, MEDIUM=%d, LOW=%d) | "
        "rule-based=%d llm-assessed=%d",
        len(all_flags), high, medium, low, rule_flag_count, llm_call_count,
    )

    return {
        "risk_flags": all_flags,
        "status": "risks_scored",
    }
