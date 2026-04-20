"""
Risk Scorer — deterministic rules layer.

This module encodes bright-line legal risk rules that do not require LLM
judgment. It runs before any LLM call in the risk scorer pipeline.

Two categories of rules:

1. MISSING_CLAUSE_RISK — absent clauses that are always a risk.
   "No Governing Law clause" is high risk regardless of context. An LLM
   cannot add useful nuance here, so we skip the call entirely.

2. PRESENCE_FLAGS — clauses where finding the clause IS the risk.
   "Uncapped Liability" found in a contract → high risk.
   "Joint IP Ownership" found → medium risk (neither party can license alone).
   These are inverted: absence is neutral, presence is the flag.

Why encode these as data (dicts) rather than if/elif chains?
- Easier to audit: a lawyer can read the dict and verify coverage.
- Easier to extend: add a new rule without touching control flow.
- Easier to test: pass a clause_type, assert the output.

Default risk level for missing clauses not listed: "low".
This is conservative — most CUAD categories are informational for some
contract types. We only escalate to medium/high for clauses with universal
legal significance.
"""

from __future__ import annotations

from typing import Literal

from core.models import ExtractedClause, RiskFlag

# ── Missing clause risk levels ─────────────────────────────────────────────────
# Clause absent (found=False) → emit RiskFlag at this level.
# Clauses not listed default to "low".

MISSING_CLAUSE_RISK: dict[str, Literal["high", "medium", "low"]] = {
    # High risk — legal exposure is immediate or unlimited without these clauses
    "Governing Law":               "high",   # no jurisdiction = enforcement ambiguity
    "Limitation of Liability":     "high",   # unlimited liability exposure
    "Liability Cap":               "high",   # unlimited monetary exposure
    "Indemnification":             "high",   # no indemnification = unknown cost bearing
    "IP Ownership Assignment":     "high",   # ownership unclear → disputes
    "Confidentiality":             "high",   # no data protection obligations

    # Medium risk — operationally important but not immediately catastrophic
    "Termination for Convenience": "medium", # locked in without exit right
    "Termination for Cause":       "medium", # no defined breach remedy mechanism
    "Payment Terms":               "medium", # payment timing undefined
    "Dispute Resolution":          "medium", # no agreed resolution process
    "Anti-Assignment":             "medium", # third-party assignment risk
    "Change of Control":           "medium", # silent on M&A events
    "No-Solicit of Employees":     "medium", # talent poaching unaddressed
    "Warranty Duration":           "medium", # warranty scope unclear

    # Low risk — useful but absence is not immediately dangerous
    "Renewal Term":                "low",
    "Notice Period to Terminate Renewal": "low",
    "Non-Compete":                 "low",
    "Non-Disparagement":           "low",
    "Audit Rights":                "low",
    "Post-Termination Services":   "low",
    "Third Party Beneficiary":     "low",
    "Confidentiality Survival":    "low",
    "Revenue/Profit Sharing":      "low",
    "Price Restrictions":          "low",
    "Minimum Commitment":          "low",
    "Volume Restriction":          "low",
    "Liquidated Damages":          "low",
    "Source Code Escrow":          "low",
    "IP Restriction":              "low",
    "Product Warranty":            "low",
    "Non-Transferable License":    "low",
    "Irrevocable or Perpetual License": "low",
    "Joint IP Ownership":          "low",
    "License Grant":               "low",
    "Exclusivity":                 "low",
    "No-Solicit of Customers":     "low",
}

# Default for any clause_type not listed above
_MISSING_CLAUSE_DEFAULT: Literal["low"] = "low"


# ── Presence flags — finding the clause IS the risk ────────────────────────────
# Maps clause_type → (risk_level, reason template).
# Triggered when found=True, regardless of content details.
# Content-level nuance (e.g. "how uncapped?") is handled by the LLM layer.

PRESENCE_FLAGS: dict[str, tuple[Literal["high", "medium", "low"], str]] = {
    "Uncapped Liability": (
        "high",
        "Uncapped liability clause found — explicit unlimited liability exposure. "
        "Review scope carefully; gross negligence and fraud carve-outs are standard "
        "but any broader uncapped obligation is high risk.",
    ),
    "Joint IP Ownership": (
        "medium",
        "Joint IP ownership clause found — neither party can license or assign the "
        "jointly owned IP without the other's consent. This restricts commercialization "
        "options and can deadlock product development.",
    ),
    "Liquidated Damages": (
        "medium",
        "Liquidated damages clause found — predetermined penalty amounts can exceed "
        "actual harm and may be construed as punitive. Verify the amounts are a "
        "reasonable pre-estimate of loss.",
    ),
    "Irrevocable or Perpetual License": (
        "medium",
        "Irrevocable or perpetual license clause found — once granted, this license "
        "cannot be revoked even on breach or termination. Review scope carefully.",
    ),
}

# ── Low-confidence threshold ───────────────────────────────────────────────────
# found=True but confidence below this → "clause exists but is ambiguous" flag.
CONFIDENCE_THRESHOLD = 0.4


def score_missing_clause(clause: ExtractedClause) -> RiskFlag | None:
    """
    Return a RiskFlag for an absent clause, or None if absence is low risk.

    We skip emitting low-risk missing-clause flags to avoid drowning the
    report in noise. Many CUAD categories are simply not applicable to a
    given contract type (e.g. Source Code Escrow in an MSA).

    Args:
        clause: An ExtractedClause with found=False.

    Returns:
        RiskFlag if the missing clause warrants attention, else None.
    """
    risk_level = MISSING_CLAUSE_RISK.get(clause.clause_type, _MISSING_CLAUSE_DEFAULT)

    # Only emit flags for high and medium — low-risk absences are informational,
    # not actionable. They would be reported in the "not found" summary anyway.
    if risk_level == "low":
        return None

    return RiskFlag(
        document_id=clause.document_id,
        clause_type=clause.clause_type,
        risk_level=risk_level,
        reason=(
            f"{clause.clause_type} clause is absent. "
            f"This is {risk_level} risk: "
            + _missing_reason(clause.clause_type, risk_level)
        ),
        is_missing_clause=True,
        source_clause_id=clause.source_chunk_id,
    )


def score_presence_flag(clause: ExtractedClause) -> RiskFlag | None:
    """
    Return a RiskFlag if the clause's presence is itself a risk indicator.

    Args:
        clause: An ExtractedClause with found=True.

    Returns:
        RiskFlag if the clause type triggers a presence-based rule, else None.
    """
    if clause.clause_type not in PRESENCE_FLAGS:
        return None

    risk_level, reason = PRESENCE_FLAGS[clause.clause_type]
    return RiskFlag(
        document_id=clause.document_id,
        clause_type=clause.clause_type,
        risk_level=risk_level,
        reason=reason,
        is_missing_clause=False,
        source_clause_id=clause.source_chunk_id,
    )


def score_low_confidence(clause: ExtractedClause) -> RiskFlag | None:
    """
    Return a medium-risk flag if a found clause has suspiciously low confidence.

    A low-confidence extraction means the clause text is ambiguous, poorly
    drafted, or the model was uncertain. Ambiguity in contract language is
    itself a legal risk — courts may interpret it against the drafter.

    Only fires for clause types with medium or high missing-clause risk,
    since ambiguity matters more in high-stakes clause types.

    Args:
        clause: An ExtractedClause with found=True and confidence < threshold.

    Returns:
        RiskFlag(medium) or None.
    """
    if clause.confidence >= CONFIDENCE_THRESHOLD:
        return None

    # Only flag low confidence for clauses that matter
    risk_level = MISSING_CLAUSE_RISK.get(clause.clause_type, _MISSING_CLAUSE_DEFAULT)
    if risk_level == "low":
        return None

    return RiskFlag(
        document_id=clause.document_id,
        clause_type=clause.clause_type,
        risk_level="medium",
        reason=(
            f"{clause.clause_type} clause found but with low extraction confidence "
            f"({clause.confidence:.2f}). The clause language may be ambiguous, "
            f"buried in boilerplate, or poorly drafted. Manual review recommended."
        ),
        is_missing_clause=False,
        source_clause_id=clause.source_chunk_id,
    )


def _missing_reason(clause_type: str, risk_level: Literal["high", "medium", "low"]) -> str:
    """Human-readable explanation for why a missing clause is risky."""
    reasons: dict[str, str] = {
        "Governing Law":               "no jurisdiction is specified — enforceability and choice of remedies are uncertain.",
        "Limitation of Liability":     "without a liability cap, exposure to consequential damages is unlimited.",
        "Liability Cap":               "no monetary ceiling on liability — any breach claim is uncapped.",
        "Indemnification":             "responsibility for third-party claims and legal costs is unallocated.",
        "IP Ownership Assignment":     "ownership of work product and deliverables is legally ambiguous.",
        "Confidentiality":             "no contractual obligation to protect confidential information.",
        "Termination for Convenience": "neither party has an exit right — both are locked in until breach or expiry.",
        "Termination for Cause":       "no defined process for terminating on material breach — dispute likely.",
        "Payment Terms":               "payment timing is undefined — cash flow and late payment remedies unclear.",
        "Dispute Resolution":          "no agreed resolution process — litigation venue and procedure uncertain.",
        "Anti-Assignment":             "either party may assign the contract without consent.",
        "Change of Control":           "the contract is silent on M&A events — assignment to acquirer is unaddressed.",
        "No-Solicit of Employees":     "no restriction on poaching key personnel during or after the engagement.",
        "Warranty Duration":           "warranty period and scope are undefined.",
    }
    return reasons.get(clause_type, f"absence of this clause introduces legal uncertainty.")
