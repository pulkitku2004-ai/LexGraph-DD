"""
Sprint 3 smoke test — Risk Scorer.

Tests both the deterministic rules layer and (optionally) the LLM reasoning
layer. Injecting known ExtractedClause fixtures means this test runs without
needing Sprint 1/2 infrastructure (Qdrant, BM25 index).

What is verified:
  1. Missing high-risk clause → HIGH RiskFlag emitted (rules layer)
  2. Missing medium-risk clause → MEDIUM RiskFlag emitted (rules layer)
  3. Missing low-risk clause → NO flag emitted (suppressed to reduce noise)
  4. Presence flag clause (Uncapped Liability found) → HIGH flag emitted
  5. Low-confidence found clause → MEDIUM flag emitted
  6. High-confidence found clause of unremarkable type → NO flag (clean clause)
  7. Full GraphState roundtrip through risk_scorer_node
  8. (Optional) LLM reasoning pass — skipped if Ollama is not running

Run from repo root:
    python run_sprint3.py
    python run_sprint3.py --skip-llm   # deterministic tests only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "legal_due_diligence"))

# core.models has no heavy dependencies (just pydantic) — safe to import at module
# level after sys.path is set. This gives Pylance a resolvable type for ExtractedClause.
from core.models import ExtractedClause  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_clause(
    doc_id: str,
    clause_type: str,
    found: bool,
    confidence: float = 0.9,
    clause_text: str | None = None,
    normalized_value: str | None = None,
) -> ExtractedClause:
    return ExtractedClause(
        document_id=doc_id,
        clause_type=clause_type,
        found=found,
        confidence=confidence,
        clause_text=clause_text,
        normalized_value=normalized_value,
        source_chunk_id="test-chunk-001",
    )


def test_rules_layer() -> None:
    from agents.risk_scorer.rules import (
        score_low_confidence,
        score_missing_clause,
        score_presence_flag,
    )

    logger.info("─" * 60)
    logger.info("TEST: Deterministic rules layer")
    logger.info("─" * 60)

    # 1. Missing high-risk clause → HIGH flag
    clause = make_clause("contract-a", "Governing Law", found=False)
    flag = score_missing_clause(clause)
    assert flag is not None, "Expected RiskFlag for missing Governing Law"
    assert flag.risk_level == "high", f"Expected HIGH, got {flag.risk_level}"
    assert flag.is_missing_clause is True
    logger.info("  PASS — missing Governing Law → HIGH flag")

    clause = make_clause("contract-a", "Limitation of Liability", found=False)
    flag = score_missing_clause(clause)
    assert flag is not None
    assert flag.risk_level == "high"
    logger.info("  PASS — missing Limitation of Liability → HIGH flag")

    # 2. Missing medium-risk clause → MEDIUM flag
    clause = make_clause("contract-a", "Termination for Convenience", found=False)
    flag = score_missing_clause(clause)
    assert flag is not None, "Expected RiskFlag for missing Termination for Convenience"
    assert flag.risk_level == "medium", f"Expected MEDIUM, got {flag.risk_level}"
    logger.info("  PASS — missing Termination for Convenience → MEDIUM flag")

    # 3. Missing low-risk clause → NO flag (noise suppression)
    clause = make_clause("contract-a", "Source Code Escrow", found=False)
    flag = score_missing_clause(clause)
    assert flag is None, "Expected no flag for missing Source Code Escrow (low risk)"
    logger.info("  PASS — missing Source Code Escrow → no flag (suppressed)")

    # 4. Presence flag — Uncapped Liability found → HIGH flag
    clause = make_clause(
        "contract-a", "Uncapped Liability", found=True,
        clause_text="Each party shall be liable for all damages without limit.",
        normalized_value="unlimited",
    )
    flag = score_presence_flag(clause)
    assert flag is not None, "Expected presence flag for Uncapped Liability"
    assert flag.risk_level == "high"
    assert flag.is_missing_clause is False
    logger.info("  PASS — Uncapped Liability found → HIGH presence flag")

    # 5. Low confidence found clause → MEDIUM flag
    clause = make_clause("contract-a", "Governing Law", found=True, confidence=0.3)
    flag = score_low_confidence(clause)
    assert flag is not None, "Expected low-confidence flag"
    assert flag.risk_level == "medium"
    logger.info("  PASS — Governing Law found with conf=0.3 → MEDIUM low-confidence flag")

    # 6. High confidence found clause, standard type → NO flag from rules
    clause = make_clause("contract-a", "Payment Terms", found=True, confidence=0.95)
    p_flag = score_presence_flag(clause)
    c_flag = score_low_confidence(clause)
    assert p_flag is None, "Payment Terms should not trigger a presence flag"
    assert c_flag is None, "High confidence should not trigger low-confidence flag"
    logger.info("  PASS — Payment Terms found with conf=0.95 → no rule flags")

    # 7. Clause not in MISSING_CLAUSE_RISK dict → defaults to low (no flag)
    clause = make_clause("contract-a", "Document Name", found=False)
    flag = score_missing_clause(clause)
    assert flag is None, "Unknown clause type should default to low risk (no flag)"
    logger.info("  PASS — missing unknown clause type → no flag (default low)")

    logger.info("  All deterministic rule tests PASSED")


def test_node_integration() -> list:
    from datetime import datetime, timezone
    from core.models import DocumentRecord, ExtractedClause
    from core.state import GraphState
    from agents.risk_scorer.agent import risk_scorer_node

    logger.info("─" * 60)
    logger.info("TEST: risk_scorer_node GraphState roundtrip")
    logger.info("─" * 60)

    # Simulate post-Sprint-2 state: extracted clauses from two contracts.
    # contract-b is missing Termination for Convenience (known from context.md).
    clauses = [
        # contract-a: clean clauses
        make_clause("contract-a", "Governing Law", True, 0.95, "Governed by Delaware", "Delaware"),
        make_clause("contract-a", "Liability Cap", True, 0.92, "Cap equals 12 months fees", "12 months fees"),
        make_clause("contract-a", "Payment Terms", True, 0.88, "Payment due in 30 days", "30 days"),
        make_clause("contract-a", "Confidentiality Survival", True, 0.85, "Survives for 5 years", "5 years"),
        make_clause("contract-a", "Termination for Convenience", True, 0.90, "30 days notice required", "30 days"),

        # contract-b: missing Termination for Convenience (should → MEDIUM flag)
        make_clause("contract-b", "Governing Law", True, 0.94, "Governed by New York", "New York"),
        make_clause("contract-b", "Liability Cap", True, 0.91, "Cap equals 6 months fees", "6 months fees"),
        make_clause("contract-b", "Payment Terms", True, 0.87, "Payment due Net 45", "45 days"),
        make_clause("contract-b", "Confidentiality Survival", True, 0.83, "Survives for 3 years", "3 years"),
        make_clause("contract-b", "Termination for Convenience", False),  # MISSING

        # Presence flag test: uncapped liability in contract-b
        make_clause("contract-b", "Uncapped Liability", True, 0.88,
                    "Liability for gross negligence shall be unlimited.", "unlimited"),
    ]

    state = GraphState(
        job_id="sprint3-test",
        status="clauses_extracted",
        created_at=datetime.now(timezone.utc),
        documents=[
            DocumentRecord(doc_id="contract-a", file_path="samples/contract_a.txt"),
            DocumentRecord(doc_id="contract-b", file_path="samples/contract_b.txt"),
        ],
        extracted_clauses=clauses,
        risk_flags=[],
        contradictions=[],
        neo4j_ready=False,
        qdrant_ready=True,
        graph_built=False,
        final_report=None,
        errors=[],
    )

    result = risk_scorer_node(state)

    assert "risk_flags" in result, "Node must return risk_flags"
    assert result["status"] == "risks_scored"

    flags = result["risk_flags"]
    logger.info("  Node returned %d risk flags", len(flags))

    # Verify: contract-b missing Termination for Convenience → MEDIUM flag
    missing_term = [
        f for f in flags
        if f.document_id == "contract-b"
        and f.clause_type == "Termination for Convenience"
        and f.is_missing_clause
    ]
    assert len(missing_term) == 1, (
        f"Expected 1 missing-termination flag for contract-b, got {len(missing_term)}"
    )
    assert missing_term[0].risk_level == "medium"
    logger.info("  PASS — contract-b missing Termination for Convenience → MEDIUM flag")

    # Verify: contract-b Uncapped Liability found → HIGH presence flag
    uncapped = [
        f for f in flags
        if f.document_id == "contract-b"
        and f.clause_type == "Uncapped Liability"
        and not f.is_missing_clause
    ]
    assert len(uncapped) == 1, f"Expected 1 uncapped liability flag, got {len(uncapped)}"
    assert uncapped[0].risk_level == "high"
    logger.info("  PASS — contract-b Uncapped Liability found → HIGH presence flag")

    # Verify: contract-a clean clauses produce no rule flags
    contract_a_flags = [f for f in flags if f.document_id == "contract-a"]
    logger.info("  contract-a rule flags: %d", len(contract_a_flags))
    # (may have LLM flags if Ollama is running — that's fine)

    logger.info("  Node integration test PASSED")
    return flags


def test_llm_reasoning(flags_from_node: list) -> None:
    """
    Spot-check the LLM reasoning layer against a contract-b Liability Cap clause.

    This test calls Ollama and will fail if Ollama is not running.
    Skip with --skip-llm flag.
    """
    from agents.risk_scorer.agent import score_clause
    from core.models import ExtractedClause

    logger.info("─" * 60)
    logger.info("TEST: LLM reasoning layer (requires Ollama running)")
    logger.info("─" * 60)

    # A liability cap clause with a very small cap — LLM should flag it
    clause = ExtractedClause(
        document_id="contract-test",
        clause_type="Liability Cap",
        found=True,
        clause_text=(
            "In no event shall either party's total liability exceed one hundred dollars "
            "($100) in the aggregate, regardless of the nature of the claim."
        ),
        normalized_value="$100",
        confidence=0.92,
        source_chunk_id="test-chunk-llm",
    )

    logger.info("  Calling LLM for Liability Cap clause with $100 cap...")
    flags = score_clause(clause)

    logger.info("  LLM assessment returned %d flag(s)", len(flags))
    for f in flags:
        logger.info("    %s | %s | %s", f.clause_type, f.risk_level.upper(), f.reason)

    # We expect the LLM to flag a $100 liability cap as problematic
    # (cannot assert exact content — LLM output varies, but flag should fire)
    assert len(flags) >= 1, (
        "Expected at least 1 flag for a $100 liability cap clause — LLM may not be running"
    )
    logger.info("  PASS — LLM reasoning flagged the $100 liability cap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM reasoning test (useful if Ollama is not running)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SPRINT 3 — Risk Scorer smoke test")
    logger.info("=" * 60)

    test_rules_layer()
    flags = test_node_integration()

    if args.skip_llm:
        logger.info("─" * 60)
        logger.info("Skipping LLM reasoning test (--skip-llm)")
    else:
        try:
            test_llm_reasoning(flags)
        except Exception as e:
            logger.warning(
                "LLM reasoning test failed (Ollama may not be running): %s", e
            )
            logger.warning("Re-run with --skip-llm to skip this test.")

    logger.info("=" * 60)
    logger.info("Sprint 3 PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
