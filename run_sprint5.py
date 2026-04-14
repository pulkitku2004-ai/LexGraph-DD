"""
Sprint 5 smoke test — Contradiction Detector.

What is verified:
  1. Node skips gracefully when graph_built=False
     a. Returns contradictions=[] and status="contradictions_detected"

  2. Cypher query correctness (requires Neo4j running)
     a. Seed Neo4j with Sprint 4 sample clauses (same fixture)
     b. Run contradiction_detector_node via GraphState
     c. Assert 4 value conflicts found:
        - Governing Law   (Delaware vs New York)
        - Liability Cap   (12 months fees vs 6 months fees)
        - Payment Terms   (30 days vs 45 days)
        - Confidentiality (5 years vs 3 years)
     d. Assert 1 absence conflict found:
        - Termination for Convenience (present in contract-a, absent in contract-b)
     e. Assert all 5 Contradiction objects have non-empty explanations
     f. Assert Contradiction model fields are correctly populated

Run from repo root:
    python run_sprint5.py
    python run_sprint5.py --skip-neo4j    # unit/stub tests only, no Docker required
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "legal_due_diligence"))

from core.models import DocumentRecord, ExtractedClause  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Fixtures (same as Sprint 4) ────────────────────────────────────────────────

def make_clause(
    doc_id: str,
    clause_type: str,
    found: bool,
    normalized_value: str | None = None,
    clause_text: str | None = None,
    confidence: float = 0.90,
) -> ExtractedClause:
    return ExtractedClause(
        document_id=doc_id,
        clause_type=clause_type,
        found=found,
        normalized_value=normalized_value,
        clause_text=clause_text,
        confidence=confidence,
        source_chunk_id="test-chunk-sprint5",
    )


SAMPLE_CLAUSES = [
    # contract-a
    make_clause("contract-a", "Governing Law", True, "Delaware",
                "This Agreement shall be governed by the laws of the State of Delaware. "
                "Acme Corp. and TechVentures Inc. agree to submit to jurisdiction."),
    make_clause("contract-a", "Liability Cap", True, "12 months fees",
                "In no event shall either party's liability exceed the fees paid in the "
                "twelve (12) months preceding the claim. Acme Corp. agrees to this cap."),
    make_clause("contract-a", "Confidentiality", True, "5 years",
                "The confidentiality obligations shall survive termination for a period "
                "of five (5) years. TechVentures Inc. acknowledges this obligation."),
    make_clause("contract-a", "Payment Terms", True, "30 days",
                "All invoices are due within thirty (30) days of receipt."),
    make_clause("contract-a", "Termination for Convenience", True, "30 days",
                "Either party may terminate this agreement with thirty (30) days written notice."),

    # contract-b
    make_clause("contract-b", "Governing Law", True, "New York",
                "This License Agreement is governed by the laws of New York. "
                "GlobalSoft LLC and BetaCorp Inc. consent to New York jurisdiction."),
    make_clause("contract-b", "Liability Cap", True, "6 months fees",
                "Liability is capped at six (6) months of fees paid. GlobalSoft LLC "
                "shall not be liable beyond this amount."),
    make_clause("contract-b", "Confidentiality", True, "3 years",
                "Confidentiality obligations survive for three (3) years after termination."),
    make_clause("contract-b", "Payment Terms", True, "45 days",
                "Payment is due Net 45 days from invoice date."),
    make_clause("contract-b", "Termination for Convenience", False,
                None,
                "Termination for convenience not found in this agreement."),
    make_clause("contract-b", "Uncapped Liability", True, "unlimited",
                "Liability for gross negligence shall be unlimited. BetaCorp Inc. "
                "accepts this provision."),
]

# Contradictions we expect to find
EXPECTED_VALUE_CONFLICTS: dict[str, tuple[str, str]] = {
    "Governing Law":   ("Delaware", "New York"),
    "Liability Cap":   ("12 months fees", "6 months fees"),
    "Confidentiality": ("5 years", "3 years"),
    "Payment Terms":   ("30 days", "45 days"),
}
EXPECTED_ABSENCE_CLAUSE = "Termination for Convenience"


# ── Test 1: Node skips when graph_built=False ──────────────────────────────────

def test_node_skips_without_graph() -> None:
    from core.state import GraphState
    from agents.contradiction_detector.agent import contradiction_detector_node

    logger.info("─" * 60)
    logger.info("TEST 1: node skips gracefully when graph_built=False")
    logger.info("─" * 60)

    state = GraphState(
        job_id="sprint5-test-no-graph",
        status="graph_built",
        created_at=datetime.now(timezone.utc),
        documents=[DocumentRecord(doc_id="contract-a", file_path="samples/contract_a.txt")],
        extracted_clauses=[],
        risk_flags=[],
        contradictions=[],
        neo4j_ready=True,
        qdrant_ready=True,
        graph_built=False,
        final_report=None,
        errors=[],
    )

    result = contradiction_detector_node(state)
    assert result["contradictions"] == [], (
        f"Expected empty contradictions when graph_built=False, got {result['contradictions']}"
    )
    assert result["status"] == "contradictions_detected"
    logger.info("  PASS — node returns contradictions=[] when graph_built=False")


# ── Test 2: Contradiction detection against live Neo4j ────────────────────────

def test_contradiction_detection() -> None:
    from core.state import GraphState
    from agents.entity_mapper.agent import entity_mapper_node
    from agents.contradiction_detector.agent import contradiction_detector_node
    from infrastructure.neo4j_client import get_neo4j_session, check_neo4j_health

    logger.info("─" * 60)
    logger.info("TEST 2: Contradiction detection against live Neo4j")
    logger.info("─" * 60)

    if not check_neo4j_health():
        raise RuntimeError(
            "Neo4j is not reachable. Start it with: docker compose up -d\n"
            "Then re-run. Use --skip-neo4j to skip this test."
        )

    # Seed Neo4j with Sprint 4 sample data
    with get_neo4j_session() as session:
        session.run(
            "MATCH (n) WHERE n.doc_id IN ['contract-a', 'contract-b'] DETACH DELETE n"
        )
    logger.info("  Cleared previous test data")

    seed_state = GraphState(
        job_id="sprint5-seed",
        status="risks_scored",
        created_at=datetime.now(timezone.utc),
        documents=[
            DocumentRecord(doc_id="contract-a", file_path="samples/contract_a.txt"),
            DocumentRecord(doc_id="contract-b", file_path="samples/contract_b.txt"),
        ],
        extracted_clauses=SAMPLE_CLAUSES,
        risk_flags=[],
        contradictions=[],
        neo4j_ready=True,
        qdrant_ready=True,
        graph_built=False,
        final_report=None,
        errors=[],
    )
    seed_result = entity_mapper_node(seed_state)
    assert seed_result["graph_built"] is True, (
        f"Seeding failed — entity_mapper_node returned graph_built=False. "
        f"Errors: {seed_result.get('errors', [])}"
    )
    logger.info("  Neo4j seeded: graph_built=True")

    # Run contradiction detector
    detect_state = GraphState(
        job_id="sprint5-test",
        status="graph_built",
        created_at=datetime.now(timezone.utc),
        documents=[
            DocumentRecord(doc_id="contract-a", file_path="samples/contract_a.txt"),
            DocumentRecord(doc_id="contract-b", file_path="samples/contract_b.txt"),
        ],
        extracted_clauses=SAMPLE_CLAUSES,
        risk_flags=[],
        contradictions=[],
        neo4j_ready=True,
        qdrant_ready=True,
        graph_built=True,
        final_report=None,
        errors=[],
    )
    result = contradiction_detector_node(detect_state)

    contradictions = result["contradictions"]
    assert result["status"] == "contradictions_detected"

    logger.info("  %d contradiction(s) returned", len(contradictions))
    for c in contradictions:
        logger.info(
            "    [%s] %s | %s=%r vs %s=%r",
            "ABSENCE" if "(clause absent)" in (c.value_a + c.value_b) else "VALUE",
            c.clause_type,
            c.document_id_a, c.value_a,
            c.document_id_b, c.value_b,
        )
        logger.info("      → %s", c.explanation[:100])

    # 2a. Total count: 4 value conflicts + 1 absence conflict
    expected_total = len(EXPECTED_VALUE_CONFLICTS) + 1
    assert len(contradictions) == expected_total, (
        f"Expected {expected_total} contradictions, got {len(contradictions)}"
    )
    logger.info("  PASS — %d contradictions found (expected %d)", len(contradictions), expected_total)

    # 2b. All expected value conflicts present
    found_types = {c.clause_type for c in contradictions}
    for clause_type, (val_a, val_b) in EXPECTED_VALUE_CONFLICTS.items():
        assert clause_type in found_types, (
            f"Expected value conflict for '{clause_type}' not found. Got: {found_types}"
        )
        match = next(c for c in contradictions if c.clause_type == clause_type)
        # Values may appear in either a/b position depending on doc_id alphabetical order
        assert {match.value_a, match.value_b} == {val_a, val_b}, (
            f"{clause_type}: expected values {{{val_a!r}, {val_b!r}}}, "
            f"got {{{match.value_a!r}, {match.value_b!r}}}"
        )
    logger.info("  PASS — all 4 value conflicts match expected values")

    # 2c. Absence conflict present
    assert EXPECTED_ABSENCE_CLAUSE in found_types, (
        f"Expected absence conflict for '{EXPECTED_ABSENCE_CLAUSE}' not found. Got: {found_types}"
    )
    absence = next(c for c in contradictions if c.clause_type == EXPECTED_ABSENCE_CLAUSE)
    assert "(clause absent)" in absence.value_a or "(clause absent)" in absence.value_b, (
        f"Expected '(clause absent)' in absence conflict values, "
        f"got value_a={absence.value_a!r}, value_b={absence.value_b!r}"
    )
    logger.info(
        "  PASS — absence conflict: %s | %s=%r vs %s=%r",
        absence.clause_type, absence.document_id_a, absence.value_a,
        absence.document_id_b, absence.value_b,
    )

    # 2d. All contradictions have non-empty explanations
    for c in contradictions:
        assert c.explanation and len(c.explanation) > 10, (
            f"Contradiction {c.clause_type} has empty/short explanation: {c.explanation!r}"
        )
    logger.info("  PASS — all %d contradictions have non-empty explanations", len(contradictions))

    # 2e. Contradiction model fields are all populated
    for c in contradictions:
        assert c.clause_type, "clause_type is empty"
        assert c.document_id_a, "document_id_a is empty"
        assert c.document_id_b, "document_id_b is empty"
        assert c.document_id_a != c.document_id_b, "document_id_a == document_id_b (same doc)"
        assert c.value_a, "value_a is empty"
        assert c.value_b, "value_b is empty"
    logger.info("  PASS — all Contradiction fields populated correctly")

    logger.info("  All contradiction detection tests PASSED")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip live Neo4j tests (stub test only — no Docker required)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SPRINT 5 — Contradiction Detector smoke test")
    logger.info("=" * 60)

    test_node_skips_without_graph()

    if args.skip_neo4j:
        logger.info("─" * 60)
        logger.info("Skipping Neo4j tests (--skip-neo4j)")
    else:
        test_contradiction_detection()

    logger.info("=" * 60)
    logger.info("Sprint 5 PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
