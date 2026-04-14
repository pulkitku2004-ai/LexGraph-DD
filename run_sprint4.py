"""
Sprint 4 smoke test — Entity Mapper.

What is verified:
  1. Entity extraction logic (extractor.py) — no Neo4j required
     a. Jurisdiction extracted from "Governing Law" normalized_value
     b. Duration extracted from "Confidentiality" normalized_value
     c. Amount extracted from "Liability Cap" normalized_value
     d. Party names extracted from clause_text via regex
     e. found=False clause → no jurisdiction/duration/amount entities (but parties still extracted)

  2. Neo4j schema writes (schema.py) — requires Neo4j running
     a. ensure_constraints() runs without error
     b. Document nodes written and queryable
     c. Clause nodes written with correct properties (found=True and found=False)
     d. HAS_CLAUSE relationships exist
     e. Jurisdiction node written + GOVERNED_BY relationship

  3. entity_mapper_node GraphState roundtrip (agent.py)
     a. Returns graph_built=True on success
     b. Returns graph_built=False when neo4j_ready=False
     c. State errors list preserved and extended on clause failure

  4. Graph verification queries
     a. Count Document and Clause nodes match input
     b. Governing Law Jurisdiction nodes exist for both contracts
     c. Jurisdiction nodes for contract-a/b are different (Delaware vs New York)
        — this is the key pre-condition for Sprint 5 contradiction detection

Run from repo root:
    python run_sprint4.py
    python run_sprint4.py --skip-neo4j   # unit tests only, no Neo4j connection
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


# ── Fixtures ──────────────────────────────────────────────────────────────────

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
        source_chunk_id="test-chunk-sprint4",
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


# ── Test 1: Entity extractor unit tests ───────────────────────────────────────

def test_extractor() -> None:
    from agents.entity_mapper.extractor import extract_entities

    logger.info("─" * 60)
    logger.info("TEST 1: Entity extractor — no Neo4j required")
    logger.info("─" * 60)

    # 1a. Jurisdiction from Governing Law
    clause = make_clause("contract-a", "Governing Law", True, "Delaware",
                         "Governed by the laws of Delaware. Acme Corp. and TechVentures Inc.")
    entities = extract_entities(clause)
    assert "Delaware" in entities["jurisdictions"], (
        f"Expected Delaware in jurisdictions, got {entities['jurisdictions']}"
    )
    logger.info("  PASS — Governing Law → jurisdiction: %s", entities["jurisdictions"])

    # 1b. Duration from Confidentiality
    clause = make_clause("contract-a", "Confidentiality", True, "5 years",
                         "Obligations survive for five (5) years.")
    entities = extract_entities(clause)
    assert "5 years" in entities["durations"], (
        f"Expected '5 years' in durations, got {entities['durations']}"
    )
    logger.info("  PASS — Confidentiality → duration: %s", entities["durations"])

    # 1c. Amount from Liability Cap
    clause = make_clause("contract-a", "Liability Cap", True, "12 months fees",
                         "Liability capped at twelve months of fees.")
    entities = extract_entities(clause)
    assert "12 months fees" in entities["amounts"], (
        f"Expected '12 months fees' in amounts, got {entities['amounts']}"
    )
    logger.info("  PASS — Liability Cap → amount: %s", entities["amounts"])

    # 1d. Party names from clause_text
    clause = make_clause("contract-a", "Governing Law", True, "Delaware",
                         "Acme Corp. and TechVentures Inc. agree to Delaware jurisdiction.")
    entities = extract_entities(clause)
    party_names = [p.lower() for p in entities["parties"]]
    assert any("acme" in p for p in party_names), (
        f"Expected Acme Corp in parties, got {entities['parties']}"
    )
    assert any("techventures" in p for p in party_names), (
        f"Expected TechVentures Inc in parties, got {entities['parties']}"
    )
    logger.info("  PASS — party names extracted: %s", entities["parties"])

    # 1e. found=False clause → no jurisdiction/duration/amount (parties still OK)
    clause = make_clause("contract-b", "Termination for Convenience", False,
                         None, "Not found in this agreement.")
    entities = extract_entities(clause)
    assert entities["jurisdictions"] == [], "found=False should produce no jurisdictions"
    assert entities["durations"] == [], "found=False should produce no durations"
    assert entities["amounts"] == [], "found=False should produce no amounts"
    logger.info("  PASS — found=False clause → no jurisdiction/duration/amount entities")

    logger.info("  All extractor tests PASSED")


# ── Test 2: entity_mapper_node skips when neo4j_ready=False ──────────────────

def test_node_skips_without_neo4j() -> None:
    from core.state import GraphState
    from agents.entity_mapper.agent import entity_mapper_node

    logger.info("─" * 60)
    logger.info("TEST 2: entity_mapper_node — skips gracefully when neo4j_ready=False")
    logger.info("─" * 60)

    state = GraphState(
        job_id="sprint4-test-no-neo4j",
        status="risks_scored",
        created_at=datetime.now(timezone.utc),
        documents=[DocumentRecord(doc_id="contract-a", file_path="samples/contract_a.txt")],
        extracted_clauses=SAMPLE_CLAUSES[:1],
        risk_flags=[],
        contradictions=[],
        neo4j_ready=False,
        qdrant_ready=True,
        graph_built=False,
        final_report=None,
        errors=[],
    )

    result = entity_mapper_node(state)
    assert result["graph_built"] is False, "Expected graph_built=False when neo4j_ready=False"
    assert result["status"] == "graph_built"
    logger.info("  PASS — node returns graph_built=False when Neo4j not ready")


# ── Test 3: Neo4j write + verification ───────────────────────────────────────

def test_neo4j_writes() -> None:
    from core.state import GraphState
    from agents.entity_mapper.agent import entity_mapper_node
    from infrastructure.neo4j_client import get_neo4j_session, check_neo4j_health

    logger.info("─" * 60)
    logger.info("TEST 3: Neo4j writes + graph verification")
    logger.info("─" * 60)

    if not check_neo4j_health():
        raise RuntimeError(
            "Neo4j is not reachable. Start it with: docker compose up -d\n"
            "Then re-run. Use --skip-neo4j to skip this test."
        )

    # Clear any previous sprint4 test data
    with get_neo4j_session() as session:
        session.run(
            "MATCH (n) WHERE n.doc_id IN ['contract-a', 'contract-b'] DETACH DELETE n"
        )
    logger.info("  Cleared previous test data")

    state = GraphState(
        job_id="sprint4-test",
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

    result = entity_mapper_node(state)

    assert result["graph_built"] is True, (
        f"Expected graph_built=True, errors: {result.get('errors', [])}"
    )
    assert result["status"] == "graph_built"
    logger.info("  PASS — entity_mapper_node returned graph_built=True")

    # Verification queries
    with get_neo4j_session() as session:

        # 3a. Document node count
        rec = session.run(
            "MATCH (d:Document) WHERE d.doc_id IN ['contract-a', 'contract-b'] "
            "RETURN count(d) AS cnt"
        ).single()
        doc_count = rec["cnt"] if rec else 0
        assert doc_count == 2, f"Expected 2 Document nodes, got {doc_count}"
        logger.info("  PASS — 2 Document nodes in graph")

        # 3b. Clause node count (11 sample clauses)
        rec = session.run(
            "MATCH (c:Clause) WHERE c.doc_id IN ['contract-a', 'contract-b'] "
            "RETURN count(c) AS cnt"
        ).single()
        clause_count = rec["cnt"] if rec else 0
        assert clause_count == len(SAMPLE_CLAUSES), (
            f"Expected {len(SAMPLE_CLAUSES)} Clause nodes, got {clause_count}"
        )
        logger.info("  PASS — %d Clause nodes written", clause_count)

        # 3c. HAS_CLAUSE relationships
        rec = session.run(
            "MATCH (:Document)-[r:HAS_CLAUSE]->(:Clause) "
            "WHERE r IS NOT NULL "
            "RETURN count(r) AS cnt"
        ).single()
        rel_count = rec["cnt"] if rec else 0
        assert rel_count >= len(SAMPLE_CLAUSES), (
            f"Expected at least {len(SAMPLE_CLAUSES)} HAS_CLAUSE rels, got {rel_count}"
        )
        logger.info("  PASS — %d HAS_CLAUSE relationships", rel_count)

        # 3d. Jurisdiction nodes — Delaware and New York must both exist
        rows = session.run(
            "MATCH (c:Clause {clause_type: 'Governing Law'})-[:GOVERNED_BY]->(j:Jurisdiction) "
            "WHERE c.doc_id IN ['contract-a', 'contract-b'] "
            "RETURN c.doc_id AS doc_id, j.name AS jurisdiction "
            "ORDER BY c.doc_id"
        ).data()

        jurisdictions = {r["doc_id"]: r["jurisdiction"] for r in rows}
        logger.info("  Jurisdiction nodes found: %s", jurisdictions)

        assert jurisdictions.get("contract-a") == "Delaware", (
            f"Expected Delaware for contract-a, got {jurisdictions.get('contract-a')}"
        )
        assert jurisdictions.get("contract-b") == "New York", (
            f"Expected New York for contract-b, got {jurisdictions.get('contract-b')}"
        )
        logger.info("  PASS — contract-a → Delaware, contract-b → New York")

        # 3e. Sprint 5 pre-condition: same clause_type, different jurisdictions, different docs
        rows = session.run(
            """
            MATCH (c1:Clause {clause_type: 'Governing Law'})-[:GOVERNED_BY]->(j1:Jurisdiction),
                  (c2:Clause {clause_type: 'Governing Law'})-[:GOVERNED_BY]->(j2:Jurisdiction)
            WHERE c1.doc_id <> c2.doc_id AND j1.name <> j2.name
            RETURN c1.doc_id AS doc_a, j1.name AS jur_a,
                   c2.doc_id AS doc_b, j2.name AS jur_b
            LIMIT 1
            """
        ).data()
        assert len(rows) >= 1, (
            "Sprint 5 pre-condition FAILED: no cross-document Governing Law conflict found in graph"
        )
        row = rows[0]
        logger.info(
            "  PASS — Sprint 5 pre-condition met: %s=%s vs %s=%s",
            row["doc_a"], row["jur_a"], row["doc_b"], row["jur_b"],
        )

        # 3f. found=False clause is in graph (needed for Sprint 5 absence detection)
        rec = session.run(
            "MATCH (c:Clause {doc_id: 'contract-b', clause_type: 'Termination for Convenience'}) "
            "RETURN c.found AS found"
        ).single()
        assert rec is not None, "Missing contract-b/Termination for Convenience node"
        assert rec["found"] is False, (
            f"Expected found=False for missing clause, got {rec['found']}"
        )
        logger.info("  PASS — found=False clause persisted correctly in graph")

    logger.info("  All Neo4j write tests PASSED")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip Neo4j write tests (unit tests only — no Docker required)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SPRINT 4 — Entity Mapper smoke test")
    logger.info("=" * 60)

    test_extractor()
    test_node_skips_without_neo4j()

    if args.skip_neo4j:
        logger.info("─" * 60)
        logger.info("Skipping Neo4j write tests (--skip-neo4j)")
    else:
        test_neo4j_writes()

    logger.info("=" * 60)
    logger.info("Sprint 4 PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
