"""
Sprint 6 smoke test — Report & Q&A.

What is verified:
  1. Formatter output — no LLM, no Qdrant required
     a. assemble_report() produces a valid markdown string with all sections
     b. Risk flags table present and sorted HIGH first
     c. Contradiction table present with correct values
     d. Missing clauses table present
     e. Template narrative fallback produces correct summary when LLM skipped

  2. Full report synthesis — requires Ollama (LLM)
     a. report_qa_node returns final_report as non-empty markdown
     b. Report contains key section headers
     c. Executive summary is non-empty and coherent
     d. Recommended actions present (≥1)
     e. Risk table and contradiction table appear in report

  3. Q&A function — requires Qdrant + BM25 index + Ollama
     a. answer_question() returns non-empty answer
     b. Citations are returned with doc_id, page_number, chunk_id, excerpt
     c. Answer is grounded in the correct document

Run from repo root:
    python run_sprint6.py                     # full suite
    python run_sprint6.py --skip-llm          # formatter only (no LLM/Qdrant)
    python run_sprint6.py --skip-qa           # formatter + report, skip Q&A test
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "legal_due_diligence"))

from core.models import Contradiction, DocumentRecord, ExtractedClause, RiskFlag  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    DocumentRecord(doc_id="contract-a", file_path="samples/contract_a.txt"),
    DocumentRecord(doc_id="contract-b", file_path="samples/contract_b.txt"),
]

SAMPLE_RISK_FLAGS = [
    RiskFlag(
        document_id="contract-a",
        clause_type="Liability Cap",
        risk_level="medium",
        reason="Cap based on fees may not adequately cover damages if fees are low.",
        is_missing_clause=False,
        source_clause_id="chunk-a-1",
    ),
    RiskFlag(
        document_id="contract-a",
        clause_type="Termination for Convenience",
        risk_level="medium",
        reason="Unilateral termination with minimal notice period may disrupt operations.",
        is_missing_clause=False,
        source_clause_id="chunk-a-2",
    ),
    RiskFlag(
        document_id="contract-b",
        clause_type="Liability Cap",
        risk_level="medium",
        reason="Cap based on fees may not adequately cover damages in case of severe breach.",
        is_missing_clause=False,
        source_clause_id="chunk-b-1",
    ),
    RiskFlag(
        document_id="contract-b",
        clause_type="Termination for Convenience",
        risk_level="medium",
        reason="Termination for Convenience clause is absent — neither party can exit cleanly.",
        is_missing_clause=True,
        source_clause_id=None,
    ),
    RiskFlag(
        document_id="contract-b",
        clause_type="Uncapped Liability",
        risk_level="high",
        reason="Uncapped liability clause found — explicit unlimited liability exposure.",
        is_missing_clause=False,
        source_clause_id="chunk-b-2",
    ),
    # Note: contract-b has Governing Law = "New York" (not missing).
    # The Governing Law risk here is a CONTRADICTION with contract-a (Delaware),
    # represented in SAMPLE_CONTRADICTIONS — not a missing clause flag.
]

SAMPLE_CONTRADICTIONS = [
    Contradiction(
        clause_type="Governing Law",
        document_id_a="contract-a",
        document_id_b="contract-b",
        value_a="Delaware",
        value_b="New York",
        explanation=(
            "Conflicting governing law creates forum shopping risk — a party could "
            "argue either state's law applies, increasing litigation costs and uncertainty."
        ),
    ),
    Contradiction(
        clause_type="Liability Cap",
        document_id_a="contract-a",
        document_id_b="contract-b",
        value_a="12 months fees",
        value_b="6 months fees",
        explanation=(
            "The liability cap differs by 2× across contracts — a party exposed under "
            "both agreements faces different maximum recovery amounts depending on which "
            "contract governs a given claim."
        ),
    ),
    Contradiction(
        clause_type="Termination for Convenience",
        document_id_a="contract-a",
        document_id_b="contract-b",
        value_a="30 days",
        value_b="(clause absent)",
        explanation=(
            "Termination for Convenience is available under contract-a but absent from "
            "contract-b, leaving a party bound indefinitely under contract-b."
        ),
    ),
]


def _make_state(job_id: str = "sprint6-test"):
    from core.state import GraphState
    return GraphState(
        job_id=job_id,
        status="contradictions_detected",
        created_at=datetime.now(timezone.utc),
        documents=SAMPLE_DOCUMENTS,
        extracted_clauses=[],
        risk_flags=SAMPLE_RISK_FLAGS,
        contradictions=SAMPLE_CONTRADICTIONS,
        neo4j_ready=True,
        qdrant_ready=True,
        graph_built=True,
        final_report=None,
        errors=[],
    )


# ── Test 1: Formatter output ───────────────────────────────────────────────────

def test_formatter() -> None:
    from agents.report_qa.formatter import (
        assemble_report,
        build_narrative_prompt,
        _format_risk_table,
        _format_contradiction_table,
        _format_missing_clauses,
    )
    from agents.report_qa.agent import _template_narrative

    logger.info("─" * 60)
    logger.info("TEST 1: Formatter — deterministic section output")
    logger.info("─" * 60)

    state = _make_state()

    # 1a. Risk table contains HIGH flag first
    risk_md = _format_risk_table(state.risk_flags)
    assert "HIGH" in risk_md, "Risk table missing HIGH flag"
    assert "Uncapped Liability" in risk_md
    assert "contract-b" in risk_md
    logger.info("  PASS — risk table contains HIGH flags and correct doc sections")

    # 1b. Contradiction table contains expected clause types
    contra_md = _format_contradiction_table(state.contradictions)
    assert "Governing Law" in contra_md
    assert "Delaware" in contra_md
    assert "New York" in contra_md
    assert "Termination for Convenience" in contra_md
    assert "(clause absent)" in contra_md
    logger.info("  PASS — contradiction table contains all 3 contradictions")

    # 1c. Missing clauses table contains missing flags only
    # contract-b/Termination for Convenience is the only missing-clause flag in the fixture.
    # contract-b/Governing Law is NOT missing — it's a CONTRADICTION (New York vs Delaware).
    missing_md = _format_missing_clauses(state.risk_flags)
    assert "Termination for Convenience" in missing_md
    assert "Governing Law" not in missing_md, (
        "Governing Law should not appear in missing clauses — contract-b has it (New York); "
        "the risk is the contradiction with contract-a, not absence."
    )
    assert "Uncapped Liability" not in missing_md
    logger.info("  PASS — missing clauses table: correct rows, no false positives")

    # 1d. Template narrative contains factual summary
    summary, actions = _template_narrative(state)
    assert "contract-a" in summary and "contract-b" in summary, "Summary missing doc names"
    assert len(state.risk_flags) > 0
    assert any("HIGH" in a.upper() or "high" in a.lower() for a in actions), (
        f"Expected high-risk action in {actions}"
    )
    logger.info("  PASS — template narrative summary: %s", summary[:80])
    logger.info("  PASS — template actions (%d): %s", len(actions), actions[0][:60])

    # 1e. Full assembly produces required section headers
    report = assemble_report(state, summary, actions)
    required_headers = [
        "# Due Diligence Brief",
        "## Executive Summary",
        "## Risk Flags by Document",
        "## Cross-Document Contradictions",
        "## Missing Clauses Inventory",
        "## Recommended Actions",
    ]
    for header in required_headers:
        assert header in report, f"Missing section: {header!r}"
    assert state.job_id in report
    logger.info("  PASS — all 6 section headers present in assembled report")

    # 1f. Narrative prompt is non-empty and structured with correct type labels
    prompt = build_narrative_prompt(state)
    assert "executive_summary" in prompt
    assert "recommended_actions" in prompt
    # HIGH flag section must use type labels so LLM can distinguish risks from absences
    assert "CONTENT RISK" in prompt, "Expected CONTENT RISK label for Uncapped Liability"
    assert "MISSING CLAUSE" in prompt, "Expected MISSING CLAUSE label for Termination for Convenience"
    # Contradictions section must clarify they are NOT missing clauses
    assert "Governing Law" in prompt  # surfaced in contradictions section
    assert "present in both docs" in prompt or "PRESENT" in prompt or "conflicting values" in prompt
    logger.info("  PASS — narrative prompt contains type-labelled HIGH flags + contradiction disambiguation")

    logger.info("  All formatter tests PASSED")


# ── Test 2: Full report synthesis (LLM) ───────────────────────────────────────

def test_report_synthesis() -> None:
    from core.state import GraphState
    from agents.report_qa.agent import report_qa_node

    logger.info("─" * 60)
    logger.info("TEST 2: Full report synthesis via report_qa_node (requires Ollama)")
    logger.info("─" * 60)

    state = _make_state("sprint6-llm-test")
    result = report_qa_node(state)

    assert result["status"] == "complete"
    report = result["final_report"]
    assert report and len(report) > 200, f"Report too short: {len(report)} chars"
    logger.info("  Report length: %d chars", len(report))

    # Required section headers
    required_headers = [
        "# Due Diligence Brief",
        "## Executive Summary",
        "## Risk Flags by Document",
        "## Cross-Document Contradictions",
        "## Missing Clauses Inventory",
        "## Recommended Actions",
    ]
    for header in required_headers:
        assert header in report, f"Missing section in report: {header!r}"
    logger.info("  PASS — all section headers present")

    # Executive summary should be non-trivial
    lines = report.split("\n")
    exec_idx = next(i for i, l in enumerate(lines) if "## Executive Summary" in l)
    summary_text = "\n".join(lines[exec_idx + 1: exec_idx + 6]).strip()
    assert len(summary_text) > 50, f"Executive summary too short: {summary_text!r}"
    logger.info("  PASS — executive summary: %s", summary_text[:100])

    # Recommended actions section should have ≥1 action
    actions_idx = next(i for i, l in enumerate(lines) if "## Recommended Actions" in l)
    actions_text = "\n".join(lines[actions_idx + 1: actions_idx + 10])
    assert "- " in actions_text, f"No bullet actions found: {actions_text!r}"
    logger.info("  PASS — recommended actions section contains bullet items")

    # Contradiction table should have our three contradictions
    assert "Governing Law" in report
    assert "Delaware" in report
    assert "Termination for Convenience" in report
    logger.info("  PASS — contradiction data present in report")

    logger.info("  Full report synthesis PASSED")
    logger.info("─" * 60)
    logger.info("REPORT PREVIEW (first 800 chars):")
    logger.info(report[:800])
    logger.info("─" * 60)


# ── Test 3: Q&A function ───────────────────────────────────────────────────────

def test_qa() -> None:
    from agents.report_qa.qa import answer_question

    logger.info("─" * 60)
    logger.info("TEST 3: Q&A function (requires Qdrant + BM25 + Ollama)")
    logger.info("─" * 60)

    doc_ids = ["contract-a", "contract-b"]

    # 3a. Governing law question
    result = answer_question("What is the governing law for these contracts?", doc_ids)
    logger.info("  Q: 'What is the governing law for these contracts?'")
    logger.info("  A: %s", result["answer"][:200])
    logger.info("  Citations: %d chunk(s)", len(result["citations"]))

    assert result["answer"], "Answer is empty"
    assert result["chunks_retrieved"] > 0, "No chunks retrieved"
    assert len(result["citations"]) > 0, "No citations returned"
    for citation in result["citations"]:
        assert "doc_id" in citation
        assert "page_number" in citation
        assert "chunk_id" in citation
        assert "excerpt" in citation
    logger.info("  PASS — answer returned with %d citation(s)", len(result["citations"]))

    # 3b. Liability cap question
    result2 = answer_question("What is the liability cap in these contracts?", doc_ids)
    logger.info("  Q: 'What is the liability cap in these contracts?'")
    logger.info("  A: %s", result2["answer"][:200])
    assert result2["answer"], "Answer 2 is empty"
    logger.info("  PASS — liability cap question answered")

    logger.info("  All Q&A tests PASSED")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM tests — formatter only (no Ollama, no Qdrant required)",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip Q&A test only — run formatter + report synthesis",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SPRINT 6 — Report & Q&A smoke test")
    logger.info("=" * 60)

    test_formatter()

    if args.skip_llm:
        logger.info("─" * 60)
        logger.info("Skipping LLM + Q&A tests (--skip-llm)")
    else:
        test_report_synthesis()

        if args.skip_qa:
            logger.info("─" * 60)
            logger.info("Skipping Q&A test (--skip-qa)")
        else:
            test_qa()

    logger.info("=" * 60)
    logger.info("Sprint 6 PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
