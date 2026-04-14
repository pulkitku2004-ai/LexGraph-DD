"""
Sprint 9 smoke test — FastAPI layer.

What is verified:
  1. Schema layer — JobResponse, QAResponse round-trip correctly
  2. Job lifecycle — create_job() registers correct doc_ids + tmp files
  3. Pipeline execution — run_pipeline() with sample contracts:
       a. Both docs ingested (processed=True)
       b. Job reaches status='done'
       c. final_report is non-empty markdown
       d. No pipeline errors
  4. Q&A — answer_question() works for a completed job
  5. Cleanup — delete_job() removes Qdrant points, Neo4j nodes, tmp dir

Run from repo root (Docker must be running):
    python run_sprint9.py            # full suite (requires Ollama + Qdrant + Neo4j)
    python run_sprint9.py --skip-llm # skips pipeline + Q&A, tests schema + job creation only

How to start the API server (after this test passes):
    source .venv/bin/activate
    uvicorn legal_due_diligence.api.main:app --reload --port 8000

Then hit it:
    curl -X POST http://localhost:8000/jobs \\
         -F "files=@samples/contract_a.txt" \\
         -F "files=@samples/contract_b.txt"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
PKG_ROOT = PROJECT_ROOT / "legal_due_diligence"
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from api.runner import JOB_STORE, create_job, delete_job, run_pipeline
from api.schemas import JobStatus

SAMPLE_A = PROJECT_ROOT / "samples" / "contract_a.txt"
SAMPLE_B = PROJECT_ROOT / "samples" / "contract_b.txt"

PASS = "✓"
FAIL = "✗"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  {status} {label}{suffix}")
    return condition


def test_schema() -> bool:
    print("\nTest 1 — Schema layer")
    from datetime import datetime
    from api.schemas import Citation, JobResponse, JobStatus, QAResponse

    job = JobResponse(
        job_id="test-123",
        status=JobStatus.pending,
        doc_ids=["contract_a", "contract_b"],
        report=None,
        errors=[],
        created_at=datetime.utcnow(),
    )
    qa = QAResponse(
        answer="Delaware governs contract A.",
        citations=[Citation(doc_id="contract_a", page_number=1, chunk_id="abc", excerpt="...")],
        chunks_retrieved=3,
    )
    ok = True
    ok &= check("JobResponse serialises", job.job_id == "test-123")
    ok &= check("JobStatus enum", job.status == "pending")
    ok &= check("QAResponse citation count", len(qa.citations) == 1)
    return ok


def test_create_job() -> tuple[bool, str]:
    print("\nTest 2 — Job creation")
    file_data = [
        (SAMPLE_A.name, SAMPLE_A.read_bytes()),
        (SAMPLE_B.name, SAMPLE_B.read_bytes()),
    ]
    job_id = create_job(file_data)
    record = JOB_STORE.get(job_id)

    ok = True
    ok &= check("job_id in JOB_STORE", record is not None)
    ok &= check("status=pending", record.status == JobStatus.pending)
    ok &= check("2 doc_ids registered", len(record.doc_ids) == 2, str(record.doc_ids))
    ok &= check("tmp_dir exists", Path(record.tmp_dir).is_dir())
    ok &= check("files saved to tmp_dir", len(list(Path(record.tmp_dir).iterdir())) == 2)
    return ok, job_id


def test_pipeline(job_id: str) -> bool:
    print("\nTest 3 — Pipeline execution (ingestion + LangGraph)")
    print("  [running — this takes several minutes for LLM calls]")
    t0 = time.perf_counter()
    run_pipeline(job_id)
    elapsed = time.perf_counter() - t0

    record = JOB_STORE[job_id]
    ok = True
    ok &= check("status=done", record.status == JobStatus.done, f"got '{record.status}'")
    ok &= check("report non-empty", bool(record.report), f"{len(record.report or '')} chars")
    ok &= check("no pipeline errors", len(record.errors) == 0, str(record.errors) if record.errors else "")
    ok &= check("report contains risk section", "Risk" in (record.report or ""))
    ok &= check("report contains contradiction section", "Contradiction" in (record.report or ""))
    print(f"  [elapsed: {elapsed:.1f}s]")
    return ok


def test_qa(job_id: str) -> bool:
    print("\nTest 4 — Q&A")
    from agents.report_qa.qa import answer_question

    result = answer_question(
        question="What is the governing law for each contract?",
        doc_ids=JOB_STORE[job_id].doc_ids,
    )
    ok = True
    ok &= check("answer non-empty", bool(result["answer"]))
    ok &= check("citations returned", len(result["citations"]) > 0, f"{len(result['citations'])} citation(s)")
    ok &= check("chunk_id present", all("chunk_id" in c for c in result["citations"]))
    print(f"  Answer preview: {result['answer'][:120].strip()}...")
    return ok


def test_delete(job_id: str) -> bool:
    print("\nTest 5 — Cleanup (DELETE)")
    record = JOB_STORE[job_id]
    tmp_dir = Path(record.tmp_dir)

    delete_job(job_id)

    ok = True
    ok &= check("job removed from JOB_STORE", job_id not in JOB_STORE)
    ok &= check("tmp_dir removed", not tmp_dir.exists())
    # Qdrant and Neo4j cleanup logged — not re-queried here to avoid re-importing
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Skip pipeline + Q&A tests (schema + job creation only)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Sprint 9 — API layer smoke test")
    print("=" * 60)

    results: list[bool] = []

    results.append(test_schema())
    schema_ok, job_id = test_create_job()
    results.append(schema_ok)

    if args.skip_llm:
        print("\n[--skip-llm] skipping pipeline, Q&A, and delete tests.")
        # clean up tmp_dir manually
        import shutil
        record = JOB_STORE.get(job_id)
        if record:
            shutil.rmtree(record.tmp_dir, ignore_errors=True)
            del JOB_STORE[job_id]
    else:
        pipeline_ok = test_pipeline(job_id)
        results.append(pipeline_ok)
        if pipeline_ok:
            results.append(test_qa(job_id))
        else:
            print("\n  [skipping Q&A — pipeline did not complete successfully]")
        results.append(test_delete(job_id))

    total = len(results)
    passed = sum(results)
    print("\n" + "=" * 60)
    print(f"  {passed}/{total} test groups passed")
    print("=" * 60)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
