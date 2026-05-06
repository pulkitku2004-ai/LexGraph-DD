"""
test_runner_cuad.py

Integration test: 100 CUAD contracts → LexGraph → ASTR-O.

Flow:
  - Splits 100 contracts (cuad_samples/) into batches of BATCH_SIZE
  - For each batch:
      1. POST /jobs  — ingest batch, run full LangGraph pipeline
      2. Poll until done
      3. Fire CUAD_QUERIES against the job via POST /jobs/{id}/qa
      4. Build ASTR-O span per query → process_lexgraph_span()
      5. Collect SAFE / FLAGGED verdicts
  - Save aggregated JSON report

Requires:
  - cuad_samples/ populated:   python setup_cuad_dataset.py
  - Infrastructure running:    docker compose up -d
  - LexGraph API running:      .venv/bin/python -m uvicorn legal_due_diligence.api.main:app --port 8000
  - ASTR-O registry updated:   python setup_astr_o_registry.py  (already done)

Run:
  .venv/bin/python test_runner_cuad.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ── ASTR_O_REPORT_SECRET ──────────────────────────────────────────────────────
os.environ.setdefault("ASTR_O_REPORT_SECRET", "lexgraph-astr-o-dev")

# ── Config ────────────────────────────────────────────────────────────────────

LEXGRAPH_BASE_URL = "http://localhost:8000"
ASTR_O_PATH       = "/Users/dr_bolty/astr-o"
CUAD_SAMPLES_DIR  = Path(__file__).parent / "cuad_samples"
BATCH_SIZE        = 10          # contracts per LexGraph job
POLL_INTERVAL_S   = 10
POLL_TIMEOUT_S    = 1200        # 20 min — 10-contract batches can be slow

# Standard CUAD clause-type questions fired against every batch
CUAD_QUERIES = [
    "What is the governing law jurisdiction?",
    "What is the liability cap or limitation of liability?",
    "What are the confidentiality obligations?",
    "What are the termination conditions?",
    "What are the indemnification obligations?",
]

# ── ASTR-O import ─────────────────────────────────────────────────────────────

sys.path.insert(0, ASTR_O_PATH)
try:
    from integration.lexgraph_to_astr_o import LexGraphToASTRO
    ASTR_O_AVAILABLE = True
except ImportError:
    ASTR_O_AVAILABLE = False
    print("[warn] ASTR-O not found — span validation only")


# ── Span helpers (same as test_runner_for_astr_o.py) ─────────────────────────

def build_span(job_id: str, qa_response: dict) -> dict:
    enriched = qa_response.get("enriched_chunks") or []
    retrieved_chunks = [
        {
            "chunk_id": c["chunk_id"],
            "source":   f"{c['doc_id']}.txt",
            "text":     c["text"],
            "metadata": {"source_tier": "SUPPORTING"},
        }
        for c in enriched
    ]
    return {
        "span_id":            str(uuid.uuid4()),
        "trace_id":           job_id,
        "retrieved_chunks":   retrieved_chunks,
        "retrieval_metadata": qa_response.get("retrieval_metadata") or {},
        "enriched_chunks":    enriched,
        "llm_response":       {"text": qa_response.get("answer", ""), "logprobs": []},
    }


def validate_span(span: dict) -> tuple[bool, str]:
    for field in ("span_id", "trace_id", "retrieved_chunks", "retrieval_metadata", "llm_response"):
        if field not in span:
            return False, f"missing field: {field}"
    meta = span["retrieval_metadata"]
    for field in ("query", "retrieval_method", "top_k", "retrieved_chunks", "all_ranked_chunks"):
        if field not in meta:
            return False, f"missing retrieval_metadata.{field}"
    if not meta.get("all_ranked_chunks"):
        return False, "all_ranked_chunks is empty"
    for chunk in meta["all_ranked_chunks"]:
        for field in ("chunk_id", "rank", "dense_score", "sparse_score", "rrf_score"):
            if field not in chunk:
                return False, f"chunk missing field: {field}"
    return True, ""


# ── Job lifecycle helpers ─────────────────────────────────────────────────────

def submit_job(client: httpx.Client, file_paths: list[Path]) -> str:
    files = [
        ("files", (p.name, p.read_bytes(), "application/octet-stream"))
        for p in file_paths
    ]
    resp = client.post("/jobs", files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()["job_id"]


def poll_until_done(client: httpx.Client, job_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        resp = client.get(f"/jobs/{job_id}", timeout=15)
        resp.raise_for_status()
        record = resp.json()
        status = record["status"]
        if status == "done":
            return record
        if status == "error":
            raise RuntimeError(f"job {job_id} failed: {record.get('errors')}")
        print(f"    [{job_id[:8]}] status={status} …")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"job {job_id} timed out after {POLL_TIMEOUT_S}s")


def run_qa(client: httpx.Client, job_id: str, question: str) -> dict:
    resp = client.post(
        f"/jobs/{job_id}/qa",
        json={"question": question},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


# ── Per-batch runner ──────────────────────────────────────────────────────────

class BatchRunner:
    def __init__(self, registry_path: str, mission_id: str):
        self.mission_id = mission_id
        self.results: list[dict] = []
        self.harness = (
            LexGraphToASTRO(
                registry_path=registry_path,
                hot_storage_path="/tmp/astr_o_hot",
                cold_storage_path="/tmp/astr_o_cold",
                mission_id=mission_id,
            )
            if ASTR_O_AVAILABLE else None
        )

    def run_batch(
        self,
        client: httpx.Client,
        batch_idx: int,
        contract_paths: list[Path],
    ) -> list[dict]:
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx+1}  ({len(contract_paths)} contracts)")
        print(f"{'='*60}")

        # ── ingest ────────────────────────────────────────────────────────────
        print("  [1/3] Submitting job …")
        job_id = submit_job(client, contract_paths)
        print(f"        job_id={job_id}")

        print("  [2/3] Waiting for pipeline …")
        job_record = poll_until_done(client, job_id)
        doc_ids = job_record.get("doc_ids", [])
        print(f"        done — {len(doc_ids)} doc(s) indexed")

        # ── Q&A + ASTR-O ─────────────────────────────────────────────────────
        print(f"  [3/3] Running {len(CUAD_QUERIES)} queries …")
        batch_results = []
        for query in CUAD_QUERIES:
            result = self._run_query(client, job_id, query, batch_idx)
            batch_results.append(result)

        safe    = sum(1 for r in batch_results if r.get("astr_o_status") == "SAFE")
        flagged = sum(1 for r in batch_results if r.get("astr_o_status") == "FLAGGED")
        print(f"        SAFE={safe}  FLAGGED={flagged}")
        return batch_results

    def _run_query(
        self,
        client: httpx.Client,
        job_id: str,
        query: str,
        batch_idx: int,
    ) -> dict:
        try:
            qa = run_qa(client, job_id, query)
            span = build_span(job_id, qa)
            is_valid, err = validate_span(span)

            if not is_valid:
                return {
                    "batch": batch_idx + 1, "query": query, "job_id": job_id,
                    "span_valid": False, "astr_o_status": "N/A", "error": err,
                }

            astr_o_status = "N/A"
            signature = "N/A"
            astr_o_err = None
            failed_criteria: list[str] = []

            if self.harness:
                astr_o_result = self.harness.process_lexgraph_span(span)
                if astr_o_result.get("error"):
                    astr_o_err = astr_o_result["error"]
                    astr_o_status = "ERROR"
                else:
                    astr_o_status = astr_o_result.get("status", "UNKNOWN")
                    signature = (
                        astr_o_result.get("signed_report", {})
                        .get("signature_metadata", {})
                        .get("signature", "N/A")
                    )
                    # Extract which criteria failed for diagnostics
                    report = (
                        astr_o_result.get("signed_report", {})
                        .get("report", {})
                    )
                    criteria = report.get("criteria_evaluation", {})
                    failed_criteria = [
                        v.get("name", k)
                        for k, v in criteria.items()
                        if not v.get("passed", True)
                    ]

            return {
                "batch":          batch_idx + 1,
                "query":          query,
                "job_id":         job_id,
                "span_id":        span["span_id"],
                "span_valid":     True,
                "astr_o_status":  astr_o_status,
                "failed_criteria": failed_criteria,
                "retrieved":      len(span["retrieved_chunks"]),
                "ranked":         len(span["retrieval_metadata"].get("all_ranked_chunks", [])),
                "report_sig":     signature[:16] + "…" if signature != "N/A" else "N/A",
                "error":          astr_o_err,
            }

        except Exception as exc:
            return {
                "batch": batch_idx + 1, "query": query, "job_id": job_id,
                "span_valid": False, "astr_o_status": "ERROR", "error": str(exc),
            }

    def summarise(self, all_results: list[dict]) -> dict:
        valid   = sum(1 for r in all_results if r.get("span_valid"))
        safe    = sum(1 for r in all_results if r.get("astr_o_status") == "SAFE")
        flagged = sum(1 for r in all_results if r.get("astr_o_status") == "FLAGGED")
        errors  = sum(1 for r in all_results if r.get("astr_o_status") == "ERROR")

        # Criteria failure frequency
        criteria_counts: dict[str, int] = {}
        for r in all_results:
            for c in r.get("failed_criteria", []):
                criteria_counts[c] = criteria_counts.get(c, 0) + 1

        return {
            "total_spans":    len(all_results),
            "valid_spans":    valid,
            "safe":           safe,
            "flagged":        flagged,
            "errors":         errors,
            "safe_rate":      round(safe / valid, 3) if valid else 0.0,
            "criteria_failure_counts": dict(
                sorted(criteria_counts.items(), key=lambda x: -x[1])
            ),
        }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # ── Verify cuad_samples/ exists ───────────────────────────────────────────
    if not CUAD_SAMPLES_DIR.exists() or not list(CUAD_SAMPLES_DIR.glob("*.txt")):
        print(f"[error] {CUAD_SAMPLES_DIR} is empty — run setup_cuad_dataset.py first")
        sys.exit(1)

    contracts = sorted(CUAD_SAMPLES_DIR.glob("*.txt"))[:30]
    batches = [contracts[i:i+BATCH_SIZE] for i in range(0, len(contracts), BATCH_SIZE)]

    mission_id    = "CUAD_ASTR_O_TEST_2026"
    registry_path = f"{ASTR_O_PATH}/reference_registry.json"

    print(f"\n{'#'*60}")
    print(f"  CUAD × ASTR-O Integration Test")
    print(f"  Contracts : {len(contracts)}")
    print(f"  Batches   : {len(batches)} × {BATCH_SIZE}")
    print(f"  Queries   : {len(CUAD_QUERIES)} per batch")
    print(f"  Total spans: {len(batches) * len(CUAD_QUERIES)}")
    print(f"  ASTR-O    : {'connected' if ASTR_O_AVAILABLE else 'NOT FOUND'}")
    print(f"{'#'*60}")

    runner = BatchRunner(registry_path=registry_path, mission_id=mission_id)
    all_results: list[dict] = []

    with httpx.Client(base_url=LEXGRAPH_BASE_URL) as client:
        for i, batch in enumerate(batches):
            batch_results = runner.run_batch(client, i, batch)
            all_results.extend(batch_results)

    summary = runner.summarise(all_results)

    report = {
        "mission_id":    mission_id,
        "contracts":     len(contracts),
        "batches":       len(batches),
        "queries":       CUAD_QUERIES,
        "results":       all_results,
        "summary":       summary,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total spans  : {summary['total_spans']}")
    print(f"  Valid spans  : {summary['valid_spans']}")
    print(f"  SAFE         : {summary['safe']}  ({summary['safe_rate']:.1%})")
    print(f"  FLAGGED      : {summary['flagged']}")
    print(f"  Errors       : {summary['errors']}")
    if summary["criteria_failure_counts"]:
        print(f"\n  Criteria failure counts:")
        for name, count in summary["criteria_failure_counts"].items():
            print(f"    {count:>3d}x  {name}")

    report_path = f"cuad_astr_o_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report → {report_path}")


if __name__ == "__main__":
    main()
