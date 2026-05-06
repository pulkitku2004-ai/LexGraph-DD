"""
test_runner_for_astr_o.py

Wraps the full LexGraph job lifecycle and pipes span dicts to ASTR-O.

Flow per run:
  1. POST /jobs       — upload documents, trigger pipeline
  2. GET  /jobs/{id}  — poll until status='done'
  3. POST /jobs/{id}/qa — fire each test query, get back answer + retrieval_metadata
  4. Build ASTR-O span dict from QA response
  5. LexGraphToASTRO.process_lexgraph_span(span) — run through ASTR-O pipeline
  6. Collect SAFE/FLAGGED verdicts, save JSON report

Requires:
  - LexGraph API server running: uvicorn legal_due_diligence.api.main:app --port 8000
  - Qdrant + Neo4j running: docker compose up -d
  - ASTR-O integration module at ASTR_O_PATH (see config below)
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

# ── ASTR_O_REPORT_SECRET — required by integrity_signer.sign_report().
# Set this in your shell or .env before running. The default is dev-only.
os.environ.setdefault("ASTR_O_REPORT_SECRET", "lexgraph-astr-o-dev")

# ── Config ────────────────────────────────────────────────────────────────────

LEXGRAPH_BASE_URL = "http://localhost:8000"
POLL_INTERVAL_S = 5       # seconds between GET /jobs/{id} polls
POLL_TIMEOUT_S  = 600     # give up after 10 min

ASTR_O_PATH = "/Users/dr_bolty/astr-o"

# ── ASTR-O import (optional — runner validates spans even without it) ─────────

try:
    sys.path.insert(0, ASTR_O_PATH)
    from integration.lexgraph_to_astr_o import LexGraphToASTRO
    ASTR_O_AVAILABLE = True
except ImportError:
    ASTR_O_AVAILABLE = False


# ── Span builder ──────────────────────────────────────────────────────────────

def build_span(job_id: str, qa_response: dict) -> dict:
    """
    Build an ASTR-O-compatible span dict from a LexGraph QA response.

    The QA response now carries:
      retrieval_metadata: {query, retrieval_method, top_k,
                           retrieved_chunks, all_ranked_chunks, retrieval_timestamp}
      enriched_chunks:    [{chunk_id, doc_id, page_number, text}, ...]
      answer:             str

    ASTR-O expects:
      span_id, trace_id, retrieval_metadata, enriched_chunks, llm_response
    """
    enriched = qa_response.get("enriched_chunks") or []

    # ASTR-O Layer 1 reads retrieved_chunks[] at the top level.
    # Map LexGraph's enriched_chunks (chunk_id, doc_id, page_number, text)
    # → ASTR-O chunk shape (chunk_id, source, text, metadata.source_tier).
    # source_tier "SUPPORTING" matches what setup_astr_o_registry.py writes
    # for legal contracts — metadata_bridge will return VERIFIED.
    retrieved_chunks = [
        {
            "chunk_id": c["chunk_id"],
            "source":   f"{c['doc_id']}.txt",
            "text":     c["text"],
            "metadata": {"source_tier": "SUPPORTING"},
        }
        for c in enriched
    ]

    # ASTR-O expects llm_response as {"text": str, "logprobs": [...]}.
    # LexGraph does not request logprobs from the LLM — empty list is safe;
    # token_confidence_analyzer handles it gracefully (no critical tokens found).
    return {
        "span_id":            str(uuid.uuid4()),
        "trace_id":           job_id,
        "retrieved_chunks":   retrieved_chunks,
        "retrieval_metadata": qa_response.get("retrieval_metadata") or {},
        "enriched_chunks":    enriched,
        "llm_response":       {"text": qa_response.get("answer", ""), "logprobs": []},
    }


# ── Span validator ────────────────────────────────────────────────────────────

def validate_span(span: dict) -> tuple[bool, str]:
    """
    Validate span has all fields ASTR-O requires before passing it in.
    Returns (is_valid, error_message).
    """
    for field in ("span_id", "trace_id", "retrieved_chunks", "retrieval_metadata", "llm_response"):
        if field not in span:
            return False, f"missing top-level field: {field}"

    meta = span["retrieval_metadata"]
    for field in ("query", "retrieval_method", "top_k", "retrieved_chunks", "all_ranked_chunks"):
        if field not in meta:
            return False, f"missing retrieval_metadata.{field}"

    ranked = meta.get("all_ranked_chunks", [])
    if not ranked:
        return False, "all_ranked_chunks is empty"

    for chunk in ranked:
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
    resp = client.post("/jobs", files=files, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"  submitted job_id={job_id}  ({len(file_paths)} doc(s))")
    return job_id


def poll_until_done(client: httpx.Client, job_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        resp = client.get(f"/jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        record = resp.json()
        status = record["status"]
        if status == "done":
            return record
        if status == "error":
            raise RuntimeError(f"Job {job_id} failed: {record.get('errors')}")
        print(f"  status={status} — waiting {POLL_INTERVAL_S}s …")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"Job {job_id} did not complete within {POLL_TIMEOUT_S}s")


def run_qa(client: httpx.Client, job_id: str, question: str) -> dict:
    resp = client.post(
        f"/jobs/{job_id}/qa",
        json={"question": question},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ── Test runner ───────────────────────────────────────────────────────────────

class LexGraphTestRunner:
    def __init__(
        self,
        document_paths: list[Path],
        registry_path: str = "/Users/dr_bolty/astr-o/reference_registry.json",
        hot_storage_path: str = "/tmp/astr_o_hot",
        cold_storage_path: str = "/tmp/astr_o_cold",
        mission_id: str = "CUAD_ASTR_O_TEST_2026",
    ):
        self.document_paths = document_paths
        self.mission_id = mission_id
        self.results: list[dict] = []

        if ASTR_O_AVAILABLE:
            self.harness = LexGraphToASTRO(
                registry_path=registry_path,
                hot_storage_path=hot_storage_path,
                cold_storage_path=cold_storage_path,
                mission_id=mission_id,
            )
        else:
            self.harness = None

    def run(self, queries: list[str]) -> dict:
        print(f"\n{'#'*60}")
        print(f"  ASTR-O Integration Test — {self.mission_id}")
        print(f"  Docs: {[p.name for p in self.document_paths]}")
        print(f"  Queries: {len(queries)}")
        print(f"  ASTR-O harness: {'connected' if ASTR_O_AVAILABLE else 'NOT FOUND — span validation only'}")
        print(f"{'#'*60}\n")

        with httpx.Client(base_url=LEXGRAPH_BASE_URL) as client:
            # ── Step 1-2: job lifecycle ─────────────────────────────────────
            print("[1/2] Submitting job …")
            job_id = submit_job(client, self.document_paths)

            print("[2/2] Waiting for pipeline to complete …")
            job_record = poll_until_done(client, job_id)
            doc_ids = job_record.get("doc_ids", [])
            print(f"  done — doc_ids: {doc_ids}\n")

            # ── Step 3-5: per-query QA + ASTR-O ────────────────────────────
            for i, query in enumerate(queries, 1):
                print(f"{'='*60}")
                print(f"Query {i}/{len(queries)}: {query}")
                print(f"{'='*60}")
                result = self._run_query(client, job_id, query)
                self.results.append(result)
                print()

        # ── Summary ─────────────────────────────────────────────────────────
        summary = self._summarise()
        report = {
            "mission_id":    self.mission_id,
            "job_id":        job_id,
            "doc_ids":       doc_ids,
            "queries_run":   len(queries),
            "results":       self.results,
            "summary":       summary,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  Valid spans:     {summary['valid_spans']} / {len(queries)}")
        if ASTR_O_AVAILABLE:
            print(f"  ASTR-O SAFE:     {summary['safe']}")
            print(f"  ASTR-O FLAGGED:  {summary['flagged']}")
            print(f"  Errors:          {summary['errors']}")
        return report

    def _run_query(self, client: httpx.Client, job_id: str, query: str) -> dict:
        try:
            # ── Q&A call ────────────────────────────────────────────────────
            print("  [1/3] POST /qa …")
            qa = run_qa(client, job_id, query)

            # ── Build span dict ──────────────────────────────────────────────
            print("  [2/3] Building span dict …")
            span = build_span(job_id, qa)

            # ── Validate ────────────────────────────────────────────────────
            is_valid, err = validate_span(span)
            if not is_valid:
                print(f"  ✗ span invalid — {err}")
                return {
                    "query": query, "span_id": span.get("span_id"),
                    "span_valid": False, "astr_o_status": "N/A", "error": err,
                }

            meta = span["retrieval_metadata"]
            print(f"  ✓ span valid")
            print(f"    retrieved_chunks:  {len(meta['retrieved_chunks'])}")
            print(f"    all_ranked_chunks: {len(meta['all_ranked_chunks'])}")

            # ── ASTR-O ──────────────────────────────────────────────────────
            print("  [3/3] ASTR-O pipeline …")
            astr_o_status = "N/A"
            signature = "N/A"
            astr_o_err = None

            if self.harness:
                astr_o_result = self.harness.process_lexgraph_span(span)
                if astr_o_result.get("error"):
                    astr_o_err = astr_o_result["error"]
                    astr_o_status = "ERROR"
                    print(f"  ✗ ASTR-O error — {astr_o_err}")
                else:
                    astr_o_status = astr_o_result.get("status", "UNKNOWN")
                    signature = (
                        astr_o_result.get("signed_report", {})
                        .get("signature_metadata", {})
                        .get("signature", "N/A")
                    )
                    print(f"  ✓ verdict={astr_o_status}  sig={signature[:16]}…")
            else:
                print("  (ASTR-O not connected — skipped)")

            return {
                "query":          query,
                "span_id":        span["span_id"],
                "span_valid":     True,
                "astr_o_status":  astr_o_status,
                "retrieved":      len(meta["retrieved_chunks"]),
                "ranked":         len(meta["all_ranked_chunks"]),
                "report_sig":     signature,
                "error":          astr_o_err,
            }

        except Exception as exc:
            print(f"  ✗ exception — {exc}")
            return {
                "query": query, "span_id": None,
                "span_valid": False, "astr_o_status": "ERROR", "error": str(exc),
            }

    def _summarise(self) -> dict:
        valid   = sum(1 for r in self.results if r.get("span_valid"))
        safe    = sum(1 for r in self.results if r.get("astr_o_status") == "SAFE")
        flagged = sum(1 for r in self.results if r.get("astr_o_status") == "FLAGGED")
        errors  = sum(1 for r in self.results if r.get("astr_o_status") == "ERROR")
        return {"valid_spans": valid, "safe": safe, "flagged": flagged, "errors": errors}

    def save_report(self, report: dict, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Documents to upload — default: sample contracts in the repo
    docs = [
        Path("samples/contract_a.txt"),
        Path("samples/contract_b.txt"),
    ]

    # Queries covering CUAD clause categories present in the sample contracts
    queries = [
        "What is the governing law jurisdiction?",
        "What are the confidentiality obligations?",
        "What is the liability cap?",
        "What are the termination conditions?",
        "What are the payment terms?",
    ]

    runner = LexGraphTestRunner(
        document_paths=docs,
        hot_storage_path="/tmp/astr_o_hot",
        cold_storage_path="/tmp/astr_o_cold",
        mission_id="CUAD_ASTR_O_TEST_2026",
    )

    report = runner.run(queries)

    report_path = f"astr_o_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    runner.save_report(report, report_path)


if __name__ == "__main__":
    main()
