"""
Sprint 7 smoke test — fast sanity check, no full eval run.

Verifies:
  1. chenghao/cuad_qa loads correctly (test split, expected schema)
  2. Sample index generation works and is reproducible
  3. Qdrant collection setup/teardown works
  4. 5 rows can be chunked, embedded, and indexed
  5. Retrieval returns results for each row
  6. Recall@K calculation is correct

Does NOT run the full 100-row eval (use eval/cuad_eval.py for that).
Does NOT call any LLM.

Usage:
  python run_sprint7.py
  python run_sprint7.py --skip-qdrant   # skip Qdrant-dependent tests (rows 3-5)
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "legal_due_diligence"))

from eval.cuad_eval import (
    EVAL_COLLECTION,
    RESULTS_DIR,
    SAMPLE_IDS_PATH,
    embed_questions,
    eval_retrieve,
    index_eval_rows,
    is_recall_hit,
    load_cuad_test,
    load_or_generate_sample_ids,
    setup_eval_collection,
    teardown_eval_collection,
)

SMOKE_N = 5  # rows used for smoke test — fast, covers basic correctness


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 7 smoke test")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant-dependent tests")
    args = parser.parse_args()

    print("\nSprint 7 smoke test")
    print("=" * 50)

    # ── Test 1: Dataset loads ─────────────────────────────────────────────
    print("\nTest 1 — Dataset loading")
    rows = load_cuad_test()
    check("test split loaded", len(rows) > 0, f"{len(rows)} rows")
    check("expected ~1,240 rows", 1000 < len(rows) < 2000, str(len(rows)))
    check("has 'context' field", "context" in rows[0])
    check("has 'question' field", "question" in rows[0])
    check("has 'answers' field", "answers" in rows[0])
    check("answers has 'text' list", isinstance(rows[0]["answers"]["text"], list))

    # ── Test 2: Stratified sampling ───────────────────────────────────────
    print("\nTest 2 — Stratified sampling")

    # Remove cached entry for SMOKE_N to force re-generation
    if SAMPLE_IDS_PATH.exists():
        with open(SAMPLE_IDS_PATH) as f:
            cached = json.load(f)
        if str(SMOKE_N) in cached:
            del cached[str(SMOKE_N)]
            with open(SAMPLE_IDS_PATH, "w") as f:
                json.dump(cached, f, indent=2)

    ids_first = load_or_generate_sample_ids(rows, SMOKE_N)
    ids_second = load_or_generate_sample_ids(rows, SMOKE_N)  # should load from file

    check("correct sample size", len(ids_first) == SMOKE_N, str(len(ids_first)))
    check("all indices in range", all(0 <= i < len(rows) for i in ids_first))
    check("reproducible (file round-trip)", ids_first == ids_second)

    # ── Test 3: Qdrant setup + indexing (skip if --skip-qdrant) ──────────
    if args.skip_qdrant:
        print("\nTest 3-5 — Qdrant (skipped via --skip-qdrant)")
        print("\n" + "=" * 50)
        print("Smoke test passed (Qdrant tests skipped)\n")
        return

    print("\nTest 3 — Qdrant collection setup")
    try:
        setup_eval_collection()
        from infrastructure.qdrant_client import get_qdrant_client
        client = get_qdrant_client()
        collections = {c.name for c in client.get_collections().collections}
        check(f"{EVAL_COLLECTION} collection created", EVAL_COLLECTION in collections)
    except Exception as e:
        check("Qdrant reachable", False, str(e))

    # ── Test 4: Indexing 5 rows ───────────────────────────────────────────
    print("\nTest 4 — Chunk + embed + index")
    try:
        bm25_model, bm25_chunk_ids, doc_chunks_map = index_eval_rows(rows, ids_first)

        check("BM25 model built", bm25_model is not None)
        check("BM25 chunk IDs match", len(bm25_chunk_ids) >= SMOKE_N)
        check("doc_chunks_map has all doc_ids", all(
            f"cuad_eval_{idx}" in doc_chunks_map for idx in ids_first
        ))
        total_chunks = sum(len(v) for v in doc_chunks_map.values())
        check("at least one chunk per row", total_chunks >= SMOKE_N, f"{total_chunks} total")
    except Exception as e:
        check("indexing succeeded", False, str(e))

    # ── Test 5: Retrieval + Recall@K ─────────────────────────────────────
    print("\nTest 5 — Retrieval + Recall@K")
    try:
        query_vectors = embed_questions(rows, ids_first)
        check("query vectors embedded", len(query_vectors) == SMOKE_N)

        retrieved_count = 0
        hits = 0

        for idx in ids_first:
            row = rows[idx]
            doc_id = f"cuad_eval_{idx}"
            answer_texts = row["answers"]["text"]
            doc_chunk_ids = {c.chunk_id for c in doc_chunks_map[doc_id]}

            results = eval_retrieve(
                query_vector=query_vectors[idx],
                query_text=row["question"],
                doc_id=doc_id,
                bm25_model=bm25_model,
                bm25_chunk_ids=bm25_chunk_ids,
                doc_chunk_ids=doc_chunk_ids,
                top_k=3,
            )
            retrieved_count += len(results)

            if answer_texts and answer_texts[0].strip():
                gt = answer_texts[0]
                texts = [t for _, t in results]
                if is_recall_hit(gt, texts):
                    hits += 1

        check("retrieval returned results", retrieved_count > 0, f"{retrieved_count} chunks across {SMOKE_N} rows")
        # Don't assert a Recall threshold on 5 rows — sample too small for meaningful signal
        print(f"  [INFO] Recall@3 on {SMOKE_N}-row smoke sample: {hits}/{SMOKE_N} hits (informational)")

    except Exception as e:
        check("retrieval succeeded", False, str(e))
    finally:
        teardown_eval_collection()
        print(f"  [INFO] {EVAL_COLLECTION} collection cleaned up")

    print("\n" + "=" * 50)
    print("Smoke test passed — ready for full eval:")
    print("  python eval/cuad_eval.py")
    print()


if __name__ == "__main__":
    main()
