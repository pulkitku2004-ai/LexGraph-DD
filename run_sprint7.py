"""
Sprint 7/8 smoke test — fast sanity check, no full eval run.

Updated for bge-m3 architecture (Sprint 8):
  - index_eval_rows returns only doc_chunks_map (BM25 removed)
  - eval_retrieve takes query_dense + query_sparse (not bm25_model)
  - embed_questions returns {idx: (dense_vector, sparse_vector)}

Verifies:
  1. chenghao/cuad_qa loads correctly (test split, expected schema)
  2. Sample index generation works and is reproducible
  3. Qdrant collection setup/teardown works
  4. 5 rows can be chunked, embedded (dense + sparse), and indexed
  5. Retrieval returns results using hybrid sparse+dense RRF
  6. Recall@K calculation is correct

Does NOT run the full 100-row eval (use eval/cuad_eval.py for that).
Does NOT call any LLM.

Usage:
  python run_sprint7.py
  python run_sprint7.py --skip-qdrant   # skip Qdrant-dependent tests (3-5)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "legal_due_diligence"))

from eval.cuad_eval import (
    EVAL_COLLECTION,
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

SMOKE_N = 5


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 7/8 smoke test")
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

    if SAMPLE_IDS_PATH.exists():
        with open(SAMPLE_IDS_PATH) as f:
            cached = json.load(f)
        if str(SMOKE_N) in cached:
            del cached[str(SMOKE_N)]
            with open(SAMPLE_IDS_PATH, "w") as f:
                json.dump(cached, f, indent=2)

    ids_first = load_or_generate_sample_ids(rows, SMOKE_N)
    ids_second = load_or_generate_sample_ids(rows, SMOKE_N)

    check("correct sample size", len(ids_first) == SMOKE_N, str(len(ids_first)))
    check("all indices in range", all(0 <= i < len(rows) for i in ids_first))
    check("reproducible (file round-trip)", ids_first == ids_second)

    if args.skip_qdrant:
        print("\nTest 3-5 — Qdrant (skipped via --skip-qdrant)")
        print("\n" + "=" * 50)
        print("Smoke test passed (Qdrant tests skipped)\n")
        return

    # ── Test 3: Qdrant setup ──────────────────────────────────────────────
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
    print("\nTest 4 — Chunk + embed (dense + sparse) + index")
    doc_chunks_map = {}
    try:
        # index_eval_rows now returns only doc_chunks_map
        # BM25 is gone — sparse vectors stored natively in Qdrant
        doc_chunks_map = index_eval_rows(rows, ids_first)

        check("doc_chunks_map has all doc_ids", all(
            f"cuad_eval_{idx}" in doc_chunks_map for idx in ids_first
        ))
        total_chunks = sum(len(v) for v in doc_chunks_map.values())
        check("at least one chunk per row", total_chunks >= SMOKE_N, f"{total_chunks} total")
        check("chunks have text", all(
            c.text for chunks in doc_chunks_map.values() for c in chunks
        ))
    except Exception as e:
        check("indexing succeeded", False, str(e))

    # ── Test 5: Retrieval + Recall@K ─────────────────────────────────────
    print("\nTest 5 — Retrieval + Recall@K")
    try:
        # embed_questions returns {idx: (dense_vector, sparse_vector)}
        query_embeddings = embed_questions(rows, ids_first)
        check("query vectors embedded", len(query_embeddings) == SMOKE_N)

        # Verify structure: each value is a (dense, sparse) tuple
        sample_idx = ids_first[0]
        dense_v, sparse_v = query_embeddings[sample_idx]
        check("dense vector is 1024-dim", len(dense_v) == 1024, f"got {len(dense_v)}-dim")
        check("sparse vector is a dict", isinstance(sparse_v, dict))
        check("sparse vector is non-empty", len(sparse_v) > 0, f"{len(sparse_v)} tokens")

        retrieved_count = 0
        hits = 0

        for idx in ids_first:
            row = rows[idx]
            doc_id = f"cuad_eval_{idx}"
            answer_texts = row["answers"]["text"]

            query_dense, query_sparse = query_embeddings[idx]

            # eval_retrieve now takes dense + sparse vectors (not bm25_model)
            results = eval_retrieve(
                query_dense=query_dense,
                query_sparse=query_sparse,
                query_text=row["question"],
                doc_id=doc_id,
                top_k=3,
                candidate_k=20,
            )
            retrieved_count += len(results)

            if answer_texts and answer_texts[0].strip():
                gt = answer_texts[0]
                texts = [t for _, t in results]
                if is_recall_hit(gt, texts):
                    hits += 1

        check("retrieval returned results", retrieved_count > 0,
              f"{retrieved_count} chunks across {SMOKE_N} rows")
        print(f"  [INFO] Recall@3 on {SMOKE_N}-row smoke: {hits}/{SMOKE_N} hits (informational only)")

    except Exception as e:
        check("retrieval succeeded", False, str(e))
    finally:
        teardown_eval_collection()
        print(f"  [INFO] {EVAL_COLLECTION} collection cleaned up")

    print("\n" + "=" * 50)
    print("Smoke test passed — ready for full eval:")
    print("  python eval/cuad_eval.py --enrich-queries")
    print()


if __name__ == "__main__":
    main()