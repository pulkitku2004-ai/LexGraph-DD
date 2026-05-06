"""
Live test: verify retrieve_with_metadata() returns correct retrieval_metadata shape.

Requires: Qdrant running with legal_clauses collection indexed (docker compose up -d + run_sprint1.py).
Tests the same code path that _extract_category_async uses before attaching to an OTel span.
"""

import json
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "legal_due_diligence"))

from agents.clause_extractor.retriever import retrieve_with_metadata

# Use a real doc_id from the indexed sample contracts and a CUAD-style clause query
DOC_ID = "contract-a"
QUERY = "governing law jurisdiction which state"

print(f"Querying doc_id={DOC_ID!r} | query={QUERY!r}")
print()

chunks, all_ranked = retrieve_with_metadata(
    query_text=QUERY,
    doc_id=DOC_ID,
    top_k=3,
    candidate_k=20,
)

# ── Assemble retrieval_metadata exactly as _extract_category_async does ─────────
retrieval_metadata = {
    "query": QUERY,
    "alt_queries": [],
    "retrieval_method": "hybrid_rrf",
    "retrieved_chunk_ids": [c.chunk_id for c in chunks],
    "all_ranked_chunks": all_ranked,
    "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
}

# ── Assertions ───────────────────────────────────────────────────────────────────
assert all_ranked, "all_ranked_chunks is empty — no results from Qdrant"

for entry in all_ranked:
    assert "dense_score" in entry,    f"dense_score missing on {entry['chunk_id']}"
    assert "sparse_score" in entry,   f"sparse_score missing on {entry['chunk_id']}"
    assert "rrf_score" in entry,      f"rrf_score missing on {entry['chunk_id']}"
    assert "rank" in entry
    assert "chunk_id" in entry
    assert "reason_for_rank" in entry

# retrieved_chunk_ids must be a subset of all_ranked_chunks
all_ids = {e["chunk_id"] for e in all_ranked}
for cid in retrieval_metadata["retrieved_chunk_ids"]:
    assert cid in all_ids, f"retrieved chunk {cid} not in all_ranked_chunks"

# Must be JSON-serializable — this is the exact call span.set_attribute makes
serialized = json.dumps(retrieval_metadata)
assert isinstance(serialized, str)

# ── Print ─────────────────────────────────────────────────────────────────────
print("✓ LexGraph emits full retrieval_metadata with all scores")
print(f"✓ Total ranked chunks captured (pre-truncation): {len(all_ranked)}")
print(f"✓ Retrieved chunk IDs sent to LLM:               {len(retrieval_metadata['retrieved_chunk_ids'])}")
print(f"✓ JSON-serializable — {len(serialized)} bytes (ready for span.set_attribute)")
print()
print("all_ranked_chunks[0]:")
for k, v in all_ranked[0].items():
    print(f"  {k}: {v}")

if len(all_ranked) > 1:
    print()
    print("all_ranked_chunks[-1] (lowest-ranked candidate):")
    for k, v in all_ranked[-1].items():
        print(f"  {k}: {v}")
