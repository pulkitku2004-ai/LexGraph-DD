"""
Sprint 1 smoke test — ingestion pipeline end-to-end.

Uses CUAD dataset text directly (no PDF needed) to test:
  chunker → embedder → indexer → Qdrant retrieval verification

Run from repo root:
    python run_sprint1.py
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "legal_due_diligence"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from ingestion.chunker import chunk_document
    from ingestion.embedder import embed_chunks
    from ingestion.indexer import index_chunks
    from ingestion.loader import load_document
    from infrastructure.qdrant_client import get_qdrant_client
    from core.config import settings

    logger.info("=" * 60)
    logger.info("SPRINT 1 — ingestion pipeline smoke test")
    logger.info("=" * 60)

    # ── Load sample contracts ─────────────────────────────────────────────
    # Two real-shaped contracts with deliberate contradictions (governing law,
    # payment terms, confidentiality survival, liability caps) for Sprint 5.
    sample_files = [
        ("./samples/contract_a.txt", "contract-a"),
        ("./samples/contract_b.txt", "contract-b"),
    ]

    documents = []
    for file_path, doc_id in sample_files:
        # loader.py handles PDF and DOCX — .txt goes through the DOCX path
        # We'll add a txt branch in loader.py or just read directly here for Sprint 1
        from ingestion.loader import LoadedDocument, PagedText
        text = Path(file_path).read_text()
        page = PagedText(
            doc_id=doc_id,
            file_path=file_path,
            page_number=1,
            text=text,
            total_pages=1,
        )
        doc = LoadedDocument(
            doc_id=doc_id,
            file_path=file_path,
            total_pages=1,
            pages=[page],
        )
        documents.append(doc)
        logger.info("Loaded: %s (%d chars)", doc_id, len(text))

    # ── Chunk ─────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Chunking documents...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        logger.info("  %s → %d chunks", doc.doc_id, len(chunks))
        all_chunks.extend(chunks)
    logger.info("Total chunks: %d", len(all_chunks))

    # ── Embed ─────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Embedding chunks (legal-bert on MPS)... this takes ~30-60s first run")
    embedded = embed_chunks(all_chunks, batch_size=32)
    logger.info("Embedded %d chunks, vector dim: %d", len(embedded), len(embedded[0].vector))

    # ── Index into Qdrant ─────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Indexing into Qdrant + BM25...")
    index_chunks(embedded)

    # ── Verify retrieval ──────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Verifying Qdrant retrieval with a test query...")

    # Embed the query using the same model
    from ingestion.chunker import Chunk
    query_chunk = Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id="query",
        file_path="",
        page_number=0,
        text="governing law jurisdiction Delaware",
        token_count=10,
        chunk_index=0,
    )
    query_embedded = embed_chunks([query_chunk])
    query_vector = query_embedded[0].vector

    client = get_qdrant_client()
    from qdrant_client.http.models import ScoredPoint
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        using="dense",
        limit=3,
    )
    results: list[ScoredPoint] = response.points

    logger.info("Top 3 results for query 'governing law jurisdiction Delaware':")
    for r in results:
        payload = r.payload or {}
        snippet = str(payload.get("text", ""))[:120].replace("\n", " ")
        logger.info("  [score=%.3f] doc=%s page=%d | %s...",
            r.score, payload.get("doc_id"), payload.get("page_number"), snippet)

    # ── Assertions ────────────────────────────────────────────────────────
    assert len(all_chunks) > 0, "No chunks produced"
    assert len(embedded) == len(all_chunks), "Embedding count mismatch"
    assert len(results) > 0, "Qdrant returned no results"
    assert all(len(e.vector) == 1024 for e in embedded), "Wrong vector dimension"
    assert all(len(e.sparse_vector) > 0 for e in embedded), "Empty sparse vectors"

    logger.info("=" * 60)
    logger.info("Sprint 1 PASSED")


if __name__ == "__main__":
    main()
