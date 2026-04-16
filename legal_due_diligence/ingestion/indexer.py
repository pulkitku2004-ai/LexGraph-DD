"""
Indexer — EmbeddedChunk → Qdrant collection (dense + sparse vectors).

Why two vector types in the same Qdrant collection?
bge-m3 produces dense (1024-dim) and sparse (learned SPLADE weights) vectors
in a single forward pass. Both are stored in the same Qdrant point so a single
collection holds both retrieval signals. At query time, we issue two separate
Qdrant queries — one dense, one sparse — then fuse their ranked lists with RRF.

Why named vectors ({"dense": ..., "sparse": ...}) instead of the legacy
single-vector layout?
Qdrant requires named vectors when a collection stores more than one vector type
per point. The names must match between collection creation, upsert, and query.
settings.sparse_vector_name ("sparse") is the shared constant.

Why replace BM25 with Qdrant-native sparse vectors?
The BM25 pickle had three drawbacks:
  1. Out-of-process: an extra file to manage, delete, and rebuild after re-indexing.
  2. Global index: BM25 indexed ALL documents; doc-scoping required a post-filter
     set intersection that was O(n) over the full corpus at query time.
  3. Hand-crafted tokenization: bm25_tokenize() needed to be kept in sync across
     indexer.py, retriever.py, and cuad_eval.py.
Qdrant sparse vectors are:
  - Per-point (no global index to manage)
  - Filterable by doc_id at the index level (same as dense)
  - Learned (the model allocates weight correctly without manual tokenisation)

Qdrant point structure (Sprint 8+):
  id:      chunk_id (UUID string)
  vector: {
    "dense":  1024-dim float list (cosine)
    "sparse": SparseVector(indices=[tok_ids], values=[weights])
  }
  payload: {
    doc_id, file_path, page_number, chunk_index, text, token_count
  }
"""

from __future__ import annotations

import logging

from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from core.config import settings
from infrastructure.qdrant_client import get_qdrant_client
from ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)


def _ensure_collection() -> None:
    """Create Qdrant collection if it doesn't exist.

    Uses named vectors: 'dense' (1024-dim cosine) + 'sparse' (learned weights).
    The sparse vector name is settings.sparse_vector_name so the same constant
    is used here, in retriever.py, and in cuad_eval.py.
    """
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                settings.sparse_vector_name: SparseVectorParams(),
            },
        )
        logger.info(
            "[indexer] created Qdrant collection '%s' (dense=%d-dim, sparse='%s')",
            settings.qdrant_collection,
            settings.embedding_dim,
            settings.sparse_vector_name,
        )
    else:
        logger.info("[indexer] collection '%s' already exists", settings.qdrant_collection)


def index_chunks(embedded_chunks: list[EmbeddedChunk]) -> None:
    """
    Upsert embedded chunks into Qdrant with both dense and sparse vectors.
    Safe to call multiple times — Qdrant upsert is idempotent on chunk_id.
    """
    if not embedded_chunks:
        logger.warning("[indexer] no chunks to index")
        return

    _ensure_collection()
    client = get_qdrant_client()

    points = [
        PointStruct(
            id=ec.chunk.chunk_id,
            vector={
                "dense": ec.vector,
                "sparse": SparseVector(
                    indices=list(ec.sparse_vector.keys()),
                    values=list(ec.sparse_vector.values()),
                ),
            },
            payload={
                "doc_id": ec.chunk.doc_id,
                "file_path": ec.chunk.file_path,
                "page_number": ec.chunk.page_number,
                "chunk_index": ec.chunk.chunk_index,
                "text": ec.chunk.text,
                "token_count": ec.chunk.token_count,
                # parent_text: the 512-token window this child was split from.
                # The retriever reads this and passes it to the LLM instead of
                # the 128-token child text, giving the LLM the full clause context.
                # None when chunks were not created with parent-child splitting.
                "parent_text": ec.chunk.parent_text,
                "parent_id": ec.chunk.parent_id,
                "parent_chunk_index": ec.chunk.parent_chunk_index,
            },
        )
        for ec in embedded_chunks
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=settings.qdrant_collection, points=batch)
        logger.debug(
            "[indexer] upserted %d/%d points",
            min(i + batch_size, len(points)),
            len(points),
        )

    logger.info("[indexer] indexed %d chunks into Qdrant", len(points))
