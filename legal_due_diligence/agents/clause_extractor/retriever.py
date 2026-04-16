"""
Hybrid retriever — bge-m3 learned sparse + dense vectors fused with RRF.

The retrieval problem for clause extraction:
  Given a clause category like "Governing Law", find the chunks in a specific
  document that are most likely to contain that clause.

Why two retrieval signals?
  Dense (bge-m3 1024-dim): semantic matching — "termination for convenience"
  matches "right to cancel without cause". Captures meaning even when
  exact keywords differ between query and clause text.

  Sparse (bge-m3 learned SPLADE weights): exact and near-exact term matching —
  "Section 12.3(b)", "$2,500,000", "force-majeure". Captures what BM25 did
  but with learned weights that understand legal term importance.

Why Qdrant sparse instead of BM25 pickle?
  1. Both retrievals are doc_id-filtered at the index level — no post-filter
     set intersection needed as with the BM25 global pickle.
  2. Learned sparse weights allocated by the model are more informative than
     BM25 TF-IDF for legal text (the model knows "notwithstanding" matters more
     than "the" even if TF-IDF says otherwise).
  3. No separate file to manage — sparse vectors live alongside dense in Qdrant.

How RRF works:
  1. Sparse query: Qdrant scores chunks by dot product over shared token IDs.
  2. Dense query: Qdrant scores chunks by cosine similarity of 1024-dim vectors.
  3. RRF assigns each chunk: score = 1/(k + rank_in_sparse) + 1/(k + rank_in_dense)
  4. Chunks that rank highly in BOTH lists score highest.

Why k=60?
The k constant controls rank sensitivity. At k=60, the score difference
between rank 1 and rank 2 is about 0.00027 — small. This means RRF is
not dominated by whichever list happens to put something at rank 1. It's
a consensus mechanism, not a winner-takes-all. The value 60 is the
original RRF paper's recommendation and holds up well empirically.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SparseVector,
)

from core.config import settings
from ingestion.embedder import EmbeddedChunk, embed_chunks
from ingestion.chunker import Chunk
from infrastructure.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)

# RRF rank smoothing constant — original paper value, don't change without benchmarking
RRF_K = 60


@dataclass
class RetrievedChunk:
    """A chunk returned by the hybrid retriever with its fused RRF score."""
    chunk_id: str
    doc_id: str
    page_number: int
    text: str               # child chunk text (256 tokens) — what was embedded
    rrf_score: float
    sparse_rank: int | None     # None if not in sparse top-k
    dense_rank: int | None      # None if not in dense top-k
    parent_text: str | None = None
    # parent_text: the 2048-token contiguous parent window this child belongs to.
    # The LLM receives parent_text (not text) for full clause context.
    parent_id: str | None = None
    # parent_id: UUID shared by all children of the same parent window.
    # Retriever deduplicates by parent_id so the LLM receives each parent once.
    parent_chunk_index: int | None = None
    # parent_chunk_index: document-order position of the parent (0, 1, 2…).
    # After score-order dedup, selected parents are re-sorted by this field
    # so the LLM sees them in document order (Article 2 before Article 10).


def _embed_query(query_text: str) -> tuple[list[float], dict[int, float]]:
    """
    Embed a query string using bge-m3, returning both dense and sparse vectors.
    Applies settings.embedding_query_prefix if set (empty for bge-m3).
    """
    prefix = settings.embedding_query_prefix
    text = prefix + query_text if prefix else query_text
    query_chunk = Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id="__query__",
        file_path="",
        page_number=0,
        text=text,
        token_count=0,
        chunk_index=0,
    )
    embedded: list[EmbeddedChunk] = embed_chunks([query_chunk])
    return embedded[0].vector, embedded[0].sparse_vector


def _qdrant_ranks(
    query: list[float] | SparseVector,
    using: str,
    doc_id: str,
    top_k: int,
) -> dict[str, tuple[int, str, int, str, str | None, str | None, int | None]]:
    """
    Fetch retrieval ranks from Qdrant for one query type, scoped to doc_id.

    Returns {chunk_id: (rank, text, page_number, doc_id, parent_text, parent_id, parent_chunk_index)}.
    Rank is 0-indexed (rank 0 = best). parent_* fields are None for pre-Sprint-16 collections.
    """
    response = get_qdrant_client().query_points(
        collection_name=settings.qdrant_collection,
        query=query,
        using=using,
        query_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=top_k,
        with_payload=True,
    )
    result = {}
    for rank, point in enumerate(response.points):
        payload = point.payload or {}
        result[str(point.id)] = (
            rank,
            payload.get("text", ""),
            payload.get("page_number", 0),
            payload.get("doc_id", doc_id),
            payload.get("parent_text"),
            payload.get("parent_id"),
            payload.get("parent_chunk_index"),
        )
    return result


def _dedup_and_order(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """
    Stage 1 (score order): dedup by parent_id → top_k unique parents.
      The highest-scoring child selects which parent the LLM receives.
      Falls back to chunk_id dedup for pre-Sprint-16 collections.

    Stage 2 (doc order): re-sort selected parents by parent_chunk_index ascending.
      Legal clauses cross-reference earlier sections. Sending Article 10 before
      Article 2 breaks the LLM's understanding of legal hierarchy.
    """
    seen: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        key = chunk.parent_id if chunk.parent_id is not None else chunk.chunk_id
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)
        if len(deduped) == top_k:
            break
    deduped.sort(
        key=lambda c: c.parent_chunk_index if c.parent_chunk_index is not None else float("inf")
    )
    return deduped


def retrieve(
    query_text: str,
    doc_id: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[RetrievedChunk]:
    """
    Hybrid sparse + dense retrieval with RRF fusion, scoped to a single document.

    Args:
        query_text:  Natural language description of the clause to find.
        doc_id:      Restrict results to this document.
        top_k:       Number of chunks to return after fusion (goes to LLM).
        candidate_k: Results to fetch from each retriever before fusion.
                     Higher = better recall, slower. 20 is a good default.

    Returns:
        List of RetrievedChunk sorted into document order after parent dedup.
        Returns empty list if Qdrant is unavailable.
    """
    query_dense, query_sparse = _embed_query(query_text)

    dense_results = _qdrant_ranks(query_dense, "dense", doc_id, candidate_k)
    if not dense_results:
        logger.warning("[retriever] no dense results for doc_id=%s", doc_id)
        return []

    if query_sparse:
        sparse_results = _qdrant_ranks(
            SparseVector(indices=list(query_sparse.keys()), values=list(query_sparse.values())),
            settings.sparse_vector_name,
            doc_id,
            candidate_k,
        )
    else:
        logger.warning("[retriever] empty sparse query for doc_id=%s — dense-only fallback", doc_id)
        sparse_results = {}

    fused: list[RetrievedChunk] = []
    for chunk_id in set(dense_results) | set(sparse_results):
        dense_rank = dense_results[chunk_id][0] if chunk_id in dense_results else None
        sparse_rank = sparse_results[chunk_id][0] if chunk_id in sparse_results else None
        rrf_score = (
            (1.0 / (RRF_K + dense_rank) if dense_rank is not None else 0.0) +
            (1.0 / (RRF_K + sparse_rank) if sparse_rank is not None else 0.0)
        )
        _, text, page_number, d_id, parent_text, parent_id, parent_chunk_index = (
            dense_results[chunk_id] if chunk_id in dense_results else sparse_results[chunk_id]
        )
        fused.append(RetrievedChunk(
            chunk_id=chunk_id, doc_id=d_id, page_number=page_number, text=text,
            rrf_score=rrf_score, sparse_rank=sparse_rank, dense_rank=dense_rank,
            parent_text=parent_text, parent_id=parent_id, parent_chunk_index=parent_chunk_index,
        ))

    fused.sort(key=lambda x: x.rrf_score, reverse=True)
    result = _dedup_and_order(fused, top_k)

    logger.debug(
        "[retriever] doc=%s query='%s...' → %d candidates, %d unique parents (doc order)",
        doc_id, query_text[:40], len(fused), len(result),
    )
    return result


def retrieve_multi(
    queries: list[str],
    doc_id: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[RetrievedChunk]:
    """
    Multi-query hybrid retrieval: fires each query independently, then
    merges candidates by summing RRF scores across queries.

    Why sum rather than max?
    A chunk that ranks well for multiple query angles is a stronger signal
    of relevance than one that ranks well for only one angle. Summing RRF
    scores gives a consensus boost proportional to cross-query agreement.

    When is this called?
    Only for categories in CUAD_ALT_QUERIES (15 hard categories). The
    agent falls back to plain retrieve() for all other categories.
    """
    score_sums: dict[str, float] = {}
    metadata: dict[str, RetrievedChunk] = {}

    for query in queries:
        for chunk in retrieve(query_text=query, doc_id=doc_id, top_k=candidate_k, candidate_k=candidate_k):
            score_sums[chunk.chunk_id] = score_sums.get(chunk.chunk_id, 0.0) + chunk.rrf_score
            if chunk.chunk_id not in metadata:
                metadata[chunk.chunk_id] = chunk

    # Update scores in-place and sort — avoids rebuilding every RetrievedChunk
    for cid, total in score_sums.items():
        metadata[cid].rrf_score = total
    merged = sorted(metadata.values(), key=lambda x: x.rrf_score, reverse=True)

    result = _dedup_and_order(merged, top_k)

    logger.debug(
        "[retriever] multi-query doc=%s queries=%d → %d unique candidates, %d unique parents (doc order)",
        doc_id, len(queries), len(merged), len(result),
    )
    return result
