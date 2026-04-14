"""
Hybrid retriever — bge-m3 learned sparse + dense vectors fused with RRF.

Sprint 8 upgrade: BM25 pickle → Qdrant-native sparse vectors.

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
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
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

import uuid

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# RRF rank smoothing constant — original paper value, don't change without benchmarking
RRF_K = 60

# ── BM25 tokenizer (kept for backward compatibility) ──────────────────────────
# bm25_tokenize() is still exported so cuad_eval.py can import it during the
# Sprint 7→8 transition. It is no longer used in the main retrieval path —
# learned sparse vectors from bge-m3 replace BM25 entirely.
_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 (legacy — kept for backward compat exports).

    Not used in the Sprint 8 retrieval path. Sparse retrieval is handled
    by Qdrant-native sparse vectors produced by bge-m3.
    """
    return _BM25_TOKEN_RE.findall(text.lower())


# ── Cross-encoder reranker singleton ──────────────────────────────────────────
_reranker: CrossEncoder | None = None
_reranker_loaded: bool = False


def _get_reranker() -> CrossEncoder | None:
    global _reranker, _reranker_loaded
    if _reranker_loaded:
        return _reranker

    _reranker_loaded = True
    if not settings.reranker_model:
        return None

    try:
        import torch
        from sentence_transformers import CrossEncoder

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _reranker = CrossEncoder(settings.reranker_model, device=device)
        logger.info("[retriever] reranker loaded: %s on %s", settings.reranker_model, device)
    except Exception as exc:
        logger.warning("[retriever] failed to load reranker %s: %s", settings.reranker_model, exc)
        _reranker = None

    return _reranker


@dataclass
class RetrievedChunk:
    """A chunk returned by the hybrid retriever with its fused RRF score."""
    chunk_id: str
    doc_id: str
    page_number: int
    text: str
    rrf_score: float
    sparse_rank: int | None   # None if not in sparse top-k
    dense_rank: int | None    # None if not in dense top-k


def _hyde_expand(clause_type: str, fallback_query: str) -> str:
    """
    HyDE: generate a hypothetical contract clause for dense query embedding.

    The generated text is closer to actual contract language than a keyword
    query, so its embedding lands in the same vector space as indexed clause
    chunks — improving recall for semantically variable clauses like
    "Covenant Not To Sue" where query keywords don't match clause language.

    Dense uses the hypothetical; sparse keeps the original enriched query
    (learned sparse weights are most reliable on exact legal terms).

    Falls back to fallback_query if the LLM call fails.
    """
    import litellm

    prompt = (
        f"Write a typical 2–3 sentence contract clause for '{clause_type}'. "
        "Use standard legal contract language. Return only the clause text, no explanation."
    )
    try:
        response = litellm.completion(
            model=settings.llm_extraction_model,
            fallbacks=settings.llm_extraction_fallbacks,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        generated = response.choices[0].message.content.strip()  # type: ignore[union-attr]
        logger.debug("[retriever] HyDE '%s' → %r", clause_type, generated[:80])
        return generated
    except Exception as e:
        logger.warning("[retriever] HyDE expansion failed for '%s': %s — using original query", clause_type, e)
        return fallback_query


def _embed_query(query_text: str) -> tuple[list[float], dict[int, float]]:
    """
    Embed a query string using bge-m3, returning both dense and sparse vectors.
    Applies settings.embedding_query_prefix if set (empty for bge-m3).

    Returns:
        (dense_vector, sparse_vector) — dense is 1024-dim float list,
        sparse is {token_id: weight} dict.
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


def _dense_ranks(
    query_vector: list[float],
    doc_id: str,
    top_k: int,
) -> dict[str, tuple[int, str, int, str]]:
    """
    Get dense retrieval ranks for chunks in doc_id from Qdrant.

    Returns dict: {chunk_id: (rank, text, page_number, doc_id)}
    Rank is 0-indexed (rank 0 = best).

    Uses 'dense' named vector — the bge-m3 1024-dim cosine field.
    """
    client = get_qdrant_client()

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        using="dense",
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
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
        )
    return result


def _sparse_ranks(
    sparse_vector: dict[int, float],
    doc_id: str,
    top_k: int,
) -> dict[str, tuple[int, str, int, str]]:
    """
    Get sparse retrieval ranks for chunks in doc_id from Qdrant.

    Returns dict: {chunk_id: (rank, text, page_number, doc_id)}
    Rank is 0-indexed (rank 0 = best).

    Uses settings.sparse_vector_name ('sparse') — the bge-m3 SPLADE-style field.
    Qdrant computes dot product over shared token IDs between the query sparse
    vector and each indexed sparse vector.

    Why filter by doc_id here (not post-filter)?
    Same reason as dense: Qdrant applies the filter at the index level before
    scoring, so we only score chunks belonging to the target document.
    This is much faster than fetching all chunks and filtering in Python.
    """
    client = get_qdrant_client()

    if not sparse_vector:
        # Empty sparse vector would return no results — fall back gracefully
        logger.warning("[retriever] empty sparse query vector for doc_id=%s", doc_id)
        return {}

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=SparseVector(
            indices=list(sparse_vector.keys()),
            values=list(sparse_vector.values()),
        ),
        using=settings.sparse_vector_name,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
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
        )
    return result


def retrieve(
    query_text: str,
    doc_id: str,
    top_k: int = 5,
    candidate_k: int = 20,
    clause_type: str = "",
    hyde: bool = False,
) -> list[RetrievedChunk]:
    """
    Hybrid sparse + dense retrieval with RRF fusion, scoped to a single document.

    Args:
        query_text:  Natural language description of the clause to find.
                     e.g. "governing law jurisdiction choice of law"
        doc_id:      Restrict results to this document.
        top_k:       Number of chunks to return after fusion (goes to LLM).
        candidate_k: How many results to fetch from each retriever before fusion.
                     Higher = better recall, slower. 20 is a good default.
        clause_type: CUAD category name — required for HyDE expansion.
        hyde:        If True, generate a hypothetical clause for dense embedding.
                     Sparse still uses original query_text.

    Returns:
        List of RetrievedChunk sorted by RRF score descending.
        Returns empty list if Qdrant is unavailable.
    """
    # ── Embed query (dense + sparse) ──────────────────────────────────────
    if hyde and clause_type:
        # HyDE: generate hypothetical clause → dense embedding
        # Original query → sparse embedding (exact legal terms)
        # Both embedded in one batched forward pass.
        hyde_text = _hyde_expand(clause_type, query_text)
        orig_chunk = Chunk(
            chunk_id=str(uuid.uuid4()), doc_id="__query__", file_path="",
            page_number=0, text=query_text, token_count=0, chunk_index=0,
        )
        hyde_chunk = Chunk(
            chunk_id=str(uuid.uuid4()), doc_id="__query__", file_path="",
            page_number=0, text=hyde_text, token_count=0, chunk_index=0,
        )
        embedded = embed_chunks([orig_chunk, hyde_chunk])
        query_sparse = embedded[0].sparse_vector   # original → exact terms
        query_dense = embedded[1].vector           # hypothetical → semantic
    else:
        query_dense, query_sparse = _embed_query(query_text)

    # ── Dense retrieval ───────────────────────────────────────────────────
    dense_results = _dense_ranks(query_dense, doc_id, top_k=candidate_k)

    if not dense_results:
        logger.warning("[retriever] no dense results for doc_id=%s", doc_id)
        return []

    # ── Sparse retrieval ──────────────────────────────────────────────────
    sparse_results = _sparse_ranks(query_sparse, doc_id, top_k=candidate_k)

    if not sparse_results:
        logger.warning("[retriever] no sparse results for doc_id=%s — dense-only fallback", doc_id)

    # ── RRF fusion ────────────────────────────────────────────────────────
    all_candidates: set[str] = set(dense_results.keys()) | set(sparse_results.keys())

    fused: list[RetrievedChunk] = []
    for chunk_id in all_candidates:
        dense_rank = dense_results[chunk_id][0] if chunk_id in dense_results else None
        sparse_rank = sparse_results[chunk_id][0] if chunk_id in sparse_results else None

        rrf_score = 0.0
        if dense_rank is not None:
            rrf_score += 1.0 / (RRF_K + dense_rank)
        if sparse_rank is not None:
            rrf_score += 1.0 / (RRF_K + sparse_rank)

        # Get text + metadata from dense results (primary payload source)
        if chunk_id in dense_results:
            _, text, page_number, d_id = dense_results[chunk_id]
        else:
            _, text, page_number, d_id = sparse_results[chunk_id]

        fused.append(RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=d_id,
            page_number=page_number,
            text=text,
            rrf_score=rrf_score,
            sparse_rank=sparse_rank,
            dense_rank=dense_rank,
        ))

    fused.sort(key=lambda x: x.rrf_score, reverse=True)

    # ── Cross-encoder reranking (optional) ────────────────────────────────
    reranker = _get_reranker()
    if reranker is not None and fused:
        pairs = [(query_text, c.text) for c in fused]
        scores = np.array(reranker.predict(pairs, convert_to_numpy=True))
        ranked_indices = np.argsort(scores)[::-1]
        fused = [fused[i] for i in ranked_indices]
        logger.debug("[retriever] reranked %d candidates", len(fused))

    logger.debug(
        "[retriever] doc=%s query='%s...' → %d candidates, returning top %d",
        doc_id, query_text[:40], len(fused), top_k,
    )

    return fused[:top_k]


def retrieve_multi(
    queries: list[str],
    doc_id: str,
    top_k: int = 5,
    candidate_k: int = 20,
    clause_type: str = "",
    hyde: bool = False,
) -> list[RetrievedChunk]:
    """
    Multi-query hybrid retrieval: fires each query independently, then
    merges candidates by summing RRF scores across queries.

    Why sum rather than max?
    A chunk that ranks well for multiple query angles is a stronger signal
    of relevance than one that ranks well for only one angle. Summing RRF
    scores gives a consensus boost proportional to cross-query agreement.
    Max would flatten that signal.

    Why run full retrieve() per query rather than sharing embeddings?
    Each query has different dense and sparse vectors — sharing would
    require restructuring the embedding pipeline. Running full retrieve()
    per query is simpler and the extra Qdrant calls are cheap (~10ms each).

    When is this called?
    Only for categories in CUAD_ALT_QUERIES (10 hard categories). The
    agent falls back to plain retrieve() for all other categories.
    """
    score_sums: dict[str, float] = {}
    metadata: dict[str, RetrievedChunk] = {}

    for query in queries:
        chunks = retrieve(
            query_text=query,
            doc_id=doc_id,
            top_k=candidate_k,
            candidate_k=candidate_k,
            clause_type=clause_type,
            hyde=hyde,
        )
        for chunk in chunks:
            score_sums[chunk.chunk_id] = score_sums.get(chunk.chunk_id, 0.0) + chunk.rrf_score
            if chunk.chunk_id not in metadata:
                metadata[chunk.chunk_id] = chunk

    # Re-sort by summed score and build final list
    merged = [
        RetrievedChunk(
            chunk_id=cid,
            doc_id=metadata[cid].doc_id,
            page_number=metadata[cid].page_number,
            text=metadata[cid].text,
            rrf_score=score_sums[cid],
            sparse_rank=metadata[cid].sparse_rank,
            dense_rank=metadata[cid].dense_rank,
        )
        for cid in score_sums
    ]
    merged.sort(key=lambda x: x.rrf_score, reverse=True)

    logger.debug(
        "[retriever] multi-query doc=%s queries=%d → %d unique candidates, returning top %d",
        doc_id, len(queries), len(merged), top_k,
    )
    return merged[:top_k]
