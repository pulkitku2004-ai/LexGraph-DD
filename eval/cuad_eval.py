"""
CUAD Retrieval Eval Harness — Recall@K

Measures Recall@1 and Recall@3 for the hybrid sparse+dense retriever against
the CUAD benchmark dataset (chenghao/cuad_qa, 1,244 test rows, 41 categories).

Each CUAD test row has its own context (a contract passage). All contexts are
indexed under unique doc_ids into a dedicated cuad_eval collection, keeping
the eval isolated from the production legal_clauses collection.

Retrieval uses the same path as production: bge-m3 dense + SPLADE sparse
vectors stored in Qdrant, RRF fusion, two-stage parent-id dedup + doc-order.

Usage:
  python eval/cuad_eval.py                                            # 100-row stratified sample
  python eval/cuad_eval.py --n 400 --enrich-queries --multi-query    # canonical run
  python eval/cuad_eval.py --full --enrich-queries --multi-query     # full 1,244-row set
  python eval/cuad_eval.py --no-cleanup                              # keep collection for inspection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

# Add project root and package root so imports resolve the same way as production code
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "legal_due_diligence"))

from core.config import get_settings
from ingestion.chunker import Chunk, _count_tokens, _parent_child_chunks, _merge_headings
from ingestion.embedder import EmbeddedChunk, embed_chunks
from infrastructure.qdrant_client import get_qdrant_client
from agents.clause_extractor.prompts import CUAD_ALT_QUERIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EVAL_COLLECTION = "cuad_eval"
SAMPLE_IDS_PATH = Path(__file__).parent / "sample_ids.json"
RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = Path(__file__).parent / "cache"

RRF_K = 60


# ── Embedding cache ───────────────────────────────────────────────────────────
# Chunk embeddings are expensive (~8s/row) but stable — the same CUAD context
# always produces the same child text → same vectors for a given model + chunk
# config. Cache to disk keyed on (model, chunk params, text hash).
# Cache file name encodes model + chunk params → auto-invalidates on config change.

_CacheEntry = tuple[list[float], dict[int, float]]  # (dense, sparse)


def _cache_path() -> Path:
    settings = get_settings()
    slug = (
        f"{settings.embedding_model.split('/')[-1]}"
        f"_p{settings.chunk_size}_c{settings.child_chunk_size}_o{settings.child_chunk_overlap}"
    ).replace("-", "_")
    return CACHE_DIR / f"embeddings_{slug}.pkl"


def _load_cache() -> dict[str, _CacheEntry]:
    path = _cache_path()
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info("[cache] loaded %d entries from %s", len(data), path.name)
        return data
    return {}


def _save_cache(cache: dict[str, _CacheEntry]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with open(_cache_path(), "wb") as f:
        pickle.dump(cache, f)


def _text_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── CUAD query enrichment ──────────────────────────────────────────────────────
# Maps official CUAD category labels (from chenghao/cuad_qa "question" field)
# to keyword-rich retrieval queries. Mirrors CUAD_CATEGORIES in prompts.py.
_CUAD_QUERY_ENRICHMENT: dict[str, str] = {
    "Document Name":                    "agreement title name contract type",
    "Parties":                          "party parties between company corporation LLC",
    "Agreement Date":                   "agreement date made entered into as of",
    "Effective Date":                   "effective date commencement start date",
    "Expiration Date":                  "expiration date end date termination date contract expires",
    "Renewal Term":                     "renewal term automatically renews extension period",
    "Notice Period to Terminate Renewal": "notice period terminate renewal cancel non-renewal written notice",
    "Governing Law":                    "governing law jurisdiction choice of law state",
    "Dispute Resolution":               "dispute resolution arbitration litigation venue court",
    "Non-Compete":                      "non-compete non-competition restrictive covenant shall not compete competing business competing products competing services engage in competition",
    "Exclusivity":                      "exclusivity exclusive rights sole provider",
    "No-Solicit of Customers":          "no solicit customers non-solicitation clients",
    "No-Solicit Of Customers":          "shall not directly or indirectly solicit divert customers clients accounts business of the other party",
    "No-Solicit of Employees":          "no solicit employees hire recruit personnel",
    "No-Solicit Of Employees":          "no solicit employees hire recruit personnel",
    "Non-Disparagement":                "non-disparagement disparage negative statements derogatory defamatory comments refrain from speaking negatively",
    "Covenant Not To Sue":              "covenant not to sue release claims waive right to bring action not assert legal claims discharge not contest challenge attack impair title trademark validity ownership",
    "Limitation of Liability":          "limitation of liability indirect consequential damages excluded",
    "Liability Cap":                    "liability cap maximum total aggregate liability shall not exceed",
    "Liquidated Damages":               "liquidated damages penalty breach fixed amount",
    "Uncapped Liability":               "uncapped liability unlimited liability gross negligence fraud",
    "Indemnification":                  "indemnification indemnify hold harmless defend claims losses",
    "IP Ownership Assignment":          "intellectual property ownership assignment work made for hire all right title interest vests assigns work product inventions",
    "Joint IP Ownership":               "joint IP ownership jointly owned co-owned each party owns co-invented jointly developed jointly created intellectual property shared ownership both parties",
    "License Grant":                    "license grant licensed rights use software platform",
    "Non-Transferable License":         "non-transferable license cannot assign sublicense transfer",
    "Irrevocable or Perpetual License": "irrevocable perpetual license permanent rights",
    "Source Code Escrow":               "source code escrow deposit release conditions",
    "IP Restriction":                   "intellectual property restrictions permitted use limitations",
    "Warranty Duration":                "warranty period duration months years guarantee defect rejection return expiration date outdated product",
    "Product Warranty":                 "product warranty performance fitness merchantability",
    "Payment Terms":                    "payment terms invoice due date net days fees",
    "Revenue/Profit Sharing":           "revenue sharing profit sharing royalty percentage net receipts gross revenue net sales proceeds equal to per unit",
    "Price Restrictions":               "price restrictions most favored nation pricing cap",
    "Most Favored Nation":              "most favored nation MFN no less favorable price terms any third party",
    "Minimum Commitment":               "minimum commitment purchase volume guaranteed amount",
    "Volume Restriction":               "volume restriction maximum usage limit quota not to exceed seats users copies units annual cap",
    "Audit Rights":                     "audit rights inspect records books financial",
    "Post-Termination Services":        "post-termination services transition assistance wind-down following termination survival obligations continue after termination data return",
    "Change of Control":                "change of control merger acquisition beneficial ownership voting securities controlling interest majority shares takeover consent assignment terminate",
    "Anti-Assignment":                  "anti-assignment cannot assign transfer consent required",
    "Third Party Beneficiary":          "third party beneficiary rights benefits",
    "Confidentiality":                  "confidentiality confidential information non-disclosure NDA",
    "Confidentiality Survival":         "confidentiality survives termination years obligations continue",
    "Termination for Convenience":      "termination for convenience without cause notice days",
    "Termination for Cause":            "termination for cause material breach cure period",
    "Affiliate License-Licensor":       "affiliate license licensor grant rights related entities sublicense",
    "Affiliate License-Licensee":       "affiliate license licensee sublicense grant related entities rights",
    "Competitive Restriction Exception": "notwithstanding competitive restriction exception carve-out permitted competing business activities despite exclusivity non-compete",
    "Rofr/Rofo/Rofn":                   "right of first refusal offer negotiation ROFR preemptive right purchase",
    "Cap on Liability":                 "cap on liability maximum aggregate liability shall not exceed ceiling",
    "Unlimited/All-You-Can-Eat-License": "unlimited non-exclusive perpetual irrevocable royalty free worldwide license unrestricted use all-you-can-eat",
}


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_cuad_test() -> list[dict]:
    """Load chenghao/cuad_qa test split."""
    logger.info("[eval] loading chenghao/cuad_qa test split...")
    ds = load_dataset("chenghao/cuad_qa", split="test")
    rows = [dict(row) for row in ds]
    logger.info("[eval] %d test rows loaded", len(rows))
    return rows


def load_or_generate_sample_ids(rows: list[dict], n: int) -> list[int]:
    """
    Load reproducible sample indices from file, or generate a stratified sample.
    Stratification: group rows by question text (41 groups), sample ceil(n/groups) per group.
    """
    if SAMPLE_IDS_PATH.exists():
        with open(SAMPLE_IDS_PATH) as f:
            data = json.load(f)
        cached = data.get(str(n), [])
        if cached:
            logger.info("[eval] loaded %d sample indices from %s", len(cached), SAMPLE_IDS_PATH)
            return cached

    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["question"]].append(i)

    per_group = max(1, math.ceil(n / len(groups)))
    rng = np.random.default_rng(seed=42)

    sampled: list[int] = []
    for question in sorted(groups):
        indices = groups[question]
        k = min(per_group, len(indices))
        chosen = rng.choice(indices, size=k, replace=False).tolist()
        sampled.extend(chosen)

    sampled = sorted(sampled)[:n]

    SAMPLE_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if SAMPLE_IDS_PATH.exists():
        with open(SAMPLE_IDS_PATH) as f:
            existing = json.load(f)
    existing[str(n)] = sampled
    with open(SAMPLE_IDS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(
        "[eval] generated %d stratified sample indices (%d groups, ~%d/group) → %s",
        len(sampled), len(groups), per_group, SAMPLE_IDS_PATH,
    )
    return sampled


# ── Qdrant collection ─────────────────────────────────────────────────────────

def setup_eval_collection() -> None:
    """Create (or recreate) the cuad_eval Qdrant collection with named dense + sparse vectors."""
    settings = get_settings()
    client = get_qdrant_client()

    existing = {c.name for c in client.get_collections().collections}
    if EVAL_COLLECTION in existing:
        client.delete_collection(EVAL_COLLECTION)
        logger.info("[eval] dropped existing %s collection", EVAL_COLLECTION)

    client.create_collection(
        collection_name=EVAL_COLLECTION,
        vectors_config={
            "dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            settings.sparse_vector_name: SparseVectorParams(),
        },
    )
    logger.info(
        "[eval] created %s (dense=%d-dim, sparse='%s')",
        EVAL_COLLECTION, settings.embedding_dim, settings.sparse_vector_name,
    )


def teardown_eval_collection() -> None:
    """Delete the cuad_eval collection."""
    get_qdrant_client().delete_collection(EVAL_COLLECTION)
    logger.info("[eval] deleted %s collection", EVAL_COLLECTION)


# ── Chunking + indexing ───────────────────────────────────────────────────────

def make_chunks(text: str, doc_id: str) -> list[Chunk]:
    """
    Produce Chunk objects using the production parent-child chunker.
    256-token children embedded; 2048-token contiguous parents stored in payload.
    Span-check uses parent_text — matching what the production LLM sees.
    """
    settings = get_settings()
    tuples = _parent_child_chunks(
        _merge_headings(text),
        parent_size=settings.chunk_size,
        child_size=settings.child_chunk_size,
        child_overlap=settings.child_chunk_overlap,
    )
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            file_path="cuad_eval",
            page_number=1,
            text=child_text,
            token_count=_count_tokens(child_text),
            chunk_index=i,
            parent_text=parent_text,
            parent_id=parent_id,
            parent_chunk_index=parent_chunk_index,
        )
        for i, (child_text, parent_text, parent_id, parent_chunk_index) in enumerate(tuples)
    ]


def index_eval_rows(rows: list[dict], sample_ids: list[int]) -> None:
    """
    Chunk + embed + upsert all sampled rows into cuad_eval.
    Embeddings are cached to disk — repeat runs skip the GPU step entirely.
    Only chunks not already in cache are embedded; cache is updated after.
    """
    client = get_qdrant_client()

    all_chunks: list[Chunk] = []
    for idx in sample_ids:
        all_chunks.extend(make_chunks(rows[idx]["context"], f"cuad_eval_{idx}"))

    logger.info("[eval] %d chunks from %d rows", len(all_chunks), len(sample_ids))

    cache = _load_cache()
    uncached = [c for c in all_chunks if _text_key(c.text) not in cache]

    if uncached:
        logger.info("[eval] embedding %d new chunks (%d cache hits)...",
                    len(uncached), len(all_chunks) - len(uncached))
        t0 = time.perf_counter()
        newly_embedded = embed_chunks(uncached)
        logger.info("[eval] embedding done in %.1fs", time.perf_counter() - t0)
        for chunk, ec in zip(uncached, newly_embedded):
            cache[_text_key(chunk.text)] = (ec.vector, ec.sparse_vector)
        _save_cache(cache)
        logger.info("[cache] saved %d total entries", len(cache))
    else:
        logger.info("[eval] all %d chunks loaded from cache — skipping GPU", len(all_chunks))

    embedded: list[EmbeddedChunk] = [
        EmbeddedChunk(
            chunk=chunk,
            vector=cache[_text_key(chunk.text)][0],
            sparse_vector=cache[_text_key(chunk.text)][1],
        )
        for chunk in all_chunks
    ]

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
                "text": ec.chunk.text,
                "parent_text": ec.chunk.parent_text,
                "parent_id": ec.chunk.parent_id,
                "parent_chunk_index": ec.chunk.parent_chunk_index,
                "page_number": ec.chunk.page_number,
                "chunk_index": ec.chunk.chunk_index,
            },
        )
        for ec in embedded
    ]

    for i in range(0, len(points), 100):
        client.upsert(collection_name=EVAL_COLLECTION, points=points[i : i + 100])

    logger.info("[eval] upserted %d points to %s", len(points), EVAL_COLLECTION)


# ── Query embedding ───────────────────────────────────────────────────────────

def embed_questions(
    rows: list[dict],
    sample_ids: list[int],
    query_prefix: str = "",
    enrich_queries: bool = False,
    multi_query: bool = False,
) -> tuple[dict[int, tuple[list[float], dict[int, float]]], dict[str, list[tuple[list[float], dict[int, float]]]]]:
    """
    Embed all sampled questions in one batched GPU call. Returns:
      - primary: {test_idx: (dense_vector, sparse_vector)}
      - alt_embeddings: {category: [(dense, sparse), ...]} pre-batched for multi-query,
        so the eval loop does a dict lookup instead of a per-row GPU call.
    """
    def _query_text(idx: int) -> str:
        q = rows[idx]["question"]
        enriched = _CUAD_QUERY_ENRICHMENT.get(q, q) if enrich_queries else q
        return query_prefix + enriched

    question_chunks = [
        Chunk(chunk_id=str(uuid.uuid4()), doc_id="__query__", file_path="",
              page_number=0, text=_query_text(idx), token_count=0, chunk_index=0)
        for idx in sample_ids
    ]
    logger.info("[eval] embedding %d questions...", len(question_chunks))
    embedded = embed_chunks(question_chunks)
    primary = {
        idx: (embedded[i].vector, embedded[i].sparse_vector)
        for i, idx in enumerate(sample_ids)
    }

    # Pre-batch all alt query embeddings — one GPU call for all categories combined.
    alt_embeddings: dict[str, list[tuple[list[float], dict[int, float]]]] = {}
    if multi_query and CUAD_ALT_QUERIES:
        present_categories = {rows[idx]["question"] for idx in sample_ids}
        cats_with_alts = [c for c in present_categories if c in CUAD_ALT_QUERIES]
        if cats_with_alts:
            flat: list[tuple[str, str]] = [
                (cat, (query_prefix + q) if query_prefix else q)
                for cat in cats_with_alts
                for q in CUAD_ALT_QUERIES[cat]
            ]
            alt_chunks = [
                Chunk(chunk_id=str(uuid.uuid4()), doc_id="__query__", file_path="",
                      page_number=0, text=text, token_count=0, chunk_index=0)
                for _, text in flat
            ]
            logger.info("[eval] pre-embedding %d alt queries for %d categories...",
                        len(alt_chunks), len(cats_with_alts))
            alt_embedded = embed_chunks(alt_chunks)
            for (cat, _), ec in zip(flat, alt_embedded):
                alt_embeddings.setdefault(cat, []).append((ec.vector, ec.sparse_vector))

    return primary, alt_embeddings


# ── Retrieval ─────────────────────────────────────────────────────────────────

def eval_retrieve(
    query_dense: list[float],
    query_sparse: dict[int, float],
    doc_id: str,
    top_k: int = 3,
    candidate_k: int = 20,
) -> list[tuple[str, str]]:
    """
    Hybrid sparse+dense retrieval over cuad_eval, scoped to a single doc_id.
    RRF fusion → two-stage parent-id dedup + document-order sort.
    Returns list of (chunk_id, span_text) — span_text is parent_text for recall check.
    """
    client = get_qdrant_client()
    settings = get_settings()

    dense_response = client.query_points(
        collection_name=EVAL_COLLECTION,
        query=query_dense,
        using="dense",
        query_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        limit=candidate_k,
        with_payload=True,
    )
    dense_results: dict[str, tuple[int, str, str | None, str | None, int | None]] = {
        str(p.id): (
            rank,
            (p.payload or {}).get("text", ""),
            (p.payload or {}).get("parent_text"),
            (p.payload or {}).get("parent_id"),
            (p.payload or {}).get("parent_chunk_index"),
        )
        for rank, p in enumerate(dense_response.points)
    }

    sparse_results: dict[str, tuple[int, str, str | None, str | None, int | None]] = {}
    if query_sparse:
        sparse_response = client.query_points(
            collection_name=EVAL_COLLECTION,
            query=SparseVector(
                indices=list(query_sparse.keys()),
                values=list(query_sparse.values()),
            ),
            using=settings.sparse_vector_name,
            query_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            limit=candidate_k,
            with_payload=True,
        )
        sparse_results = {
            str(p.id): (
                rank,
                (p.payload or {}).get("text", ""),
                (p.payload or {}).get("parent_text"),
                (p.payload or {}).get("parent_id"),
                (p.payload or {}).get("parent_chunk_index"),
            )
            for rank, p in enumerate(sparse_response.points)
        }

    # RRF fusion
    fused: list[tuple[float, str, str, str | None, int | None]] = []
    for cid in set(dense_results) | set(sparse_results):
        rrf = 0.0
        if cid in dense_results:
            rrf += 1.0 / (RRF_K + dense_results[cid][0])
        if cid in sparse_results:
            rrf += 1.0 / (RRF_K + sparse_results[cid][0])
        _, child_text, parent_text, parent_id, parent_chunk_index = (
            dense_results[cid] if cid in dense_results else sparse_results[cid]
        )
        span_text = parent_text if parent_text is not None else child_text
        fused.append((rrf, cid, span_text, parent_id, parent_chunk_index))

    fused.sort(key=lambda x: x[0], reverse=True)

    # Stage 1: score-order dedup by parent_id — best child selects which parent is kept
    seen: set[str] = set()
    deduped: list[tuple[str, str, int | None]] = []
    for _, cid, span_text, parent_id, parent_chunk_index in fused:
        key = parent_id if parent_id is not None else cid
        if key not in seen:
            seen.add(key)
            deduped.append((cid, span_text, parent_chunk_index))
        if len(deduped) == top_k:
            break

    # Stage 2: re-sort into document order (Article 2 before Article 10)
    deduped.sort(key=lambda x: x[2] if x[2] is not None else float("inf"))

    return [(cid, span_text) for cid, span_text, _ in deduped]


def eval_retrieve_multi(
    query_pairs: list[tuple[list[float], dict[int, float]]],
    doc_id: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[tuple[str, str]]:
    """
    Multi-query retrieval: runs eval_retrieve() for each (dense, sparse) pair,
    unions results, sums RRF scores (chunks appearing in multiple results get a
    boosted combined score), returns top_k. query_pairs[0] is the primary query.
    """
    score_sums: dict[str, float] = {}
    text_map: dict[str, str] = {}

    for dense, sparse in query_pairs:
        result = eval_retrieve(
            query_dense=dense,
            query_sparse=sparse,
            doc_id=doc_id,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )
        for rank, (cid, text) in enumerate(result):
            score_sums[cid] = score_sums.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            if cid not in text_map:
                text_map[cid] = text

    merged = sorted(score_sums.keys(), key=lambda c: score_sums[c], reverse=True)
    return list(zip(merged[:top_k], [text_map[c] for c in merged[:top_k]]))


# ── Recall@K ──────────────────────────────────────────────────────────────────

def is_recall_hit(ground_truth: str, retrieved_texts: list[str]) -> bool:
    """True if ground_truth appears as a substring in any retrieved chunk (case-insensitive)."""
    gt = ground_truth.lower().strip()
    return any(gt in text.lower() for text in retrieved_texts)


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval(
    rows: list[dict],
    sample_ids: list[int],
    query_embeddings: dict[int, tuple[list[float], dict[int, float]]],
    alt_embeddings: dict[str, list[tuple[list[float], dict[int, float]]]],
    top_k: int = 3,
    candidate_k: int = 20,
    query_prefix: str = "",
    enrich_queries: bool = False,
    multi_query: bool = False,
) -> dict[str, Any]:
    """Run retrieval eval loop. Returns results dict with per-row details and aggregate metrics."""
    settings = get_settings()
    per_row: list[dict] = []
    hits_at_1 = hits_at_3 = hits_at_topk = answered = skipped = 0

    for i, idx in enumerate(sample_ids):
        row = rows[idx]
        doc_id = f"cuad_eval_{idx}"
        question = row["question"]
        answer_texts: list[str] = row["answers"]["text"]

        if not answer_texts or not answer_texts[0].strip():
            skipped += 1
            continue

        ground_truth = answer_texts[0]
        query_dense, query_sparse = query_embeddings[idx]

        precomputed_alts = alt_embeddings.get(question, []) if multi_query else []
        if precomputed_alts:
            query_pairs = [(query_dense, query_sparse)] + precomputed_alts
            retrieved = eval_retrieve_multi(
                query_pairs=query_pairs,
                doc_id=doc_id,
                top_k=max(top_k, 5),
                candidate_k=candidate_k,
            )
        else:
            retrieved = eval_retrieve(
                query_dense=query_dense,
                query_sparse=query_sparse,
                doc_id=doc_id,
                top_k=top_k,
                candidate_k=candidate_k,
            )

        texts = [text for _, text in retrieved]
        hit1 = is_recall_hit(ground_truth, texts[:1])
        hit3 = is_recall_hit(ground_truth, texts[:min(3, top_k)])
        hit_topk = is_recall_hit(ground_truth, texts[:top_k])

        if hit1:
            hits_at_1 += 1
        if hit3:
            hits_at_3 += 1
        if hit_topk:
            hits_at_topk += 1
        answered += 1

        row_result: dict[str, Any] = {
            "test_idx": idx,
            "question": question,
            "doc_id": doc_id,
            "ground_truth": ground_truth,
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "retrieved_texts": [t[:300] for t in texts],
        }
        if top_k != 3:
            row_result[f"hit_at_{top_k}"] = hit_topk
        per_row.append(row_result)

        if (i + 1) % 10 == 0 or (i + 1) == len(sample_ids):
            logger.info(
                "[eval] %3d/%d | Recall@1=%.3f  Recall@3=%.3f%s  (answered=%d skipped=%d)",
                i + 1, len(sample_ids),
                hits_at_1 / max(answered, 1),
                hits_at_3 / max(answered, 1),
                f"  Recall@{top_k}={hits_at_topk / max(answered, 1):.3f}" if top_k != 3 else "",
                answered, skipped,
            )

    metrics: dict[str, Any] = {
        "recall_at_1": round(hits_at_1 / answered, 4) if answered else 0.0,
        "recall_at_3": round(hits_at_3 / answered, 4) if answered else 0.0,
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "answered_rows": answered,
        "skipped_no_answer": skipped,
        "total_sampled": len(sample_ids),
    }
    if top_k != 3:
        metrics[f"recall_at_{top_k}"] = round(hits_at_topk / answered, 4) if answered else 0.0
        metrics[f"hits_at_{top_k}"] = hits_at_topk

    return {
        "model": settings.embedding_model,
        "query_prefix": query_prefix,
        "enrich_queries": enrich_queries,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "dataset": "chenghao/cuad_qa",
        "split": "test",
        "sample_size": len(sample_ids),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": metrics,
        "per_row": per_row,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CUAD retrieval eval — Recall@K")
    parser.add_argument("--n", type=int, default=100,
                        help="Rows to sample (default: 100). Ignored when --full is set.")
    parser.add_argument("--full", action="store_true",
                        help="Run the full 1,244-row test set instead of a sample.")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Chunks to retrieve per row (default: 3).")
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Candidates fetched per retriever before RRF fusion (default: 20).")
    parser.add_argument("--query-prefix", type=str, default="",
                        help="Prefix prepended to each question before embedding.")
    parser.add_argument("--enrich-queries", action="store_true",
                        help="Map CUAD category labels to keyword-rich retrieval queries.")
    parser.add_argument("--multi-query", action="store_true",
                        help="Fire alt queries for hard categories (CUAD_ALT_QUERIES), union with summed RRF.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path. Default: auto-generated from model name and flags.")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep the cuad_eval Qdrant collection after the run.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    if args.output:
        output_path = Path(args.output)
    else:
        model_slug = settings.embedding_model.split("/")[-1].lower().replace("-", "_")
        prefix_tag = "_prefix" if args.query_prefix else ""
        enrich_tag = "_enriched" if args.enrich_queries else ""
        mq_tag = "_mq" if args.multi_query else ""
        output_path = RESULTS_DIR / f"{model_slug}_topk{args.top_k}{prefix_tag}{enrich_tag}{mq_tag}.json"

    rows = load_cuad_test()
    sample_ids = list(range(len(rows))) if args.full else load_or_generate_sample_ids(rows, args.n)
    logger.info(
        "[eval] running %d rows | model: %s | top_k: %d | candidate_k: %d | enrich: %s | multi_query: %s",
        len(sample_ids), settings.embedding_model, args.top_k, args.candidate_k,
        args.enrich_queries, args.multi_query,
    )

    setup_eval_collection()

    try:
        index_eval_rows(rows, sample_ids)
        query_embeddings, alt_embeddings = embed_questions(
            rows, sample_ids,
            query_prefix=args.query_prefix,
            enrich_queries=args.enrich_queries,
            multi_query=args.multi_query,
        )

        t0 = time.perf_counter()
        results = run_eval(
            rows, sample_ids, query_embeddings, alt_embeddings,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            query_prefix=args.query_prefix,
            enrich_queries=args.enrich_queries,
            multi_query=args.multi_query,
        )
        elapsed = time.perf_counter() - t0

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        m = results["metrics"]
        print("\n" + "=" * 62)
        print(f"  CUAD Retrieval Eval")
        print(f"  Model     : {results['model']}")
        print(f"  Prefix    : {results['query_prefix']!r}" if results["query_prefix"] else f"  Prefix    : (none)")
        print(f"  Enrich    : {'yes (production queries)' if results['enrich_queries'] else 'no (bare CUAD labels)'}")
        print(f"  Top-K     : {results['top_k']}  (candidate_k={results['candidate_k']})")
        print(f"  Dataset   : {results['dataset']} (test split)")
        print("=" * 62)
        print(f"  Sample  : {results['sample_size']} rows  ({m['answered_rows']} with answers, {m['skipped_no_answer']} skipped)")
        print(f"  Recall@1: {m['recall_at_1']:.1%}  ({m['hits_at_1']}/{m['answered_rows']})")
        print(f"  Recall@3: {m['recall_at_3']:.1%}  ({m['hits_at_3']}/{m['answered_rows']})")
        if args.top_k != 3:
            k = args.top_k
            print(f"  Recall@{k}: {m[f'recall_at_{k}']:.1%}  ({m[f'hits_at_{k}']}/{m['answered_rows']})")
        print(f"  Time    : {elapsed:.1f}s")
        print(f"  Output  : {output_path}")
        print("=" * 62 + "\n")

    finally:
        if not args.no_cleanup:
            teardown_eval_collection()


if __name__ == "__main__":
    main()
