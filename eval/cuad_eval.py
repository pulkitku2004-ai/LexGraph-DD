"""
Sprint 7/8 — CUAD Retrieval Eval Harness

Measures Recall@1 and Recall@3 for the hybrid sparse+dense retriever against
the CUAD benchmark dataset.

Sprint 8 change: BM25 in-memory index replaced by bge-m3 learned sparse vectors
stored in Qdrant. The eval collection now uses named vectors (dense + sparse),
mirroring the production indexer schema.

Dataset: chenghao/cuad_qa (Parquet, datasets 4.x compatible)
  - 1,240 test rows | 41 CUAD question categories
  - schema: context, question, answers.text, answers.answer_start

Design notes:

Why a separate cuad_eval Qdrant collection?
  Each CUAD test row has its own context (a contract passage). We index them
  under unique doc_ids into a dedicated collection so the eval doesn't
  pollute the main legal_clauses collection and teardown is a single delete.

Why embed sparse vectors for queries?
  bge-m3 produces sparse vectors for both passages and queries in the same
  encode() call. The sparse query vector is the natural complement to the
  indexed sparse vectors — Qdrant scores them by dot product over shared tokens.
  This is the exact same path as production retrieval.

Why batch-embed queries upfront?
  100 individual embed_chunks() calls would each spin up a tiny GPU batch.
  Batching all questions in one call amortises the kernel launch overhead
  and fully utilises MPS throughput.

Why Recall@K as the primary metric?
  Recall@K directly measures whether the correct answer span is available in
  the retrieved context before the LLM sees it. If the right chunk isn't
  retrieved, no amount of LLM quality helps.

Usage:
  python eval/cuad_eval.py                    # 100-row stratified sample
  python eval/cuad_eval.py --n 20             # quick sanity check
  python eval/cuad_eval.py --full             # full 1,240-row test set
  python eval/cuad_eval.py --output results/bge_m3_baseline.json
  python eval/cuad_eval.py --no-cleanup       # keep cuad_eval collection for inspection
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

# CrossEncoder is an optional import — only used when --reranker is passed.
try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _HAVE_ST = True
except ImportError:
    _HAVE_ST = False

# ── CUAD query enrichment ──────────────────────────────────────────────────────
# Maps official CUAD category labels (from chenghao/cuad_qa "question" field)
# to keyword-rich retrieval queries. Same vocabulary used by the production
# clause extractor — see agents/clause_extractor/prompts.py CUAD_CATEGORIES.
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
    "Non-Compete":                      "non-compete non-competition competitive business restrict",
    "Exclusivity":                      "exclusivity exclusive rights sole provider",
    "No-Solicit of Customers":          "no solicit customers non-solicitation clients",
    "No-Solicit Of Customers":          "no solicit customers non-solicitation clients",
    "No-Solicit of Employees":          "no solicit employees hire recruit personnel",
    "No-Solicit Of Employees":          "no solicit employees hire recruit personnel",
    "Non-Disparagement":                "non-disparagement disparage negative statements derogatory defamatory comments refrain from speaking negatively",
    "Covenant Not To Sue":              "covenant not to sue release claims waive right to bring action not assert legal claims discharge",
    "Limitation of Liability":          "limitation of liability indirect consequential damages excluded",
    "Liability Cap":                    "liability cap maximum total aggregate liability shall not exceed",
    "Liquidated Damages":               "liquidated damages penalty breach fixed amount",
    "Uncapped Liability":               "uncapped liability unlimited liability gross negligence fraud",
    "Indemnification":                  "indemnification indemnify hold harmless defend claims losses",
    "IP Ownership Assignment":          "intellectual property ownership assignment work made for hire all right title interest vests assigns work product inventions",
    "Joint IP Ownership":               "joint ownership jointly owned intellectual property jointly developed co-developed each party owns shared ownership",
    "License Grant":                    "license grant licensed rights use software platform",
    "Non-Transferable License":         "non-transferable license cannot assign sublicense transfer",
    "Irrevocable or Perpetual License": "irrevocable perpetual license permanent rights",
    "Source Code Escrow":               "source code escrow deposit release conditions",
    "IP Restriction":                   "intellectual property restrictions permitted use limitations",
    "Warranty Duration":                "warranty period duration months years guarantee",
    "Product Warranty":                 "product warranty performance fitness merchantability",
    "Payment Terms":                    "payment terms invoice due date net days fees",
    "Revenue/Profit Sharing":           "revenue sharing profit sharing royalty percentage",
    "Price Restrictions":               "price restrictions most favored nation pricing cap",
    "Minimum Commitment":               "minimum commitment purchase volume guaranteed amount",
    "Volume Restriction":               "volume restriction maximum usage limit quota not to exceed seats users copies units annual cap",
    "Audit Rights":                     "audit rights inspect records books financial",
    "Post-Termination Services":        "post-termination services transition assistance wind-down following termination survival obligations continue after termination data return",
    "Change of Control":                "change of control acquisition merger assignment consent",
    "Anti-Assignment":                  "anti-assignment cannot assign transfer consent required",
    "Third Party Beneficiary":          "third party beneficiary rights benefits",
    "Confidentiality":                  "confidentiality confidential information non-disclosure NDA",
    "Confidentiality Survival":         "confidentiality survives termination years obligations continue",
    "Termination for Convenience":      "termination for convenience without cause notice days",
    "Termination for Cause":            "termination for cause material breach cure period",
    "Affiliate License-Licensor":       "affiliate license licensor grant rights related entities sublicense",
    "Affiliate License-Licensee":       "affiliate license licensee sublicense grant related entities rights",
    "Competitive Restriction Exception": "competitive restriction exception carve-out notwithstanding competing business allowed",
    "Rofr/Rofo/Rofn":                   "right of first refusal offer negotiation ROFR preemptive right purchase",
    "Cap on Liability":                 "cap on liability maximum aggregate liability shall not exceed ceiling",
    "Most Favored Nation":              "most favored nation MFN best pricing equivalent terms comparable customer",
}

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
from ingestion.chunker import Chunk, _count_tokens, _token_chunks, _merge_headings
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

RRF_K = 60


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

    Stratification: group rows by question text (41 groups, one per CUAD category),
    sample ceil(n / num_groups) rows from each.
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


# ── Chunking helpers ──────────────────────────────────────────────────────────

def make_chunks(text: str, doc_id: str) -> list[Chunk]:
    """Produce Chunk objects from raw text using the production chunker."""
    settings = get_settings()
    raw_texts = _token_chunks(
        _merge_headings(text),
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            file_path="cuad_eval",
            page_number=1,
            text=chunk_text,
            token_count=_count_tokens(chunk_text),
            chunk_index=i,
        )
        for i, chunk_text in enumerate(raw_texts)
    ]


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_eval_rows(
    rows: list[dict],
    sample_ids: list[int],
) -> dict[str, list[Chunk]]:
    """
    Chunk + embed + upsert all sampled rows into cuad_eval with dense and sparse vectors.
    Returns doc_chunks_map: {doc_id: list[Chunk]}.

    All contexts are chunked first, then embedded in one batched call to
    fully utilise MPS throughput. Qdrant upsert follows in batches of 100.

    Sprint 8: no in-memory BM25 built here — sparse vectors from bge-m3 stored
    in Qdrant replace it. The BM25 return value is removed from this function.
    """
    client = get_qdrant_client()
    settings = get_settings()

    all_chunks: list[Chunk] = []
    doc_chunks_map: dict[str, list[Chunk]] = {}

    for idx in sample_ids:
        doc_id = f"cuad_eval_{idx}"
        chunks = make_chunks(rows[idx]["context"], doc_id)
        all_chunks.extend(chunks)
        doc_chunks_map[doc_id] = chunks

    logger.info("[eval] %d chunks from %d rows", len(all_chunks), len(sample_ids))

    logger.info("[eval] embedding chunks (dense + sparse)...")
    t0 = time.perf_counter()
    embedded: list[EmbeddedChunk] = embed_chunks(all_chunks)
    logger.info("[eval] embedding done in %.1fs", time.perf_counter() - t0)

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
                "page_number": ec.chunk.page_number,
                "chunk_index": ec.chunk.chunk_index,
            },
        )
        for ec in embedded
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=EVAL_COLLECTION, points=points[i : i + batch_size])

    logger.info("[eval] upserted %d points to %s", len(points), EVAL_COLLECTION)

    return doc_chunks_map


def _hyde_generate(clause_type: str, fallback: str) -> str:
    """
    Generate a hypothetical contract clause for HyDE dense embedding.
    Uses the extraction LLM (fast model). Falls back to the enriched query on failure.
    """
    import litellm
    settings = get_settings()
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("[eval] HyDE generation failed for '%s': %s", clause_type, e)
        return fallback


def embed_questions(
    rows: list[dict],
    sample_ids: list[int],
    query_prefix: str = "",
    enrich_queries: bool = False,
    use_hyde: bool = False,
) -> dict[int, tuple[list[float], dict[int, float]]]:
    """
    Embed all sampled questions. Returns {test_idx: (dense_vector, sparse_vector)}.

    HyDE mode (use_hyde=True):
      Dense  ← embedding of a hypothetical contract clause generated by the LLM.
               The hypothetical is in the same vocabulary space as indexed chunks.
      Sparse ← embedding of the original enriched query (exact legal terms).
    One hypothetical is generated per unique CUAD category (≤41 LLM calls total).

    Standard mode:
      Both dense and sparse come from embedding the enriched query string.
    """
    def _query_text(idx: int) -> str:
        q = rows[idx]["question"]
        enriched = _CUAD_QUERY_ENRICHMENT.get(q, q) if enrich_queries else q
        return query_prefix + enriched

    if use_hyde:
        # Generate one hypothetical per unique category (saves LLM calls)
        unique_categories = {rows[idx]["question"] for idx in sample_ids}
        logger.info("[eval] HyDE: generating %d hypothetical clauses...", len(unique_categories))
        hypotheticals: dict[str, str] = {}
        for cat in unique_categories:
            fallback = _CUAD_QUERY_ENRICHMENT.get(cat, cat)
            hypotheticals[cat] = _hyde_generate(cat, fallback)

        # Embed original queries (→ sparse) and hypotheticals (→ dense) in two batches
        orig_chunks = [
            Chunk(chunk_id=str(uuid.uuid4()), doc_id="__query__", file_path="",
                  page_number=0, text=_query_text(idx), token_count=0, chunk_index=0)
            for idx in sample_ids
        ]
        hyde_chunks = [
            Chunk(chunk_id=str(uuid.uuid4()), doc_id="__query__", file_path="",
                  page_number=0, text=hypotheticals[rows[idx]["question"]],
                  token_count=0, chunk_index=0)
            for idx in sample_ids
        ]
        logger.info("[eval] embedding %d original queries (sparse) ...", len(orig_chunks))
        orig_embedded = embed_chunks(orig_chunks)
        logger.info("[eval] embedding %d hypothetical clauses (dense) ...", len(hyde_chunks))
        hyde_embedded = embed_chunks(hyde_chunks)
        return {
            idx: (hyde_embedded[i].vector, orig_embedded[i].sparse_vector)
            for i, idx in enumerate(sample_ids)
        }

    # Standard: both signals from the same enriched query embedding
    question_chunks = [
        Chunk(chunk_id=str(uuid.uuid4()), doc_id="__query__", file_path="",
              page_number=0, text=_query_text(idx), token_count=0, chunk_index=0)
        for idx in sample_ids
    ]
    logger.info("[eval] embedding %d questions...", len(question_chunks))
    embedded = embed_chunks(question_chunks)
    return {
        idx: (embedded[i].vector, embedded[i].sparse_vector)
        for i, idx in enumerate(sample_ids)
    }


# ── Eval retriever ────────────────────────────────────────────────────────────

def eval_retrieve(
    query_dense: list[float],
    query_sparse: dict[int, float],
    query_text: str,
    doc_id: str,
    top_k: int = 3,
    candidate_k: int = 20,
    reranker: Any = None,
) -> list[tuple[str, str]]:
    """
    Hybrid sparse+dense retrieval over cuad_eval, scoped to a single doc_id.
    Returns list of (chunk_id, text) sorted by RRF (or reranker) score descending.

    Sprint 8: BM25 parameters removed. Sparse retrieval uses Qdrant-native
    sparse vectors (bge-m3 SPLADE weights) filtered by doc_id at index level.
    """
    client = get_qdrant_client()
    settings = get_settings()

    # ── Dense retrieval ───────────────────────────────────────────────────
    dense_response = client.query_points(
        collection_name=EVAL_COLLECTION,
        query=query_dense,
        using="dense",
        query_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=candidate_k,
        with_payload=True,
    )
    dense_results: dict[str, tuple[int, str]] = {
        str(p.id): (rank, (p.payload or {}).get("text", ""))
        for rank, p in enumerate(dense_response.points)
    }

    # ── Sparse retrieval ──────────────────────────────────────────────────
    sparse_results: dict[str, tuple[int, str]] = {}
    if query_sparse:
        sparse_response = client.query_points(
            collection_name=EVAL_COLLECTION,
            query=SparseVector(
                indices=list(query_sparse.keys()),
                values=list(query_sparse.values()),
            ),
            using=settings.sparse_vector_name,
            query_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=candidate_k,
            with_payload=True,
        )
        sparse_results = {
            str(p.id): (rank, (p.payload or {}).get("text", ""))
            for rank, p in enumerate(sparse_response.points)
        }

    # ── RRF fusion ────────────────────────────────────────────────────────
    all_candidates = set(dense_results) | set(sparse_results)
    fused: list[tuple[float, str, str]] = []
    for cid in all_candidates:
        rrf = 0.0
        if cid in dense_results:
            rrf += 1.0 / (RRF_K + dense_results[cid][0])
        if cid in sparse_results:
            rrf += 1.0 / (RRF_K + sparse_results[cid][0])
        text = dense_results[cid][1] if cid in dense_results else sparse_results[cid][1]
        fused.append((rrf, cid, text))

    fused.sort(key=lambda x: x[0], reverse=True)

    # ── Cross-encoder reranking (optional) ───────────────────────────────
    if reranker is not None and fused:
        pairs = [(query_text, text) for _, _, text in fused]
        scores = np.array(reranker.predict(pairs, convert_to_numpy=True))
        ranked_indices = np.argsort(scores)[::-1]
        fused = [fused[i] for i in ranked_indices]

    return [(cid, text) for _, cid, text in fused[:top_k]]


def eval_retrieve_multi(
    query_pairs: list[tuple[list[float], dict[int, float]]],
    query_text: str,
    doc_id: str,
    top_k: int = 5,
    candidate_k: int = 20,
    reranker: Any = None,
) -> list[tuple[str, str]]:
    """
    Multi-query retrieval for the eval harness.

    Runs eval_retrieve() for each (dense, sparse) pair in query_pairs,
    unions results by chunk_id, sums RRF scores across queries (chunks
    appearing in multiple results get a boosted combined score), returns top_k.

    query_pairs[0] is the primary (enriched) query.
    query_pairs[1:] are alternative angle queries from CUAD_ALT_QUERIES.
    """
    score_sums: dict[str, float] = {}
    text_map: dict[str, str] = {}

    for dense, sparse in query_pairs:
        result = eval_retrieve(
            query_dense=dense,
            query_sparse=sparse,
            query_text=query_text,
            doc_id=doc_id,
            top_k=candidate_k,
            candidate_k=candidate_k,
            reranker=None,  # reranker applied once on merged list below
        )
        # Each tuple is (chunk_id, text); reconstruct RRF score from rank
        for rank, (cid, text) in enumerate(result):
            rrf = 1.0 / (RRF_K + rank)
            score_sums[cid] = score_sums.get(cid, 0.0) + rrf
            if cid not in text_map:
                text_map[cid] = text

    merged = sorted(score_sums.keys(), key=lambda c: score_sums[c], reverse=True)
    merged_texts = [text_map[c] for c in merged]

    if reranker is not None and merged:
        pairs = [(query_text, t) for t in merged_texts]
        scores = np.array(reranker.predict(pairs, convert_to_numpy=True))
        ranked_indices = list(np.argsort(scores)[::-1])
        merged = [merged[i] for i in ranked_indices]
        merged_texts = [merged_texts[i] for i in ranked_indices]

    return list(zip(merged[:top_k], merged_texts[:top_k]))


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
    doc_chunks_map: dict[str, list[Chunk]],
    top_k: int = 3,
    candidate_k: int = 20,
    query_prefix: str = "",
    enrich_queries: bool = False,
    multi_query: bool = False,
    reranker: Any = None,
    reranker_name: str = "",
) -> dict[str, Any]:
    """
    Run retrieval eval loop. Returns full results dict with per-row details and
    aggregate metrics suitable for JSON serialisation.
    """
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

        query_text = (
            _CUAD_QUERY_ENRICHMENT.get(question, question) if enrich_queries else question
        )

        alt_queries = CUAD_ALT_QUERIES.get(question, []) if multi_query else []
        if alt_queries:
            # Embed alt queries on-the-fly (bge-m3 already loaded)
            alt_chunks = [
                Chunk(
                    chunk_id=str(uuid.uuid4()), doc_id="__query__", file_path="",
                    page_number=0, text=(query_prefix + q) if query_prefix else q,
                    token_count=0, chunk_index=0,
                )
                for q in alt_queries
            ]
            alt_embedded = embed_chunks(alt_chunks)
            query_pairs = [(query_dense, query_sparse)] + [
                (e.vector, e.sparse_vector) for e in alt_embedded
            ]
            retrieved = eval_retrieve_multi(
                query_pairs=query_pairs,
                query_text=query_text,
                doc_id=doc_id,
                top_k=max(top_k, 5),  # multi-query returns up to 5 for better recall
                candidate_k=candidate_k,
                reranker=reranker,
            )
        else:
            retrieved = eval_retrieve(
                query_dense=query_dense,
                query_sparse=query_sparse,
                query_text=query_text,
                doc_id=doc_id,
                top_k=top_k,
                candidate_k=candidate_k,
                reranker=reranker,
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
        "reranker": reranker_name if reranker_name else None,
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
    parser.add_argument(
        "--n", type=int, default=100,
        help="Rows to sample (default: 100). Ignored when --full is set.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run the full 1,240-row test set instead of a sample.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Chunks to retrieve per row (default: 3).",
    )
    parser.add_argument(
        "--candidate-k", type=int, default=20,
        help=(
            "Candidates fetched from each retriever (sparse and dense) before RRF fusion "
            "and optional reranking (default: 20)."
        ),
    )
    parser.add_argument(
        "--query-prefix", type=str, default="",
        help="Prefix prepended to each question before embedding (not to indexed chunks).",
    )
    parser.add_argument(
        "--enrich-queries", action="store_true",
        help=(
            "Map CUAD category labels to keyword-rich retrieval queries before embedding. "
            "Makes eval representative of what the production clause extractor queries."
        ),
    )
    parser.add_argument(
        "--reranker", type=str, default="",
        help="Cross-encoder model for re-ranking RRF candidates. Example: 'BAAI/bge-reranker-base'.",
    )
    parser.add_argument(
        "--hyde", action="store_true",
        help=(
            "Enable HyDE (Hypothetical Document Embeddings). "
            "Generates a hypothetical contract clause per category via LLM, "
            "embeds it for dense retrieval. Sparse uses the original enriched query. "
            "Costs ~41 extra LLM calls (one per unique CUAD category)."
        ),
    )
    parser.add_argument(
        "--multi-query", action="store_true",
        help=(
            "Enable multi-query retrieval for hard categories (those in CUAD_ALT_QUERIES). "
            "Fires 2-3 alternative queries per category, unions results with summed RRF scores. "
            "Only applies to ~10 low-recall categories; all others use single-query retrieval."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path. Default: auto-generated from model name and flags.",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep the cuad_eval Qdrant collection after the run.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    if args.output:
        output_path = Path(args.output)
    else:
        model_slug = settings.embedding_model.split("/")[-1].lower().replace("-", "_")
        prefix_tag = "_prefix" if args.query_prefix else ""
        enrich_tag = "_enriched" if args.enrich_queries else ""
        reranker_tag = "_reranked" if args.reranker else ""
        hyde_tag = "_hyde" if args.hyde else ""
        mq_tag = "_mq" if args.multi_query else ""
        output_path = RESULTS_DIR / f"{model_slug}_topk{args.top_k}{prefix_tag}{enrich_tag}{hyde_tag}{mq_tag}{reranker_tag}.json"

    reranker = None
    if args.reranker:
        if not _HAVE_ST:
            logger.error("[eval] --reranker requires sentence-transformers: pip install sentence-transformers")
            sys.exit(1)
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("[eval] loading reranker %s on %s ...", args.reranker, device)
        reranker = _CrossEncoder(args.reranker, device=device)
        logger.info("[eval] reranker ready")

    rows = load_cuad_test()
    sample_ids = list(range(len(rows))) if args.full else load_or_generate_sample_ids(rows, args.n)
    logger.info(
        "[eval] running %d rows | model: %s | top_k: %d | candidate_k: %d | enrich: %s | hyde: %s | multi_query: %s | reranker: %s",
        len(sample_ids), settings.embedding_model, args.top_k, args.candidate_k,
        args.enrich_queries, args.hyde, args.multi_query, args.reranker or "(none)",
    )

    setup_eval_collection()

    try:
        doc_chunks_map = index_eval_rows(rows, sample_ids)
        query_embeddings = embed_questions(
            rows, sample_ids,
            query_prefix=args.query_prefix,
            enrich_queries=args.enrich_queries,
            use_hyde=args.hyde,
        )

        t0 = time.perf_counter()
        results = run_eval(
            rows, sample_ids, query_embeddings, doc_chunks_map,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            query_prefix=args.query_prefix,
            enrich_queries=args.enrich_queries,
            multi_query=args.multi_query,
            reranker=reranker,
            reranker_name=args.reranker,
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
        print(f"  Reranker  : {results['reranker']}" if results["reranker"] else f"  Reranker  : (none)")
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
