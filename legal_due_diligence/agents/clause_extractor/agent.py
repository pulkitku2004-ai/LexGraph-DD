"""
Clause Extractor Agent — Sprint 12 async upgrade.

Sprint 2 established the sequential extraction flow:
  For each of 41 CUAD categories per document:
    1. retrieve() — hybrid sparse+dense+RRF, top 3 chunks
    2. LLM call (Groq Llama fast) — extract clause + normalize value
    3. Parse JSON → ExtractedClause

Sprint 12 makes both axes concurrent:

  Previously: docs × categories = sequential × sequential
              50 docs × 41 cats × 300ms = ~10 min

  Now:        docs × categories = concurrent × concurrent (under semaphore)
              wall time ≈ max(doc_extraction_times) ≈ ceil(41/10) × 300ms ≈ 1.5s
              50 docs ≈ 1.5–5s (limited by Groq rate limits and Ollama throughput)

Why asyncio.gather() across categories rather than across documents?
Both axes are now parallelised. The single global semaphore (EXTRACTION_CONCURRENCY)
caps total concurrent LLM calls regardless of how many docs are in the job.
This prevents rate-limit explosions on multi-doc jobs without needing per-doc logic.

Why asyncio.to_thread() for retrieval?
retrieve() is synchronous (Qdrant HTTP + bge-m3 MPS forward pass). Running it
directly inside an async function would block the event loop. to_thread() offloads
it to the default ThreadPoolExecutor. Multiple retrieve() calls run concurrently
in separate threads; PyTorch's Metal command queue serializes the GPU portion.

Why asyncio.run() inside clause_extractor_node() rather than making the node async?
The pipeline runs via graph.invoke() inside a FastAPI BackgroundTasks thread, which
has no running event loop. asyncio.run() creates a fresh event loop for the
extraction phase and exits cleanly. If the graph is ever converted to graph.ainvoke(),
replace asyncio.run() with await.

Why keep the global semaphore across docs?
Per-doc semaphores would allow N_docs × concurrency simultaneous Groq calls.
A global semaphore keeps the burst rate bounded regardless of job size.

Why keep _call_llm() (sync)?
The fallback chain (litellm.completion with fallbacks=) is synchronous.
acompletion + async fallbacks is used in _call_llm_async(). Both exist because
the sync path is used in legacy callers and tests; the async path is the
hot path in the extraction pipeline.

Why catch ALL exceptions around the LLM call?
Same as Sprint 2: one flaky response must not abort 40 remaining categories
or 49 remaining documents. found=False is conservative (over-flags risk).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import litellm

from agents.clause_extractor.prompts import (
    CUAD_ALT_QUERIES,
    CUAD_CATEGORIES,
    SYSTEM_PROMPT,
    build_extraction_prompt,
)
from agents.clause_extractor.retriever import retrieve, retrieve_multi
from core.config import settings
from core.models import ExtractedClause
from core.state import GraphState

logger = logging.getLogger(__name__)

# Suppress litellm's verbose request logging — we log at our own level
litellm.suppress_debug_info = True


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(user_prompt: str) -> Optional[str]:
    """
    Synchronous LLM call — kept for legacy callers and direct tests.
    Hot path uses _call_llm_async() instead.
    """
    try:
        response = litellm.completion(
            model=settings.llm_extraction_model,
            fallbacks=settings.llm_extraction_fallbacks,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            api_key=settings.groq_api_key or None,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("[clause_extractor] LLM call failed across all providers: %s", e)
        return None


async def _call_llm_async(user_prompt: str) -> Optional[str]:
    """
    Async LLM call via litellm.acompletion().

    Same fallback chain as the sync version — LiteLLM handles the cascade
    (primary Groq → scout Groq → Ollama) transparently in async mode.
    Called under the global semaphore in _extract_category_async() to cap
    total concurrent Groq requests within rate-limit budget.
    """
    try:
        response = await litellm.acompletion(
            model=settings.llm_extraction_model,
            fallbacks=settings.llm_extraction_fallbacks,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            api_key=settings.groq_api_key or None,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("[clause_extractor] async LLM call failed across all providers: %s", e)
        return None


# ── Response parsing (unchanged from Sprint 2) ────────────────────────────────

def _parse_response(
    raw: Optional[str],
    clause_type: str,
    doc_id: str,
    source_chunk_id: str,
) -> ExtractedClause:
    """
    Parse LLM JSON response into ExtractedClause.
    Returns a safe found=False clause on any parse failure.

    Why strip markdown fences before parsing?
    Some models (especially smaller ones) wrap JSON in ```json ... ```
    even when instructed not to. Stripping fences makes the parser robust
    without needing a separate prompt instruction for every model.
    """
    if raw is None:
        return _missing_clause(clause_type, doc_id, source_chunk_id)

    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        data = json.loads(text)
        return ExtractedClause(
            document_id=doc_id,
            clause_type=clause_type,
            found=bool(data.get("found", False)),
            clause_text=data.get("clause_text"),
            normalized_value=data.get("normalized_value"),
            confidence=float(data.get("confidence", 0.0)),
            source_chunk_id=source_chunk_id,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(
            "[clause_extractor] JSON parse failed for %s/%s: %s | raw=%s",
            doc_id, clause_type, e, raw[:120],
        )
        return _missing_clause(clause_type, doc_id, source_chunk_id)


def _missing_clause(
    clause_type: str,
    doc_id: str,
    source_chunk_id: str,
) -> ExtractedClause:
    return ExtractedClause(
        document_id=doc_id,
        clause_type=clause_type,
        found=False,
        clause_text=None,
        normalized_value=None,
        confidence=0.0,
        source_chunk_id=source_chunk_id,
    )


# ── Async extraction core ─────────────────────────────────────────────────────

async def _extract_category_async(
    clause_type: str,
    retrieval_query: str,
    doc_id: str,
    sem: asyncio.Semaphore,
) -> ExtractedClause:
    """
    Async extraction for one (doc, category) pair.

    Retrieval runs in a thread pool (sync Qdrant + bge-m3) so it doesn't
    block the event loop. Multiple retrieve() calls run concurrently;
    PyTorch MPS serializes GPU ops internally.

    The LLM call is gated by the shared semaphore — this is the rate-limit
    control point. Retrieval is allowed to run ahead (no semaphore there)
    so chunks are ready the moment the semaphore opens.
    """
    # ── Retrieval (sync, offloaded to thread pool) ─────────────────────────
    alt_queries = CUAD_ALT_QUERIES.get(clause_type, [])
    if alt_queries:
        all_queries = [retrieval_query] + alt_queries
        chunks = await asyncio.to_thread(
            retrieve_multi,
            all_queries,
            doc_id,
            5,           # top_k: more chunks to LLM when multi-querying
            20,          # candidate_k per query
            clause_type,
            settings.use_hyde,
        )
    else:
        chunks = await asyncio.to_thread(
            retrieve,
            query_text=retrieval_query,
            doc_id=doc_id,
            top_k=3,
            candidate_k=20,
            clause_type=clause_type,
            hyde=settings.use_hyde,
        )

    if not chunks:
        logger.warning(
            "[clause_extractor] no chunks retrieved for %s/%s", doc_id, clause_type
        )
        return _missing_clause(clause_type, doc_id, "")

    source_chunk_id = chunks[0].chunk_id
    prompt = build_extraction_prompt(clause_type, [c.text for c in chunks], doc_id)

    # ── LLM call (async, rate-limited by semaphore) ────────────────────────
    async with sem:
        raw_response = await _call_llm_async(prompt)

    clause = _parse_response(raw_response, clause_type, doc_id, source_chunk_id)
    logger.info(
        "[clause_extractor] %s | %s | found=%s | value=%s | conf=%.2f",
        doc_id, clause_type, clause.found,
        clause.normalized_value or "—", clause.confidence,
    )
    return clause


async def _extract_document_async(
    doc_id: str,
    sem: asyncio.Semaphore,
) -> list[ExtractedClause]:
    """
    Async extraction for one document — all 41 categories launched concurrently.
    The shared semaphore across all docs ensures total LLM concurrency stays bounded.
    """
    logger.info(
        "[clause_extractor] extracting %d categories from %s",
        len(CUAD_CATEGORIES), doc_id,
    )
    tasks = [
        _extract_category_async(clause_type, query, doc_id, sem)
        for clause_type, query in CUAD_CATEGORIES.items()
    ]
    clauses: list[ExtractedClause] = list(await asyncio.gather(*tasks))

    found_count = sum(1 for c in clauses if c.found)
    logger.info(
        "[clause_extractor] %s complete — %d/%d clauses found",
        doc_id, found_count, len(clauses),
    )
    return clauses


async def _run_extraction_async(docs: list) -> list[ExtractedClause]:
    """
    Run extraction for all unprocessed documents concurrently.
    One global semaphore caps LLM concurrency regardless of doc count.
    """
    unprocessed = [d for d in docs if not d.processed]
    if not unprocessed:
        return []

    sem = asyncio.Semaphore(settings.extraction_concurrency)
    doc_tasks = [_extract_document_async(doc.doc_id, sem) for doc in unprocessed]
    results = await asyncio.gather(*doc_tasks)
    return [clause for doc_clauses in results for clause in doc_clauses]


# ── LangGraph node ────────────────────────────────────────────────────────────

def clause_extractor_node(state: GraphState) -> dict:
    """
    LangGraph node — async extraction, sync interface.

    Calls asyncio.run() to create a dedicated event loop for the extraction
    phase. Safe because graph.invoke() runs inside a FastAPI BackgroundTasks
    thread (no parent event loop). If the graph is ever migrated to
    graph.ainvoke(), replace asyncio.run() with await here.
    """
    if not state.qdrant_ready:
        err = "[clause_extractor] Qdrant not ready — skipping extraction"
        logger.error(err)
        return {"errors": state.errors + [err], "status": "clauses_extracted"}

    unprocessed = [d for d in state.documents if not d.processed]
    if not unprocessed:
        return {
            "extracted_clauses": list(state.extracted_clauses),
            "status": "clauses_extracted",
        }

    logger.info(
        "[clause_extractor] starting async extraction: %d doc(s) × %d categories "
        "(concurrency=%d)",
        len(unprocessed), len(CUAD_CATEGORIES), settings.extraction_concurrency,
    )

    new_clauses = asyncio.run(_run_extraction_async(unprocessed))
    all_clauses = list(state.extracted_clauses) + new_clauses

    return {
        "extracted_clauses": all_clauses,
        "status": "clauses_extracted",
    }
