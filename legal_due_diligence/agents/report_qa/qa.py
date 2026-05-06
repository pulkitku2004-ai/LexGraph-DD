"""
Q&A module — RAG question answering with source citations.

Architecture:
  question + doc_ids → hybrid retrieval across all docs → LLM answer → citations

Retrieval strategy for Q&A vs clause extraction:

Clause extraction uses per-document retrieval (one doc at a time). Q&A is
cross-document: "What does the Liability Cap say across all contracts?" should
return results from every document.

Rather than a new retriever, this module calls the existing per-document
retrieve() for each doc_id and merges the results by RRF score. For 2–50
documents with 2–3 chunks each, this is fast enough and avoids maintaining
a separate retrieval path.

Why not query Qdrant without a doc_id filter?
For a multi-job system, the collection accumulates chunks from every job ever
run. An unscoped query would return chunks from unrelated jobs. Scoping to
doc_ids ensures Q&A answers are grounded in the current job's documents only.

Citation format: {doc_id, page_number, chunk_id, excerpt}
chunk_id is the source_chunk_id that traces directly back to the Qdrant point
and therefore to a page in the original PDF — the same lineage chain used by
RiskFlag.source_clause_id.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import litellm

from agents.clause_extractor.retriever import RetrievedChunk, retrieve_with_metadata
from core.config import settings

logger = logging.getLogger(__name__)

_QA_SYSTEM_PROMPT = """You are a legal document retrieval system.
Output ONLY verbatim sentences or phrases copied character-for-character from the excerpts that answer the question.
Do not write any introduction, framing, bullets, numbers, document names, or words of your own.
If no relevant text exists in the excerpts, output exactly: NOT_FOUND"""


def _retrieve_across_docs(
    question: str,
    doc_ids: list[str],
    top_k_per_doc: int = 3,
) -> tuple[list[RetrievedChunk], dict]:
    """
    Retrieve relevant chunks across all documents in doc_ids.

    Calls per-document retrieve_with_metadata() for each doc_id, merges results
    by RRF score, and returns both the final chunk list and the full retrieval
    metadata (all pre-truncation ranked chunks per doc) for ASTR-O span emission.

    Returns:
        (chunks, retrieval_metadata)
        retrieval_metadata shape matches ASTR-O's expected span structure:
          {query, retrieval_method, top_k, retrieved_chunks, all_ranked_chunks, retrieval_timestamp}
    """
    all_chunks: list[RetrievedChunk] = []
    all_ranked_flat: list[dict] = []

    for doc_id in doc_ids:
        chunks, ranked = retrieve_with_metadata(question, doc_id, top_k=top_k_per_doc)
        all_chunks.extend(chunks)
        # Tag each ranked entry with its doc_id so ASTR-O can trace cross-doc conflicts
        for entry in ranked:
            all_ranked_flat.append({**entry, "doc_id": doc_id})

    all_chunks.sort(key=lambda c: c.rrf_score, reverse=True)
    final_chunks = all_chunks[: top_k_per_doc * 2]

    retrieval_metadata = {
        "query": question,
        "retrieval_method": "hybrid_rrf",
        "top_k": top_k_per_doc,
        "retrieved_chunks": [c.chunk_id for c in final_chunks],
        "all_ranked_chunks": all_ranked_flat,
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return final_chunks, retrieval_metadata


def _build_qa_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Excerpt {i}]\n{chunk.text.strip()}")
    context = "\n\n".join(context_parts)

    return f"""Contract excerpts:

{context}

---

Question: {question}

Copy verbatim sentences from the excerpts above that answer the question. No other words."""


def _call_qa_llm(prompt: str) -> str | None:
    """
    Call the reasoning model for Q&A.
    temperature=0.1 — slight randomness for prose fluency, still grounded.
    max_tokens=300 — answers should be concise.
    """
    try:
        response = litellm.completion(
            model=settings.llm_reasoning_model,
            messages=[
                {"role": "system", "content": _QA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()  # type: ignore[union-attr]
    except Exception as e:
        logger.error("[qa] LLM call failed: %s", e)
        return None


def answer_question(
    question: str,
    doc_ids: list[str],
    top_k_per_doc: int = 3,
) -> dict:
    """
    Answer a legal question using hybrid retrieval across the given documents.

    Returns:
        {
            "answer": str,        — plain text answer grounded in the retrieved excerpts
            "citations": [        — source chunks used to generate the answer
                {
                    "doc_id": str,
                    "page_number": int,
                    "chunk_id": str,   — Qdrant point ID → PDF page citation chain
                    "excerpt": str,    — first 200 chars of the chunk text
                }
            ],
            "chunks_retrieved": int,
        }

    On retrieval failure (empty Qdrant): returns answer explaining no context found.
    On LLM failure: returns answer noting the model was unavailable.
    """
    logger.info("[qa] question=%r across %d doc(s): %s", question[:60], len(doc_ids), doc_ids)

    chunks, retrieval_metadata = _retrieve_across_docs(question, doc_ids, top_k_per_doc=top_k_per_doc)
    if not chunks:
        return {
            "answer": "No relevant contract excerpts were found for this question.",
            "citations": [],
            "chunks_retrieved": 0,
            "retrieval_metadata": retrieval_metadata,
            "enriched_chunks": [],
        }

    prompt = _build_qa_prompt(question, chunks)
    answer = _call_qa_llm(prompt)

    if answer is None:
        answer = "The Q&A model was unavailable. Please retry or query the contracts manually."

    citations = [
        {
            "doc_id": c.doc_id,
            "page_number": c.page_number,
            "chunk_id": c.chunk_id,
            "excerpt": c.text[:200].strip(),
        }
        for c in chunks
    ]

    # enriched_chunks = the content actually sent to the LLM (child chunk text + provenance)
    enriched_chunks = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "page_number": c.page_number,
            "text": c.text.strip(),
        }
        for c in chunks
    ]

    logger.info("[qa] answered with %d citation(s)", len(citations))
    return {
        "answer": answer,
        "citations": citations,
        "chunks_retrieved": len(chunks),
        "retrieval_metadata": retrieval_metadata,
        "enriched_chunks": enriched_chunks,
    }
