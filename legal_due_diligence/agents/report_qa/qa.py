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

import litellm

from agents.clause_extractor.retriever import RetrievedChunk, retrieve
from core.config import settings

logger = logging.getLogger(__name__)

_QA_SYSTEM_PROMPT = """You are a legal due diligence assistant answering questions about contracts.
Answer based ONLY on the contract excerpts provided. Be precise and cite which document you are drawing from.
If the answer cannot be determined from the provided excerpts, say so explicitly — do not speculate."""


def _retrieve_across_docs(
    question: str,
    doc_ids: list[str],
    top_k_per_doc: int = 3,
) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks across all documents in doc_ids.

    Calls per-document retrieve() for each doc_id and merges results by RRF
    score. Each doc contributes up to top_k_per_doc chunks; the final list is
    sorted by descending RRF score so the most relevant passages surface first.

    Why per-doc calls rather than a single cross-doc Qdrant query?
    The BM25 index has no filter API — it requires per-doc post-filtering.
    Using the existing retrieve() keeps both BM25 and dense retrieval
    consistent between extraction and Q&A. The overhead is negligible for ≤50
    documents.
    """
    all_chunks: list[RetrievedChunk] = []
    for doc_id in doc_ids:
        chunks = retrieve(question, doc_id, top_k=top_k_per_doc)
        all_chunks.extend(chunks)

    # Re-rank by RRF score descending — same signal used within each doc
    all_chunks.sort(key=lambda c: c.rrf_score, reverse=True)

    # Return at most top_k_per_doc * 2 chunks (enough context, not too many tokens)
    return all_chunks[: top_k_per_doc * 2]


def _build_qa_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Excerpt {i} — {chunk.doc_id}, page {chunk.page_number}]\n{chunk.text.strip()}"
        )
    context = "\n\n".join(context_parts)

    return f"""Contract excerpts:

{context}

---

Question: {question}

Answer based only on the excerpts above. Reference the document and page where relevant."""


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

    chunks = _retrieve_across_docs(question, doc_ids, top_k_per_doc=top_k_per_doc)
    if not chunks:
        return {
            "answer": "No relevant contract excerpts were found for this question.",
            "citations": [],
            "chunks_retrieved": 0,
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

    logger.info("[qa] answered with %d citation(s)", len(citations))
    return {
        "answer": answer,
        "citations": citations,
        "chunks_retrieved": len(chunks),
    }
