"""
Text chunker — LoadedDocument → list of Chunk objects.

Why token-based chunking instead of character or sentence-based?
legal-bert has a 512-token hard limit. Character-based chunking can produce
chunks that exceed this limit for dense legal text (long sentences, no
whitespace). Token-based chunking guarantees every chunk fits in the model
without truncation artifacts.

Why 512 tokens with 128 overlap?
- 512 = legal-bert's max sequence length. Using the full window maximises
  the semantic content per embedding.
- 128 token overlap (25%) prevents clause boundary splits. A limitation-of-
  liability clause that starts at token 490 of one chunk and ends at token
  50 of the next will be fully represented in the overlapping region.
  Without overlap, retrieval for that clause would fail regardless of
  query quality.

Why split at page boundaries before chunking?
Contracts have natural section boundaries at page breaks. Crossing a page
boundary in a chunk risks embedding two unrelated clauses together, which
degrades retrieval precision. We chunk within each page first, then
concatenate residual text across pages only when a page is shorter than
the minimum chunk size (to avoid tiny orphan chunks).

Why store page_number and char_start/end in the chunk?
source_chunk_id must resolve to a specific location in the original document
for the citation system in Sprint 6. page_number + char_start is the
minimal information needed to highlight the source text in the PDF viewer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from core.config import settings
from ingestion.loader import LoadedDocument, PagedText

# Load the tokenizer once at module level — it's stateless and thread-safe.
# We use the tokenizer only for counting tokens (encode), not for inference.
# Typed as PreTrainedTokenizerBase (the actual base class) rather than AutoTokenizer
# (a metaclass factory) so Pylance can resolve .encode() / .decode() instance methods.
_tokenizer: PreTrainedTokenizerBase | None = None


def _get_tokenizer() -> PreTrainedTokenizerBase:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    return _tokenizer  # type: ignore[return-value]


def _count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _merge_headings(text: str, heading_max_chars: int = 80) -> str:
    """
    Merge short paragraphs (clause headings) forward into the next paragraph.

    Legal contracts use patterns like:
        "12. INDEMNIFICATION\\n\\nEach party shall indemnify and hold harmless..."

    Without this merge, token-window chunking may split the heading into its own
    chunk — so the word "INDEMNIFICATION" appears only in the heading chunk (12
    tokens) and NOT in the obligation-text chunk. A BM25 query for "indemnification"
    then scores zero on the chunk containing the actual obligation.

    Any paragraph whose stripped length ≤ heading_max_chars is treated as a
    heading and prepended to the following paragraph. Multiple consecutive short
    paragraphs accumulate (e.g. "ARTICLE 12\\n\\nINDEMNIFICATION\\n\\n...") and
    are all prepended together.

    Edge case — trailing short paragraph (e.g., a signature line at page end):
    appended to the preceding paragraph rather than lost.
    """
    paragraphs = text.split("\n\n")
    merged: list[str] = []
    pending: str = ""

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if len(stripped) <= heading_max_chars:
            pending = (pending + "\n\n" + stripped).lstrip("\n") if pending else stripped
        else:
            merged.append((pending + "\n\n" + stripped).lstrip("\n") if pending else stripped)
            pending = ""

    # Trailing short paragraph — attach to last full paragraph rather than drop
    if pending:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + pending
        else:
            merged.append(pending)

    return "\n\n".join(merged)


def _token_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into token-bounded chunks with overlap.
    Returns a list of text strings, each fitting within chunk_size tokens.
    """
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(token_ids), step):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text: str = tokenizer.decode(chunk_ids, skip_special_tokens=True)  # type: ignore[assignment]
        chunks.append(chunk_text)
        if end == len(token_ids):
            break

    return chunks


@dataclass
class Chunk:
    """A single indexable unit from a document."""
    chunk_id: str           # UUID — used as Qdrant point ID and source_chunk_id
    doc_id: str
    file_path: str
    page_number: int
    text: str               # child chunk text — embedded into Qdrant for retrieval
    token_count: int
    chunk_index: int        # position within document (0-indexed, child-level)
    parent_text: str | None = field(default=None)
    parent_id: str | None = field(default=None)
    # parent_id: UUID shared by all children of the same parent window.
    # Used by the retriever to deduplicate: multiple children from the same
    # parent → send parent_text once to the LLM, not N times.
    parent_chunk_index: int | None = field(default=None)
    # parent_chunk_index: ordinal position of the parent in the document (0, 1, 2…).
    # After dedup by parent_id (score order), the retriever re-sorts selected
    # parents by parent_chunk_index ascending so the LLM receives them in
    # document order (Article 2 before Article 10, not by retrieval score).


def _parent_child_chunks(
    text: str,
    parent_size: int,
    child_size: int,
    child_overlap: int,
) -> list[tuple[str, str, str, int]]:
    """
    Split text into (child_text, parent_text, parent_id, parent_chunk_index) tuples.

    Parents — contiguous, non-overlapping windows of parent_size tokens:
      Non-overlapping parents guarantee no text ever appears in two parent
      windows. If parents overlapped, the retriever could send the same clause
      twice (once from each parent) — wasting LLM context and confusing output.

    Children — overlapping windows of child_size tokens within each parent:
      Overlap is LOCAL within the parent (children never cross parent boundaries).
      This prevents a clause-boundary split where the key term is at the very
      end of one child and missing from the next.

    parent_id — UUID shared by all children belonging to the same parent.
      The retriever uses this to deduplicate: if child A and child B both hit
      and share a parent_id, the LLM receives the parent text only once.

    parent_chunk_index — ordinal position of the parent in the document (0, 1, 2…).
      After dedup (score-order selection), the retriever re-sorts selected parents
      by parent_chunk_index ascending so the LLM sees them in document order.

    Why work at the token-ID level rather than calling _token_chunks twice?
      _token_chunks decodes token IDs back to text, then re-encodes for the child
      pass. Decode→encode roundtrips can shift token boundaries. Working at the
      token-ID level throughout ensures parent and child windows are exact slices
      of the same token sequence — no boundary drift.
    """
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        return []

    child_step = max(1, child_size - child_overlap)
    triples: list[tuple[str, str, str, int]] = []

    for parent_idx, p_start in enumerate(range(0, len(token_ids), parent_size)):
        p_end = min(p_start + parent_size, len(token_ids))
        p_tokens = token_ids[p_start:p_end]
        parent_text: str = tokenizer.decode(p_tokens, skip_special_tokens=True)  # type: ignore[assignment]
        parent_id = str(uuid.uuid4())

        for c_start in range(0, len(p_tokens), child_step):
            c_end = min(c_start + child_size, len(p_tokens))
            child_text: str = tokenizer.decode(p_tokens[c_start:c_end], skip_special_tokens=True)  # type: ignore[assignment]
            triples.append((child_text, parent_text, parent_id, parent_idx))
            if c_end == len(p_tokens):
                break

    return triples


def chunk_document(document: LoadedDocument) -> list[Chunk]:
    """
    Chunk a loaded document into parent-child token-bounded segments.

    Child chunks (child_chunk_size=256 tokens, 51-token overlap) are embedded
    into Qdrant for precise retrieval. Each child carries parent_text (2048
    tokens, contiguous), parent_id (UUID shared across siblings), and
    parent_chunk_index (document-order position of the parent).

    The retriever uses parent_id to deduplicate and parent_chunk_index to
    reorder selected parents into document order before the LLM sees them.

    Returns chunks ordered by (page_number, child position within page).
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        if not page.text.strip():
            continue

        tuples = _parent_child_chunks(
            _merge_headings(page.text),
            parent_size=settings.chunk_size,
            child_size=settings.child_chunk_size,
            child_overlap=settings.child_chunk_overlap,
        )

        for child_text, parent_text, parent_id, parent_chunk_index in tuples:
            token_count = _count_tokens(child_text)
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.doc_id,
                file_path=document.file_path,
                page_number=page.page_number,
                text=child_text,
                token_count=token_count,
                chunk_index=chunk_index,
                parent_text=parent_text,
                parent_id=parent_id,
                parent_chunk_index=parent_chunk_index,
            ))
            chunk_index += 1

    return chunks
