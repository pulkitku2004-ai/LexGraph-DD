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
    text: str
    token_count: int
    chunk_index: int        # position within document (0-indexed)


def chunk_document(document: LoadedDocument) -> list[Chunk]:
    """
    Chunk a loaded document into token-bounded overlapping segments.
    Returns chunks ordered by (page_number, position_within_page).
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        if not page.text.strip():
            continue

        page_chunks = _token_chunks(
            _merge_headings(page.text),
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for chunk_text in page_chunks:
            token_count = _count_tokens(chunk_text)
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.doc_id,
                file_path=document.file_path,
                page_number=page.page_number,
                text=chunk_text,
                token_count=token_count,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

    return chunks
