"""
Embedder — Chunk → dense (1024-dim) + sparse (learned weights) using BAAI/bge-m3.

Why bge-m3 over bge-base-en-v1.5?
bge-m3 produces two complementary representations in a single forward pass:
  1. Dense vector (1024-dim, cosine similarity) — semantic matching, same role
     as the old bge-base 768-dim vector but from a stronger model.
  2. Sparse vector (learned weights per token) — captures exact and near-exact
     term matching. Replaces the hand-crafted BM25 pickle with a data-driven
     sparse retriever stored natively in Qdrant.

Why not use FlagEmbedding (the official bge-m3 library)?
FlagEmbedding >= 1.2 is incompatible with transformers >= 5.x (several symbols
were removed in the transformers 5.0 API overhaul). FlagEmbedding <= 1.1.9 pre-
dates BGEM3FlagModel. Rather than downgrading transformers (which would break
the rest of the project), we implement bge-m3 encoding directly using only:
  - transformers.AutoModel / AutoTokenizer
  - huggingface_hub.hf_hub_download (to fetch sparse_linear.pt on first use)
This is a thin, stable implementation that only uses the stable HuggingFace API.

How bge-m3 sparse encoding works:
  1. Run XLM-RoBERTa forward pass → last_hidden_state (batch, seq_len, 1024)
  2. Apply sparse_linear (Linear 1024→1): relu(sparse_linear(hidden)) → token weights
     shape: (batch, seq_len, 1)
  3. For each unique token_id in input_ids, take max weight across all positions
     where that token appears.
  4. Result: {token_id: max_weight} sparse dict — non-zero entries only.
This is the exact algorithm from the BGEM3ForInference source in FlagEmbedding.

How bge-m3 dense encoding works:
  CLS token of the last_hidden_state → L2-normalized 1024-dim vector.
  (bge-m3 uses CLS pooling unlike bge-base which needed mean pooling.)

Why use_fp16 via half()?
bge-m3 is 570M params (vs 109M for bge-base). fp16 halves memory footprint
and is faster on MPS. Sparse output is always computed in float32 for precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from core.config import settings
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ── Device selection ───────────────────────────────────────────────────────────
def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Model singletons ──────────────────────────────────────────────────────────
# Both components loaded lazily and cached for the process lifetime.
_tokenizer: PreTrainedTokenizerBase | None = None
_model: PreTrainedModel | None = None
_sparse_linear: torch.nn.Linear | None = None
_device: torch.device | None = None


def _load_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.nn.Linear, torch.device]:
    global _tokenizer, _model, _sparse_linear, _device

    if _model is None:
        from huggingface_hub import hf_hub_download

        device = _get_device()
        _device = device

        logger.info("[embedder] loading %s ...", settings.embedding_model)
        _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

        # Load base XLM-RoBERTa encoder
        # fp16 on MPS/CUDA: halves memory (2.24GB → 1.12GB) and speeds up
        # matrix multiplications significantly. float32 fallback on CPU.
        _model = AutoModel.from_pretrained(settings.embedding_model)
        if device.type in ("mps", "cuda"):
            _model = _model.half()  # type: ignore[assignment]  # float16; .half() returns Module
        _model = _model.to(device)  # type: ignore[assignment]
        _model.eval()

        # Load bge-m3 sparse_linear head (Linear 1024→1, trained weights)
        # sparse_linear.pt is a separate file on the HuggingFace hub — downloaded
        # once and cached alongside the main model weights.
        sparse_pt_path = hf_hub_download(
            repo_id=settings.embedding_model,
            filename="sparse_linear.pt",
        )
        sl_state = torch.load(sparse_pt_path, map_location="cpu", weights_only=True)
        _sparse_linear = torch.nn.Linear(
            _model.config.hidden_size, 1, bias=True
        )
        _sparse_linear.load_state_dict(sl_state)
        # Keep sparse_linear on CPU — it's a tiny Linear(1024, 1) and MPS kernel-
        # launch overhead dominates for such small ops. We transfer hidden states
        # to CPU in _compute_sparse() before applying it.
        _sparse_linear.eval()

        logger.info("[embedder] model loaded on %s (hidden_size=%d)", device, _model.config.hidden_size)

    return _tokenizer, _model, _sparse_linear, _device  # type: ignore[return-value]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]              # 1024-dim dense, L2-normalised
    sparse_vector: dict[int, float]  # {token_id: weight} learned sparse weights


# ── Sparse vector helper ──────────────────────────────────────────────────────

def _compute_sparse(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sparse_linear: torch.nn.Linear,
) -> list[dict[int, float]]:
    """
    Compute SPLADE-style sparse vectors for a batch — fully on CPU.

    Why CPU instead of MPS for this step?
    sparse_linear is a single linear layer (1024→1). The MPS kernel launch overhead
    for such a tiny operation dominates the actual compute. Running it on CPU with
    pre-fetched tensors is ~6x faster than MPS for this specific bottleneck.

    Algorithm (vectorised over the full batch):
    1. Transfer last_hidden_state to CPU as float32 in one call.
    2. Apply sparse_linear on CPU → token weights (batch, seq_len, 1) → squeeze.
    3. For each item, collect (token_id, weight) pairs where weight > 0 and
       attention_mask == 1, then max-aggregate per unique token_id.
    """
    import numpy as np

    # Move GPU tensors to CPU in bulk — fewer device→host transfers
    hidden_cpu = last_hidden_state.detach().cpu().float()  # (batch, seq_len, 1024)
    ids_cpu = input_ids.detach().cpu()                      # (batch, seq_len)
    mask_cpu = attention_mask.detach().cpu().bool()         # (batch, seq_len)

    # Sparse linear on CPU (sparse_linear is kept on CPU — see _load_model)
    with torch.no_grad():
        token_weights = torch.relu(sparse_linear(hidden_cpu)).squeeze(-1)  # (batch, seq_len)

    batch_sparse: list[dict[int, float]] = []
    for i in range(hidden_cpu.shape[0]):
        mask_i = mask_cpu[i]
        ids_np = ids_cpu[i][mask_i].numpy()          # non-padding token ids
        weights_np = token_weights[i][mask_i].numpy() # corresponding weights

        nonzero = weights_np > 0.0
        if not nonzero.any():
            batch_sparse.append({})
            continue

        ids_nz = ids_np[nonzero]
        weights_nz = weights_np[nonzero]

        # Max-aggregate weights per unique token_id using numpy for speed
        unique_ids, first_idx = np.unique(ids_nz, return_index=True)
        # For max: group by token_id, find max weight per group
        sparse: dict[int, float] = {}
        for uid, w in zip(ids_nz.tolist(), weights_nz.tolist()):
            if w > sparse.get(uid, 0.0):
                sparse[uid] = w

        batch_sparse.append(sparse)

    return batch_sparse


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[Chunk], batch_size: int = 24) -> list[EmbeddedChunk]:
    """
    Embed a list of chunks using bge-m3, returning both dense and sparse vectors.

    batch_size=24: empirically optimal on M4 MPS (16GB). The dominant cost is the
    MPS→CPU synchronization overhead (~3s regardless of batch size). Larger batches
    amortise this fixed cost; batch_size=24 gives ~140ms/chunk vs 271ms/chunk at 12.

    Dense: CLS token → L2-normalised 1024-dim vector.
    Sparse: sparse_linear projection over all token positions → {token_id: max_weight}.
    """
    if not chunks:
        return []

    tokenizer, model, sparse_linear, device = _load_model()
    results: list[EmbeddedChunk] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c.text for c in batch]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        last_hidden = output.last_hidden_state  # (batch, seq_len, 1024)

        # ── Dense: CLS token, L2-normalised ──────────────────────────────
        cls = last_hidden[:, 0, :]                              # (batch, 1024)
        cls_normed = F.normalize(cls, p=2, dim=1)               # unit sphere
        dense_list: list[list[float]] = cls_normed.cpu().float().tolist()

        # ── Sparse: token importance weights ─────────────────────────────
        # _compute_sparse transfers to CPU and casts to float32 internally.
        batch_sparse = _compute_sparse(
            last_hidden, encoded["input_ids"], encoded["attention_mask"], sparse_linear
        )

        for chunk, dense, sparse in zip(batch, dense_list, batch_sparse):
            results.append(EmbeddedChunk(
                chunk=chunk,
                vector=dense,
                sparse_vector=sparse,
            ))

        logger.debug(
            "[embedder] embedded batch %d/%d",
            min(batch_start + batch_size, len(chunks)),
            len(chunks),
        )

    return results
