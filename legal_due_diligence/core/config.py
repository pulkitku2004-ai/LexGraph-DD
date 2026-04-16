"""
Central configuration loaded once at import time via pydantic-settings.

Why a settings singleton rather than os.getenv() scattered across agents?
- Type coercion and validation at startup — you catch "QDRANT_PORT=abc" immediately,
  not when the first vector query fires.
- Single source of truth: every module imports `settings`, not raw env vars.
- pydantic-settings reads from environment first, then falls back to .env —
  so the same code works in Docker (env vars) and local dev (.env file).

Why not a dataclass or plain dict?
- Pydantic gives you validators and computed fields for free.
- IDEs autocomplete settings.neo4j_uri instead of config["NEO4J_URI"].
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently drop unknown env vars — important in Docker
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Provider API keys — pydantic-settings reads these from .env automatically.
    #
    # THREE PROVIDERS, THREE ROLES — why each one:
    #
    # 1. GROQ  →  extraction (high-volume, latency-sensitive)
    #    - llama-3.1-8b-instant: fastest available inference, ~300ms/call
    #    - Free tier: 6000 TPM — sufficient with fallback chain
    #    - 41 categories × 50 docs = 2050 calls per job; cloud quality is overkill
    #      for structured JSON extraction; speed and throughput matter more
    #    - DO NOT use Groq for reasoning: 6000 TPM drains fast at reasoning
    #      token lengths, and Groq models are optimised for speed not nuance
    #
    # 2. OLLAMA  →  reasoning fallback + large batch jobs (zero cost, zero rate limit)
    #    - mistral-nemo 7.1GB on M4: reliable when cloud limits are hit
    #    - No internet required, full privacy — useful when testing offline
    #    - Ceiling: 7B params — quality degrades on nuanced legal risk assessment
    #      compared to 70B+ cloud models; acceptable for dev and large batches
    #    - DO NOT use Ollama for extraction at scale: slower than Groq, blocks
    #      the M4 GPU for embedding if both run simultaneously
    #    - Best role: final fallback that never fails (no rate limit, no cost)
    #
    # 3. OPENROUTER (FREE TIER)  →  reasoning quality upgrade for small/medium jobs
    #    - Free tier specifics:
    #        * Requires `:free` suffix on model names (e.g. deepseek/deepseek-r1:free)
    #        * Rate limits: ~20 req/min, ~200 req/day depending on model
    #        * No cost, but no guaranteed throughput — same constraint as Groq free
    #    - IS IT BENEFICIAL for small jobs (≤25 docs)? YES:
    #        * deepseek-r1:free and llama-3.3-70b:free reason far better than
    #          mistral-nemo 7B on nuanced legal language at zero cost
    #        * ~200 reasoning calls (8 cats × 25 docs) fits within daily free limit
    #        * Single key, no separate Anthropic/OpenAI account needed
    #    - IS IT BENEFICIAL for large jobs (50 docs)? NO:
    #        * 8 categories × 50 docs = ~400 reasoning calls → hits daily limit mid-run
    #        * Use Ollama directly for 50-doc batches — unlimited, no interruption
    #    - NOT beneficial for extraction at any scale: rate limits make it unreliable
    #      at 2050 calls/job; Groq free tier is strictly better for extraction volume
    #
    # DECISION GUIDE — set LLM_REASONING_MODEL in .env:
    #   Small batch (≤25 docs):  openrouter/nvidia/nemotron-3-super-120b-a12b:free  ← verified working
    #                            openrouter/nousresearch/hermes-3-llama-3.1-405b:free  ← 405B, may rate-limit
    #                            openrouter/meta-llama/llama-3.3-70b-instruct:free    ← may rate-limit
    #   Large batch (50 docs):   ollama/mistral-nemo                                  ← rate-limit safe
    #   Offline / no internet:   ollama/mistral-nemo
    #   Extraction always:       groq/llama-3.1-8b-instant → fallbacks (no change needed)
    #
    # Note: deepseek/deepseek-r1:free is NOT available on this account's free tier.

    groq_api_key: str = Field(default="", description="Groq API key — extraction only")
    openrouter_api_key: str = Field(
        default="",
        description=(
            "OpenRouter API key (free tier) — reasoning for small/medium jobs (≤25 docs). "
            "Use `:free` suffix on model names. Rate limits: ~200 req/day. "
            "Set LLM_REASONING_MODEL=openrouter/deepseek/deepseek-r1:free to activate. "
            "For 50-doc batches, use ollama/mistral-nemo instead (no rate limit)."
        ),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL — local reasoning fallback, zero cost, zero rate limit",
    )

    # LiteLLM model strings — prefix tells LiteLLM which provider to route to.
    # Two tiers:
    #   Extraction (high volume, ~2050 calls for 50 docs): Groq Llama fast inference
    #   Reasoning (low volume, quality matters): Ollama locally (dev) or OpenRouter (prod)
    llm_reasoning_model: str = Field(
        default="ollama/mistral-nemo",
        description=(
            "Model for risk scoring and report synthesis — quality over speed. "
            "Default: ollama/mistral-nemo (local, unlimited, good for large batches). "
            "Small job upgrade (≤25 docs): openrouter/deepseek/deepseek-r1:free "
            "or openrouter/meta-llama/llama-3.3-70b-instruct:free — "
            "note `:free` suffix required; ~200 req/day limit applies."
        ),
    )
    llm_extraction_model: str = Field(
        default="groq/llama-3.1-8b-instant",
        description="Primary extraction model — Groq fast inference, free tier sufficient at this volume",
    )
    # Fallback chain for extraction — each has its own rate limit bucket.
    # LiteLLM tries these in order when the primary is rate limited.
    # OpenRouter is deliberately excluded from extraction fallbacks: per-token
    # cost at 2050 calls/job is not justified when Ollama (free, local) is available.
    llm_extraction_fallbacks: list[str] = Field(
        default=["groq/meta-llama/llama-4-scout-17b-16e-instruct", "ollama/mistral-nemo"],
        description="Fallback models for extraction in priority order",
    )

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="legal_clauses")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # ── Embeddings ───────────────────────────────────────────────────────────
    # bge-m3 is a multi-functionality model that produces both dense (1024-dim)
    # and learned sparse (SPLADE-style) vectors in a single forward pass.
    # The sparse vectors replace the BM25 pickle — they capture exact and
    # near-exact term matches the same way BM25 does, but learned from data
    # rather than a hand-crafted TF-IDF formula. Both vectors are stored in
    # Qdrant and fused with RRF at query time.
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dim: int = Field(default=1024)
    sparse_vector_name: str = Field(
        default="sparse",
        description="Name of the Qdrant sparse vector field. Must match the name used at collection creation.",
    )
    # bge-m3 does not require an asymmetric query prefix — the model is trained
    # with internal instruction tuning that handles query vs passage distinction
    # without an explicit prefix string.
    embedding_query_prefix: str = Field(
        default="",
        description=(
            "Prefix prepended to query text only (not to indexed chunks). "
            "Empty for bge-m3 (handles query/passage asymmetry internally). "
            "bge-base/large-en-v1.5: 'Represent this sentence for searching relevant passages: '"
        ),
    )

    # ── Re-ranking ───────────────────────────────────────────────────────────
    # Cross-encoder re-ranker applied after RRF fusion.
    # Empty string = disabled (default — preserves current behaviour).
    # Recommended: "BAAI/bge-reranker-base" (same ecosystem as bge-base embeddings).
    # The reranker scores (query, chunk) pairs jointly, catching relevance that
    # bi-encoder cosine similarity misses (e.g. category-label queries like
    # "No-Solicit Of Employees" vs clause text that never uses those exact words).
    # Cost: ~candidate_k cross-encoder calls per retrieval (fast on M4 MPS).
    reranker_model: str = Field(
        default="",
        description=(
            "Cross-encoder model for re-ranking RRF candidates. "
            "Empty = disabled. "
            "Recommended: 'BAAI/bge-reranker-base'. "
            "Loaded lazily on first retrieval call."
        ),
    )

    # ── Chunking strategy ────────────────────────────────────────────────────
    # Parent-child chunking (Sprint 16 revision):
    #
    # Parents (chunk_size = 2048 tokens, contiguous — NO overlap between parents):
    #   - Large window gives the LLM a full legal section with surrounding context.
    #   - Contiguous (non-overlapping) parents mean the same text never appears in
    #     two parents — no duplicate context sent to the LLM.
    #   - Each parent gets a UUID parent_id + sequential parent_chunk_index (its
    #     position in the document). Both stored in every child's Qdrant payload.
    #
    # Children (child_chunk_size = 256 tokens, 20% overlap = 51 tokens):
    #   - Embedded into Qdrant for precise retrieval.
    #   - Overlap is LOCAL within the parent — children do not cross parent
    #     boundaries, so no child ever references text from two parents.
    #
    # Retriever deduplication (two-stage sort):
    #   Stage 1 — Score order: sort all child hits by RRF score descending.
    #             Walk the list; when a new parent_id is seen, add that parent
    #             to the selected set. Stop when top_k unique parents collected.
    #             (Best child per parent determines which parents make the cut.)
    #   Stage 2 — Document order: sort the selected parents by parent_chunk_index
    #             ascending before passing to the LLM.
    #             Legal clauses reference prior sections; sending Article 10 before
    #             Article 2 breaks the LLM's understanding of legal hierarchy.
    #
    # chunk_overlap: 0 — parents are contiguous; field kept for backward compat.
    chunk_size: int = Field(default=2048)
    chunk_overlap: int = Field(
        default=0,
        description="Unused post-Sprint-16 — parents are contiguous (no inter-parent overlap).",
    )
    child_chunk_size: int = Field(
        default=256,
        description="Token size of child chunks embedded into Qdrant for retrieval.",
    )
    child_chunk_overlap: int = Field(
        default=51,
        description="Token overlap between consecutive child chunks within a parent (≈20% of 256).",
    )

    # ── Async extraction ─────────────────────────────────────────────────────
    # Max concurrent LLM calls during clause extraction.
    # Categories within a document AND documents across each other all share
    # this single semaphore, so the actual Groq request rate stays bounded.
    #
    # Groq free tier: 6000 TPM, ~20 req/min sustained.
    # At 10 concurrent × ~300ms avg latency → ~33 req/min peak.
    # The fallback chain (llama-4-scout → ollama) absorbs rate-limit errors,
    # so 10 is safe — burst hits Groq, overflow routes to Ollama locally.
    # Lower to 5 if you see frequent 429s with no Ollama fallback configured.
    extraction_concurrency: int = Field(
        default=10,
        description=(
            "Max concurrent LLM calls during clause extraction (global semaphore). "
            "Retrieval (Qdrant + bge-m3) runs concurrently without this limit. "
            "10 is safe for Groq free tier with Ollama fallback."
        ),
    )

    # ── Application ──────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached singleton. lru_cache(maxsize=1) means this constructs Settings
    exactly once per process, regardless of how many modules call get_settings().
    In tests you can call get_settings.cache_clear() and patch env vars to get
    a fresh config without restarting the interpreter.
    """
    return Settings()


# Module-level alias so callers can do `from core.config import settings`
# instead of calling get_settings() every time. Both patterns are fine.
settings = get_settings()

# Propagate API keys into os.environ for LiteLLM provider routing.
# pydantic-settings reads .env into the settings object but does NOT write
# back to os.environ, so LiteLLM (which reads os.environ directly) would
# miss keys that are only in .env and not already exported in the shell.
if settings.openrouter_api_key:
    os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
if settings.groq_api_key:
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
