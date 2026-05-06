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

from pydantic import Field, SecretStr, computed_field
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
    # EXTRACTION CHAIN (41 cats × 50 docs = ~2050 calls/job):
    #   gpt-4o-mini (primary) → ollama/mistral-nemo (fallback)
    #
    #   gpt-4o-mini is the primary: Cond. F1 0.617 vs 0.42 for llama-3.1-8b.
    #   Groq was removed in Sprint 26 — all roles now use gpt-4o-mini for
    #   consistent output quality; Groq fallback caused ASTR-O groundedness
    #   failures due to paraphrasing. Ollama is the sole offline fallback.
    #
    # REASONING (risk scorer, report narrative, Q&A — low volume):
    #   gpt-4o-mini (default) — same key, separate role.
    #   Override with LLM_REASONING_MODEL in .env for large batches or offline use:
    #     Large batch (50 docs):  ollama/mistral-nemo                                 ← unlimited
    #     Small batch (≤25 docs): openrouter/nvidia/nemotron-3-super-120b-a12b:free  ← verified working
    #     Offline:                ollama/mistral-nemo
    #   Note: deepseek/deepseek-r1:free is NOT available on this account's free tier.

    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key — primary extraction model (gpt-4o-mini). Required for ASTR-O.",
    )
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Groq API key — extraction fallback",
    )
    openrouter_api_key: SecretStr = Field(
        default=SecretStr(""),
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

    llm_reasoning_model: str = Field(
        default="gpt-4o-mini",
        description="Model for risk scoring, report synthesis, and Q&A.",
    )
    llm_extraction_model: str = Field(
        default="gpt-4o-mini",
        description="Primary extraction model. Override via LLM_EXTRACTION_MODEL in .env.",
    )
    llm_extraction_fallbacks: list[str] = Field(
        default=["ollama/mistral-nemo"],
        description=(
            "Ordered fallback chain for extraction. Sprint 26: Groq removed — "
            "all roles use gpt-4o-mini; Ollama is the sole local fallback with "
            "no rate limit and no quality mismatch risk for ASTR-O groundedness."
        ),
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
    chunk_size: int = Field(default=2048)
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

    # ── Auth ─────────────────────────────────────────────────────────────────
    # Static Bearer token for API authentication.
    # Empty string (default) = auth disabled — safe for local dev.
    # Set API_KEY=<secret> in .env to enable. All endpoints require
    # Authorization: Bearer <key>. The Streamlit UI reads the same var.
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Bearer token for API auth. Empty = disabled (local dev).",
    )

    # ── Observability ────────────────────────────────────────────────────────
    # OTLP HTTP endpoint for OpenTelemetry span export.
    # Empty (default) = no export; spans are created but dropped (zero overhead).
    # Example: http://localhost:4318 for a local OTel collector.
    otel_endpoint: str = Field(
        default="",
        description="OTLP HTTP endpoint for OTel span export. Empty = disabled.",
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
if settings.openai_api_key.get_secret_value():
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())
if settings.openrouter_api_key.get_secret_value():
    os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key.get_secret_value())
if settings.groq_api_key.get_secret_value():
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key.get_secret_value())

# Suppress litellm's verbose request/response logging globally.
# Set once here so agent modules don't each repeat it.
import litellm  # noqa: E402
litellm.suppress_debug_info = True
