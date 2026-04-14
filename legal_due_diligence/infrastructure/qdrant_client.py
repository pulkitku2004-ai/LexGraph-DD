"""
Qdrant client singleton.

Why a module-level singleton with lazy initialization?
- Instantiating a QdrantClient is cheap but connecting is not. We initialize
  on first use so import time stays fast and tests can patch the singleton
  before any connection is attempted.
- A singleton means all agents share one connection pool rather than each
  agent spinning up its own — critical when you're running 6 agents
  concurrently under asyncio.

Why not AsyncQdrantClient?
The qdrant-client library's async client requires the full grpc extra and
has a different API surface. For Sprint 0 we use the sync client wrapped
in asyncio.to_thread() at the call site. We'll evaluate switching if
profiling shows the thread pool is a bottleneck (unlikely for ≤50 docs).
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from core.config import settings

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def check_qdrant_health() -> bool:
    """
    Returns True if Qdrant is reachable and responsive.
    Does NOT raise — callers check the bool and set state.errors if needed.
    """
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception:
        return False
