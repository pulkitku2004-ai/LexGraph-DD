"""
GraphState — the single object that flows through the LangGraph state machine.

Architectural rationale for a fat state object vs. passing individual fields:

LangGraph's state machine is fundamentally a reducer over a shared state type.
Every node receives the full state and returns a dict of fields to update.
This means:
  1. Every agent can read any prior agent's output without explicit message passing.
  2. The graph can checkpoint the entire state to disk (LangGraph's built-in
     checkpointing) for free — useful for resuming long jobs.
  3. Debugging is simple: inspect the state at any node boundary to see
     exactly what changed.

The alternative — passing only the fields each agent needs — sounds cleaner
but creates hidden coupling. When the contradiction detector needs both
extracted_clauses AND the entity graph, you'd have to thread those fields
through every intermediate node anyway.

Why `errors: list[str]` instead of raising exceptions?
LangGraph stops the graph on an unhandled exception. For a multi-document job
where one PDF might be malformed, we want the graph to continue processing
the other documents and collect errors, then report them in the final brief.
Agents append to errors; the orchestrator checks it before routing.

Why not `Annotated[list[X], operator.add]` (LangGraph's append reducer)?
Because we want full control over deduplication and ordering. Agents return
updated full lists, not just new items. The slight memory overhead for 50
documents is negligible compared to the debugging clarity you gain.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from core.models import (
    Contradiction,
    DocumentRecord,
    ExtractedClause,
    RiskFlag,
)


class GraphState(BaseModel):
    job_id: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Document registry ─────────────────────────────────────────────────
    documents: list[DocumentRecord] = []

    # ── Agent outputs (accumulated) ───────────────────────────────────────
    extracted_clauses: list[ExtractedClause] = []
    risk_flags: list[RiskFlag] = []
    contradictions: list[Contradiction] = []

    # ── Infrastructure readiness flags ────────────────────────────────────
    # These gate routing decisions in the orchestrator. The health check node
    # sets them; downstream agents trust them rather than re-checking.
    neo4j_ready: bool = False
    qdrant_ready: bool = False
    graph_built: bool = False  # True after entity_mapper writes to Neo4j

    # ── Terminal outputs ──────────────────────────────────────────────────
    final_report: Optional[str] = None

    # ── Error accumulation ────────────────────────────────────────────────
    errors: list[str] = []
