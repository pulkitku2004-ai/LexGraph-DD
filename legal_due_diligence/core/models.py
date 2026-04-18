"""
Domain models — the shared vocabulary for every agent.

These live in core/ because they cross agent boundaries. An agent should never
define its own output schema — it uses these. This is how you prevent the
slow death of a multi-agent system: type drift where each agent invents its
own representation of the same concept.

Design decisions worth understanding:
- All IDs are strings, not UUIDs. This lets us use doc filename as the ID
  during development and swap to uuid4 for production without breaking
  the schema (UUID is a string subtype for our purposes).
- normalized_value on ExtractedClause is the key field for contradiction
  detection. Raw clause_text is messy; normalized_value is what the
  contradiction detector compares across documents. E.g. "30 days", "Delaware".
- is_missing_clause on RiskFlag lets the risk scorer distinguish between
  "found a bad clause" and "found no clause at all" — these have different
  legal weight and different report formatting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    doc_id: str
    file_path: str
    processed: bool = False
    page_count: Optional[int] = None


class ExtractedClause(BaseModel):
    document_id: str
    clause_type: str           # one of CUAD's 41 categories
    found: bool                # False = missing clause = risk signal
    clause_text: Optional[str] = None
    normalized_value: Optional[str] = None  # e.g. "Delaware", "30 days"
    confidence: float
    source_chunk_id: str       # for citation tracing back to Qdrant chunk


class RiskFlag(BaseModel):
    document_id: str
    clause_type: str
    risk_level: Literal["high", "medium", "low"]
    reason: str
    is_missing_clause: bool
    source_clause_id: Optional[str] = None  # ExtractedClause.source_chunk_id → Qdrant → page citation


class Contradiction(BaseModel):
    clause_type: str
    document_id_a: str
    document_id_b: str
    value_a: str
    value_b: str
    explanation: str
    risk_level: Literal["high", "medium", "low"] = "medium"
