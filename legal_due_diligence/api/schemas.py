"""
API request/response schemas.

Kept separate from core/models.py because these are transport-layer shapes —
they may diverge from internal domain models (e.g. JobResponse flattens
GraphState fields the API caller needs, omitting internals like graph_built).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    doc_ids: list[str]
    report: str | None
    errors: list[str]
    created_at: datetime


class QARequest(BaseModel):
    question: str


class Citation(BaseModel):
    doc_id: str
    page_number: int
    chunk_id: str
    excerpt: str


class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    chunks_retrieved: int
