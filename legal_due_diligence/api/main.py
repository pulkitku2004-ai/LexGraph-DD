"""
Sprint 9 — FastAPI layer over the legal due diligence pipeline.

Endpoints:
  POST   /jobs              Upload 1–50 files, kick off pipeline, return job_id
  GET    /jobs/{job_id}     Poll job status + retrieve report when done
  POST   /jobs/{job_id}/qa  Ask a question grounded in a completed job's documents
  DELETE /jobs/{job_id}     Remove Qdrant points, Neo4j nodes, temp files

Start the server:
  uvicorn legal_due_diligence.api.main:app --reload --port 8000

Or from within legal_due_diligence/:
  uvicorn api.main:app --reload --port 8000

Design notes:

Why 202 Accepted for POST /jobs?
The pipeline takes minutes (LLM calls × 41 categories × N documents).
202 signals "received and processing" — the caller polls GET /jobs/{id}
for the result. 200 would imply the work is done.

Why BackgroundTasks instead of Celery/RQ?
For ≤50 documents in a dev/demo context, FastAPI's built-in background
task execution is sufficient. No broker, no worker process, no infra.
Swap for a task queue when you need job persistence across restarts or
horizontal scaling.

Why DELETE runs in BackgroundTasks?
Qdrant deletion + Neo4j deletion can take a few seconds for large jobs.
Returning 204 immediately while cleanup runs in the background keeps the
client responsive. If cleanup fails, it's logged but doesn't affect the
client.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensures bare imports (from core.config import ...) resolve correctly regardless
# of how uvicorn is started (from project root or from legal_due_diligence/).
_PKG_ROOT = Path(__file__).parent.parent  # → legal_due_diligence/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.runner import JOB_STORE, create_job, delete_job, run_pipeline
from api.schemas import Citation, JobResponse, JobStatus, QARequest, QAResponse
from core.config import settings
from infrastructure.neo4j_client import close_neo4j_driver

# ── Auth ──────────────────────────────────────────────────────────────────────
# auto_error=False so we return a clean 401 instead of FastAPI's default 403.
_bearer = HTTPBearer(auto_error=False)


def _verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """Dependency — enforces Bearer token auth when API_KEY is configured."""
    if not settings.api_key:
        return  # auth disabled in dev (API_KEY not set)
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)

app = FastAPI(
    title="Legal Due Diligence API",
    version="0.1.0",
    description="Multi-agent legal contract analysis — clause extraction, risk scoring, contradiction detection, Q&A.",
)


@app.on_event("shutdown")
def _shutdown() -> None:
    """Drain the Neo4j connection pool cleanly on server stop."""
    close_neo4j_driver()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/jobs",
    response_model=JobResponse,
    status_code=202,
    summary="Submit a due diligence job",
    description="Upload 1–50 PDF or DOCX files. Returns immediately with a job_id. Poll GET /jobs/{job_id} for status.",
)
async def submit_job(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="PDF or DOCX contract files"),
    _: None = Depends(_verify_api_key),
) -> JobResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    file_data: list[tuple[str, bytes]] = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="All uploaded files must have a filename.")
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in (".pdf", ".docx", ".doc", ".txt"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}' for '{upload.filename}'. Expected .pdf, .docx, or .txt",
            )
        contents = await upload.read()
        file_data.append((upload.filename, contents))

    job_id = create_job(file_data)
    background_tasks.add_task(run_pipeline, job_id)

    record = JOB_STORE[job_id]
    return JobResponse(
        job_id=record.job_id,
        status=record.status,
        doc_ids=record.doc_ids,
        report=record.report,
        errors=record.errors,
        created_at=record.created_at,
    )


@app.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Poll job status",
    description="Returns current status. When status='done', the 'report' field contains the full markdown brief.",
)
def get_job(job_id: str, _: None = Depends(_verify_api_key)) -> JobResponse:
    record = JOB_STORE.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse(
        job_id=record.job_id,
        status=record.status,
        doc_ids=record.doc_ids,
        report=record.report,
        errors=record.errors,
        created_at=record.created_at,
    )


@app.post(
    "/jobs/{job_id}/qa",
    response_model=QAResponse,
    summary="Ask a question about a completed job",
    description="Runs hybrid retrieval across all documents in the job and returns a grounded answer with citations.",
)
def qa(job_id: str, request: QARequest, _: None = Depends(_verify_api_key)) -> QAResponse:
    record = JOB_STORE.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if record.status != JobStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not complete yet (status='{record.status}'). Wait for status='done'.",
        )
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    from agents.report_qa.qa import answer_question

    result: dict[str, Any] = answer_question(question=request.question, doc_ids=record.doc_ids)
    return QAResponse(
        answer=result["answer"],
        citations=[
            Citation(
                doc_id=c["doc_id"],
                page_number=c["page_number"],
                chunk_id=c["chunk_id"],
                excerpt=c["excerpt"],
            )
            for c in result["citations"]
        ],
        chunks_retrieved=result["chunks_retrieved"],
    )


@app.delete(
    "/jobs/{job_id}",
    status_code=204,
    summary="Delete a job and all its data",
    description="Removes Qdrant vectors, Neo4j graph nodes, and uploaded files for this job.",
)
def delete(job_id: str, background_tasks: BackgroundTasks, _: None = Depends(_verify_api_key)) -> Response:
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    record = JOB_STORE[job_id]
    if record.status == JobStatus.running:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a running job. Wait for it to complete first.",
        )
    background_tasks.add_task(delete_job, job_id)
    return Response(status_code=204)
