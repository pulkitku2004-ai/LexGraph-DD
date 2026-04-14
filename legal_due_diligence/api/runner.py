"""
Background job runner — ingestion pipeline + LangGraph execution.

Why a module-level JOB_STORE dict rather than a database?
Sprint 9 is a dev/demo API. Jobs live in memory for the server lifetime —
sufficient for interactive use and smoke tests. Swap for Redis or Postgres
when persistence across restarts is needed.

Why run_pipeline() is synchronous (not async):
FastAPI BackgroundTasks executes tasks in a thread pool after returning the
response. The LangGraph pipeline and ingestion are CPU/IO bound and
fully synchronous — there is nothing to await. Running them in a thread
via BackgroundTasks is correct; wrapping in asyncio.run() would deadlock.

Why imports are deferred inside run_pipeline():
The embedder loads bge-m3 (~1.1GB) on first call. Importing at module level
would trigger the load on server startup before any job is submitted.
Deferred imports let the server start instantly and load the model on demand.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from api.schemas import JobStatus

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    doc_ids: list[str]
    tmp_dir: str
    report: str | None = None
    errors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


# ── In-memory job store ────────────────────────────────────────────────────────
JOB_STORE: dict[str, JobRecord] = {}


def create_job(file_data: list[tuple[str, bytes]]) -> str:
    """
    Persist uploaded file bytes to a temp directory, register the job.

    Args:
        file_data: list of (filename, raw_bytes) pairs

    Returns:
        job_id (UUID string)
    """
    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f"legal_dd_{job_id}_")

    doc_ids: list[str] = []
    for filename, data in file_data:
        dest = Path(tmp_dir) / filename
        dest.write_bytes(data)
        doc_ids.append(dest.stem)

    JOB_STORE[job_id] = JobRecord(
        job_id=job_id,
        status=JobStatus.pending,
        doc_ids=doc_ids,
        tmp_dir=tmp_dir,
    )
    logger.info("[runner] created job %s with %d doc(s): %s", job_id, len(doc_ids), doc_ids)
    return job_id


def run_pipeline(job_id: str) -> None:
    """
    Run ingestion + LangGraph pipeline for a job. Updates JOB_STORE in place.
    Called by FastAPI BackgroundTasks — must not raise (errors → record.errors).
    """
    # Deferred imports — avoids loading bge-m3 at server startup
    from agents.orchestrator.graph import build_graph
    from core.models import DocumentRecord
    from ingestion.chunker import chunk_document
    from ingestion.embedder import embed_chunks
    from ingestion.indexer import index_chunks
    from ingestion.loader import load_document

    record = JOB_STORE.get(job_id)
    if not record:
        logger.error("[runner] job %s not found in store", job_id)
        return

    record.status = JobStatus.running
    tmp_dir = Path(record.tmp_dir)
    documents: list[DocumentRecord] = []

    # ── Ingestion ─────────────────────────────────────────────────────────────
    for doc_id in record.doc_ids:
        matches = list(tmp_dir.glob(f"{doc_id}.*"))
        if not matches:
            msg = f"File not found for doc_id '{doc_id}' in {tmp_dir}"
            logger.error("[runner] %s", msg)
            record.errors.append(msg)
            documents.append(DocumentRecord(doc_id=doc_id, file_path="", processed=False))
            continue

        file_path = str(matches[0])
        try:
            loaded = load_document(file_path, doc_id=doc_id)
            chunks = chunk_document(loaded)
            embedded = embed_chunks(chunks)
            index_chunks(embedded)
            documents.append(DocumentRecord(
                doc_id=doc_id,
                file_path=file_path,
                processed=True,
                page_count=loaded.total_pages,
            ))
            logger.info("[runner] ingested %s (%d chunks)", doc_id, len(chunks))
        except Exception as e:
            msg = f"Ingestion failed for '{doc_id}': {e}"
            logger.error("[runner] %s", msg)
            record.errors.append(msg)
            documents.append(DocumentRecord(doc_id=doc_id, file_path=file_path, processed=False))

    if not any(d.processed for d in documents):
        record.errors.append("All documents failed ingestion — pipeline aborted.")
        record.status = JobStatus.error
        return

    # ── LangGraph pipeline ────────────────────────────────────────────────────
    try:
        graph = build_graph()
        result = graph.invoke({"job_id": job_id, "documents": documents})  # type: ignore[arg-type]
        record.report = result.get("final_report")
        record.errors.extend(result.get("errors", []))
        record.status = JobStatus.done
        logger.info("[runner] job %s complete", job_id)
    except Exception as e:
        msg = f"Pipeline error: {e}"
        logger.error("[runner] job %s — %s", job_id, msg)
        record.errors.append(msg)
        record.status = JobStatus.error


def delete_job(job_id: str) -> None:
    """
    Delete all data associated with a job:
      1. Qdrant points (filtered by doc_id)
      2. Neo4j Document + Clause nodes (DETACH DELETE), then orphan cleanup
      3. Temp directory with uploaded files
      4. JOB_STORE entry
    """
    record = JOB_STORE.get(job_id)
    if not record:
        return

    _delete_qdrant(record.doc_ids)
    _delete_neo4j(record.doc_ids)

    shutil.rmtree(record.tmp_dir, ignore_errors=True)
    logger.info("[runner] removed tmp_dir %s", record.tmp_dir)

    del JOB_STORE[job_id]
    logger.info("[runner] deleted job %s", job_id)


def _delete_qdrant(doc_ids: list[str]) -> None:
    from core.config import settings
    from infrastructure.qdrant_client import get_qdrant_client
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_qdrant_client()
    for doc_id in doc_ids:
        try:
            client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
            logger.info("[runner] deleted Qdrant points for doc_id=%s", doc_id)
        except Exception as e:
            logger.warning("[runner] Qdrant delete failed for %s: %s", doc_id, e)


def _delete_neo4j(doc_ids: list[str]) -> None:
    from infrastructure.neo4j_client import get_neo4j_session

    try:
        with get_neo4j_session() as session:
            for doc_id in doc_ids:
                # DETACH DELETE removes the Document node, all HAS_CLAUSE
                # relationships, and the Clause nodes (which are doc_id-scoped).
                session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_CLAUSE]->(c:Clause)
                    DETACH DELETE d, c
                    """,
                    doc_id=doc_id,
                )
                logger.info("[runner] deleted Neo4j nodes for doc_id=%s", doc_id)

            # Remove entity nodes (Party, Jurisdiction, Duration, MonetaryAmount)
            # that have no remaining relationships after the clause deletions.
            session.run(
                """
                MATCH (n)
                WHERE (n:Party OR n:Jurisdiction OR n:Duration OR n:MonetaryAmount)
                AND NOT (n)--()
                DELETE n
                """
            )
            logger.info("[runner] orphan entity cleanup done")
    except Exception as e:
        logger.warning("[runner] Neo4j delete failed: %s", e)
