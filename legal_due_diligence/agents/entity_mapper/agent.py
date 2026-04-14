"""
Entity Mapper Agent — Sprint 4.

Reads state.extracted_clauses, extracts typed entities, and writes the
knowledge graph to Neo4j. Sets state.graph_built = True on success.

Pipeline position: after risk_scorer, before contradiction_detector.
Only runs if state.neo4j_ready = True (set by health_check_node).

Graph written per run:
    (:Document {doc_id})           — one per unique document_id
        -[:HAS_CLAUSE]->
    (:Clause {doc_id, clause_type, normalized_value, confidence, found, source_chunk_id})
        -[:INVOLVES]->    (:Party {name})
        -[:GOVERNED_BY]-> (:Jurisdiction {name})
        -[:HAS_DURATION]->(:Duration {value})
        -[:HAS_AMOUNT]->  (:MonetaryAmount {value})

Error handling: per-clause exceptions are caught, logged, appended to
state.errors, and the loop continues. A single bad clause does not abort
the graph build — partial graphs are still useful to contradiction_detector.
A Neo4j session failure is a hard stop (returns graph_built=False).

Why write all clauses (found=True AND found=False)?
The contradiction detector needs to distinguish two cases:
    1. Contract A says "Delaware", contract B says "New York" → conflict
    2. Contract A has governing law, contract B is missing it → different risk
Both cases require a Clause node to exist for each document.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from core.state import GraphState
from infrastructure.neo4j_client import get_neo4j_session

from .extractor import extract_entities
from .schema import (
    ensure_constraints,
    write_amount,
    write_clause,
    write_document,
    write_duration,
    write_jurisdiction,
    write_party,
)

logger = logging.getLogger(__name__)


def entity_mapper_node(state: GraphState) -> dict:
    """
    LangGraph node: build the Neo4j knowledge graph from extracted clauses.

    Returns dict with:
        graph_built: bool
        status: "graph_built"
        errors: updated list (may append new entries)
    """
    if not state.neo4j_ready:
        logger.warning("[entity_mapper] Neo4j not ready — skipping graph build")
        return {"graph_built": False, "status": "graph_built"}

    if not state.extracted_clauses:
        logger.info("[entity_mapper] No extracted clauses in state — nothing to write")
        return {"graph_built": True, "status": "graph_built"}

    errors = list(state.errors)
    clauses_written = 0
    entity_counts: dict[str, int] = defaultdict(int)

    try:
        with get_neo4j_session() as session:
            ensure_constraints(session)

            # Write Document nodes first — Clause writes MATCH on them.
            doc_ids = {c.document_id for c in state.extracted_clauses}
            for doc_id in sorted(doc_ids):
                write_document(session, doc_id)
            logger.info("[entity_mapper] wrote %d Document node(s): %s", len(doc_ids), sorted(doc_ids))

            for clause in state.extracted_clauses:
                try:
                    write_clause(
                        session,
                        doc_id=clause.document_id,
                        clause_type=clause.clause_type,
                        normalized_value=clause.normalized_value,
                        confidence=clause.confidence,
                        found=clause.found,
                        source_chunk_id=clause.source_chunk_id,
                    )
                    clauses_written += 1

                    entities = extract_entities(clause)

                    for party in entities["parties"]:
                        write_party(session, clause.document_id, clause.clause_type, party)
                        entity_counts["parties"] += 1

                    for jur in entities["jurisdictions"]:
                        write_jurisdiction(session, clause.document_id, clause.clause_type, jur)
                        entity_counts["jurisdictions"] += 1

                    for dur in entities["durations"]:
                        write_duration(session, clause.document_id, clause.clause_type, dur)
                        entity_counts["durations"] += 1

                    for amt in entities["amounts"]:
                        write_amount(session, clause.document_id, clause.clause_type, amt)
                        entity_counts["amounts"] += 1

                except Exception as exc:
                    msg = (
                        f"[entity_mapper] clause write failed "
                        f"{clause.document_id}/{clause.clause_type}: {exc}"
                    )
                    logger.warning(msg)
                    errors.append(msg)

    except Exception as exc:
        msg = f"[entity_mapper] Neo4j session error: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {"graph_built": False, "status": "graph_built", "errors": errors}

    logger.info(
        "[entity_mapper] graph build complete — clauses=%d | parties=%d "
        "jurisdictions=%d durations=%d amounts=%d",
        clauses_written,
        entity_counts["parties"],
        entity_counts["jurisdictions"],
        entity_counts["durations"],
        entity_counts["amounts"],
    )

    return {"graph_built": True, "status": "graph_built", "errors": errors}
