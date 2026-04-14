"""
Health check node — the first node in the LangGraph state machine.

Why make health checking a graph node rather than a startup check in main.py?
Two reasons:
1. The graph can route around failures. If Qdrant is down but Neo4j is up,
   the orchestrator can still run entity mapping on previously indexed data.
   A startup check that raises on any failure would kill the entire job.
2. State-driven observability. The GraphState records neo4j_ready and
   qdrant_ready, so the final report can include infrastructure context:
   "Contradiction detection skipped — Neo4j unavailable during job xyz."

The node returns a dict (not a GraphState) because LangGraph merges the
returned dict into the existing state. Returning the full state object would
overwrite fields other nodes have already set — a subtle bug that's painful
to debug.
"""

from __future__ import annotations

import logging

from core.state import GraphState
from infrastructure.neo4j_client import check_neo4j_health
from infrastructure.qdrant_client import check_qdrant_health

logger = logging.getLogger(__name__)


def health_check_node(state: GraphState) -> dict:
    """
    LangGraph node: checks infrastructure and gates downstream routing.

    Returns only the fields that this node owns. LangGraph merges this dict
    into the existing GraphState — other fields are untouched.
    """
    logger.info("[health_check] running infrastructure checks")

    qdrant_ok = check_qdrant_health()
    neo4j_ok = check_neo4j_health()

    errors = list(state.errors)  # copy — never mutate state in place

    if not qdrant_ok:
        msg = "Qdrant unreachable — clause extraction and retrieval will fail"
        logger.warning("[health_check] %s", msg)
        errors.append(msg)

    if not neo4j_ok:
        msg = "Neo4j unreachable — entity mapping and contradiction detection will be skipped"
        logger.warning("[health_check] %s", msg)
        errors.append(msg)

    if qdrant_ok and neo4j_ok:
        logger.info("[health_check] all systems healthy")

    return {
        "qdrant_ready": qdrant_ok,
        "neo4j_ready": neo4j_ok,
        "status": "infrastructure_checked",
        "errors": errors,
    }
