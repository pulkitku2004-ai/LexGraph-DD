"""
LangGraph state machine — the orchestrator.

Graph topology (Sprint 0, linear):

  START
    │
    ▼
  health_check          ← sets qdrant_ready, neo4j_ready
    │
    ▼ (route_after_health)
  clause_extractor      ← if qdrant_ready; else → report_qa (fast-fail)
    │
    ▼
  risk_scorer
    │
    ▼ (route_after_risk)
  entity_mapper         ← if neo4j_ready; else skip to contradiction_detector
    │
    ▼
  contradiction_detector
    │
    ▼
  report_qa
    │
    ▼
  END

Why conditional routing after health_check rather than failing the whole job?
Because a document review job should be partially useful even when one
infrastructure component is down. If Qdrant is unavailable, the graph routes
directly to report_qa which generates a brief explaining what was attempted
and what failed. That's more valuable to an operator than a stack trace.

Why not parallel edges from health_check to clause_extractor AND entity_mapper?
LangGraph supports parallel fan-out but it requires the parallel branches
to not write to the same state fields, or you need to define merge reducers.
For Sprint 0 the linear graph is correct. We'll evaluate parallelizing
clause_extractor + entity_mapper in Sprint 4 once we understand the
write patterns. Premature parallelism here is a source of subtle merge bugs.

Why add_conditional_edges instead of just add_edge?
add_edge is unconditional — the next node always runs. add_conditional_edges
lets us inspect state between nodes and route to different next nodes.
The routing function receives the GraphState and returns the name of the
next node as a string. This is the hook where all orchestration logic lives.

Note on node naming: LangGraph node names are strings used as dict keys in
add_conditional_edges's path_map. Keep them lowercase with underscores —
they appear in logs and checkpoint filenames.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.clause_extractor.agent import clause_extractor_node
from agents.contradiction_detector.agent import contradiction_detector_node
from agents.entity_mapper.agent import entity_mapper_node
from agents.report_qa.agent import report_qa_node
from agents.risk_scorer.agent import risk_scorer_node
from core.state import GraphState
from infrastructure.health_check import health_check_node

logger = logging.getLogger(__name__)

# ── Routing functions ──────────────────────────────────────────────────────────
# Each router receives the *current* GraphState (after the preceding node has
# run and its updates have been merged) and returns the name of the next node.


def route_after_health(state: GraphState) -> str:
    """
    If Qdrant is unreachable, clause extraction is impossible.
    Skip directly to report_qa which will document the failure.
    """
    if not state.qdrant_ready:
        logger.warning(
            "[orchestrator] Qdrant not ready — routing directly to report_qa"
        )
        return "report_qa"
    return "clause_extractor"


def route_after_risk(state: GraphState) -> str:
    """
    Entity mapping requires Neo4j. If it's not ready, skip entity_mapper
    and contradiction_detector — both depend on the graph.
    """
    if not state.neo4j_ready:
        logger.warning(
            "[orchestrator] Neo4j not ready — skipping entity_mapper and "
            "contradiction_detector"
        )
        return "contradiction_detector"
    return "entity_mapper"


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Constructs and compiles the LangGraph state machine.

    Returns the compiled graph (a Pregel instance). Callers invoke it with:
        app = build_graph()
        result = app.invoke({"job_id": "...", "documents": [...]})

    Why return the compiled graph rather than calling compile() at module level?
    Module-level compilation runs at import time. If any agent import fails,
    the entire application fails to start with a cryptic error. Returning from
    a function defers compilation and gives better error messages.
    """
    graph = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("health_check", health_check_node)
    graph.add_node("clause_extractor", clause_extractor_node)
    graph.add_node("risk_scorer", risk_scorer_node)
    graph.add_node("entity_mapper", entity_mapper_node)
    graph.add_node("contradiction_detector", contradiction_detector_node)
    graph.add_node("report_qa", report_qa_node)

    # ── Wire edges ────────────────────────────────────────────────────────
    graph.add_edge(START, "health_check")

    # Conditional: health_check → clause_extractor OR report_qa
    graph.add_conditional_edges(
        "health_check",
        route_after_health,
        {
            "clause_extractor": "clause_extractor",
            "report_qa": "report_qa",
        },
    )

    graph.add_edge("clause_extractor", "risk_scorer")

    # Conditional: risk_scorer → entity_mapper OR contradiction_detector
    graph.add_conditional_edges(
        "risk_scorer",
        route_after_risk,
        {
            "entity_mapper": "entity_mapper",
            "contradiction_detector": "contradiction_detector",
        },
    )

    graph.add_edge("entity_mapper", "contradiction_detector")
    graph.add_edge("contradiction_detector", "report_qa")
    graph.add_edge("report_qa", END)

    return graph.compile()
