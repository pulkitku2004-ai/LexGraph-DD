"""
Sprint 0 smoke test — run the full graph with dummy input and print state
at each transition.

Run from the repo root:
    cd /Users/dr_bolty/legal_dd
    python run_sprint0.py

What "success" looks like for Sprint 0:
- All 6 nodes execute in order (or the conditional path is taken correctly).
- State transitions are logged at each step.
- Final state shows final_report set to the stub string.
- Errors list shows infrastructure warnings if Qdrant/Neo4j are unreachable
  (expected in local dev without Docker).

Why is this a standalone script rather than a pytest test?
For Sprint 0 we want visible, human-readable output showing state at each
node. pytest suppresses stdout by default (-s shows it). A script also
makes it easy to step through in a debugger without pytest scaffolding.
Sprint 7 will convert this to a proper DeepEval test.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
# We're running from the repo root, so add legal_due_diligence to sys.path
# so that `from core.state import GraphState` resolves correctly.
# In production this is handled by the package install (pip install -e .).
sys.path.insert(0, str(Path(__file__).parent / "legal_due_diligence"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def serialize_state(state: dict) -> str:
    """JSON-serialize state for pretty printing, handling datetime objects."""
    def default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    return json.dumps(state, indent=2, default=default)


def main() -> None:
    from agents.orchestrator.graph import build_graph
    from core.models import DocumentRecord
    from core.state import GraphState

    logger.info("=" * 60)
    logger.info("SPRINT 0 — LangGraph state machine smoke test")
    logger.info("=" * 60)

    # ── Build dummy input state ───────────────────────────────────────────
    job_id = f"sprint0-{uuid.uuid4().hex[:8]}"
    dummy_docs = [
        DocumentRecord(
            doc_id="doc-001",
            file_path="./samples/contract_a.pdf",
            processed=False,
        ),
        DocumentRecord(
            doc_id="doc-002",
            file_path="./samples/contract_b.pdf",
            processed=False,
        ),
    ]

    initial_state = GraphState(
        job_id=job_id,
        documents=dummy_docs,
    )

    logger.info("Initial state:\n%s", serialize_state(initial_state.model_dump()))
    logger.info("-" * 60)

    # ── Compile and run graph ─────────────────────────────────────────────
    app = build_graph()

    # stream() yields (node_name, state_after_node) tuples so we can print
    # state at each transition. invoke() would only give us the final state.
    logger.info("Streaming graph execution...\n")

    final_state: dict | None = None
    for step in app.stream(initial_state.model_dump()):  # type: ignore[union-attr]
        # step is a dict: {node_name: state_dict_after_node}
        node_name = next(iter(step))
        node_output = step[node_name]
        logger.info("▶ Node completed: [%s]", node_name)
        logger.info("  Output fields: %s", list(node_output.keys()))
        final_state = node_output  # last step's full state is the final state

    logger.info("=" * 60)
    logger.info("FINAL STATE:")
    logger.info(serialize_state(final_state or {}))

    # ── Assertions ────────────────────────────────────────────────────────
    assert final_state is not None, "Graph produced no output"
    assert final_state.get("status") == "complete", (
        f"Expected status='complete', got '{final_state.get('status')}'"
    )
    assert final_state.get("final_report") is not None, "final_report is None"

    logger.info("=" * 60)
    logger.info("Sprint 0 PASSED — graph ran end-to-end")
    logger.info("final_report: %s", final_state.get("final_report"))
    if final_state.get("errors"):
        logger.info(
            "Infrastructure warnings (expected if Docker is not running):\n  %s",
            "\n  ".join(final_state["errors"]),
        )


if __name__ == "__main__":
    main()
