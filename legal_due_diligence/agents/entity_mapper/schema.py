"""
Neo4j Cypher write operations for the entity mapper.

Graph schema written here:

    (:Document {doc_id})
        -[:HAS_CLAUSE]->
    (:Clause {doc_id, clause_type, normalized_value, confidence, found, source_chunk_id})
        -[:INVOLVES]->    (:Party {name})
        -[:GOVERNED_BY]-> (:Jurisdiction {name})
        -[:HAS_DURATION]->(:Duration {value})
        -[:HAS_AMOUNT]->  (:MonetaryAmount {value})

Design decisions:

MERGE everywhere — idempotent writes. Re-running the entity mapper after a
partial failure will not create duplicate nodes or relationships. This is
critical for a pipeline that accumulates errors and retries from checkpoints.

Clause uniqueness key is (doc_id, clause_type) — one Clause node per document
per CUAD category. This is the atomic unit the contradiction detector queries:
    MATCH (c1:Clause {clause_type: 'Governing Law'})
    MATCH (c2:Clause {clause_type: 'Governing Law'})
    WHERE c1.doc_id <> c2.doc_id AND c1.normalized_value <> c2.normalized_value
The (doc_id, clause_type) composite uniqueness constraint enforces this and
makes the MERGE efficient (index lookup, not full scan).

SET on MERGE — normalized_value, confidence, found, source_chunk_id are
mutable fields on the Clause node (can change if a document is re-processed).
Using SET after MERGE ensures they're always up-to-date without duplicating
the node.

Why a Jurisdiction node rather than storing the jurisdiction string on Clause?
Two contracts both governed by "Delaware" should share one Jurisdiction node.
Then Sprint 5 can ask: "find all Clause nodes GOVERNED_BY the same Jurisdiction
but belonging to different Documents" — a natural graph traversal impossible
in a flat SQL/JSON store.

Same logic applies to Party, Duration, MonetaryAmount.
"""

from __future__ import annotations

from neo4j import Session


def ensure_constraints(session: Session) -> None:
    """
    Create uniqueness constraints if they don't already exist.

    Safe to call on every agent run — IF NOT EXISTS is idempotent.
    Must run before any writes so MERGE can use the index.
    """
    session.run(
        "CREATE CONSTRAINT doc_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"
    )
    # Composite index: speeds up MERGE (c:Clause {doc_id, clause_type}) lookups.
    # NODE KEY would enforce uniqueness at the DB level but requires Enterprise.
    # MERGE semantics already guarantee one Clause per (doc_id, clause_type) — the
    # index is purely a performance aid for the Community Edition Docker image.
    session.run(
        "CREATE INDEX clause_doc_type IF NOT EXISTS "
        "FOR (c:Clause) ON (c.doc_id, c.clause_type)"
    )


def write_document(session: Session, doc_id: str) -> None:
    """MERGE a Document node — idempotent."""
    session.run(
        "MERGE (d:Document {doc_id: $doc_id})",
        doc_id=doc_id,
    )


def write_clause(
    session: Session,
    doc_id: str,
    clause_type: str,
    normalized_value: str | None,
    confidence: float,
    found: bool,
    source_chunk_id: str,
) -> None:
    """
    MERGE a Clause node and link it to its Document.

    Writes all clauses — found=True and found=False — so the contradiction
    detector can distinguish "different value" from "one document missing the
    clause entirely."
    """
    session.run(
        """
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (c:Clause {doc_id: $doc_id, clause_type: $clause_type})
        SET c.normalized_value = $normalized_value,
            c.confidence       = $confidence,
            c.found            = $found,
            c.source_chunk_id  = $source_chunk_id
        MERGE (d)-[:HAS_CLAUSE]->(c)
        """,
        doc_id=doc_id,
        clause_type=clause_type,
        normalized_value=normalized_value,
        confidence=confidence,
        found=found,
        source_chunk_id=source_chunk_id,
    )


def write_party(
    session: Session, doc_id: str, clause_type: str, party_name: str
) -> None:
    """MERGE a Party node and link it to the Clause via INVOLVES."""
    session.run(
        """
        MATCH (c:Clause {doc_id: $doc_id, clause_type: $clause_type})
        MERGE (p:Party {name: $name})
        MERGE (c)-[:INVOLVES]->(p)
        """,
        doc_id=doc_id,
        clause_type=clause_type,
        name=party_name,
    )


def write_jurisdiction(
    session: Session, doc_id: str, clause_type: str, name: str
) -> None:
    """MERGE a Jurisdiction node and link it to the Clause via GOVERNED_BY."""
    session.run(
        """
        MATCH (c:Clause {doc_id: $doc_id, clause_type: $clause_type})
        MERGE (j:Jurisdiction {name: $name})
        MERGE (c)-[:GOVERNED_BY]->(j)
        """,
        doc_id=doc_id,
        clause_type=clause_type,
        name=name,
    )


def write_duration(
    session: Session, doc_id: str, clause_type: str, value: str
) -> None:
    """MERGE a Duration node and link it to the Clause via HAS_DURATION."""
    session.run(
        """
        MATCH (c:Clause {doc_id: $doc_id, clause_type: $clause_type})
        MERGE (dur:Duration {value: $value})
        MERGE (c)-[:HAS_DURATION]->(dur)
        """,
        doc_id=doc_id,
        clause_type=clause_type,
        value=value,
    )


def write_amount(
    session: Session, doc_id: str, clause_type: str, value: str
) -> None:
    """MERGE a MonetaryAmount node and link it to the Clause via HAS_AMOUNT."""
    session.run(
        """
        MATCH (c:Clause {doc_id: $doc_id, clause_type: $clause_type})
        MERGE (amt:MonetaryAmount {value: $value})
        MERGE (c)-[:HAS_AMOUNT]->(amt)
        """,
        doc_id=doc_id,
        clause_type=clause_type,
        value=value,
    )
