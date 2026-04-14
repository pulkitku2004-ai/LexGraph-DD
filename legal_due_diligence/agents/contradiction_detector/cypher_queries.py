"""
Cypher queries for the Contradiction Detector.

Two structural patterns are detected:

1. Value conflicts — both documents have the clause (found=True) but with
   different normalized_values. Example: contract-a Governing Law = "Delaware",
   contract-b Governing Law = "New York".

2. Absence conflicts — one document has the clause (found=True) and another
   doesn't (found=False). Example: contract-a has Termination for Convenience,
   contract-b does not.

Design decisions:

docA.doc_id < docB.doc_id — prevents returning the same pair twice as (A,B)
and (B,A). The less-than comparison is string-ordered (deterministic across
all Neo4j versions). Without this, 4 value conflicts become 8 rows.

toLower(trim(...)) comparison in VALUE_CONFLICT_QUERY — normalizes away
casing and whitespace so "Delaware" and "delaware" are not falsely reported
as a conflict. Extraction model output is not always consistently cased.

Both queries return doc_id_a, doc_id_b, clause_type so the Python layer
can build Contradiction objects without needing to re-query.

find_absence_conflicts returns found_a and found_b so the Python layer can
correctly assign "(clause absent)" to whichever side is missing, regardless
of which doc_id is alphabetically first.
"""

from __future__ import annotations

from neo4j import Session


def find_value_conflicts(session: Session, doc_ids: list[str]) -> list[dict]:
    """
    Find clause types where both documents have the clause but with different
    normalized_values, scoped to the provided document set.

    Scoping to doc_ids is required: the graph accumulates documents across
    jobs. Without a filter, queries from job B would return contradictions
    involving job A's documents, producing spurious cross-job results.

    Returns rows with: doc_id_a, doc_id_b, clause_type, value_a, value_b.
    """
    rows = session.run(
        """
        MATCH (docA:Document)-[:HAS_CLAUSE]->(a:Clause),
              (docB:Document)-[:HAS_CLAUSE]->(b:Clause)
        WHERE a.clause_type = b.clause_type
          AND docA.doc_id < docB.doc_id
          AND docA.doc_id IN $doc_ids
          AND docB.doc_id IN $doc_ids
          AND a.found = true
          AND b.found = true
          AND a.normalized_value IS NOT NULL
          AND b.normalized_value IS NOT NULL
          AND toLower(trim(a.normalized_value)) <> toLower(trim(b.normalized_value))
        RETURN docA.doc_id  AS doc_id_a,
               docB.doc_id  AS doc_id_b,
               a.clause_type AS clause_type,
               a.normalized_value AS value_a,
               b.normalized_value AS value_b
        ORDER BY a.clause_type
        """,
        doc_ids=doc_ids,
    ).data()
    return rows


def find_absence_conflicts(session: Session, doc_ids: list[str]) -> list[dict]:
    """
    Find clause types where one document has the clause and another doesn't,
    scoped to the provided document set.

    Returns rows with: doc_id_a, doc_id_b, clause_type, found_a, value_a,
    found_b, value_b.

    found_a / found_b are returned so the caller can correctly label which
    side is "(clause absent)" regardless of alphabetical doc_id ordering.
    """
    rows = session.run(
        """
        MATCH (docA:Document)-[:HAS_CLAUSE]->(a:Clause),
              (docB:Document)-[:HAS_CLAUSE]->(b:Clause)
        WHERE a.clause_type = b.clause_type
          AND docA.doc_id < docB.doc_id
          AND docA.doc_id IN $doc_ids
          AND docB.doc_id IN $doc_ids
          AND a.found <> b.found
        RETURN docA.doc_id   AS doc_id_a,
               docB.doc_id   AS doc_id_b,
               a.clause_type  AS clause_type,
               a.found        AS found_a,
               a.normalized_value AS value_a,
               b.found        AS found_b,
               b.normalized_value AS value_b
        ORDER BY a.clause_type
        """,
        doc_ids=doc_ids,
    ).data()
    return rows
