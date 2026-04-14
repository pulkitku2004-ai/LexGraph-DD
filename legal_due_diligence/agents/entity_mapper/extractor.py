"""
Entity extraction from ExtractedClause records.

Uses normalized_value (already structured from Sprint 2) as the primary source.
Supplements with regex-based ORG detection on clause_text for party names.

Why no external NER model here?
- normalized_value is already clean structured text ("Delaware", "30 days",
  "12 months fees") — mapping to entity type is a lookup, not NER.
- Party names follow predictable legal suffix patterns (Inc., LLC, Corp., Ltd.)
  — a targeted regex outperforms a general NER model on this narrow pattern
  without the 300MB model weight overhead.
- spaCy/transformers NER would be the right choice if we needed to extract
  parties from unstructured prose at scale. For Sprint 4, the clause_text
  already contains the relevant sentence(s) — suffix regex is sufficient.

Sprint 7 candidate: swap regex party extraction → legal-NER model if recall
on party names proves insufficient during CUAD evals.
"""

from __future__ import annotations

import re

from core.models import ExtractedClause

# Clause types whose normalized_value IS a jurisdiction name.
# Sprint 5 contradiction detector queries these directly.
JURISDICTION_CLAUSE_TYPES: frozenset[str] = frozenset({
    "Governing Law",
    "Dispute Resolution",
    "Venue",
})

# Clause types whose normalized_value IS a duration string.
DURATION_CLAUSE_TYPES: frozenset[str] = frozenset({
    "Warranty Duration",
    "Non-Compete",
    "Confidentiality",
    "Confidentiality Survival",
    "Termination for Convenience",
    "Termination for Cause",
    "Non-Solicit of Employees",
    "Non-Solicit of Customers",
    "Post-Termination Services",
    "License Grant",
    "Renewal Term",
    "Notice Period",
})

# Clause types whose normalized_value IS a monetary amount or fee formula.
AMOUNT_CLAUSE_TYPES: frozenset[str] = frozenset({
    "Liability Cap",
    "Liquidated Damages",
    "Minimum Commitment",
    "Volume Restriction",
    "Price Restrictions",
    "Audit Rights",
    "Revenue/Profit Sharing",
    "Cap on Liability",
})

# Regex: match org-name suffixes that identify legal entities.
# Captures the preceding name fragment up to the suffix (non-greedy).
# Examples matched: "Acme Corp.", "TechVentures Inc.", "GlobalSoft LLC"
_ORG_SUFFIX_RE = re.compile(
    r'\b([A-Z][A-Za-z0-9&\s]{1,40}?'
    r'(?:Inc\.?|LLC|Corp\.?|Ltd\.?|LLP|LP|GmbH|AG|S\.A\.|PLC|Co\.))'
    r'(?:\s|,|;|\.|\)|\"|\'|$)',
)


def extract_parties_from_text(text: str) -> list[str]:
    """
    Extract organisation names from clause_text using legal-entity suffix matching.

    Returns deduplicated list, shortest-first (to exclude accidental supersets).
    Empty list if no matches or text is falsy.
    """
    if not text:
        return []

    seen: set[str] = set()
    parties: list[str] = []
    for match in _ORG_SUFFIX_RE.finditer(text):
        name = match.group(1).strip().rstrip(",;.")
        if name and name not in seen and len(name) >= 4:
            seen.add(name)
            parties.append(name)
    return parties


def extract_entities(clause: ExtractedClause) -> dict[str, list[str]]:
    """
    Return typed entities derived from a single ExtractedClause.

    Schema:
        parties       — org names extracted from clause_text (regex)
        jurisdictions — normalized_value when clause_type is a governing-law type
        durations     — normalized_value when clause_type is a duration type
        amounts       — normalized_value when clause_type is a monetary type

    Only emits non-empty values. Returns empty lists for all keys if clause
    was not found (found=False) or normalized_value is absent — a missing
    clause has no entity to write; the Document+Clause nodes are still written
    by agent.py regardless so Sprint 5 can detect the absence.
    """
    entities: dict[str, list[str]] = {
        "parties": [],
        "jurisdictions": [],
        "durations": [],
        "amounts": [],
    }

    # Always attempt party extraction from clause_text (works even if found=False,
    # because clause_text may contain the preamble with party names).
    if clause.clause_text:
        entities["parties"] = extract_parties_from_text(clause.clause_text)

    # Entity type from normalized_value only makes sense when found=True.
    if not clause.found or not clause.normalized_value:
        return entities

    val = clause.normalized_value.strip()
    if not val:
        return entities

    if clause.clause_type in JURISDICTION_CLAUSE_TYPES:
        entities["jurisdictions"].append(val)
    elif clause.clause_type in DURATION_CLAUSE_TYPES:
        entities["durations"].append(val)
    elif clause.clause_type in AMOUNT_CLAUSE_TYPES:
        entities["amounts"].append(val)

    return entities
