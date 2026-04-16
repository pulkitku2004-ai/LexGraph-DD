"""
Clause extraction prompts — CUAD 41-category taxonomy.

Two things live here:
  1. CUAD_CATEGORIES — maps each category name to a retrieval query string
  2. build_extraction_prompt() — the LLM prompt that turns retrieved chunks
     into a structured ExtractedClause

Why separate retrieval queries from category names?
The category name "Governing Law" is a legal term of art. The retrieval
query "governing law jurisdiction choice of law state" is optimised for
BM25 + dense recall — it adds synonyms and related terms that appear in
contracts but not in the category name itself. These are different problems:
one is classification vocabulary, the other is retrieval vocabulary.

Why does the prompt ask for normalized_value separately from clause_text?
clause_text = the raw sentence(s) from the contract (verbatim, for citation)
normalized_value = the extracted fact (for contradiction detection)

Example for Governing Law:
  clause_text:    "This Agreement shall be governed by the laws of the State
                   of Delaware, without regard to conflict of law principles."
  normalized_value: "Delaware"

The contradiction detector compares normalized_values across documents —
"Delaware" vs "New York". It cannot meaningfully compare two 200-word
governing law paragraphs. Normalization is what makes contradiction
detection tractable.

Why JSON output from the LLM?
Structured output lets us parse the response deterministically. Free-text
responses require another LLM call to extract the fields, doubling cost.
We use a strict JSON schema with a known set of keys. If the model produces
invalid JSON, we catch it in the agent and emit found=False with low confidence.

Why confidence as a float?
The risk scorer uses confidence to weight its decisions. A found=True clause
with confidence=0.4 is treated differently from confidence=0.95. Low
confidence is itself a risk signal — it means the clause exists but is
ambiguous or poorly written.
"""

from __future__ import annotations

# ── CUAD 41-category taxonomy ─────────────────────────────────────────────────
# Format: "category_name": "retrieval query string"
# Retrieval queries are BM25+dense optimised — more verbose than category names.

CUAD_CATEGORIES: dict[str, str] = {
    # ── Parties & Dates ───────────────────────────────────────────────────
    "Document Name":               "agreement title name contract type",
    "Parties":                     "party parties between company corporation LLC",
    "Agreement Date":              "agreement date made entered into as of",
    "Effective Date":              "effective date commencement start date",
    "Expiration Date":             "expiration date end date termination date contract expires",
    "Renewal Term":                "renewal term automatically renews extension period",
    "Notice Period to Terminate Renewal": "notice period terminate renewal cancel non-renewal written notice",

    # ── Jurisdiction & Dispute ─────────────────────────────────────────────
    "Governing Law":               "governing law jurisdiction choice of law state",
    "Dispute Resolution":          "dispute resolution arbitration litigation venue court",

    # ── Restrictive Covenants ──────────────────────────────────────────────
    "Non-Compete":                 "non-compete non-competition restrictive covenant shall not compete competing business competing products competing services engage in competition",
    "Exclusivity":                 "exclusivity exclusive rights sole provider co-exclusive basis exclusive right to promote distribute sell",
    "No-Solicit of Customers":     "shall not solicit divert business customers clients accounts of the other party directly or indirectly",
    "No-Solicit of Employees":     "no solicit employees hire recruit personnel",
    "Non-Disparagement":           "non-disparagement shall not make disparaging statements derogatory defamatory comments refrain from speaking negatively about",
    "Covenant Not To Sue":         "covenant not to sue release claims waive right to bring action not assert legal claims discharge not contest challenge attack impair title trademark validity ownership",

    # ── Liability & Indemnification ────────────────────────────────────────
    "Limitation of Liability":     "limitation of liability indirect consequential damages excluded",
    "Liability Cap":               "liability cap maximum total aggregate liability shall not exceed",
    "Liquidated Damages":          "liquidated damages penalty breach fixed amount",
    "Uncapped Liability":          "uncapped liability unlimited liability gross negligence fraud",
    "Indemnification":             "indemnification indemnify hold harmless defend claims losses",

    # ── Intellectual Property ──────────────────────────────────────────────
    "IP Ownership Assignment":     "intellectual property ownership assignment work made for hire all right title interest vests assigns work product inventions",
    "Joint IP Ownership":          "joint IP ownership jointly owned co-owned each party owns co-invented jointly developed jointly created intellectual property shared ownership both parties",
    "License Grant":               "license grant licensed rights use software platform",
    "Non-Transferable License":    "non-transferable license cannot assign sublicense transfer",
    "Irrevocable or Perpetual License": "irrevocable perpetual license permanent rights",
    "Unlimited/All-You-Can-Eat License": "unlimited non-exclusive perpetual irrevocable royalty free worldwide license unrestricted use",
    "Source Code Escrow":          "source code escrow deposit release conditions",
    "IP Restriction":              "intellectual property restrictions permitted use limitations",

    # ── Warranties ─────────────────────────────────────────────────────────
    "Warranty Duration":           "warranty period duration months years guarantee defect rejection return expiration",
    "Product Warranty":            "product warranty performance fitness merchantability",

    # ── Financial ─────────────────────────────────────────────────────────
    "Payment Terms":               "payment terms invoice due date net days fees",
    "Revenue/Profit Sharing":      "revenue sharing profit sharing royalty percentage net receipts gross revenue net sales proceeds equal to per unit",
    "Price Restrictions":          "price restrictions pricing cap floor ceiling most favored nation",
    "Most Favored Nation":         "most favored nation MFN no less favorable price terms any third party",
    "Competitive Restriction Exception": "notwithstanding competitive restriction exception carve-out permitted competing business activities despite exclusivity non-compete",
    "Minimum Commitment":          "minimum commitment shall maintain at least minimum number sales representatives staff guaranteed amount purchase volume",
    "Volume Restriction":          "volume restriction maximum usage limit quota not to exceed maximum number up to limit per month year capacity ceiling threshold",

    # ── Operational ────────────────────────────────────────────────────────
    "Audit Rights":                "audit rights inspect records books financial",
    "Post-Termination Services":   "post-termination services step-in right authorized to maintain service continue perform obligations following termination wind-down transition",
    "Change of Control":           "change of control merger acquisition beneficial ownership voting securities controlling interest majority shares takeover consent assignment terminate",
    "Anti-Assignment":             "anti-assignment cannot assign transfer consent required",
    "Third Party Beneficiary":     "third party beneficiary rights benefits",

    # ── Confidentiality ────────────────────────────────────────────────────
    "Confidentiality":             "confidentiality confidential information non-disclosure NDA",
    "Confidentiality Survival":    "confidentiality survives termination years obligations continue",

    # ── Termination ────────────────────────────────────────────────────────
    "Termination for Convenience": "termination for convenience without cause notice days",
    "Termination for Cause":       "termination for cause material breach cure period",
}


# ── Multi-query retrieval: alternative queries for hard categories ─────────────
#
# For categories where a single query consistently misses relevant clauses,
# we fire additional queries and union the results (RRF scores summed).
# Chunks appearing in multiple query results get a boosted combined score.
#
# Why these specific categories?
# Derived from per-category Recall@3 on the full 1,244-row CUAD eval:
# categories below ~35% R@3 with n>10 have vocabulary mismatch between
# the category name / primary query and the actual contract language used.
#
# Alt queries are grounded in real ground-truth text from eval misses —
# they capture the surface forms that contracts actually use for each clause.
CUAD_ALT_QUERIES: dict[str, list[str]] = {
    "Covenant Not To Sue": [
        # Contracts rarely say "covenant not to sue" — they say:
        "not directly or indirectly attack challenge oppose contest acknowledge affirm rights",
        "not contest challenge dispute ownership validity registration intellectual property trademark",
    ],
    "IP Ownership Assignment": [
        # "shall become our property", "concept process improvement shall be property of"
        "shall become property of company may utilize disclose concept process improvement developed during",
        "acknowledges agrees work product invention improvement shall be owned by licensor franchisor all right title",
    ],
    "Minimum Commitment": [
        # "shall maintain at least N sales representatives", "shall ensure sufficient staff"
        "shall maintain at least minimum number sales representatives field force staff",
        "shall ensure sufficient experienced staff available production schedule minimum forecast",
    ],
    "Exclusivity": [
        # "co-exclusive basis", "exclusive retail music store sponsor"
        "exclusive retail sponsor exclusive right to promote detail distribute sell in territory",
        "co-exclusive basis solely with affiliates sole and exclusive right license supplier",
    ],
    "Revenue/Profit Sharing": [
        # "royalty equal to $X per Y", "share of all gross margins on transactions"
        "royalty equal to per unit dose vial payment fee earned during term",
        "share of gross margin gross revenue profit generated transactions advertising sponsorship",
    ],
    "Non-Compete": [
        # "shall not market distribute or sell a Competing Product"
        "shall not market distribute sell competing product in territory during term",
        "irrevocably undertakes not to compete conduct any other business commercial arrangement",
    ],
    "Change of Control": [
        # "assign in whole or in part to Affiliate", "upon any Change of Control of X"
        "terminate upon change of control merger acquisition takeover controlling interest ownership",
        "assign agreement in whole or in part to affiliate successor acquirer without consent",
    ],
    "Post-Termination Services": [
        # "step-in right", "perform obligations until earlier of", "maintain and service contracts"
        "step-in right grant access rights following termination authorized to maintain service",
        "shall perform obligations until earlier of continue service after expiration wind-down",
    ],
    "Volume Restriction": [
        # Diverse ground truths: social media posts, pipeline capacity, training courses
        "maximum number of posts shares per month social media limit up to N times",
        "maximum equivalent percent daily capacity crude oil pipeline capacity ceiling threshold",
    ],
    "Joint IP Ownership": [
        # "jointly developed", "each party shall own"
        "jointly created developed each party shall own co-inventors shared patent rights",
        "joint development agreement both parties own intellectual property jointly created work",
    ],
    "Most Favored Nation": [
        # CUAD: "prices shall be no less favorable than those at which seller sells to any other party"
        "prices shall be no less favorable than prices offered to any other customer distributor party",
        "if licenses to any third party on more favorable terms licensee entitled to same terms conditions",
    ],
    "Competitive Restriction Exception": [
        # "Notwithstanding the foregoing, may make available from parties other than [exclusive party]"
        "notwithstanding the foregoing may engage purchase from parties other than exclusive sponsor exception to restriction",
        "has no control influence over decisions third party arrangement carve out exception permitted notwithstanding",
    ],
    "Warranty Duration": [
        # "within two months prior or six months past expiration", "within seven days after receipt defective"
        "eligible return reimbursement months prior past expiration date defective product outdated reject",
        "inspect products receipt reject defective within days after receipt warranty claim return period",
    ],
    "No-Solicit of Customers": [
        # "except as otherwise expressly provided, agent shall not solicit"
        "shall not directly or indirectly solicit divert customers clients accounts business of the other party",
        "except as otherwise provided agent shall not solicit compete with customers clients of the other party",
    ],
    "Unlimited/All-You-Can-Eat License": [
        # "unlimited, non-exclusive, perpetual, irrevocable, royalty free, worldwide right and license"
        "unlimited non-exclusive perpetual irrevocable royalty free worldwide right license unrestricted use content",
        "all-you-can-eat unlimited seats users copies unrestricted enterprise wide license without limitation",
    ],
}


# ── Extraction prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal contract analysis assistant. Your job is to extract specific clause information from contract text and return structured JSON.

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no code blocks.
- Look for the SUBSTANCE of the clause, not just an explicitly labeled section. Contracts rarely title their clauses with the exact category name — look for the underlying right, obligation, or fact.
- Set "found" to true if the text contains ANY relevant information about the clause, even if it uses different terminology or does not explicitly label the clause by name.
- Only set "found" to false if the text contains absolutely no information relevant to this clause type.
- "clause_text" must be the verbatim sentence(s) from the contract that best express this clause.
- "normalized_value" must be a short extracted fact (e.g. "Delaware", "30 days", "$500,000"), not a full sentence.
- "confidence" must be between 0.0 and 1.0.
- If found is false, set clause_text and normalized_value to null."""


def build_extraction_prompt(
    clause_type: str,
    retrieved_chunks: list[str],
    doc_id: str,
) -> str:
    """
    Build the user-turn prompt for clause extraction.

    Args:
        clause_type:      One of CUAD_CATEGORIES keys.
        retrieved_chunks: Top-k chunk texts from hybrid retrieval.
        doc_id:           Document identifier (for context in the prompt).

    Returns:
        User message string to send to the LLM alongside SYSTEM_PROMPT.

    Why pass multiple chunks?
    The top-1 chunk might contain the clause but also contain surrounding
    noise. Passing top-3 gives the model enough context to find the clause
    even if it spans a chunk boundary, without overwhelming the context window.
    Three chunks × 512 tokens = ~1536 tokens of context, well within Haiku/
    Llama limits.
    """
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{text}" for i, text in enumerate(retrieved_chunks)
    )

    # normalized_value examples are clause-type specific — they guide the model
    # toward the right level of abstraction (a state name, not a paragraph)
    normalized_examples = _normalized_value_examples.get(
        clause_type, "the key extracted value (short, specific fact)"
    )

    hint = _extraction_hints.get(clause_type, "")
    hint_line = f"\nHint: {hint}" if hint else ""

    return f"""Extract the "{clause_type}" clause from the following contract text.{hint_line}

Document: {doc_id}

CONTRACT TEXT:
{chunks_text}

Return JSON with exactly these fields:
{{
  "found": true or false,
  "clause_text": "verbatim sentence(s) from the contract that constitute this clause, or null if not found",
  "normalized_value": "{normalized_examples}, or null if not found",
  "confidence": 0.0 to 1.0
}}"""


# Per-category extraction hints injected into the user prompt.
# Used for categories where the LLM systematically returns found=False
# even when the right chunk is retrieved — because the clause substance
# appears under different wording than the category name implies.
_extraction_hints: dict[str, str] = {
    "Agreement Date":
        "This is often just a date in the preamble, e.g. 'This Agreement is dated as of November 20, 2007' or 'made and entered into as of [date]'. It does not need to be a labeled clause section.",
    "Document Name":
        "This is the title of the agreement, typically at the top of the document, e.g. 'JOINT FILING AGREEMENT' or 'SOFTWARE LICENSE AGREEMENT'. Extract the title as found.",
    "Termination for Convenience":
        "Look for any right to terminate without cause, without giving a specific reason, or upon notice alone. Phrases like 'terminate at any time', 'terminate without cause', or 'terminate upon [N] days notice' all qualify — even if not labeled 'for convenience'.",
    "Post-Termination Services":
        "Look for any obligation or right that continues or takes effect AFTER the agreement ends — e.g. transition assistance, completing in-flight work, returning data, wind-down periods, or survival of specific obligations.",
    "Non-Transferable License":
        "Look for any restriction on sublicensing, assigning, or transferring the license. Phrases like 'may not sublicense', 'no right to grant sublicenses', or 'non-transferable' all qualify.",
    "Minimum Commitment":
        "Look for any guaranteed minimum — a minimum purchase, minimum royalty, minimum volume, or minimum revenue obligation. Phrases like 'guaranteed minimum', 'shall purchase at least', or 'minimum royalty' all qualify.",
    "Rofr/Rofo/Rofn":
        "Look for any right of first refusal, right of first offer, or right of first negotiation. Phrases like 'right to match', 'first right to purchase', or 'right of first offer' all qualify.",
    "Non-Disparagement":
        "Look for any obligation to refrain from making negative, derogatory, or disparaging statements about the other party. Does not need the word 'disparagement'.",
    "Covenant Not To Sue":
        "Look for any release of claims, waiver of right to sue, or covenant not to assert legal claims. Phrases like 'shall not challenge', 'release all claims', 'waives any right to bring action', or 'shall not contest' all qualify.",
    "Revenue/Profit Sharing":
        "Look for any payment based on a percentage of revenue, profit, sales, or gross margin — royalties, revenue shares, or profit participation all qualify.",
    "Volume Restriction":
        "Look for any cap or ceiling on usage, quantity, number of users, seats, copies, or transactions — e.g. 'not to exceed', 'up to N units', 'maximum of'.",
    "Change of Control":
        "Look for any clause triggered by a merger, acquisition, change of ownership, or transfer of control — e.g. termination rights, assignment rights, or consent requirements upon such events.",
    "IP Ownership Assignment":
        "Look for any transfer of ownership of intellectual property, work product, or inventions to the other party. Phrases like 'shall become our property', 'assigns all right title and interest', or 'work made for hire' all qualify.",
    "Joint IP Ownership":
        "Look for any clause where both parties share or jointly own intellectual property, inventions, or developments created together.",
    "Source Code Escrow":
        "Look for mentions of source code deposit, escrow agent, escrow agreement, or software repository held by a third party.",
}


# Clause-specific guidance for normalized_value extraction.
# These examples anchor the model to produce consistent, comparable values —
# critical for the contradiction detector which does string comparison.
_normalized_value_examples: dict[str, str] = {
    "Governing Law":               "state/jurisdiction name only, e.g. 'Delaware' or 'New York'",
    "Dispute Resolution":          "method and venue, e.g. 'AAA arbitration, Wilmington Delaware'",
    "Expiration Date":             "date or duration, e.g. '2026-01-01' or '2 years'",
    "Renewal Term":                "duration, e.g. '1 year' or '6 months'",
    "Notice Period to Terminate Renewal": "notice period, e.g. '30 days' or '90 days'",
    "Liability Cap":               "amount or formula, e.g. '$500,000' or '12 months fees'",
    "Payment Terms":               "due date, e.g. '30 days' or 'Net 45'",
    "Confidentiality Survival":    "duration after termination, e.g. '3 years' or '5 years'",
    "Termination for Convenience": "notice period, e.g. '30 days' or '60 days'",
    "Termination for Cause":       "cure period, e.g. '15 days' or '30 days'",
    "Non-Compete":                 "duration and scope, e.g. '2 years, same industry'",
    "No-Solicit of Employees":     "duration, e.g. '1 year' or '2 years'",
    "Warranty Duration":           "duration, e.g. '90 days' or '1 year'",
    "Minimum Commitment":          "amount or volume, e.g. '$100,000/year' or '1000 units'",
    "IP Ownership Assignment":     "who owns IP, e.g. 'Client owns all work product'",
    "License Grant":               "scope, e.g. 'non-exclusive, non-transferable, internal use'",
}
