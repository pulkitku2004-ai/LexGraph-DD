# LexGraph-DD — Master Context

**Last updated:** 2026-04-20
**Status:** PROJECT CLOSED. Sprint 23 complete. Sprint 24 (JOB_STORE SQLite persistence) was in scope but project was called off after final extraction quality evaluation confirmed a model capability ceiling — not addressable without a better extraction model or contrastive fine-tuning. All retrieval and extraction improvement avenues exhausted. System is production-ready for a prototype; limitations documented below and in README.

**Post-close cleanup (2026-04-20):** Low-risk code quality pass — dead field removal (`chunk_overlap`, `reranker_model`), `litellm.suppress_debug_info` centralized to `core/config.py`, duplicate JSON fence-stripping extracted to `core/utils.strip_json_fence()`, `Optional[X]` modernized to `X | None` throughout.

---

## What This System Does

Ingests 1–50 PDF/DOCX/TXT contracts → 6 LangGraph agents → structured due diligence brief:
- Clause extraction across 41 CUAD categories
- Risk scoring (rules + LLM reasoning)
- Entity mapping → Neo4j knowledge graph
- **Cross-document contradiction detection via shared Neo4j graph** (key differentiator — no open-source implementation does this)
- Interactive Q&A with page-level citations

---

## Environment

| Item | Value |
|---|---|
| Machine | MacBook Air M4, 16GB RAM |
| Python | 3.12.1 (pyenv) |
| Venv | `/path/to/legal_dd/.venv` |
| Project root | `/path/to/legal_dd/` |
| Package root | `/path/to/legal_dd/legal_due_diligence/` |

```bash
source /path/to/legal_dd/.venv/bin/activate   # every session
/path/to/legal_dd/.venv/bin/python -m uvicorn legal_due_diligence.api.main:app --port 8000
/path/to/legal_dd/.venv/bin/python -m streamlit run legal_due_diligence/ui/app.py
```

`pyrightconfig.json` at root sets `extraPaths: ["legal_due_diligence"]`.

---

## Tech Stack

| Component | Choice | Note |
|---|---|---|
| Orchestration | LangGraph 1.1.6 | State machine, conditional routing |
| Vector store | Qdrant (Docker) | use `query_points()` not `search()` |
| Knowledge graph | Neo4j 5 (Docker) | Bolt :7687, browser :7474 |
| LLM routing | LiteLLM 1.83.4 | Provider portability + fallback chain |
| LLM extraction | groq/llama-3.1-8b-instant → groq/llama-4-scout-17b → ollama/mistral-nemo | Fallback chain |
| LLM reasoning | ollama/mistral-nemo (large batch) / OpenRouter free (≤25 docs) | |
| Embeddings | BAAI/bge-m3 | 1024-dim dense + learned SPLADE sparse; no BM25 pickle |
| ML | PyTorch 2.11.0, MPS active | bge-m3 fp16 on MPS; sparse_linear on CPU |
| PDF/DOCX | pymupdf 1.27.2.2, python-docx 1.2.0 | |
| API | FastAPI 0.128.0 | |
| UI | Streamlit | |

**Infrastructure:**
```bash
docker compose up -d          # start Qdrant + Neo4j
docker compose down -v        # wipe all data (re-index required after)
python run_sprint1.py         # re-index after wipe (no pickle — sparse in Qdrant)
```

---

## Sprint Plan

| Sprint | Goal | Status |
|---|---|---|
| 0–6 | Core pipeline: scaffold → ingestion → all 6 agents → report + Q&A | ✅ DONE |
| 7 | CUAD evals + model search: legal-bert(9%) → bge-base+prefix(15%) | ✅ DONE |
| 8 | bge-m3 hybrid dense+sparse: R@3 15%→42% | ✅ DONE |
| 9 | FastAPI: POST/GET/DELETE /jobs + POST /jobs/{id}/qa | ✅ DONE |
| 10 | Streamlit UI: upload→running→done, Report+Q&A tabs | ✅ DONE |
| 11 | HyDE: disabled — hurts R@3 52.1%→40.8% | ✅ DONE |
| 12 | Async extraction: 50 docs ~10 min → ~3–10s | ✅ DONE |
| 13 | Multi-query for hard categories (CUAD_ALT_QUERIES) | ✅ DONE |
| 14 | e2e eval (e2e_eval.py) + SYSTEM_PROMPT rewrite + extraction hints | ✅ DONE |
| 15 | Parent-child chunking v1: 128-child/512-parent; F1 mean → 0.444 | ✅ DONE |
| 16 | Parent-child v2: 256-child/2048-parent, contiguous parents, parent_id dedup, doc-order delivery | ✅ DONE |
| 17 | Retrieval ceiling: HyDE (−3.9pp), reranker (−9.5pp), MMR (N/A), CUAD def analysis | ✅ DONE |
| 18 | CUAD definition-based query enrichment for bottom-tier categories | ✅ DONE |
| 19 | Embedding cache: materialize chunk embeddings to disk — ~50 min eval → ~2 min repeat runs | ✅ DONE |
| 20 | Anchor word injection: official CUAD definition phrases → R@3 67.5%→68.0% (+0.5pp) | ✅ DONE |
| 21 | Hybrid alpha tuning (sparse-heavy RRF sweep α=0.7/0.8/0.9) — rejected: all worse than equal weight | ✅ DONE |
| 22 | Case-insensitive enrichment/alt-query lookup fix + Affiliate License-Licensor alt queries → R@3 68.0%→68.3% | ✅ DONE |
| 23 | Pipeline quality: e2e baseline ✅ + `trim_clause_text()` ✅ + risk scorer category prompts ✅ + contradiction detector fixes ✅ + extraction hints ✅ + auth ✅ | ✅ DONE |
| 24 | Verbatim prompt test (Cond. F1 0.421→0.373, −4.8pp, rejected) → extraction quality ceiling confirmed → project closed | ✅ DONE (closed) |

---

## Current Benchmark (canonical)

```
Eval: chenghao/cuad_qa, 1244 rows, enrich-queries + multi-query
R@1:  39.6%    R@3:  68.3%   (Sprint 22 — full 1244 rows, Apr 17, definitive)
```

**Full progression:**
| Config | R@3 | Rows |
|---|---|---|
| legal-bert baseline | 9% | ~100 |
| bge-base + prefix | 15% | ~100 |
| bge-m3, ck=20, enriched | 42% → **52.1%** | 100 → 1244 |
| + multi-query + Sprint 16 chunking | **61.7%** | 360 |
| Sprint 18 query enrichment | **61.4%** | 360 |
| Sprint 19 — full dataset | **67.5%** | 1244 |
| Sprint 20 — anchor word injection | 68.0% | 1244 |
| **Sprint 22 — enrichment fix + Affiliate License-Licensor alt** | **68.3%** | **1244** |

**Rejected retrieval improvements (benchmarked):**
- HyDE: −3.9pp (llama-3.1-8b generates boilerplate, shifts embeddings away from contract language)
- Cross-encoder reranker (bge-reranker-v2-m3): −9.5pp (MS-MARCO trained, domain mismatch on legal text)
- MMR: N/A (parent_id dedup already prevents clumping)
- candidate_k=50 vs ck=20: +1pp, within noise — ck=20 confirmed
- Hybrid alpha tuning (Sprint 21): α=0.7 → 67.1% (−0.9pp), α=0.8 → 67.3% (−0.7pp), α=0.9 → 66.7% (−1.3pp). bge-m3's dense and sparse vectors are jointly trained for equal-weight RRF — tilting toward sparse overweights exact-term matching and loses the semantic clustering dense provides. Equal weight (α=0.5) is optimal.
- Multi-query expansion to new categories (Sprint 22 attempt): License Grant, Non-Transferable License, Irrevocable/Perpetual License, Non-Disparagement, ROFR/ROFO/ROFN, Termination for Convenience all caused regressions (−3 to −18pp). Change of Control alt queries caused −23.1pp (2nd alt query "assign agreement to affiliate" pulled Anti-Assignment chunks). Multi-query hurts when the added queries overlap vocabulary with adjacent categories.
- Change of Control alt queries: removed permanently. The primary enriched query already covers the space; alt queries caused Anti-Assignment contamination.
- Chunk size variants: 128/1024 and 512/2048 both benchmarked — 256/2048 (current) is optimal.
- RRF_K tuning: not attempted — marginal expected gain not worth the effort at this ceiling.

**Retrieval ceiling declared at R@3 = 68.3%.** Further gains require contrastive fine-tuning (paid GPU) or a fundamentally different retrieval architecture.

**Rejected extraction quality improvements (benchmarked):**
- `trim_clause_text()` (Sprint 23): negligible effect on Token F1 (±0.003, within noise). Kept for cleaner risk/contradiction input.
- Extraction hints for 19 categories (Sprint 23): partial run confounded by model mixing — not cleanly measurable.
- Verbatim copy instruction in SYSTEM_PROMPT (Sprint 24): Cond. F1 0.421 → 0.373 (−4.8pp). llama-3.1-8b treats the instruction as additional constraint and degrades. **This is a model capability ceiling**, not a prompt engineering problem.

**Extraction ceiling declared at Cond. F1 ≈ 0.42 (llama-3.1-8b-instant).** Improving this requires either a larger model (e.g. llama-3.1-70b) or a CUAD fine-tuned extraction model.

**Per-category breakdown (Sprint 20, full 1244-row, Apr 17 — authoritative):**

| Category | R@3 | n | vs Sprint 19 |
|---|---|---|---|
| Most Favored Nation | 0% | 3 | — |
| Non-Compete | 35% | 23 | +9pp ✅ |
| Joint IP Ownership | 29% | 7 | — |
| Unlimited/All-You-Can-Eat License | 33% | 3 | — |
| Warranty Duration | 40% | 10 | — |
| Revenue/Profit Sharing | 46% | 35 | +6pp ✅ |
| Non-Disparagement | 43% | 7 | — |
| Competitive Restriction Exception | 50% | 16 | +6pp ✅ |
| Post-Termination Services | 52% | 29 | +7pp ✅ |
| Covenant Not To Sue | 46% | 24 | — |
| ROFR/ROFO/ROFN | 53% | 17 | +6pp ✅ |
| Parties | 97% | 102 | — |
| Document Name | 87% | 102 | — |
| Agreement Date | 87% | 93 | — |
| Anti-Assignment | 81% | 72 | — |

Sprint 18 query enrichment changes (6 categories): net neutral on overall R@3 (61.7% → 61.7%). Per-category impact cannot be determined without a clean Sprint 17 baseline — that run also used the stale file. This is the new authoritative baseline.

**E2E metrics (200-row):**

| Sprint | Found Rate | Token F1 mean | Cond. F1 | Substring Match | Notes |
|---|---|---|---|---|---|
| 15 (baseline) | 78.1% | 0.444 | — | — | 128-child / 512-parent, SYSTEM_PROMPT rewrite |
| 23 (pre-trim) | 88.5% | 0.386 | — | 28.1% | 256-child / 2048-parent; Groq primary throughout |
| 23 (trimmed, cached A/B) | 81.8% | 0.375 | — | 24.0% | trim=yes; same 192 LLM responses as no-trim |
| 23 (no-trim, cached A/B) | 82.3% | 0.378 | — | 24.5% | trim=no; clean baseline |
| 24 (model-mixed baseline) | 83.3% | 0.350 | 0.421 | 22.4% | Groq TPD hit mid-run → ollama fallback. Not authoritative. |
| 24 (verbatim prompt, model-mixed) | 86.5% | 0.322 | 0.373 | 18.8% | Verbatim instruction added. Same confound. **Rejected −4.8pp Cond. F1.** |

**Sprint 23 trimmer finding:** `trim_clause_text()` has negligible effect on token F1: +0.003 when disabled (within noise). Kept — logically correct for cleaner risk/contradiction input even if CUAD token F1 can't measure it.

**Sprint 24 extraction ceiling finding:** Adding an explicit verbatim copy instruction to SYSTEM_PROMPT ("Copy the text character-for-character exactly as it appears. Do not paraphrase, summarize, or reword.") caused Cond. F1 to drop 0.421 → 0.373 (−4.8pp). The instruction confused the model — llama-3.1-8b's paraphrasing is a model capability ceiling, not a prompt engineering problem. Fixing it requires either a larger extraction model or a fine-tuned one. **Prompt reverted. Project closed.**

**Note on eval confound:** Both Sprint 24 runs are model-mixed — Groq TPD exhausted (~500k tokens/day on scout-17b) and ollama/mistral-nemo handled the remainder. Clean authoritative numbers require waiting for TPD reset and using the Groq LLM cache. The cache (259 entries, `eval/cache/llm_responses_llama_3.1_8b_instant.pkl`) is restored to its pre-test state.

---

## Folder Structure

```
legal_dd/
├── .env                    ← API keys (never commit)
├── docker-compose.yml
├── pyrightconfig.json
├── run_sprint{0,1,3,4,5,6,7,9}.py   ← smoke tests
├── analyze_categories.py   ← per-category R@3 from eval JSON; accepts file arg (fixed Sprint 18 — was hardcoded to wrong file)
├── samples/contract_{a,b}.txt        ← deliberate contradictions for testing
├── eval/
│   ├── cuad_eval.py        ← Recall@K harness (Sprint 19: embedding cache, dead code removed)
│   ├── e2e_eval.py         ← end-to-end extraction eval (Token F1 + found rate)
│   ├── sample_ids.json
│   ├── cache/              ← Sprint 19: chunk embedding cache (pkl, keyed by model+chunk params)
│   │   └── embeddings_bge_m3_p2048_c256_o51.pkl   ← 5,199 entries, ~50 min → ~2 min repeat
│   └── results/
└── legal_due_diligence/
    ├── core/config.py, models.py, state.py, utils.py
    │         └── utils.py  ← strip_json_fence() shared by clause_extractor, risk_scorer, report_qa
    ├── infrastructure/qdrant_client.py, neo4j_client.py, health_check.py
    ├── ingestion/loader.py, chunker.py, embedder.py, indexer.py
    ├── agents/
    │   ├── orchestrator/graph.py
    │   ├── clause_extractor/retriever.py, prompts.py, agent.py
    │   ├── risk_scorer/rules.py, agent.py
    │   ├── entity_mapper/extractor.py, schema.py, agent.py
    │   ├── contradiction_detector/cypher_queries.py, agent.py
    │   └── report_qa/formatter.py, qa.py, agent.py
    ├── api/main.py, schemas.py, runner.py
    └── ui/app.py
```

---

## Core Data Models (`core/models.py`, `core/state.py`)

```python
# core/models.py — Pydantic BaseModel, Python 3.12 union syntax throughout

class DocumentRecord(BaseModel):
    doc_id: str
    file_path: str
    processed: bool = False
    page_count: int | None = None        # set after successful load

class ExtractedClause(BaseModel):
    document_id: str
    clause_type: str                     # one of CUAD's 41 categories
    found: bool                          # False = missing clause = risk signal
    clause_text: str | None = None       # verbatim extracted text
    normalized_value: str | None = None  # e.g. "Delaware", "30 days", "$1M"
    confidence: float                    # 0.0–1.0
    source_chunk_id: str                 # child_chunk_id → Qdrant → page → PDF

class RiskFlag(BaseModel):
    document_id: str
    clause_type: str
    risk_level: Literal["high", "medium", "low"]
    reason: str
    is_missing_clause: bool              # distinguishes missing vs. bad content
    source_clause_id: str | None = None  # ExtractedClause.source_chunk_id → page citation

class Contradiction(BaseModel):
    clause_type: str
    document_id_a: str
    document_id_b: str
    value_a: str
    value_b: str
    explanation: str                     # LLM-generated plain-English risk explanation
    risk_level: Literal["high", "medium", "low"] = "medium"

# core/state.py — the single object that flows through the LangGraph machine
class GraphState(BaseModel):
    job_id: str
    status: str = "pending"
    created_at: datetime
    documents: list[DocumentRecord] = []
    extracted_clauses: list[ExtractedClause] = []
    risk_flags: list[RiskFlag] = []
    contradictions: list[Contradiction] = []
    neo4j_ready: bool = False
    qdrant_ready: bool = False
    graph_built: bool = False            # True after entity_mapper writes to Neo4j
    final_report: str | None = None
    errors: list[str] = []
```

**Critical:** LangGraph nodes return `dict` (not GraphState). Return only changed fields.

**Utility:** `core/utils.py` — `strip_json_fence(text: str) -> str` strips ` ```...``` ` wrappers from LLM responses before `json.loads()`. Used by clause_extractor, risk_scorer, and report_qa parsers.

---

## LangGraph Topology

```
START → health_check
  ├─ qdrant_ready=False ──────────────────────────► report_qa → END
  └─ qdrant_ready=True ─► clause_extractor → risk_scorer
                              ├─ neo4j_ready=False ──► contradiction_detector
                              └─ neo4j_ready=True ──► entity_mapper
                                                            └─► contradiction_detector
                                                                      └─► report_qa → END
```

---

## System DFD (Level 1 — Production)

```mermaid
graph TD
    %% ── External entities ──────────────────────────────────────────────────────
    CLIENT(["`**API Client**
    browser / curl / Streamlit`"])
    LLM(["`**LLM Providers**
    Groq · Ollama · OpenRouter`"])

    %% ── Data stores ────────────────────────────────────────────────────────────
    FS[("`**Temp FS**
    /tmp/legal_dd_{job}/
    raw PDF·DOCX·TXT`")]

    QDRANT[("`**Qdrant**
    collection: legal_clauses
    dense 1024-dim + sparse SPLADE`")]

    NEO4J[("`**Neo4j**
    Document·Clause·Party
    Jurisdiction·Duration·Amount`")]

    JOBSTORE[("`**JOB_STORE**
    in-memory dict
    job_id → JobRecord`")]

    %% ── Processes ──────────────────────────────────────────────────────────────
    P0["`**0 · Ingestion**
    loader → chunker → embedder → indexer`"]

    P1["`**1 · Health Check**
    ping Qdrant + Neo4j`"]

    P2["`**2 · Clause Extractor**
    41 cats × N docs · async · Semaphore(10)`"]

    P3["`**3 · Risk Scorer**
    rules pass + LLM pass (8 cats)`"]

    P4["`**4 · Entity Mapper**
    MERGE nodes + relationships`"]

    P5["`**5 · Contradiction Detector**
    Cypher value + absence queries`"]

    P6["`**6 · Report + Q&A**
    deterministic formatter + LLM narrative`"]

    %% ── Data flows ─────────────────────────────────────────────────────────────

    CLIENT -->|"multipart file bytes"| FS
    FS -->|"file path + doc_id"| P0
    P0 -->|"EmbeddedChunk[]\n{child_vec[1024], sparse_vec,\nparent_text, page_number, doc_id}"| QDRANT

    P1 -->|"HTTP GET /healthz"| QDRANT
    P1 -->|"Bolt PING"| NEO4J
    P1 -->|"qdrant_ready: bool\nneo4j_ready: bool"| P2

    QDRANT -->|"RetrievedChunk[]\n{text, parent_text,\nrrf_score, page_number,\nchunk_id, doc_id}"| P2
    P2 <-->|"extraction prompt →\n← JSON {found, clause_text,\nnormalized_value, confidence}"| LLM
    P2 -->|"ExtractedClause[]\n41 × N objects"| P3

    P3 <-->|"nuance prompt (8 cats) →\n← JSON {flag, risk_level, reason}"| LLM
    P3 -->|"RiskFlag[]"| P4

    P4 -->|"MERGE Document, Clause,\nParty, Jurisdiction,\nDuration, MonetaryAmount"| NEO4J
    P4 -->|"graph_built: bool"| P5

    NEO4J -->|"value_conflict rows\n{clause_type, doc_id_a/b,\nvalue_a/b}"| P5
    NEO4J -->|"absence_conflict rows\n{clause_type, found_a,\nvalue_a/b}"| P5
    P5 <-->|"explanation prompt →\n← JSON {risk_level, explanation}"| LLM
    P5 -->|"Contradiction[]"| P6

    P6 <-->|"narrative prompt →\n← JSON {executive_summary,\nrecommended_actions}"| LLM
    P6 -->|"final_report: str (Markdown)"| JOBSTORE

    JOBSTORE -->|"JobRecord\n{status, report, errors}"| CLIENT
    CLIENT -->|"POST /jobs/{id}/qa\n{question: str}"| P6
    QDRANT -->|"RetrievedChunk[] (cross-doc\nscoped to job doc_ids)"| P6
    P6 -->|"answer + citations[]"| CLIENT
```

> **Scope:** Production path only. No caching — every ingestion embeds fresh via GPU.
> **Neo4j bypass:** If `neo4j_ready=False`, P4 is skipped; P5 returns `[]` immediately; report notes the skip.
> **Qdrant bypass:** If `qdrant_ready=False`, P2/P3/P4/P5 all skipped; P6 generates report from empty state.

---

### Eval Harness DFD (with Embedding Cache)

```mermaid
graph TD
    CUAD(["`**CUAD Dataset**
    chenghao/cuad_qa
    1,244 test rows`"])

    CACHE[("`**Embedding Cache**
    eval/cache/embeddings_{slug}.pkl
    key: MD5(child_text)`")]

    QDRANT2[("`**Qdrant**
    ephemeral per-eval run
    scoped to eval doc_ids`")]

    E0["`**Load + Chunk**
    same Chunker config as prod
    256-child / 2048-parent`"]
    E1["`**Cache Check**
    slug = model + chunk params
    auto-invalidates on config change`"]
    E2["`**BGE-M3 GPU Pass**
    MPS fp16 · batch=24
    ~8s/row cold`"]
    E3["`**Embed Questions**
    primary + alt queries (15 hard cats)
    batched upfront — not per-row`"]
    E4["`**Recall@K Loop**
    enrich-queries + multi-query RRF
    hit@1, hit@3 per row`"]
    E5["`**Results JSON**
    eval/results/{name}.json`"]
    E6["`**analyze_categories.py**
    per-category R@3
    worst-first sort`"]

    CUAD --> E0
    E0 -->|"child chunks"| E1
    E1 -->|"cache miss\nuncached texts"| E2
    E2 -->|"dense+sparse tensors"| CACHE
    E1 -->|"cache hit\n~1s warm load"| QDRANT2
    CACHE -->|"tensors from disk"| QDRANT2
    QDRANT2 --> E3
    E3 --> E4
    E4 -->|"per-row results"| E5
    E5 --> E6
```

> **Speedup:** Cold run ≈ 50 min. Warm run (cache hit) ≈ 2 min (30×). Question embeddings not cached — always re-embeds (~22s for 360 questions).

---

## Ingestion Pipeline

**Chunking (Sprint 16 — current):**
- Parents: 2048 tokens, contiguous (no overlap). Each gets UUID `parent_id` + `parent_chunk_index`.
- Children: 256 tokens, 51-token overlap within each parent. Children never cross parent boundaries.
- Embedded: child text only. Parent text stored in Qdrant payload.
- `_merge_headings()`: short paragraphs (≤80 chars) merged forward before chunking.
- Everything at token-ID level — no decode→re-encode drift.

**Embedder:** bge-m3, MPS fp16, CLS-pool L2-norm dense (1024-dim) + sparse_linear head (SPLADE). sparse_linear on CPU (MPS overhead dominates for Linear(1024,1)). Batch=24.

**Qdrant point:** `id=child_chunk_id`, vectors `{"dense": float[1024], "sparse": SparseVector}`, payload `{text, parent_text, parent_id, parent_chunk_index, doc_id, page_number, ...}`

---

## Clause Extractor (Sprint 16)

**Retrieval per (doc × category):**
1. Dense query: Qdrant cosine top-20 (doc_id filter)
2. Sparse query: Qdrant SPLADE top-20 (doc_id filter)
3. RRF fusion k=60 → ranked children
4. **Stage 1 (score order):** dedup by parent_id → top-k unique parents
5. **Stage 2 (doc order):** re-sort by parent_chunk_index ascending
6. LLM receives parent_text (2048 tokens) per unique parent

**Multi-query (`CUAD_ALT_QUERIES`):** 15 hard categories fire 2–3 alt queries, sum RRF scores (consensus boost), same two-stage dedup. Confirmed +6.1pp R@3. Alt query embeddings pre-batched upfront in `embed_questions()` (Sprint 18 fix — was per-row GPU call in eval loop).

**LLM:** groq/llama-3.1-8b → groq/llama-4-scout → ollama/mistral-nemo. temperature=0, max_tokens=300. JSON output → parse → ExtractedClause. Any failure → found=False, confidence=0.0.

**Post-extraction trimming (`trim_clause_text()` — Sprint 23):** Applied at parse time in `_parse_response()`. Strips section headers that LLM includes when 2048-token parents give it too much context. Logic:
1. Strip leading section numbers: `12.1 ` / `12.1. ` / `3. ` (two-arm regex to avoid stripping "3 months")
2. Strip leading `Article 12.` / `SECTION 3.1` keywords
3. **Only if** step 1 or 2 matched: strip ALL-CAPS headers (`INDEMNIFICATION. `) and title-case headers (`Change of Control. `) — guard prevents stripping clause subjects like `LICENSEE shall not...` or `THIS AGREEMENT is made...`
4. Strip trailing orphan numbers (`13.` at end)
Fallback: return original text if trimming produces empty string.

**Async:** asyncio.gather per doc, Semaphore(10) global cap. 50 docs: ~3–10s wall time.

---

## Risk Scorer

**Rules pass** (O(1), no LLM): MISSING_CLAUSE_RISK dict (HIGH: Limitation of Liability, Governing Law, etc. MEDIUM: 8 more. LOW: suppressed). PRESENCE_FLAGS: Uncapped Liability found=HIGH, Joint IP/Liquidated Damages/Irrevocable License found=MEDIUM. confidence<0.4 on medium/high categories → MEDIUM flag.

**LLM pass** (8 categories only): Limitation of Liability, Liability Cap, Indemnification, IP Ownership Assignment, Non-Compete, Governing Law, Termination for Convenience, Confidentiality.

---

## Entity Mapper

Reads extracted_clauses → MERGE to Neo4j:
```
(:Document)-[:HAS_CLAUSE]->(:Clause)-[:INVOLVES]->(:Party)
                                                  -[:GOVERNED_BY]->(:Jurisdiction)
                                                  -[:HAS_DURATION]->(:Duration)
                                                  -[:HAS_AMOUNT]->(:MonetaryAmount)
```
Sets `graph_built=True` on state. Fully idempotent (MERGE).

---

## Contradiction Detector

Queries Neo4j (scoped to job's `$doc_ids`):
1. `find_value_conflicts()` — same clause_type, both found, different normalized_value
2. `find_absence_conflicts()` — same clause_type, one found/one missing

LLM explanation per conflict (template fallback on failure). Returns `list[Contradiction]`.

---

## Report + Q&A

**Report:** Deterministic formatter builds risk table + contradiction table (no LLM). One LLM call → JSON `{executive_summary, recommended_actions}`. `_template_narrative()` fallback if LLM fails — `final_report` always populated.

**Q&A** (`POST /jobs/{id}/qa`): hybrid retrieval per doc → merge by RRF → LLM answer + page-level citations. Citations trace back via `source_chunk_id → Qdrant → page_number`.

---

## Agent Contracts

Structured I/O contracts for every node in the LangGraph state machine. All nodes receive the full `GraphState` and return a `dict` of changed fields only.

---

### 1. Health Check (`infrastructure/health_check.py`)

| | |
|---|---|
| **Reads** | `state.errors` (copied, not mutated) |
| **Writes** | `qdrant_ready: bool`, `neo4j_ready: bool`, `status: "infrastructure_checked"`, `errors: list[str]` |
| **Contract downstream** | Both booleans read exclusively by the orchestrator routing functions — never by downstream agents. `qdrant_ready` gates `route_after_health`; `neo4j_ready` gates `route_after_risk`. |
| **Transition from** | `START` (always first node) |
| **Transition to** | `clause_extractor` if `qdrant_ready=True`; `report_qa` if `qdrant_ready=False` |
| **If failure** | HTTP ping to Qdrant raises → `qdrant_ready=False`, error appended. Bolt ping to Neo4j raises → `neo4j_ready=False`, error appended. Never raises — always returns a complete dict so the graph can route gracefully. |

---

### 2. Clause Extractor (`agents/clause_extractor/agent.py`)

| | |
|---|---|
| **Reads** | `state.documents` (for `doc_id` list + `processed` flag), `state.extracted_clauses` (to extend), `state.qdrant_ready` (guard) |
| **Writes** | `extracted_clauses: list[ExtractedClause]`, `status: "clauses_extracted"`, `errors: list[str]` |
| **Contract downstream** | `ExtractedClause[]` — one object per `(doc_id × clause_type)` pair. Risk Scorer reads `found`, `confidence`, `clause_text`, `clause_type`, `document_id`. Entity Mapper reads all fields. Format: `ExtractedClause(document_id, clause_type, found: bool, clause_text: str\|None, normalized_value: str\|None, confidence: float, source_chunk_id: str)` |
| **Transition from** | `health_check` (only when `qdrant_ready=True`) |
| **Transition to** | `risk_scorer` (unconditional) |
| **If failure** | Per-`(doc, category)` exception → `_missing_clause()` returned (found=False, confidence=0.0) — conservative, over-flags risk rather than silently dropping. LLM total failure across all providers → all 41 categories for that doc return found=False. `qdrant_ready=False` guard at node entry → returns immediately with error. One bad doc never aborts others. |

---

### 3. Risk Scorer (`agents/risk_scorer/agent.py`)

| | |
|---|---|
| **Reads** | `state.extracted_clauses` |
| **Writes** | `risk_flags: list[RiskFlag]`, `status: "risks_scored"` |
| **Contract downstream** | `RiskFlag[]` — Report+Q&A reads `risk_level`, `reason`, `is_missing_clause`, `clause_type`, `document_id`, `source_clause_id`. Format: `RiskFlag(document_id, clause_type, risk_level: "high"\|"medium"\|"low", reason: str, is_missing_clause: bool, source_clause_id: str\|None)` |
| **Transition from** | `clause_extractor` (unconditional) |
| **Transition to** | `entity_mapper` if `neo4j_ready=True`; `contradiction_detector` if `neo4j_ready=False` |
| **If failure** | Per-clause exception → logged, that clause skipped entirely (no flag emitted — conservative). LLM call failure for nuanced categories → LLM flag skipped; deterministic rule flags from Pass 1 still stand. Low-confidence clauses (`confidence < 0.4`) skip LLM pass — unreliable clause_text would produce unreliable risk assessment. |

---

### 4. Entity Mapper (`agents/entity_mapper/agent.py`)

| | |
|---|---|
| **Reads** | `state.extracted_clauses`, `state.neo4j_ready`, `state.errors` |
| **Writes** | `graph_built: bool`, `status: "graph_built"`, `errors: list[str]`; **side-effect:** Neo4j MERGE of Document, Clause, Party, Jurisdiction, Duration, MonetaryAmount nodes |
| **Contract downstream** | Neo4j graph consumed by Contradiction Detector via two Cypher queries. Schema: `(:Document {doc_id})-[:HAS_CLAUSE]->(:Clause {doc_id, clause_type, normalized_value, confidence, found, source_chunk_id})-[:INVOLVES]->(:Party {name})` etc. All writes are `MERGE` (idempotent — safe to re-run). Writes both `found=True` AND `found=False` clauses — Contradiction Detector needs the absence records. |
| **Transition from** | `risk_scorer` (only when `neo4j_ready=True`) |
| **Transition to** | `contradiction_detector` (unconditional) |
| **If failure** | Per-clause write exception → logged, appended to errors, loop continues (partial graph is still useful — Contradiction Detector works on whatever nodes exist). Neo4j session failure (hard stop) → `graph_built=False`, errors updated; Contradiction Detector checks `graph_built` and returns `[]` immediately. |

---

### 5. Contradiction Detector (`agents/contradiction_detector/agent.py`)

| | |
|---|---|
| **Reads** | `state.graph_built` (guard), `state.documents` (for `doc_ids` scope), `state.errors`; queries Neo4j directly via two Cypher queries |
| **Writes** | `contradictions: list[Contradiction]`, `status: "contradictions_detected"`, `errors: list[str]` |
| **Contract downstream** | `Contradiction[]` — Report+Q&A builds contradiction table. Format: `Contradiction(clause_type, document_id_a, document_id_b, value_a: str, value_b: str, explanation: str, risk_level: "high"\|"medium"\|"low")`. All Cypher queries are scoped to `$doc_ids` — graph accumulates across jobs but detection never leaks cross-job. |
| **Transition from** | `entity_mapper` (neo4j path) **or** `risk_scorer` (shortcut when `neo4j_ready=False`) |
| **Transition to** | `report_qa` (unconditional) |
| **If failure** | `graph_built=False` → returns `{"contradictions": [], "status": ...}` immediately (no Neo4j call attempted). Neo4j query exception → logged, appended to errors, returns empty list. LLM explanation failure per conflict → `_template_narrative`-style template fallback used; `Contradiction` object always emitted with explanation string. Value normalization (`_normalize_for_comparison`) collapses false positives ("thirty days" vs "30 days") before any LLM call. |

---

### 6. Report + Q&A (`agents/report_qa/agent.py`, `agents/report_qa/qa.py`)

| | |
|---|---|
| **Reads** | Full `GraphState`: `extracted_clauses`, `risk_flags`, `contradictions`, `documents`, `errors`, `job_id` |
| **Writes** | `final_report: str`, `status: "complete"`, `errors: list[str]`; **side-effect (runner):** `JOB_STORE[job_id].report` set by `api/runner.py` from `result["final_report"]` |
| **Contract downstream** | `final_report` is a Markdown string — returned verbatim via `GET /jobs/{id}` in `JobResponse.report`. Q&A endpoint (`POST /jobs/{id}/qa`) returns `{answer: str, citations: [{doc_id, page_number, chunk_id, excerpt}], chunks_retrieved: int}`. |
| **Transition from** | `contradiction_detector` (normal path) **or** `health_check` (fast-fail when `qdrant_ready=False`) |
| **Transition to** | `END` |
| **If failure** | Deterministic formatter (`formatter.py`) cannot fail — pure Python on state data. LLM narrative call fails → `_template_narrative(state)` generates a factually accurate generic summary from state counts. Outer `try/except` in `report_qa_node` calls template fallback as last resort. `final_report` is **always** a non-None string — the pipeline never returns without a report. |

---

## FastAPI

| Endpoint | Behaviour |
|---|---|
| POST /jobs | Multipart upload, BackgroundTasks pipeline, returns job_id (202) |
| GET /jobs/{id} | Poll: pending/running/done/error; report when done |
| POST /jobs/{id}/qa | Q&A on completed job |
| DELETE /jobs/{id} | Wipes Qdrant points + Neo4j nodes + tempfiles (async cleanup) |

`JOB_STORE`: in-memory dict. BackgroundTasks (not Celery). `python -m uvicorn` via venv Python (not pyenv shim).

---

## LLM Provider Strategy

| Role | Model | Limit |
|---|---|---|
| Extraction primary | groq/llama-3.1-8b-instant | 6000 TPM |
| Extraction fallback 1 | groq/llama-4-scout-17b | separate bucket |
| Extraction fallback 2 | ollama/mistral-nemo | none |
| Reasoning ≤25 docs | openrouter/nvidia/nemotron-3-super-120b-a12b:free | ~200 req/day |
| Reasoning large batch | ollama/mistral-nemo | none |

`config.py` pushes keys to `os.environ` — LiteLLM reads from env directly. OpenRouter models require `:free` suffix. `deepseek-r1:free` NOT available on this account.

---

## Key Architectural Decisions (reference)

**Pipeline / LangGraph:**
- **LangGraph nodes return `dict`:** only changed fields — returning the full GraphState would overwrite fields already set by prior nodes
- **Health check as graph node:** routes around infra failures gracefully; partial results more useful than a stack trace
- **Conditional routing after health and risk:** two infra flags (`qdrant_ready`, `neo4j_ready`) gate the two expensive external dependencies independently
- **errors accumulate, never raise:** one bad PDF must not abort 49 others; pipeline always terminates at report_qa with whatever it has

**Retrieval:**
- **Full list replacement in state:** explicit dedup vs LangGraph's blind append reducer — agents return updated full lists
- **parent_id dedup Stage 1 (score order):** best child per parent selected by RRF score before passing to LLM
- **doc-order Stage 2 (parent_chunk_index):** LLM gets Article 2 before Article 10 — legal clauses cross-reference earlier sections
- **Equal-weight RRF (α=0.5):** bge-m3 dense and sparse vectors are jointly trained; tilting toward sparse overweights exact-match and loses semantic clustering (benchmarked — all alpha variants worse)
- **Multi-query only for 15 hard categories:** alt queries for easy categories contaminate adjacent ones (Change of Control alt caused −23pp Anti-Assignment regression)

**Graph / Contradiction:**
- **Cypher scoped to `$doc_ids`:** Neo4j graph accumulates across jobs — unscoped query returns cross-job contradictions
- **Entity Mapper writes found=False clauses too:** Contradiction Detector distinguishes value conflicts (both found, different values) from absence conflicts (one found, one missing)

**Report:**
- **Formatter-first:** deterministic risk table and contradiction table are built in pure Python before any LLM call — tables cannot be hallucinated or truncated
- **`final_report` always populated:** three-layer fallback — LLM narrative → `_template_narrative()` → outer except; report_qa_node never returns None

**LLM:**
- **temperature=0 everywhere except Q&A (0.1):** deterministic extraction = reproducible evals; slight randomness in Q&A for prose fluency
- **LiteLLM fallback chain handles rate limits:** no custom retry logic in agent code; Groq primary → Groq scout → Ollama local

**Code quality (post-close cleanup 2026-04-20):**
- **`litellm.suppress_debug_info = True` in `core/config.py`:** set once at import time, takes effect process-wide — not repeated per agent
- **`core/utils.strip_json_fence()`:** shared across clause_extractor, risk_scorer, report_qa; was a copy-pasted 3-liner in three separate parse functions
- **`X | None` over `Optional[X]`:** Python 3.10+ union syntax used consistently throughout codebase; `Optional` import removed from all modules
- **`chunk_overlap` and `reranker_model` removed from config:** both were unused — `chunk_overlap` explicitly documented as unused since Sprint 16; `reranker_model` benchmarked and rejected Sprint 17 (−9.5pp), never wired to the retriever

---

## Sprint 18 — Query Enrichment Changes

Updated in both `eval/cuad_eval.py` (`_CUAD_QUERY_ENRICHMENT`) and `agents/clause_extractor/prompts.py` (`CUAD_CATEGORIES` + `CUAD_ALT_QUERIES`):

| Category | Key additions | Result |
|---|---|---|
| Revenue/Profit Sharing | net receipts, gross revenue, net sales, proceeds | 20% R@3 (authoritative) |
| Non-Compete | restrictive covenant, competing business/products/services | 10% R@3 (authoritative) |
| Joint IP Ownership | co-invented, co-owned, jointly created, both parties | 29% R@3 (authoritative) |
| Change of Control | beneficial ownership, voting securities, controlling interest, majority shares | 40% R@3 (authoritative) |
| Covenant Not To Sue | release claims, discharge, waive right to bring action; not contest/challenge/attack validity | 40% R@3 (authoritative) |
| Most Favored Nation | minimized to: `most favored nation MFN no less favorable price terms any third party`; alt queries: "no less favorable than prices offered to any other customer/third party" | 0% — retrieval ceiling, not a query problem |

⚠ No clean Sprint 17 baseline exists — per-category delta vs pre-Sprint-18 is not measurable. Authoritative numbers above are from the Apr 17 360-row cache run.

**Eval bottleneck (pre-Sprint-19):** chunk embedding = ~8s/row → 360 rows ≈ 50 min.

---

## Sprint 19 — Embedding Cache

Cache location: `eval/cache/embeddings_{slug}.pkl`
Slug encodes: `{model}_{parent_chunk_size}_{child_chunk_size}_{child_overlap}` → auto-invalidates on config change.

Key: `MD5(child_text)` → value: `(dense_vector: list[float], sparse_vector: dict[int, float])`

Flow:
1. `_load_cache()` → load pkl (or empty dict if missing)
2. Diff against all chunks from eval rows → find uncached texts
3. Embed only uncached → update dict → `_save_cache()`
4. All rows reconstructed from cache: `EmbeddedChunk(chunk, vector, sparse_vector)`

Cold run (360 rows, 31,273 unique children): ~50 min → populates 5,199 entries.
Warm run: cache loaded in ~1s, GPU skipped, total eval ~2 min (30× speedup).

**Eval-only** — production ingestion pipeline unchanged. Alt query embeddings (question-side) are NOT cached; they re-embed each run (~22s for 360 questions).

---

## Known Issues

| Issue | Fix |
|---|---|
| `query_points()` rejects `NamedSparseVector` | Use `SparseVector(indices=..., values=...) + using="sparse"` |
| FlagEmbedding ≥1.2 incompatible with transformers 5.x | Implemented bge-m3 directly via AutoModel + hf_hub_download |
| uvicorn pyenv shim misses venv packages | Always use `python -m uvicorn` via venv Python |
| HuggingFace unauth rate limit warning | Set `HF_TOKEN` env var |
| bm25 pickle duplicate on re-run | Gone — sparse vectors in Qdrant, no pickle |
| Groq 6000 TPM per model | Fallback chain eliminates wait |
| `e2e_eval.py` stale API (Sprint 23 fix) | `embed_questions()` returns `(primary_dict, alt_dict)` tuple — was captured as single value, crashing on `query_embeddings[idx]` for idx > 1. Fixed: `query_embeddings, _ = embed_questions(...)`. Also removed stale `query_text=` and `reranker=` kwargs from `eval_retrieve()` / `eval_retrieve_multi()` calls, and removed `CrossEncoder` import and `--reranker` CLI arg. |

---

## Common Commands

```bash
# Start everything
docker compose up -d
source /path/to/legal_dd/.venv/bin/activate

# Smoke tests
python run_sprint1.py     # ingestion + retrieval
python run_sprint9.py     # full API lifecycle

# Eval
python eval/cuad_eval.py --n 400 --enrich-queries --multi-query
python eval/e2e_eval.py --n 200 --enrich-queries --multi-query
python analyze_categories.py eval/results/FILENAME.json

# Full reset
docker compose down -v && docker compose up -d && python run_sprint1.py
```

---

## Sample Contracts (deliberate contradictions)

| Clause | contract_a.txt | contract_b.txt |
|---|---|---|
| Governing Law | Delaware | New York |
| Liability Cap | 12 months fees | 6 months fees |
| Payment Terms | 30 days | 45 days |
| Confidentiality | 5 years | 3 years |
| Termination | 30 days notice | **missing** |


