# LexGraph-DD — Master Context

**Last updated:** 2026-05-24
**Status:** ACTIVE — Sprint 26 (ASTR-O full integration) complete. Project was closed after Sprint 24 confirmed a llama-3.1-8b extraction ceiling (Cond. F1 ≈ 0.42). Reopened 2026-05-02: Sprint 25 added ASTR-O observability and switched extraction to `gpt-4o-mini` (Cond. F1 0.617, +19.6pp). Sprint 26 unified all LLM roles to gpt-4o-mini and ran a 30-contract CUAD integration test against ASTR-O (13/15 spans SAFE, 86.7%). Sprint 27 planned (2026-05-24): log-centric architecture upgrades — LangGraph SqliteSaver checkpoint, SQLite event log for JOB_STORE, content-hash dedup, internal event bus, SSE streaming. Sprint 28 planned (2026-05-31): Hexagonal + Event-Sourced architecture refactor — `core/ports.py` ABCs, `adapters/` directory, `JsonlEventLog`, domain events, crash recovery via log replay.

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
| LLM extraction | gpt-4o-mini (primary) → ollama/mistral-nemo | Sprint 26: Groq removed; all roles unified to gpt-4o-mini |
| LLM reasoning | gpt-4o-mini | Sprint 26: unified with extraction; Ollama offline fallback only |
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
| 25 | ASTR-O compatibility: `retrieval_metadata` per retrieval call, extraction LLM → gpt-4o-mini, minimal OTel, HTTP test runner → `process_lexgraph_span()` | ✅ DONE |
| 26 | ASTR-O full integration: all LLM roles → gpt-4o-mini (Groq removed), verbatim-only Q&A prompt, 30-contract CUAD integration test → 13/15 SAFE (86.7%) | ✅ DONE |
| 27 | Log-centric upgrades: LangGraph SqliteSaver checkpoint, SQLite event log for JOB_STORE, content-hash dedup, internal event bus, SSE streaming | 🔜 PLANNED |
| 28 | Hexagonal + Event-Sourced refactor: `core/ports.py` (IEventLog, IVectorDB, IKnowledgeGraph), `adapters/` directory, `JsonlEventLog`, domain events, crash recovery via log replay | 🔜 PLANNED |

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

**Extraction ceiling broken (Sprint 25):** Switching to gpt-4o-mini as the extraction primary yields Cond. F1 0.617 (+19.6pp), Token F1 0.540, Substring Match 42.7%, Found Rate 87.5% — clean 192-row run, no fallback, fresh cache (`eval/cache/llm_responses_gpt_4o_mini.pkl`). This confirms the ceiling was model-specific to llama-3.1-8b, not a retrieval or prompt problem.

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
| **25 (gpt-4o-mini, clean, 192 rows)** | **87.5%** | **0.540** | **0.617** | **42.7%** | OpenAI primary, no fallback, clean cache. **Breaks llama ceiling by +19.6pp Cond. F1.** |

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
├── astro_req.md            ← Sprint 25 ASTR-O change log + retrieval_metadata shapes + live test results
├── test_runner_for_astr_o.py  ← HTTP job lifecycle → ASTR-O span dict → process_lexgraph_span()
├── test_retrieval_metadata.py ← live Qdrant test for retrieve_with_metadata() shape
├── eval/
│   ├── cuad_eval.py        ← Recall@K harness (Sprint 19: embedding cache, dead code removed)
│   ├── e2e_eval.py         ← end-to-end extraction eval (Token F1 + found rate)
│   ├── sample_ids.json
│   ├── cache/              ← chunk embedding cache + LLM response cache (keyed by model slug)
│   │   ├── embeddings_bge_m3_p2048_c256_o51.pkl   ← 5,199 entries, ~50 min → ~2 min repeat
│   │   ├── llm_responses_llama_3.1_8b_instant.pkl ← 259 entries (Groq runs)
│   │   └── llm_responses_gpt_4o_mini.pkl           ← Sprint 25 clean OpenAI run
│   └── results/
└── legal_due_diligence/
    ├── core/config.py, models.py, state.py, utils.py
    │         └── utils.py  ← strip_json_fence() shared by clause_extractor, risk_scorer, report_qa
    ├── infrastructure/qdrant_client.py, neo4j_client.py, health_check.py, observability.py
    │         └── observability.py  ← Sprint 25: OTel TracerProvider; no-op if OTEL_ENDPOINT unset
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

## System DFD — Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INGESTION  (runs once per job; results persisted to Qdrant)            │
└─────────────────────────────────────────────────────────────────────────┘

 Raw files  (PDF / DOCX / TXT)
   │
   │  loader.py  (PyMuPDF page-by-page / python-docx paragraphs)
   ▼
 LoadedDocument
   { doc_id: str,  file_path: str,  total_pages: int,
     pages: list[PagedText { page_number: int,  text: str }] }
   │
   │  chunker.py  _merge_headings() → _parent_child_chunks()
   │  bge-m3 tokenizer — token-ID level slicing, no roundtrip drift
   │  Parents: 2048 tokens, contiguous (no inter-parent overlap)
   │    └─ Children: 256 tokens, 51-token overlap within each parent
   ▼
 list[Chunk]  (one entry per child chunk)
   { chunk_id: UUID,               ← Qdrant point ID
     doc_id: str,
     page_number: int,
     text: str,                    ← 256-token child — embedded for retrieval
     parent_text: str,             ← 2048-token parent — passed to LLM
     parent_id: UUID,              ← dedup key in retriever Stage 1
     parent_chunk_index: int,      ← doc-order re-sort key in Stage 2
     token_count: int }
   │
   │  embedder.py  (BAAI/bge-m3, MPS fp16, batch=24)
   │  dense:  CLS-pool → L2-norm → float[1024]
   │  sparse: sparse_linear head (CPU) → SPLADE weights
   │  child text only — parent_text stored in payload, not embedded
   ▼
 list[EmbeddedChunk]
   { chunk_id: UUID,
     vector: float[1024],            ← dense cosine
     sparse_vector: dict[int,float]  ← sparse SPLADE }
   │
   │  indexer.py  (batches of 100, idempotent on chunk_id)
   ▼
 Qdrant  collection: legal_clauses
   point { id: chunk_id,
           vectors: { "dense": float[1024],  "sparse": SparseVector },
           payload: { text, parent_text, parent_id, parent_chunk_index,
                      doc_id, file_path, page_number, token_count } }
```

---

## System DFD — LangGraph Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LANGGRAPH PIPELINE  (per job; single GraphState flows node to node)    │
└─────────────────────────────────────────────────────────────────────────┘

 GraphState  { job_id: str,  documents: list[DocumentRecord],
               errors: list[str],  status: str,  ... }
   │
   │  health_check  (infrastructure/health_check.py)
   │  HTTP ping → Qdrant     Bolt ping → Neo4j
   ▼
 + qdrant_ready: bool
 + neo4j_ready: bool
   │
   ├─ qdrant_ready=False ──────────────────────────────────────────────┐
   │                                                                    │
   │  clause_extractor  (agents/clause_extractor/agent.py)             │
   │  per (doc × category): bge-m3 query → Qdrant hybrid RRF top-20   │
   │  Stage 1: score-order parent_id dedup (best child per parent)     │
   │  Stage 2: doc-order re-sort by parent_chunk_index                 │
   │  15 hard categories: 2–3 alt queries, RRF scores summed           │
   │  LLM: gpt-4o-mini → ollama/mistral-nemo (fallback)               │
   │  asyncio.gather per doc × category;  Semaphore(10) global cap     │
   ▼                                                                    │
 + extracted_clauses: list[ExtractedClause]                            │
     { document_id, clause_type,  found: bool,                         │
       clause_text: str|None,  normalized_value: str|None,             │
       confidence: float,  source_chunk_id: str }   ← 41 × N objects  │
   │                                                                    │
   │  risk_scorer  (agents/risk_scorer/agent.py)                       │
   │  Pass 1: deterministic rules — missing clauses + presence flags   │
   │  Pass 2: LLM reasoning for 8 nuanced categories only              │
   ▼                                                                    │
 + risk_flags: list[RiskFlag]                                          │
     { document_id, clause_type,                                        │
       risk_level: "high"|"medium"|"low",                               │
       reason: str,  is_missing_clause: bool,                          │
       source_clause_id: str|None }                                    │
   │                                                                    │
   ├─ neo4j_ready=False ─────────────────────────────────┐             │
   │                                                     │             │
   │  entity_mapper  (agents/entity_mapper/agent.py)     │             │
   │  MERGE Document + Clause nodes for all clauses      │             │
   │  extracts Party, Jurisdiction, Duration,            │             │
   │  MonetaryAmount from normalized_value               │             │
   │  fully idempotent (MERGE, not CREATE)               │             │
   ▼                                                     │             │
 + graph_built: True  (Neo4j graph populated)            │             │
   │                                                     │             │
   │  contradiction_detector  ◄──────────────────────────┘             │
   │  (agents/contradiction_detector/agent.py)                         │
   │  Cypher 1: value conflicts — same clause_type, different value    │
   │  Cypher 2: absence conflicts — present in A, missing in B         │
   │  both queries scoped to $doc_ids (no cross-job leakage)           │
   │  LLM: plain-English risk explanation per conflict found           │
   ▼                                                                    │
 + contradictions: list[Contradiction]                                 │
     { clause_type,  document_id_a,  document_id_b,                    │
       value_a: str,  value_b: str,                                    │
       explanation: str,  risk_level: "high"|"medium"|"low" }          │
   │                                                                    │
   │  report_qa  ◄──────────────────────────────────────────────────────┘
   │  (agents/report_qa/agent.py)
   │  Step 1: deterministic formatter — risk table + contradiction table
   │          (pure Python, no LLM; tables cannot be hallucinated)
   │  Step 2: one LLM call → { executive_summary, recommended_actions }
   │          _template_narrative() fallback if LLM fails
   │  Step 3: assemble_report() slots narrative into fixed Markdown template
   ▼
 + final_report: str  (Markdown brief — always populated, never None)
   │
   ▼  END
 runner.py:  JOB_STORE[job_id].report = final_report
             record.status = JobStatus.done
```

---

### Eval Harness DFD

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EVAL HARNESS  (eval-only; embedding cache not used in production)      │
└─────────────────────────────────────────────────────────────────────────┘

 CUAD test rows  (chenghao/cuad_qa · 1,244 rows · fixed sample_ids.json)
   │
   │  load + chunk  (same Chunker config as production)
   │  256-child / 2048-parent
   ▼
 list[Chunk]
   │
   │  cache check  eval/cache/embeddings_{slug}.pkl
   │  slug encodes model + chunk_size + child_size + overlap
   │  key: MD5(child_text)   auto-invalidates on config change
   │
   ├─ cache miss ──────────────────────────────────────────────────────┐
   │                                                                    │
   │  BGE-M3 GPU pass  (MPS fp16 · batch=24 · ~8s/row)                │
   ▼                                                                    │
 tensors saved to cache  ◄──────────────────────────────────────────────┘
   │
   │  cache hit: warm load ~1s  (GPU skipped entirely)
   ▼
 Qdrant  ephemeral upsert  (scoped to eval doc_ids)
   │
   │  embed_questions()
   │  primary queries + alt queries for 15 hard categories
   │  batched upfront — not re-embedded per row
   ▼
 Recall@K eval loop  (enrich-queries + multi-query RRF)
   { hit_at_1: bool,  hit_at_3: bool }  per row
   │
   ▼
 eval/results/{name}.json  →  analyze_categories.py
                               per-category R@3 · worst-first sort
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

**LLM (Sprint 26):** gpt-4o-mini (primary) → ollama/mistral-nemo (fallback). Groq removed in Sprint 26 — all roles unified to gpt-4o-mini for consistent output quality. temperature=0, max_tokens=300. JSON output → parse → ExtractedClause. Any failure → found=False, confidence=0.0. LiteLLM reads all provider keys from `os.environ` (set by config.py) — no `api_key=` kwarg passed to `litellm.completion()`.

**Retrieval metadata (Sprint 25):** `_extract_category_async` now calls `retrieve_with_metadata()` / `retrieve_multi_with_metadata()` instead of the plain variants. Assembles `retrieval_metadata` dict and attaches to OTel span as `span.set_attribute("retrieval_metadata", json.dumps(...))`. Original `retrieve()` / `retrieve_multi()` signatures unchanged — eval harness unaffected.

**New retriever internals (Sprint 25):**
- `RetrievedChunk` gets `dense_score: float | None` and `sparse_score: float | None` fields
- `_retrieve_fused(query, doc_id, candidate_k)` — private; shared by both retrieve() and retrieve_with_metadata()
- `_retrieve_multi_fused(queries, doc_id, candidate_k)` — private; shared similarly
- `_build_ranking_metadata(fused)` — serializes pre-truncation list: chunk_id, rank, dense_score, sparse_score, rrf_score, dense_rank, sparse_rank, reason_for_rank
- `retrieve_with_metadata(query, doc_id, top_k, candidate_k) → (chunks, all_ranked)`
- `retrieve_multi_with_metadata(queries, doc_id, top_k, candidate_k) → (chunks, all_ranked)`

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

**Sprint 25 Q&A additions:** `_retrieve_across_docs()` now calls `retrieve_with_metadata()` per doc and tags each ranked chunk with `doc_id`. `answer_question()` returns `retrieval_metadata` and `enriched_chunks` (list of `{chunk_id, doc_id, page_number, text}` — the parent texts actually sent to the LLM). `QAResponse` schema exposes both as optional fields. These feed directly into the ASTR-O span dict built by `test_runner_for_astr_o.py`.

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
| POST /jobs/{id}/qa | Q&A on completed job; response now includes `retrieval_metadata` and `enriched_chunks` (Sprint 25) |
| DELETE /jobs/{id} | Wipes Qdrant points + Neo4j nodes + tempfiles (async cleanup) |

`JOB_STORE`: in-memory dict. BackgroundTasks (not Celery). `python -m uvicorn` via venv Python (not pyenv shim).

---

## LLM Provider Strategy

| Role | Model | Limit |
|---|---|---|
| All roles (extraction + reasoning + Q&A) | gpt-4o-mini | OpenAI pay-per-token; ~300-token extraction outputs → low cost |
| Offline fallback | ollama/mistral-nemo | none — local, no rate limit, lower quality |

Sprint 26: Groq removed from all roles. ASTR-O groundedness requires verbatim output; smaller models (Groq llama-3.1-8b, mistral-nemo) paraphrase and cause failures. `config.py` pushes `OPENAI_API_KEY` to `os.environ` — LiteLLM reads from env directly.

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
- **LiteLLM fallback chain handles outages:** no custom retry logic in agent code; gpt-4o-mini primary → Ollama local (Sprint 26: Groq removed)
- **No `api_key=` kwarg to `litellm.completion()`:** config.py pushes all provider keys to `os.environ` at startup; passing a Groq key when the primary model is gpt-4o-mini fails authentication

**ASTR-O / Observability (Sprint 25):**
- **`infrastructure/observability.py`:** Configures OTel `TracerProvider` at import time. If `OTEL_ENDPOINT` set in env → `BatchSpanProcessor` + `OTLPSpanExporter` (HTTP). If not set → default no-op tracer (zero overhead, no code path changes needed).
- **`retrieval_metadata` on every span:** clause_extractor attaches full pre-truncation ranked list as `span.set_attribute("retrieval_metadata", json.dumps(...))`. Shape: `{query, alt_queries, retrieval_method, retrieved_chunk_ids, all_ranked_chunks, retrieval_timestamp}`. Each ranked chunk carries `chunk_id, rank, dense_score, sparse_score, rrf_score, dense_rank, sparse_rank, reason_for_rank`.
- **ASTR-O span dict:** `test_runner_for_astr_o.py` assembles `{span_id, trace_id, retrieval_metadata, enriched_chunks, llm_response}` from the HTTP QA response and passes to `LexGraphToASTRO.process_lexgraph_span()`. Graceful degradation: runs span validation-only if `ASTR_O_PATH` not set.
- **`retrieve()` / `retrieve_multi()` signatures preserved:** eval harness and qa.py call these unchanged. ASTR-O variants are additive (`retrieve_with_metadata`, `retrieve_multi_with_metadata`).

**Code quality (post-close cleanup 2026-04-20):**
- **`litellm.suppress_debug_info = True` in `core/config.py`:** set once at import time, takes effect process-wide — not repeated per agent
- **`core/utils.strip_json_fence()`:** shared across clause_extractor, risk_scorer, report_qa; was a copy-pasted 3-liner in three separate parse functions
- **`X | None` over `Optional[X]`:** Python 3.10+ union syntax used consistently throughout codebase; `Optional` import removed from all modules
- **`chunk_overlap` and `reranker_model` removed from config:** both were unused — `chunk_overlap` explicitly documented as unused since Sprint 16; `reranker_model` benchmarked and rejected Sprint 17 (−9.5pp), never wired to the retriever

---

## Sprint 27 — Planned: Log-Centric Architecture Upgrades

**Analysis date:** 2026-05-24. After studying Jay Kreps' "The Log" (LinkedIn Engineering), five structural gaps were identified in LexGraph's architecture that map directly to log theory anti-patterns. All five have concrete, low-to-medium-effort fixes. No code has been written yet — this section captures the full theory and implementation plan so it can be executed in a future session.

---

### The Log — Theory Summary

The log is an append-only, totally ordered sequence of records ordered by time. Each record gets a unique sequential log entry number — that number **is** the clock. The log is the authoritative source of truth; every table or index is a derived projection of that history into a useful data structure.

**Core principles:**

- **Log → Table:** apply changes in order → current state (credits/debits → account balance; events → JobRecord)
- **Table → Log (CDC):** record every mutation as a changelog → replication, audit trail, time-travel
- **State machine replication:** two identical deterministic processes given the same inputs in the same order produce the same output and end in the same state. Distributed consistency = a consistent log feeding them the same inputs in order
- **Replica as single number:** `cursor_position + log = entire state of the replica`. A replica can be fully described by the max log entry number it has processed
- **N×M → N+M:** Without a central log: N sources × M destinations = N×M point-to-point connections. With a log: N+M connections total. Adding a new consumer touches only its connection to the log — not every producer
- **Log as buffer:** decouples producers from consumers; consumers can fail or restart without slowing the rest of the processing graph
- **Stream processing = log processing:** a stream processor reads from and writes to logs; produces output at user-controlled frequency without requiring a static snapshot
- **Log compaction:** remove records whose primary key has a more recent update → log becomes a complete backup of current state without storing full history; Kafka implements this natively

The log's role in a system: sequencing concurrent updates, replication between nodes, commit semantics for writers, restoring failed replicas, external subscription feeds, data rebalancing.

---

### The 5 Gaps Found in LexGraph

**Gap 1 — JOB_STORE is the source of truth, not a derived view (`api/runner.py:48`)**

```python
JOB_STORE: dict[str, JobRecord] = {}
```

This dict IS the authoritative state — there is nothing behind it. Server restart wipes every job and every report. The correct model: an append-only event log is the source of truth; JOB_STORE is the materialized current-state view rebuilt from it. Acknowledged as a known limitation in the runner.py docstring and CONTEXT Known Issues table.

**Gap 2 — No pipeline checkpointing — crash = restart from zero (`agents/orchestrator/graph.py`)**

`run_pipeline()` is a single monolithic function: ingest → all 6 LangGraph agents → done. No node-level cursor, no resume point. LangGraph supports `SqliteSaver` as a checkpointer natively, but `build_graph()` is called with no checkpointer argument. Enabling it takes ~10 lines.

**Gap 3 — N×M integration: Sprint 25 ASTR-O violated the integration rule**

Adding ASTR-O as a consumer required modifying 5+ existing files:
- `clause_extractor/retriever.py` — new `retrieve_with_metadata()` / `retrieve_multi_with_metadata()` variants + `_build_ranking_metadata()`
- `clause_extractor/agent.py` — switch calls to new variants
- `report_qa/qa.py` — return `retrieval_metadata` + `enriched_chunks`
- `api/schemas.py` — new optional fields on `QAResponse`
- `test_runner_for_astr_o.py` — assemble span dict from new fields

The integration rule says adding a new consumer should create work only to connect it to a single pipeline — not to every producing system. Zero of those files should have changed; ASTR-O should have subscribed to a retrieval event log.

**Gap 4 — No document deduplication (log compaction missing)**

Uploading the same contract twice creates duplicate Qdrant vectors, inflating retrieval noise. Acknowledged in Known Issues. Log compaction fix: content hash as primary key, dedup before embedding. Same primary key (content hash) → latest record wins.

**Gap 5 — Batch pipeline, no incremental output to client**

Extraction is already concurrent internally (`asyncio.gather` per doc × category, `Semaphore(10)`) but all output is accumulated and returned as a batch at the end. The user sees nothing for 10 minutes on a 50-doc job. Stream processing principle: produce output at user-controlled frequency, no static snapshot required.

---

### Upgrade 1 — LangGraph SqliteSaver (checkpoint log)

**Files:** `agents/orchestrator/graph.py`, `api/runner.py` | **Effort:** ~10 lines | **Prerequisite:** none

Every LangGraph node transition is checkpointed to SQLite. On crash or restart, `graph.invoke()` with the same `thread_id` (= `job_id`) resumes from the last completed node — not from ingestion.

```python
# agents/orchestrator/graph.py
from langgraph.checkpoint.sqlite import SqliteSaver

def build_graph():
    graph = StateGraph(GraphState)
    # ... add_node / add_edge / add_conditional_edges unchanged ...
    checkpointer = SqliteSaver.from_conn_string("jobs.db")
    return graph.compile(checkpointer=checkpointer)

# api/runner.py — add config kwarg to graph.invoke()
result = graph.invoke(
    {"job_id": job_id, "documents": documents},
    config={"configurable": {"thread_id": job_id}}
)
```

**What this unlocks:**
- Crash recovery: 50-doc job crashes after 40 min of extraction → resumes from the last completed node
- Stage replay: re-run `risk_scorer` with new rules on already-extracted clauses by invoking with the same thread_id from the `risks_scored` checkpoint — no re-ingestion, no re-extraction
- Log principle applied: `job_id (thread_id) + checkpoint DB = entire pipeline state`. Single cursor offset = full replay position. This is exactly "replica described by a single number."

---

### Upgrade 2 — SQLite Event Log for JOB_STORE

**Files:** `api/runner.py` (new `EventLog` class, replace dict usages) | **Effort:** ~150 lines | **Prerequisite:** none

Replace the in-memory `JOB_STORE` dict with an append-only event log. JOB_STORE becomes a derived view rebuilt by replaying events on startup.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS job_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,  -- log entry number / clock
    job_id  TEXT    NOT NULL,
    event   TEXT    NOT NULL,
    payload TEXT    NOT NULL,                   -- JSON blob
    ts      TEXT    NOT NULL                    -- ISO-8601 timestamp
);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id);
```

**Event types:**

| event | payload fields |
|---|---|
| `job_created` | `doc_ids, tmp_dir, created_at` |
| `doc_ingested` | `doc_id, page_count, chunk_count` |
| `ingestion_failed` | `doc_id, error` |
| `pipeline_complete` | `final_report` |
| `pipeline_error` | `error` |

**Usage pattern:**
```python
class EventLog:
    def append(self, job_id: str, event: str, payload: dict) -> int:
        # INSERT INTO job_events ... → return rowid (the log entry number)
        ...
    def replay(self) -> dict[str, JobRecord]:
        # SELECT * FROM job_events ORDER BY id → fold into dict[job_id, JobRecord]
        ...

event_log = EventLog("jobs.db")
# At startup:
JOB_STORE = event_log.replay()
# On any status change instead of direct dict mutation:
event_log.append(job_id, "pipeline_complete", {"final_report": report})
```

**Log principle applied:** Log → Table. The event log is the source of truth; JOB_STORE is the materialized current-state projection. Full audit trail: every state transition is timestamped and persisted. Every previous state of every job is recoverable. Server restart is safe.

---

### Upgrade 3 — Content-Hash Deduplication

**Files:** `api/runner.py` (or `ingestion/indexer.py`) | **Effort:** ~30 lines | **Prerequisite:** none

Before ingesting, SHA-256 hash the file bytes. Store `content_hash` in each Qdrant point's payload. On upload, scroll Qdrant for an existing point with that hash. If found, reuse the existing `doc_id` and skip embedding entirely.

```python
import hashlib

def find_existing_doc(file_bytes: bytes, qdrant_client) -> str | None:
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    results, _ = qdrant_client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))]),
        limit=1,
    )
    return results[0].payload["doc_id"] if results else None

# In ingestion path:
existing_doc_id = find_existing_doc(file_bytes, qdrant)
if existing_doc_id:
    # reuse — skip load/chunk/embed/index entirely
    doc_id = existing_doc_id
else:
    doc_id = str(uuid.uuid4())
    # proceed with ingestion, store content_hash in payload
```

Also requires adding `content_hash` to the Qdrant payload in `indexer.py` at index time.

**Log principle applied:** Log compaction — same primary key (content hash) → latest record wins, duplicate discarded. Eliminates the "same contract uploaded twice" known limitation and the retrieval noise that comes with it.

---

### Upgrade 4 — Internal Event Bus

**Files:** `core/events.py` (new), agent files (add `bus.publish()` calls), `api/main.py` or startup (add `bus.subscribe()` calls) | **Effort:** ~100 lines + migration | **Prerequisite:** none

A thin synchronous event bus. Agents publish events with no knowledge of who consumes them. New consumers register at startup — zero changes to agent code when adding a new consumer.

```python
# core/events.py
from collections import defaultdict
from typing import Callable

class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event_type: str, payload: dict) -> None:
        for h in self._handlers[event_type]:
            h(payload)

bus = EventBus()  # module-level singleton
```

**Publish calls to add in agents (one line each):**
```python
# clause_extractor/agent.py — after each successful extraction
bus.publish("clause_extracted", {
    "job_id": job_id, "doc_id": doc_id,
    "clause": clause.model_dump(), "retrieval_metadata": metadata
})

# risk_scorer/agent.py — after each flag
bus.publish("risk_flagged", {"job_id": job_id, "flag": flag.model_dump()})

# contradiction_detector/agent.py — after each contradiction
bus.publish("contradiction_found", {"job_id": job_id, "contradiction": c.model_dump()})
```

**Consumer registration replaces Sprint 25 plumbing (in startup, not in agent files):**
```python
# ASTR-O — instead of adding retrieval_metadata plumbing through 5 files:
bus.subscribe("clause_extracted", astro_handler.on_retrieval_event)

# Future consumers — zero changes to any agent file:
bus.subscribe("risk_flagged", lambda e: slack.post(e) if e["flag"]["risk_level"] == "high" else None)
bus.subscribe("contradiction_found", compliance_exporter.on_contradiction)
```

**Log principle applied:** N×M → N+M. Adding a new consumer = one `bus.subscribe()` call in one file. At LexGraph's current scale a synchronous Python bus is sufficient. Upgrade path: swap for Redis Pub/Sub to get cross-process fanout; swap for Kafka if throughput requires partitioned logs.

---

### Upgrade 5 — SSE Streaming for Live Extraction Results

**Files:** `api/main.py` (new endpoint), `ui/app.py` (live table component) | **Effort:** ~80 lines | **Prerequisite:** Upgrade 4 (event bus)

New endpoint `GET /jobs/{id}/stream` returns Server-Sent Events. The event bus feeds an `asyncio.Queue` per connected client. Extracted clauses and risk flags stream to the client as they complete — no waiting for the full pipeline.

```python
# api/main.py
from sse_starlette.sse import EventSourceResponse

@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, _: None = Depends(_verify_api_key)):
    async def generator():
        queue: asyncio.Queue = asyncio.Queue()

        def on_event(payload: dict) -> None:
            if payload.get("job_id") == job_id:
                asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, payload)

        bus.subscribe("clause_extracted", on_event)
        bus.subscribe("risk_flagged", on_event)
        bus.subscribe("contradiction_found", on_event)

        while True:
            event = await queue.get()
            if event.get("event") == "pipeline_complete":
                yield {"data": json.dumps(event)}
                break
            yield {"data": json.dumps(event)}

    return EventSourceResponse(generator())
```

Streamlit UI update: replace the polling `GET /jobs/{id}` loop with an SSE listener that renders a live clause table, highlighting HIGH risks in red as each `risk_flagged` event arrives.

**Log principle applied:** Stream processing — produce output at user-controlled frequency, no static snapshot required. The user sees extraction results for doc 1 while doc 2 is still being processed.

---

### Upgrade Priority Order

| # | Upgrade | Effort | Impact | Files Changed |
|---|---|---|---|---|
| 1 | LangGraph SqliteSaver | ~10 lines | Crash recovery, stage replay | `orchestrator/graph.py`, `api/runner.py` |
| 3 | Content-hash dedup | ~30 lines | Eliminates known dupe limitation | `api/runner.py`, `ingestion/indexer.py` |
| 2 | SQLite event log | ~150 lines | Job durability across restarts | `api/runner.py` |
| 4 | Internal event bus | ~100 + migration | All future consumers free, ASTR-O decoupled | `core/events.py` + agent publish calls |
| 5 | SSE streaming | ~80 lines | Live progress UX | `api/main.py`, `ui/app.py` |

**Start with Upgrade 1.** LangGraph already has the machinery — `SqliteSaver` is a first-party LangGraph package. One argument to `graph.compile()`, one `config=` kwarg to `graph.invoke()`. A 50-doc job that crashes after 40 minutes of extraction resumes from the last node instead of restarting from ingestion. Everything else is additive.

---

## Sprint 28 — Planned: Hexagonal + Event-Sourced Architecture

**Analysis date:** 2026-05-31. After studying FLARE's architecture (spacecraft telemetry anomaly detection) and the underlying theory (Hexagonal Architecture — Cockburn; DDIA Ch11 — Kleppmann; The Log — Kreps), three architectural patterns were identified as directly applicable to LexGraph. Sprint 28 applies all three in a phased migration. No code has been written yet — this section captures the full plan so it can be executed in a future session.

**Prerequisite note:** Sprint 27 and Sprint 28 target overlapping concerns (both address JOB_STORE durability and event persistence). Sprint 28's `JsonlEventLog` supersedes Sprint 27's `SQLite event log` — they solve the same Gap 2 with different mechanisms. Sprint 27 Upgrades 1 (SqliteSaver), 3 (content-hash dedup), and 5 (SSE streaming) remain independent and can be applied alongside Sprint 28. Sprint 27 Upgrade 4 (event bus) is absorbed into Sprint 28's event-driven pipeline.

---

### The Three Patterns and What Question Each Answers

These are not competing choices. They answer different questions at different levels and stack cleanly:

| Pattern | Question | Where It Sits |
|---|---|---|
| **Hexagonal Architecture (Ports & Adapters)** | What is allowed to know about what? | Structure — the container |
| **Event-Driven Pipeline** | How does computation flow inside the container? | Behavior — inside the container |
| **Event Sourcing (Log-Based State)** | How is state preserved at the persistence boundary? | Persistence — the adapter boundary |

---

### Current Architecture: What Is Broken

**Gap A — JOB_STORE is the source of truth (`api/runner.py`)**
```python
JOB_STORE: dict[str, JobRecord] = {}
```
Mutable in-memory dict. Server restart wipes every job and every report. No history, no audit trail, no recovery.

**Gap B — No crash recovery for the pipeline**
`run_pipeline()` is monolithic: ingest → 6 agents → done. No checkpoint. Crash after 40 minutes of extraction on a 50-doc job = restart from ingestion.

**Gap C — Infrastructure imported directly inside agents (N×M coupling)**
`get_qdrant_client()` inside `retriever.py`. `neo4j_client.py` imported inside `entity_mapper/agent.py` and `contradiction_detector/agent.py`. Swapping the vector store requires touching agent files. Sprint 25's ASTR-O integration required modifying 5 existing files because there was no clean boundary.

**Gap D — State transitions leave no trace**
`GraphState` is mutated in-place via LangGraph's dict merging. There is no record of what changed, when, or in what order. The only artifact is the final `final_report` string.

---

### Target Architecture

**`GraphState` stays as the LangGraph interface** — LangGraph requires it. What changes: it becomes a *materialized view* derived from the event log, not the source of truth.

**`runner.py` becomes the composition root** — the only file that instantiates adapters and injects them. Agents import ports only, never adapters, never infrastructure.

```
core/ports.py        — ABCs (IEventLog, IVectorDB, IKnowledgeGraph)
                       BOUNDARY: no qdrant, no neo4j, no litellm imports here

adapters/            — Implementations of the ABCs
  qdrant_adapter.py  — QdrantAdapter implements IVectorDB
  neo4j_adapter.py   — Neo4jAdapter implements IKnowledgeGraph
  jsonl_event_log.py — JsonlEventLog implements IEventLog
                       BOUNDARY: no LangGraph state imports here

agents/              — Orchestration only. Imports ports, not adapters.
                       BOUNDARY: inject adapters from runner.py, never instantiate here

api/runner.py        — Composition root: instantiates adapters, injects into agents, creates
                       JsonlEventLog per job, wires everything together
```

---

### Port Definitions (`core/ports.py`)

```python
from abc import ABC, abstractmethod

class IEventLog(ABC):
    @abstractmethod
    def append(self, event: DomainEvent) -> None: ...
    @abstractmethod
    def replay(self, job_id: str) -> list[DomainEvent]: ...
    @abstractmethod
    def fold(self, job_id: str) -> GraphState: ...

class IVectorDB(ABC):
    @abstractmethod
    def upsert(self, points: list[EmbeddedChunk]) -> None: ...
    @abstractmethod
    def query(self, dense: list[float], sparse: dict, doc_id: str, top_k: int) -> list[Chunk]: ...

class IKnowledgeGraph(ABC):
    @abstractmethod
    def merge_clause(self, clause: ExtractedClause) -> None: ...
    @abstractmethod
    def find_contradictions(self, doc_ids: list[str]) -> list[Contradiction]: ...
```

---

### Domain Events (`core/events.py`)

Immutable frozen dataclasses. The log stores these in order. `fold()` consumes them in order to reconstruct `GraphState`.

```python
@dataclass(frozen=True)
class JobStarted:
    job_id: str
    file_paths: list[str]
    timestamp: datetime

@dataclass(frozen=True)
class DocumentIngested:
    job_id: str
    doc_id: str
    page_count: int
    chunk_count: int
    timestamp: datetime

@dataclass(frozen=True)
class ClauseExtracted:
    job_id: str
    document_id: str
    clause_type: str
    found: bool
    confidence: float
    timestamp: datetime

@dataclass(frozen=True)
class RiskFlagRaised:
    job_id: str
    document_id: str
    clause_type: str
    risk_level: str
    timestamp: datetime

@dataclass(frozen=True)
class ContradictionDetected:
    job_id: str
    clause_type: str
    document_id_a: str
    document_id_b: str
    timestamp: datetime

@dataclass(frozen=True)
class ReportGenerated:
    job_id: str
    timestamp: datetime
```

`fold(events) → GraphState`: apply events in append order to reconstruct current state. On crash → replay → resume. The `job_id.jsonl` file is the source of truth; `GraphState` in memory is the projection.

---

### What Does NOT Change

| Component | Why untouched |
|---|---|
| 6 agent node bodies and their logic | Ports are injected — node code calls port methods, not infrastructure directly |
| LangGraph topology (6 nodes, conditional routing) | Unchanged — same `add_node` / `add_edge` / `add_conditional_edges` |
| Retrieval pipeline (bge-m3, RRF, parent-child, CUAD_ALT_QUERIES) | Lives inside `QdrantAdapter.query()` — same Qdrant calls, same parameters |
| CUAD eval harness | Calls retriever directly, bypasses adapters; unchanged |
| FastAPI endpoints and schemas | Unchanged |
| Streamlit UI | Unchanged |
| R@3 = 68.3% benchmark | `QdrantAdapter` must be a faithful forwarding wrapper — any query parameter change would regress this |

---

### Migration Phases

**Each phase ends with the smoke test passing. Do not proceed to the next phase until the current phase is verified.**

**Phase 1 — Define ports and adapters, wire nothing** (~30 lines)
- Create `core/ports.py` with three ABCs
- Create `adapters/` directory
- Move `qdrant_client.py`, `neo4j_client.py` into `adapters/` as `QdrantAdapter`, `Neo4jAdapter`
- Create `JsonlEventLog` that writes `{job_id}.jsonl`
- Existing agent code unchanged — no imports modified yet
- **Smoke test:** still passes. R@3: not benchmarked (no retrieval code changed)

**Phase 2 — Wire IEventLog into the orchestrator** (~50 lines)
- `runner.py` instantiates `JsonlEventLog` per job
- Orchestrator emits `JobStarted` on job creation
- On crash and restart: `fold()` reconstructs `GraphState` from JSONL log, `graph.invoke()` resumes
- Agents unchanged
- **Smoke test:** still passes. **Crash recovery check:** kill process mid-job, restart, confirm state reconstructed correctly from JSONL

**Phase 3 — Wire IVectorDB and IKnowledgeGraph into agents** (~80 lines)
- Replace `get_qdrant_client()` calls in `retriever.py` with `IVectorDB` port
- Replace `neo4j_client` calls in `entity_mapper/agent.py` and `contradiction_detector/agent.py` with `IKnowledgeGraph` port
- Inject adapters from `runner.py` (composition root) — no agent instantiates infrastructure
- **Critical:** `QdrantAdapter.query()` must forward identical parameters to `query_points()` — no semantic change allowed
- **Smoke test:** still passes. **Benchmark:** run `cuad_eval.py --n 400` to verify R@3 = 68.3% ± noise

**Phase 4 — Emit domain events from agents** (~60 lines)
- Each agent emits the relevant `DomainEvent` to `IEventLog` after completing its work
- `ClauseExtracted` after each extraction, `RiskFlagRaised` after each flag, `ContradictionDetected` per contradiction, `ReportGenerated` at pipeline end
- Additive — no existing logic removed or modified
- **Smoke test:** still passes. **Verification:** inspect `{job_id}.jsonl` — events appear in correct order with correct payloads

---

### Migration Risk Summary

| Phase | Risk | Primary concern |
|---|---|---|
| 1 | **Low** | Pure structural — zero behavior change |
| 2 | **Medium** | First real behavior change; crash recovery path must be verified explicitly |
| 3 | **Medium** | Adapter must not silently alter Qdrant query parameters — R@3 regression would be the symptom |
| 4 | **Low** | Purely additive — no existing code path modified |

**Highest risk:** Phase 3 adapter wiring. If `QdrantAdapter.query()` changes `query_points()` call semantics (different `candidate_k`, missing `using=` parameter, altered filter), retrieval quality regresses. Run the full CUAD eval at the end of Phase 3 before proceeding.

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
| `e2e_eval.py` unbound `key` variable (Sprint 25 fix) | `key = _llm_key(prompt)` was defined inside `if llm_cache is not None:` block — unbound if cache was None. Fixed: hoisted `key = _llm_key(prompt)` before the cache check; combined condition to `if llm_cache is not None and key in llm_cache:`. Also added `raw: str = ... # type: ignore[union-attr]` to silence Pyright on `choices[0].message.content` (`str \| None`). |
| Groq key used as OpenAI key during Sprint 25 first eval | `.env` had `gsk_...` set as `OPENAI_API_KEY`. Caused AuthenticationError on gpt-4o-mini, fell back to Groq, contaminated the gpt-4o-mini LLM cache. Fix: replaced with real OpenAI key, cleared cache with `rm eval/cache/llm_responses_gpt_4o_mini.pkl`, re-ran clean. |

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


