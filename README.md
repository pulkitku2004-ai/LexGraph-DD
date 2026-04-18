# LexGraph-DD — Multi-Agent Legal Due Diligence Engine

> Ingest contracts. Extract every clause. Score every risk. Detect contradictions across documents. Ask questions. Get a brief.

LexGraph-DD is an end-to-end legal due diligence pipeline built on a multi-agent LangGraph state machine. Upload 1–50 contracts (PDF, DOCX, TXT), and the system produces a structured brief covering clause extraction across all 41 CUAD categories, risk scoring, entity mapping, and — its defining capability — **cross-document contradiction detection via a shared Neo4j knowledge graph**.

---

## Why This Matters for Enterprise

Legal due diligence is one of the most time-intensive bottlenecks in M&A, venture investment, procurement, and compliance reviews. A typical deal involving 30–50 contracts requires a team of lawyers to:

- Manually read every document to find clauses that may not use standard legal terminology
- Cross-reference obligations, governing laws, IP assignments, and liability caps across all contracts
- Identify where two contracts say conflicting things about the same party or obligation
- Produce a written brief summarising risk exposure

**What LexGraph-DD automates:**

| Task | Manual Time | With LexGraph-DD |
|---|---|---|
| Clause extraction (41 categories × 50 docs) | 3–5 days | ~15 minutes |
| Risk flagging (missing / dangerous clauses) | 1–2 days | Instant (deterministic rules) |
| Cross-document contradiction detection | Extremely error-prone manually | Automated via graph queries |
| Due diligence brief | Half a day of writing | Generated, structured Markdown |
| Follow-up Q&A on specific clauses | Hours of re-reading | Sub-second RAG with citations |

**Who benefits:**
- **M&A legal teams** reviewing acquisition targets with hundreds of contracts
- **VC/PE firms** running portfolio-level contract audits
- **Corporate legal ops** standardising contract review across subsidiaries
- **Compliance teams** checking obligations against regulatory changes
- **Law firms** accelerating associate work on large deal rooms

---

## Key Differentiator

**Cross-document contradiction detection.** No existing open-source legal AI implementation does this — tools like contract-bert, LegalBench extractors, and GPT-based reviewers all analyse documents in isolation.

LexGraph-DD writes every extracted clause, party, jurisdiction, duration, and monetary amount into a shared Neo4j knowledge graph. Two Cypher queries then systematically find:

1. **Value conflicts** — same clause type, different values across documents (Governing Law: *Delaware* vs *New York*; Liability Cap: *$1M* vs *uncapped*)
2. **Absence conflicts** — clause present in one document, structurally absent in another (Document A has a Non-Compete; Document B — signed by the same party — does not)

Each conflict surfaces in the report with an LLM-generated plain-English explanation written for a lawyer.

---

## Architecture

```
PDF / DOCX / TXT
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Ingestion                                                            │
│  PyMuPDF / python-docx → heading merge → parent-child chunker        │
│  (2048-tok parents / 256-tok children, 51-tok child overlap)         │
│  → bge-m3 embed children (dense 1024-dim CLS + learned sparse)       │
│  → Qdrant (child vectors + parent text in payload)                   │
└──────┬───────────────────────────────────────────────────────────────┘
       │
       │  LangGraph state machine (6 agents, conditional routing)
       │
       ▼
┌─────────────────┐
│  Health Check   │  Qdrant + Neo4j reachability → route or degrade gracefully
└────────┬────────┘
         ▼
┌─────────────────┐
│ Clause          │  41 CUAD categories × hybrid retrieval (bge-m3 sparse+dense+RRF)
│ Extractor       │  → parent-child dedup → async LLM extraction → ExtractedClause[]
└────────┬────────┘
         ▼
┌─────────────────┐
│ Risk Scorer     │  Deterministic rules + LLM reasoning → RiskFlag[]
└────────┬────────┘
         ▼
┌─────────────────┐
│ Entity Mapper   │  Parties / jurisdictions / durations / amounts → Neo4j graph
└────────┬────────┘
         ▼
┌─────────────────┐
│ Contradiction   │  Cypher queries → value conflicts + absence conflicts
│ Detector        │  → LLM explanations → Contradiction[]
└────────┬────────┘
         ▼
┌─────────────────┐
│ Report + Q&A    │  Structured Markdown brief + RAG Q&A with page-level citations
└─────────────────┘
         │
    FastAPI REST  ──►  Streamlit UI
```

---

## Agents

### 1. [Health Check](https://github.com/pulkitku2004-ai/LexGraph-DD/blob/main/legal_due_diligence/infrastructure/health_check.py)
Pings Qdrant and Neo4j before any work begins and writes two boolean flags (`qdrant_ready`, `neo4j_ready`) into shared graph state. The orchestrator reads these immediately and routes conditionally — if Qdrant is down, the pipeline skips to the Report agent and documents what failed rather than crashing. **Partial results are more useful to an operator than a stack trace.**

### 2. [Clause Extractor](https://github.com/pulkitku2004-ai/LexGraph-DD/blob/main/legal_due_diligence/agents/clause_extractor/agent.py)
The most computationally intensive agent. For each document × each of the 41 CUAD clause categories, it:

1. Embeds the category query with `bge-m3` → dense (1024-dim) + learned sparse vector pair
2. Runs hybrid retrieval against Qdrant — dense cosine + sparse SPLADE weights, fused with Reciprocal Rank Fusion (RRF k=60), top-20 candidates each
3. **Two-stage parent dedup:** stage 1 (score order) deduplicates by `parent_id` to select the best child per parent; stage 2 (doc order) re-sorts surviving parents by `parent_chunk_index` so the LLM sees Article 2 before Article 10
4. Calls an LLM with the full parent text (up to 2048 tokens per parent), returning `{"found": bool, "clause_text": "...", "normalized_value": "...", "confidence": 0.0–1.0}`

All 41 categories run **concurrently per document** via `asyncio.gather`. All documents run concurrently too. A global `asyncio.Semaphore(10)` caps LLM concurrency to stay within rate limits. 15 hard categories (e.g. *Covenant Not To Sue*, *Revenue/Profit Sharing*, *Non-Compete*) fire 2–3 alternative queries each — RRF scores are summed across queries before dedup — solving the vocabulary gap between CUAD category names and real contract language.

Output: `list[ExtractedClause]` — one object per (document × category) pair.

### 3. [Risk Scorer](https://github.com/pulkitku2004-ai/LexGraph-DD/blob/main/legal_due_diligence/agents/risk_scorer/agent.py)
Two-pass scoring:

- **Rules pass** (deterministic, no LLM): flags missing high-stakes clauses (no Limitation of Liability → `high`), detects dangerous normalized values (uncapped liability, perpetual lock-in), checks structural clause patterns
- **LLM reasoning pass** (8 nuanced categories only): for clauses where risk depends on content — indemnification scope, IP assignment breadth, termination triggers — the LLM reads the extracted text and returns a structured risk assessment

Output: `list[RiskFlag]` with `risk_level` (high / medium / low), a human-readable `reason`, and a `source_clause_id` tracing back to the originating chunk for citations.

### 4. [Entity Mapper](https://github.com/pulkitku2004-ai/LexGraph-DD/blob/main/legal_due_diligence/agents/entity_mapper/agent.py)
Reads all extracted clauses and writes a structured knowledge graph into Neo4j using `MERGE` (fully idempotent — safe to re-run without duplicating nodes).

```
(Document)-[:HAS_CLAUSE]->(Clause)-[:INVOLVES]----->(Party)
                                  -[:GOVERNED_BY]-->(Jurisdiction)
                                  -[:LASTS]-------->(Duration)
                                  -[:WORTH]-------->(MonetaryAmount)
```

Party names, jurisdictions, durations, and monetary amounts are extracted from `normalized_value` fields via regex + LLM. **This graph is what makes cross-document contradiction detection possible** — without it, documents remain isolated blobs of text.

Output: `graph_built: bool` in state. The actual data lives in Neo4j.

### 5. [Contradiction Detector](https://github.com/pulkitku2004-ai/LexGraph-DD/blob/main/legal_due_diligence/agents/contradiction_detector/agent.py)
Queries the Neo4j graph directly via two Cypher queries — never re-reads clause state:

1. **Value conflicts** — `MATCH` clauses of the same type across different documents where `normalized_value` differs (e.g. Governing Law: *Delaware* vs *New York*)
2. **Absence conflicts** — clause present in document A, missing in document B (asymmetric obligations)

All Cypher queries are scoped to the current job's `$doc_ids` — the graph accumulates across jobs but contradiction detection never leaks cross-job results. Each conflict is passed to the LLM for a plain-English explanation. If Neo4j is unavailable (`graph_built=False`), returns an empty list immediately and the report notes the skip.

Output: `list[Contradiction]` with document pair, conflicting values, and explanation.

### 6. [Report + Q&A](https://github.com/pulkitku2004-ai/LexGraph-DD/blob/main/legal_due_diligence/agents/report_qa/agent.py)
Terminal agent — reads full state, produces two outputs:

**Structured Report**: A deterministic formatter builds the risk table and contradiction table first (pure Python, no LLM, no failure mode). One LLM call generates an executive summary and recommended actions, which slot into a fixed Markdown template. If the LLM call fails, a template narrative is generated from state counts — `final_report` is **always populated**.

**Interactive Q&A** (`POST /jobs/{id}/qa`): After a job completes, hybrid retrieval runs scoped to that job's documents and returns a grounded answer with page-level citations. Same bge-m3 + RRF pipeline as the Clause Extractor.

---

## Tech Stack

| Component | Technology | Role |
|---|---|---|
| Orchestration | LangGraph | State machine with conditional routing between agents |
| Vector store | Qdrant (Docker) | Stores dense + sparse vectors, hybrid search |
| Knowledge graph | Neo4j 5 (Docker) | Entity graph for cross-document contradiction detection |
| Embeddings | `BAAI/bge-m3` | Single model: dense (1024-dim CLS, L2-norm) + learned sparse (SPLADE-style) |
| LLM routing | LiteLLM | Provider abstraction + automatic fallback chain |
| LLM — extraction | `groq/llama-3.1-8b-instant` → `groq/llama-4-scout-17b` → `ollama/mistral-nemo` | Speed-first, with local fallback |
| LLM — reasoning | `ollama/mistral-nemo` / OpenRouter free tier | Quality-first for risk scoring and report narrative |
| PDF parsing | PyMuPDF | Fastest parser, preserves reading order |
| DOCX parsing | python-docx | Direct XML access, no external tools |
| API | FastAPI | Async REST, 202 Accepted pattern for long jobs |
| UI | Streamlit | Upload interface, job polling, report viewer |
| Eval | Custom CUAD harness + embedding cache | Recall@K across 41 categories; cached chunk embeddings for fast iteration |

---

## Performance

### Retrieval (CUAD Benchmark)

Evaluated on [`chenghao/cuad_qa`](https://huggingface.co/datasets/chenghao/cuad_qa) — 1,244 test rows across 41 legal clause categories.

| Retrieval Setup | R@1 | R@3 | Notes |
|---|---|---|---|
| legal-bert baseline | — | 9% | Dense only |
| bge-base dense-only | ~10% | ~15% | With query prefix |
| bge-m3 hybrid sparse+dense+RRF | 33.3% | 52.1% | 1,244 rows |
| + parent-child chunking (256/2048) | — | 61.7% | Sprint 16 |
| + CUAD definition query enrichment | — | 67.5% | Sprint 18–20 |
| + enrichment fix + alt query tuning | **39.6%** | **68.3%** | **Sprint 22 — final, 1,244 rows** |

**3× improvement** over the dense-only baseline. Parent-child chunking keeps embeddings focused on 256-token children while delivering 2048-token parent context to the LLM. Multi-query retrieval (+6.1pp) fires 2–3 alternative phrasings for 15 hard categories and sums RRF scores before deduplication.

**Retrieval ceiling declared at R@3 = 68.3%.** Further gains require contrastive fine-tuning on CUAD or a fundamentally different architecture.

**Per-category breakdown (1,244-row run, selected categories):**

| Category | R@3 | n |
|---|---|---|
| Most Favored Nation | 0% | 3 |
| Non-Compete | 35% | 23 |
| Joint IP Ownership | 29% | 7 |
| Revenue/Profit Sharing | 46% | 35 |
| Post-Termination Services | 52% | 29 |
| Covenant Not To Sue | 46% | 24 |
| Anti-Assignment | 81% | 72 |
| Agreement Date | 87% | 93 |
| Document Name | 87% | 102 |
| Parties | 97% | 102 |

Hard categories (0–35%) are vocabulary-gap problems — the clause is present but uses language far removed from its CUAD category name. Query enrichment and multi-query recovered most of the gap; the remainder requires fine-tuned embeddings.

### Extraction Quality (End-to-End)

Evaluated on 192 rows from the same CUAD test set — measures whether the LLM correctly extracts the clause text given retrieved context.

| Metric | Value |
|---|---|
| Found Rate | ~83% |
| Token F1 (mean) | ~0.35 |
| Conditional F1 (when found=True) | ~0.42 |
| Substring Match | ~22% |

**Extraction ceiling declared at Cond. F1 ≈ 0.42.** The primary limiting factor is the extraction model (llama-3.1-8b-instant) paraphrasing rather than copying verbatim. This is a model capability ceiling, not a retrieval or prompt engineering problem — see [What Was Evaluated](#what-was-evaluated) below.

---

## What Was Evaluated

Every approach that was tested and either kept or rejected, in the order it was tried.

### Retrieval

| Approach | R@3 Delta | Verdict | Why |
|---|---|---|---|
| Dense-only (legal-bert) | baseline 9% | Rejected | Domain-specific but small model, no sparse signal |
| bge-base + query prefix | +6pp | Superseded | Replaced by bge-m3 |
| bge-m3 hybrid dense+sparse RRF | +37pp over baseline | **Kept** | Single model, jointly trained dense+sparse vectors — no need for a separate BM25 index |
| HyDE (Hypothetical Document Embeddings) | −3.9pp | **Rejected** | llama-3.1-8b generates generic boilerplate; shifts query embeddings away from real contract language |
| Cross-encoder reranker (bge-reranker-v2-m3) | −9.5pp | **Rejected** | MS-MARCO trained on web/news; legal text domain mismatch causes consistent regression |
| MMR (Maximal Marginal Relevance) | N/A | **Rejected** | parent_id dedup already prevents the clumping MMR solves — no additional gain |
| candidate_k=50 vs k=20 | +1pp | **Rejected** | Within noise; k=20 confirmed optimal |
| Parent-child chunking 128/1024 | baseline 61.7% | Superseded | Replaced by 256/2048 |
| Parent-child chunking 512/2048 | worse | **Rejected** | 512-child embeddings too broad; lose precision |
| Parent-child chunking 256/2048 | +6.1pp | **Kept** | Optimal: focused child embeddings + long parent context for LLM |
| Multi-query for 15 hard categories | +6.1pp | **Kept** | RRF score summing across alt queries gives vocabulary coverage without false positive leakage |
| Multi-query expansion to 6 new categories | −3 to −23pp | **Rejected** | Alt queries for License Grant, Non-Disparagement, ROFR/ROFO/ROFN, Change of Control all caused leakage into adjacent categories (Anti-Assignment contamination worst case: −23pp) |
| Hybrid alpha tuning (α=0.7/0.8/0.9) | −0.7 to −1.3pp | **Rejected** | bge-m3 dense+sparse are jointly trained for equal-weight fusion; tilting toward sparse overweights exact-term matching and loses semantic clustering |
| CUAD definition query enrichment (6 categories) | +0.8pp net | **Kept** | Anchor words from official CUAD definitions injected into queries for bottom-tier categories |
| Case-insensitive enrichment lookup fix | +0.3pp | **Kept** | Bug fix; enrichment was silently failing on mixed-case category names |
| RRF_K parameter tuning | Not attempted | Skipped | Marginal expected gain insufficient to justify the effort at R@3 = 68.3% |

### Extraction Quality

| Approach | Cond. F1 Delta | Verdict | Why |
|---|---|---|---|
| SYSTEM_PROMPT rewrite (Sprint 14) | baseline established | **Kept** | Structured JSON schema, found/clause_text/normalized_value separation, substance-over-label instruction |
| Extraction hints for 19 categories (Sprint 23) | Confounded | **Kept** | Run was model-mixed (Groq TPD hit → ollama fallback); improvement not cleanly measurable but logically sound |
| `trim_clause_text()` — strip section headers (Sprint 23) | ±0.003 (noise) | **Kept** | Negligible Token F1 effect, but produces cleaner input for risk scorer and contradiction detector |
| Verbatim copy instruction in SYSTEM_PROMPT (Sprint 24) | −4.8pp | **Rejected** | "Copy character-for-character, do not paraphrase" confused the model — llama-3.1-8b degraded under the additional constraint. This is a model capability ceiling, not a prompt problem. |

---

## Current Limitations

This is a working research prototype. Before deploying in a production legal environment, the following gaps should be addressed:

| Limitation | Detail |
|---|---|
| **Extraction quality ceiling** | Conditional F1 ≈ 0.42 on CUAD. The extraction LLM (llama-3.1-8b-instant) paraphrases rather than copying verbatim. Fixing this requires a larger model (70B+) or a CUAD fine-tuned extraction model — neither is addressable with prompt engineering alone. |
| **Retrieval ceiling on hard categories** | Most Favored Nation (0%), Non-Compete (35%), Joint IP Ownership (29%) remain low after all enrichment. Root cause: clause vocabulary diverges significantly from category name. Requires contrastive fine-tuning on CUAD (paid GPU). |
| **In-memory job store** | Jobs live in a Python dict — lost on server restart. A SQLite or Postgres-backed store is needed for production durability. |
| **Rate-limit dependency** | The extraction pipeline makes ~2,050 LLM calls per 50-document job. Groq's free tier (500k tokens/day on scout-17b) can exhaust mid-run on large batches; the local Ollama fallback is slower and produces lower-quality output. |
| **English-only** | bge-m3 is multilingual but the CUAD prompts and risk rules are English-only. Non-English contracts will retrieve correctly but extract poorly. |
| **No document deduplication** | Uploading the same contract twice creates duplicate vectors. Qdrant MERGE handles the graph safely, but vector duplicates inflate retrieval noise. |
| **No PDF table/form extraction** | PyMuPDF extracts text flow only. Contracts with obligation tables or signature blocks in PDF form fields may lose structured data. |
| **Contradiction detection depends on Neo4j** | If Neo4j is unavailable, the contradiction detector returns an empty list silently. The report notes the skip, but the user may not realise contradictions were not checked. |
| **LLM-generated explanations are not verified** | Risk flag reasons and contradiction explanations are LLM outputs — they can hallucinate. All LLM-generated text should be treated as a starting point for lawyer review, not a legal conclusion. |

---

## Quickstart

### Prerequisites
- Docker (for Qdrant + Neo4j)
- Python 3.12
- [Ollama](https://ollama.com) with `mistral-nemo` pulled (`ollama pull mistral-nemo`)
- A free [Groq API key](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/pulkitku2004-ai/LexGraph-DD.git
cd LexGraph-DD

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY at minimum
```

### 3. Start infrastructure

```bash
docker compose up -d
```

Starts Qdrant (`:6333`) and Neo4j (`:7687`, browser at `:7474`).

### 4. Run the API

```bash
.venv/bin/python -m uvicorn legal_due_diligence.api.main:app --port 8000
```

### 5. Run the UI

```bash
.venv/bin/python -m streamlit run legal_due_diligence/ui/app.py
```

Open `http://localhost:8501`, upload contracts, get a due diligence brief.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/jobs` | Upload contracts (multipart/form-data), returns `job_id` immediately (202 Accepted) |
| `GET` | `/jobs/{id}` | Poll status: `pending → running → done / error`. Report in `report` field when done. |
| `POST` | `/jobs/{id}/qa` | Ask a free-text question; returns answer + page-level citations |
| `DELETE` | `/jobs/{id}` | Remove job, Qdrant vectors, Neo4j nodes, and temp files |

```bash
# Upload contracts
curl -X POST http://localhost:8000/jobs \
  -F "files=@contract_a.pdf" -F "files=@contract_b.pdf" | jq .

# Poll until done
curl http://localhost:8000/jobs/{job_id} | jq .status

# Ask a question
curl -X POST http://localhost:8000/jobs/{job_id}/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "Which contract has an uncapped liability clause?"}' | jq .
```

---

## Project Structure

```
LexGraph-DD/
├── legal_due_diligence/
│   ├── agents/
│   │   ├── clause_extractor/        # bge-m3 hybrid retrieval + parent-child dedup + async LLM extraction
│   │   ├── risk_scorer/             # Deterministic rules + LLM reasoning
│   │   ├── entity_mapper/           # Neo4j knowledge graph writer
│   │   ├── contradiction_detector/  # Cypher queries + LLM explanations
│   │   ├── report_qa/               # Markdown formatter + RAG Q&A
│   │   └── orchestrator/            # LangGraph state machine + conditional routing
│   ├── ingestion/                   # PDF/DOCX/TXT loader, parent-child chunker, bge-m3 embedder, indexer
│   ├── core/                        # Settings (pydantic-settings), domain models, GraphState
│   ├── api/                         # FastAPI endpoints + background job runner
│   ├── ui/                          # Streamlit interface
│   └── infrastructure/              # Qdrant + Neo4j client singletons, health check
├── eval/
│   ├── cuad_eval.py                 # Retrieval eval harness — Recall@K on CUAD benchmark
│   ├── e2e_eval.py                  # End-to-end extraction eval (Token F1 + found rate)
│   ├── cache/                       # Chunk embedding cache — keyed by model+chunk params (30× speedup)
│   └── results/                     # JSON result files per run
├── analyze_categories.py            # Per-category R@3 breakdown from eval JSON
├── docker-compose.yml               # Qdrant + Neo4j services
├── .env.example                     # Environment variable template
└── requirements.txt                 # Direct dependencies
```

---

## LLM Provider Strategy

| Role | Provider | Rationale |
|---|---|---|
| Extraction (~2,050 calls/job) | Groq free tier (`llama-3.1-8b-instant`) | ~300ms/call, 6,000 TPM sufficient with semaphore cap |
| Extraction fallback 1 | `groq/llama-4-scout-17b` | Separate rate-limit bucket from primary model |
| Extraction fallback 2 | `ollama/mistral-nemo` (local) | Zero rate limit, fully offline, runs on M-series via MPS |
| Reasoning (report, risk) | `ollama/mistral-nemo` | Unlimited local inference for quality-sensitive passes |
| Reasoning (small jobs) | OpenRouter free tier | Access to 120B+ models at zero cost, ~200 req/day |

The fallback chain is handled automatically by LiteLLM — no custom retry logic in the application code.

---

## Running the Retrieval Eval

```bash
source .venv/bin/activate

# Quick sanity check (50 rows, ~2 min warm / ~7 min cold)
python eval/cuad_eval.py --n 50 --enrich-queries --multi-query

# Standard benchmark run (360 rows, ~2 min warm / ~50 min cold)
python eval/cuad_eval.py --n 400 --enrich-queries --multi-query

# Per-category breakdown
python analyze_categories.py eval/results/<result_file>.json
```

The eval harness caches all chunk embeddings to `eval/cache/` on the first run. Subsequent runs with the same model and chunk settings skip GPU embedding entirely — a 360-row eval drops from ~50 minutes to ~2 minutes. The cache filename encodes the model and chunk parameters and auto-invalidates on configuration change.

Results saved to `eval/results/` as JSON.

---

## Engineering Notes

For a full walkthrough of every sprint, all design decisions, retrieval benchmark results, agent I/O contracts, and known failure modes, see [CONTEXT.md](CONTEXT.md).
