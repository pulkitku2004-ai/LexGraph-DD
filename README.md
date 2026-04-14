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
┌──────────────────────────────────────────────────────────────────┐
│  Ingestion                                                        │
│  PyMuPDF / python-docx → token chunker (512 tok, 128 overlap)   │
│  → bge-m3 embed (dense 1024-dim + learned sparse) → Qdrant      │
└──────┬───────────────────────────────────────────────────────────┘
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
│ Extractor       │  → async LLM extraction (asyncio.gather) → ExtractedClause[]
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
2. Runs hybrid retrieval against Qdrant — dense cosine similarity + sparse SPLADE weights, fused with Reciprocal Rank Fusion (RRF k=60)
3. Calls an LLM with a structured prompt, returning `{"found": bool, "clause_text": "...", "normalized_value": "...", "confidence": 0.0–1.0}`

All 41 categories run **concurrently per document** via `asyncio.gather`. All documents run concurrently too. A global `asyncio.Semaphore(10)` caps LLM concurrency to stay within rate limits. 10 hard categories (e.g. *Covenant Not To Sue*, *IP Ownership Assignment*) fire 2 alternative queries each with RRF score summing — solving the vocabulary gap between CUAD category names and real contract language.

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

Each conflict is passed to the LLM for a plain-English explanation written for a lawyer. If Neo4j is unavailable (`graph_built=False`), returns an empty list immediately and the report notes the skip.

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
| Embeddings | `BAAI/bge-m3` | Single model: dense (1024-dim CLS) + learned sparse (SPLADE-style) |
| LLM routing | LiteLLM | Provider abstraction + automatic fallback chain |
| LLM — extraction | `groq/llama-3.1-8b-instant` → `groq/llama-4-scout-17b` → `ollama/mistral-nemo` | Speed-first, with local fallback |
| LLM — reasoning | `ollama/mistral-nemo` / OpenRouter free tier | Quality-first for risk scoring and report narrative |
| PDF parsing | PyMuPDF | Fastest parser, preserves reading order |
| DOCX parsing | python-docx | Direct XML access, no external tools |
| API | FastAPI | Async REST, 202 Accepted pattern for long jobs |
| UI | Streamlit | Upload interface, job polling, report viewer |
| Eval | Custom CUAD harness | Recall@K across 1,244 test rows, 41 categories |

---

## Retrieval Performance (CUAD Benchmark)

Evaluated on [`chenghao/cuad_qa`](https://huggingface.co/datasets/chenghao/cuad_qa) — 1,244 test rows across 41 legal clause categories.

| Retrieval Setup | Recall@1 | Recall@3 |
|---|---|---|
| bge-base dense-only (baseline) | ~10% | ~16% |
| bge-m3 hybrid sparse+dense+RRF | **33.3%** | **52.1%** |
| + Multi-query for hard categories | TBD | TBD |

Hybrid retrieval (bge-m3 learned sparse + dense, fused with RRF k=60) gives a **3× improvement** over the dense-only baseline. HyDE (Hypothetical Document Embeddings) was evaluated and **found to hurt** — R@3 dropped to 40.8% — so it is disabled by default.

---

## Current Limitations

This is a working research prototype. Before deploying in a production legal environment, the following gaps should be addressed:

| Limitation | Detail |
|---|---|
| **No authentication** | The FastAPI layer has no auth. All jobs and reports are accessible to anyone who can reach the server. |
| **In-memory job store** | Jobs live in a Python dict — they are lost on server restart. A Redis or Postgres-backed store is needed for production. |
| **No job persistence across restarts** | Closely related: if the server crashes mid-pipeline, the job is gone. |
| **Rate-limit dependency** | The extraction pipeline makes ~2,050 LLM calls per 50-document job. Groq's free tier (6,000 TPM) can throttle large batches without the local Ollama fallback running. |
| **English-only** | bge-m3 is multilingual but the CUAD prompts and risk rules are English-only. Non-English contracts will retrieve correctly but extract poorly. |
| **No document deduplication** | Uploading the same contract twice creates duplicate vectors and graph nodes. Qdrant `MERGE` handles the graph safely, but vector duplicates inflate retrieval noise. |
| **Eval gap on hard categories** | 10 categories (Covenant Not To Sue, IP Ownership Assignment, etc.) still have low R@3 despite multi-query retrieval. Multi-query eval result is TBD. |
| **No PDF table/form extraction** | PyMuPDF extracts text flow only. Contracts with obligation tables or signature blocks in PDF form fields may lose structured data. |
| **No fine-tuned extraction model** | Extraction uses a general-purpose LLM. A fine-tuned model on CUAD (e.g. legal-bert or a LoRA-tuned Llama) would improve accuracy significantly, especially on low-recall categories. |

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
│   │   ├── clause_extractor/        # bge-m3 hybrid retrieval + async LLM extraction
│   │   ├── risk_scorer/             # Deterministic rules + LLM reasoning
│   │   ├── entity_mapper/           # Neo4j knowledge graph writer
│   │   ├── contradiction_detector/  # Cypher queries + LLM explanations
│   │   ├── report_qa/               # Markdown formatter + RAG Q&A
│   │   └── orchestrator/            # LangGraph state machine + conditional routing
│   ├── ingestion/                   # PDF/DOCX/TXT loader, chunker, bge-m3 embedder, indexer
│   ├── core/                        # Settings (pydantic-settings), domain models, GraphState
│   ├── api/                         # FastAPI endpoints + background job runner
│   ├── ui/                          # Streamlit interface
│   └── infrastructure/              # Qdrant + Neo4j client singletons, health check
├── eval/
│   └── cuad_eval.py                 # Retrieval eval harness — Recall@K on CUAD benchmark
├── docker-compose.yml               # Qdrant + Neo4j services
├── .env.example                     # Environment variable template
└── requirements.txt                 # Direct dependencies
```

---

## LLM Provider Strategy

| Role | Provider | Rationale |
|---|---|---|
| Extraction (~2,050 calls/job) | Groq free tier | ~300ms/call, 6,000 TPM sufficient with semaphore cap |
| Extraction fallback | `groq/llama-4-scout-17b` | Separate rate-limit bucket from primary model |
| Extraction fallback 2 | `ollama/mistral-nemo` (local) | Zero rate limit, fully offline, runs on M-series via MPS |
| Reasoning (report, risk) | `ollama/mistral-nemo` | Unlimited local inference for quality-sensitive passes |
| Reasoning (small jobs) | OpenRouter free tier | Access to 120B–405B models at zero cost, ~200 req/day |

The fallback chain is handled automatically by LiteLLM — no custom retry logic in the application code.

---

## Running the Retrieval Eval

```bash
source .venv/bin/activate

# Sanity check (50 rows, ~2 min)
python eval/cuad_eval.py --n 50 --enrich-queries

# Full benchmark (1,244 rows)
python eval/cuad_eval.py --full --enrich-queries

# With multi-query retrieval for hard categories
python eval/cuad_eval.py --n 400 --enrich-queries --multi-query
```

Results saved to `eval/results/` as JSON.

---

## Engineering Notes

For a full walkthrough of every sprint, all design decisions, retrieval benchmark results, agent I/O contracts, and known failure modes, see [CONTEXT.md](CONTEXT.md).
