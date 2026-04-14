# Legal Due Diligence Engine

A multi-agent pipeline that ingests 1–50 contracts (PDF/DOCX/TXT) and produces a structured due diligence brief — clause extraction across 41 CUAD categories, risk scoring, entity mapping, and **cross-document contradiction detection via a shared Neo4j knowledge graph**.

**Key differentiator:** Cross-document contradiction detection. No existing open-source legal AI implementation does this — most tools analyze documents in isolation.

---

## Architecture

```
PDF / DOCX / TXT
       │
       ▼
┌──────────────┐
│   Ingestion  │  PyMuPDF → chunker (512 tok, 128 overlap) → bge-m3 embed → Qdrant
└──────┬───────┘
       │  LangGraph state machine
       ▼
┌──────────────┐
│   Clause     │  41 CUAD categories × hybrid retrieval (bge-m3 sparse+dense+RRF)
│  Extractor   │  → async LLM extraction (Groq llama-3.1-8b) → ExtractedClause[]
└──────┬───────┘
       ▼
┌──────────────┐
│ Risk Scorer  │  Deterministic rules + LLM reasoning → RiskFlag[]
└──────┬───────┘
       ▼
┌──────────────┐
│Entity Mapper │  Parties / obligations / jurisdictions → Neo4j graph (MERGE, idempotent)
└──────┬───────┘
       ▼
┌──────────────┐
│Contradiction │  Cypher queries over shared graph → value conflicts + absence conflicts
│  Detector    │  → LLM explanations → Contradiction[]
└──────┬───────┘
       ▼
┌──────────────┐
│  Report +    │  Markdown brief + RAG Q&A with page-level citations
│    Q&A       │
└──────────────┘
       │
  FastAPI  ──►  Streamlit UI
```

## Agents

### 1. Health Check
Pings Qdrant and Neo4j before any work begins and writes two boolean flags (`qdrant_ready`, `neo4j_ready`) into shared graph state. The LangGraph orchestrator reads these flags immediately and routes conditionally — if Qdrant is down the pipeline skips directly to the Report agent and documents what failed, rather than crashing. Partial results are more useful than a stack trace.

### 2. Clause Extractor
The most computationally intensive agent. For each document × each of the 41 CUAD clause categories, it:
1. Embeds the category query with `bge-m3` to produce a dense (1024-dim) + learned sparse vector pair.
2. Runs hybrid retrieval against Qdrant — dense cosine similarity + sparse SPLADE weights, fused with Reciprocal Rank Fusion (RRF k=60) — to find the most relevant chunks.
3. Calls an LLM (`groq/llama-3.1-8b-instant`) with a structured prompt asking it to find the clause and return `{"found": bool, "clause_text": "...", "normalized_value": "...", "confidence": 0.0–1.0}`.

All 41 categories are extracted **concurrently per document** (`asyncio.gather`), and all documents run concurrently too — a global `asyncio.Semaphore(10)` caps LLM concurrency to stay within Groq's rate limit. 10 categories with low recall also fire 2 alternative queries each (multi-query RRF) to overcome the vocabulary gap between CUAD category names and real contract language.

Output: `list[ExtractedClause]` — one object per (document, category) pair.

### 3. Risk Scorer
Two-pass scoring — deterministic rules first, LLM reasoning second:

- **Rules pass** (O(1), no LLM): flags missing high-stakes clauses (no Limitation of Liability → high risk), detects specific dangerous normalized values (uncapped liability, perpetual terms), and checks clause presence patterns.
- **LLM reasoning pass** (8 nuanced categories only): for clauses where risk depends on *content* — indemnification scope, IP assignment breadth, termination triggers — the LLM reads the extracted clause text and returns a structured risk assessment.

Output: `list[RiskFlag]` with `risk_level` (high / medium / low), a human-readable `reason`, and a `source_clause_id` that traces back to the originating chunk for citations.

### 4. Entity Mapper
Reads all extracted clauses and writes a structured knowledge graph into Neo4j using `MERGE` (fully idempotent — safe to re-run). The graph schema:

```
(Document)-[:HAS_CLAUSE]->(Clause)-[:INVOLVES]->(Party)
                                               -[:GOVERNED_BY]->(Jurisdiction)
                                               -[:LASTS]->(Duration)
                                               -[:WORTH]->(MonetaryAmount)
```

Party names, jurisdictions, durations, and monetary amounts are extracted from `normalized_value` fields via lightweight regex + LLM. This graph is what makes cross-document contradiction detection possible — without it, documents are isolated blobs of text.

Output: `graph_built: bool` in state. The actual data lives in Neo4j.

### 5. Contradiction Detector
Queries the Neo4j graph directly with two Cypher queries — it never re-reads extracted clauses from state:

1. **Value conflicts**: `MATCH` clauses of the same type across different documents where `normalized_value` differs (e.g. Governing Law: *Delaware* vs *New York*).
2. **Absence conflicts**: clause present in document A, missing in document B — structurally asymmetric obligations.

Each detected conflict is passed to the LLM for a plain-English explanation written for a lawyer, not an engineer. If Neo4j is unavailable (`graph_built=False`), returns an empty list immediately — the report notes the skip.

Output: `list[Contradiction]` with document pair, conflicting values, and explanation.

### 6. Report + Q&A
Terminal agent — reads the full state and produces two things:

**Report**: A deterministic Python formatter builds the risk table and contradiction table first (no LLM, no failure mode). One LLM call then generates `{"executive_summary": "...", "recommended_actions": [...]}` which slots into a fixed Markdown template. If the LLM call fails, a template narrative is generated from state counts — `final_report` is always populated.

**Q&A** (separate endpoint): After a job completes, `POST /jobs/{id}/qa` runs hybrid retrieval scoped to that job's documents and returns a grounded answer with page-level citations. Retrieval uses the same bge-m3 + RRF pipeline as the Clause Extractor.

---

## Tech Stack

| Component | Library |
|---|---|
| Orchestration | LangGraph |
| Vector store | Qdrant (Docker) |
| Knowledge graph | Neo4j 5 (Docker) |
| Embeddings | `BAAI/bge-m3` — single model, dense (1024-dim) + learned sparse |
| LLM routing | LiteLLM with fallback chain |
| LLM extraction | `groq/llama-3.1-8b-instant` → `groq/llama-4-scout-17b` → `ollama/mistral-nemo` |
| LLM reasoning | `ollama/mistral-nemo` (local) or OpenRouter free tier |
| API | FastAPI |
| UI | Streamlit |

---

## Retrieval Performance (CUAD Benchmark)

Evaluated on the [chenghao/cuad_qa](https://huggingface.co/datasets/chenghao/cuad_qa) dataset (1,244 test rows, 41 legal clause categories).

| Retrieval Setup | Recall@1 | Recall@3 |
|---|---|---|
| bge-base (baseline) | ~10% | ~16% |
| bge-m3 hybrid sparse+dense+RRF | **33.3%** | **52.1%** |
| + Multi-query for hard categories | TBD | TBD |

Hybrid retrieval (bge-m3 learned sparse + dense, fused with RRF k=60) gives a **3× improvement** over dense-only bge-base.

---

## Quickstart

### Prerequisites
- Docker (for Qdrant + Neo4j)
- Python 3.12
- [Ollama](https://ollama.com) with `mistral-nemo` pulled (`ollama pull mistral-nemo`)
- A free [Groq API key](https://console.groq.com)

### 1. Clone and set up environment

```bash
git clone https://github.com/pulkitku2004-ai/LexGraph-DD.git
cd legal-dd

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
# Must use venv Python directly — pyenv shim misses venv packages
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
| `POST` | `/jobs` | Upload contracts (multipart), start pipeline (202 Accepted) |
| `GET` | `/jobs/{id}` | Poll status: `pending → running → done / error` |
| `POST` | `/jobs/{id}/qa` | Ask a question about the uploaded contracts |
| `DELETE` | `/jobs/{id}` | Delete job + Qdrant vectors + Neo4j nodes |

### Example

```bash
# Upload a contract
curl -X POST http://localhost:8000/jobs \
  -F "files=@contract.pdf" | jq .

# Poll until done
curl http://localhost:8000/jobs/{job_id} | jq .status

# Ask a question
curl -X POST http://localhost:8000/jobs/{job_id}/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the governing law?"}' | jq .
```

---

## Project Structure

```
legal_due_diligence/
├── agents/
│   ├── clause_extractor/        # Hybrid retrieval + async LLM extraction
│   ├── risk_scorer/             # Deterministic rules + LLM reasoning
│   ├── entity_mapper/           # Neo4j graph writer
│   ├── contradiction_detector/  # Cypher queries + LLM explanations
│   └── report_qa/               # Markdown formatter + RAG Q&A
├── ingestion/                   # PDF/DOCX/TXT loader, chunker, bge-m3 embedder
├── core/                        # Config, Pydantic models, LangGraph state
├── api/                         # FastAPI endpoints + background runner
├── ui/                          # Streamlit interface
└── infrastructure/              # Qdrant + Neo4j client singletons

eval/
└── cuad_eval.py                 # Retrieval eval harness (Recall@K on CUAD benchmark)
```

---

## LLM Provider Strategy

Three providers, three roles — optimised for cost and throughput:

| Role | Provider | Why |
|---|---|---|
| Extraction (~2050 calls/job) | Groq free tier | ~300ms/call, sufficient TPM at this volume |
| Reasoning (quality matters) | Ollama local | Unlimited, no rate limit, works offline |
| Reasoning (small jobs ≤25 docs) | OpenRouter free tier | 120B–405B models at zero cost |

The fallback chain (`groq/llama-3.1-8b → groq/llama-4-scout → ollama/mistral-nemo`) is handled automatically by LiteLLM — no custom retry logic needed.

---

## Contradiction Detection

After clause extraction, normalized values are written to Neo4j. Two Cypher queries then find:

1. **Value conflicts** — same clause type, different values across documents (e.g. Governing Law: Delaware vs New York)
2. **Absence conflicts** — clause present in one document, absent in another

Each conflict gets an LLM-generated plain-English explanation surfaced in the report.

---

## Running the Retrieval Eval

```bash
source .venv/bin/activate

# Quick sanity check (50 rows, ~2 min)
python eval/cuad_eval.py --n 50 --enrich-queries

# Full benchmark (1,244 rows, ~90s retrieval after indexing)
python eval/cuad_eval.py --full --enrich-queries

# With multi-query retrieval for hard categories
python eval/cuad_eval.py --n 400 --enrich-queries --multi-query
```

Results saved to `eval/results/` as JSON.

---

## Engineering Notes

For a deep dive into the sprint history, retrieval benchmarks, agent contracts, and design decisions, see [CONTEXT.md](CONTEXT.md).
