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
