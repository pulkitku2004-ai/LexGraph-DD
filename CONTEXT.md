# Master Context — Multi-Agent Legal Due Diligence Engine

Last updated: 2026-04-15
Status: Sprint 13 done (eval pending)

---

## What This System Does

Ingests 1–50 PDF/DOCX contracts → processes through 6 specialist agents orchestrated by LangGraph → outputs a structured due diligence brief with:
- Risk scores per document
- Clause extraction across 41 CUAD categories
- Entity mapping (parties, obligations, jurisdictions)
- Cross-document contradiction detection
- Interactive Q&A with clause-level citations

**Key differentiator:** Cross-document contradiction detection via a shared Neo4j knowledge graph. No existing open source implementation does this.

---

## Environment

| Item | Value |
|---|---|
| Machine | MacBook Air M4, 16GB RAM, 512GB |
| OS | macOS Darwin 25.3.0 |
| Python | 3.12.1 (pyenv) |
| Shell | zsh |
| Virtual env | `/path/to/legal_dd/.venv` |
| Project root | `/path/to/legal_dd/` |
| Package root | `/path/to/legal_dd/legal_due_diligence/` |

**Every session — activate venv first:**
```bash
source /path/to/legal_dd/.venv/bin/activate
```

**VS Code interpreter:** `/path/to/legal_dd/.venv/bin/python3`
**Pylance config:** `pyrightconfig.json` at project root — sets `extraPaths: ["legal_due_diligence"]` so imports resolve.

---

## Tech Stack

| Component | Library | Version | Why |
|---|---|---|---|
| Orchestration | LangGraph | 1.1.6 | State machine with conditional routing, checkpointing |
| LangChain Core | langchain-core | 1.2.27 | LangGraph dependency |
| Vector store | Qdrant (Docker) | latest | Dense vector search with payload filtering |
| Qdrant client | qdrant-client | 1.17.0 | Note: use `query_points()` not `search()` (removed) |
| Knowledge graph | Neo4j (Docker) | 5 | Graph traversal for cross-doc contradiction detection |
| Neo4j driver | neo4j | 6.1.0 | Bolt protocol, connection pool management |
| LLM routing | LiteLLM | 1.83.4 | Provider portability + native fallback chain |
| LLM extraction | groq/llama-3.1-8b-instant | — | Primary: fast, 6000 TPM free tier |
| LLM fallback 1 | groq/llama-4-scout-17b | — | Separate rate limit bucket from primary |
| LLM fallback 2 | ollama/mistral-nemo | 7.1GB local | Zero rate limit, runs on M4 via Ollama |
| LLM reasoning (dev/large batch) | ollama/mistral-nemo | — | Local, unlimited, no rate limit |
| LLM reasoning (small batch) | OpenRouter free tier | — | 120B–405B models via single key; ~200 req/day limit |
| Embeddings | bge-m3 | 1024-dim dense + learned sparse | Single model produces both; replaces bge-base (dense) + BM25 (sparse) |
| ML framework | PyTorch | 2.11.0 | MPS backend active — runs on M4 GPU |
| Sparse retrieval | bge-m3 (Qdrant-native) | — | SPLADE-style learned weights via sparse_linear.pt head; no BM25 pickle |
| PDF parsing | pymupdf (fitz) | 1.27.2.2 | Fastest, preserves reading order |
| DOCX parsing | python-docx | 1.2.0 | Reads XML directly, no external tools |
| Settings | pydantic-settings | 2.12.0 | Type-safe env var loading from .env |
| Data models | pydantic | 2.12.5 | All domain models and GraphState |
| API | FastAPI | 0.128.0 | Sprint 6+ |
| UI | Streamlit | — | Sprint 6+ |
| Evals | DeepEval vs CUAD | — | Sprint 7 |

---

## Infrastructure (Docker)

```bash
docker compose up -d        # start Qdrant + Neo4j
docker compose down         # stop
docker compose down -v      # stop + wipe ALL data (re-index needed after)
docker compose ps           # check status
```

| Service | Port | Credentials |
|---|---|---|
| Qdrant | 6333 | none |
| Neo4j | 7687 (bolt), 7474 (browser UI) | neo4j / password |

**Neo4j browser:** `http://localhost:7474`

**After `docker compose down -v`:** re-run `python run_sprint1.py` to re-index (no pickle to delete — sparse vectors live in Qdrant)

---

## Folder Structure

```
legal_dd/
├── .env                           ← API keys + infra config (never commit)
├── context.md                     ← this file
├── README.md
├── docker-compose.yml             ← Qdrant + Neo4j services
├── pyrightconfig.json             ← Pylance import resolution
├── run_sprint0.py                 ← Sprint 0 smoke test
├── run_sprint1.py                 ← Sprint 1 smoke test
├── run_sprint3.py                 ← Sprint 3 smoke test (--skip-llm flag available)
├── run_sprint4.py                 ← Sprint 4 smoke test (--skip-neo4j flag available)
├── run_sprint5.py                 ← Sprint 5 smoke test (--skip-neo4j flag available)
├── run_sprint6.py                 ← Sprint 6 smoke test (--skip-llm, --skip-qa flags available)
├── run_sprint7.py                 ← Sprint 7 smoke test: 5-row dataset sanity check
├── run_sprint9.py                 ← Sprint 9 smoke test: schema + job lifecycle + pipeline + Q&A + delete
├── analyze_categories.py          ← per-category Recall@3 breakdown from eval JSON results
├── samples/
│   ├── contract_a.txt             ← Acme/TechVentures MSA (Delaware)
│   └── contract_b.txt             ← GlobalSoft/BetaCorp License (New York)
├── eval/
│   ├── cuad_eval.py               ← DeepEval harness: Recall@K + Faithfulness + AnswerRelevancy
│   ├── sample_ids.json            ← 100-row stratified sample indices (reproducible)
│   └── results/
│       ├── baseline_legal_bert.json  ← Recall@3=9% baseline
│       └── baseline_bge.json         ← Recall@3=15% after bge-base swap
└── legal_due_diligence/
    ├── core/
    │   ├── config.py              ← pydantic-settings singleton + LLM provider config + auto-propagates API keys to os.environ
    │   ├── models.py              ← domain models: ExtractedClause, RiskFlag, etc.
    │   └── state.py               ← GraphState (LangGraph state machine envelope)
    ├── infrastructure/
    │   ├── qdrant_client.py       ← Qdrant singleton + health check bool
    │   ├── neo4j_client.py        ← Neo4j driver singleton + session context manager
    │   └── health_check.py        ← LangGraph node: checks infra, sets state flags
    ├── ingestion/
    │   ├── loader.py              ← PDF/DOCX → LoadedDocument (page-level text)
    │   ├── chunker.py             ← LoadedDocument → list[Chunk] (512 tok, 128 overlap)
    │   ├── embedder.py            ← list[Chunk] → list[EmbeddedChunk] (1024-dim dense + sparse, bge-m3, MPS fp16)
    │   └── indexer.py             ← EmbeddedChunk → Qdrant upsert (named dense+sparse vectors, no BM25 pickle)
    ├── agents/
    │   ├── orchestrator/
    │   │   └── graph.py           ← LangGraph StateGraph: all 6 nodes + routing logic
    │   ├── clause_extractor/
    │   │   ├── retriever.py       ← Hybrid sparse (bge-m3 learned) + dense retrieval with RRF fusion
    │   │   ├── prompts.py         ← CUAD 41 categories + extraction prompt builder
    │   │   └── agent.py           ← LangGraph node: retrieval → LLM → ExtractedClause
    │   ├── risk_scorer/
    │   │   ├── rules.py           ← Deterministic rules: MISSING_CLAUSE_RISK, PRESENCE_FLAGS, confidence threshold
    │   │   └── agent.py           ← LangGraph node: rules pass → LLM reasoning pass → RiskFlag list
    │   ├── entity_mapper/
    │   │   ├── extractor.py       ← Entity classification from normalized_value + regex party NER
    │   │   ├── schema.py          ← Neo4j Cypher MERGE writes (Document, Clause, Party, Jurisdiction, etc.)
    │   │   └── agent.py           ← LangGraph node: ExtractedClause → Neo4j graph
    │   ├── contradiction_detector/
    │   │   ├── cypher_queries.py  ← find_value_conflicts() + find_absence_conflicts() scoped to doc_ids
    │   │   └── agent.py           ← LangGraph node: Cypher queries → LLM explanations → Contradiction list
    │   └── report_qa/
    │       ├── formatter.py       ← deterministic section builder: risk table, contradiction table, missing clauses, assemble_report()
    │       ├── qa.py              ← answer_question(): cross-doc hybrid retrieval → LLM → citations
    │       └── agent.py           ← LangGraph node: formatter → LLM narrative → markdown brief
    ├── api/
    │   ├── main.py                ← FastAPI app: POST /jobs, GET /jobs/{id}, POST /jobs/{id}/qa, DELETE /jobs/{id}
    │   ├── schemas.py             ← JobResponse, QAResponse, JobStatus, Citation Pydantic models
    │   └── runner.py              ← JOB_STORE (in-memory), create_job(), run_pipeline(), delete_job()
    ├── evals/                     ← stub (Sprint 7+)
    └── ui/
        └── app.py                 ← Streamlit app: upload → running (5s poll) → done (Report + Q&A tabs)
```

---

## Core Data Models

### `core/models.py` — shared domain vocabulary

```python
DocumentRecord
  doc_id: str           # filename stem or uuid
  file_path: str
  processed: bool       # True after ingestion complete
  page_count: int | None

ExtractedClause         # one per (document × CUAD category)
  document_id: str
  clause_type: str      # one of 41 CUAD categories
  found: bool           # False = missing clause = risk signal
  clause_text: str | None     # verbatim from contract (for citation)
  normalized_value: str | None  # short extracted fact ("Delaware", "30 days")
  confidence: float     # 0.0–1.0, used by risk scorer
  source_chunk_id: str  # UUID → Qdrant point → page in original PDF

RiskFlag
  document_id: str
  clause_type: str
  risk_level: "high" | "medium" | "low"
  reason: str
  is_missing_clause: bool
  source_clause_id: str | None  # ExtractedClause.source_chunk_id → Qdrant → page citation (Sprint 6)   # distinguishes absent vs bad clause

Contradiction
  clause_type: str
  document_id_a: str
  document_id_b: str
  value_a: str              # normalized_value from doc A
  value_b: str              # normalized_value from doc B
  explanation: str          # LLM-generated plain-language risk explanation
```

### `core/state.py` — GraphState

The single object flowing through LangGraph. Every node receives it, returns a dict of changed fields, LangGraph merges the dict back in.

```python
GraphState
  job_id: str
  status: str           # tracks pipeline position
  created_at: datetime
  documents: list[DocumentRecord]
  extracted_clauses: list[ExtractedClause]   # flat list, all docs combined
  risk_flags: list[RiskFlag]
  contradictions: list[Contradiction]
  neo4j_ready: bool     # set by health_check node
  qdrant_ready: bool    # set by health_check node
  graph_built: bool     # set by entity_mapper when Neo4j write complete
  final_report: str | None
  errors: list[str]     # accumulates failures — never raises, always continues
```

**Critical rule:** LangGraph nodes return `dict`, not GraphState instances. Returning a model instance causes a merge failure. Return only the fields you changed.

---

## LangGraph State Machine

### Graph topology

```
START
  │
  ▼
health_check
  │
  ├─ qdrant_ready=False ──────────────────────────────────► report_qa ──► END
  │
  └─ qdrant_ready=True ──► clause_extractor
                                │
                                ▼
                           risk_scorer
                                │
                                ├─ neo4j_ready=False ──────► contradiction_detector
                                │                                    │
                                └─ neo4j_ready=True ──► entity_mapper
                                                              │
                                                              ▼
                                                     contradiction_detector
                                                              │
                                                              ▼
                                                          report_qa
                                                              │
                                                              ▼
                                                             END
```

### Why health check is a node (not a startup gate)
The graph routes around failures gracefully. If Qdrant is down, the graph skips 4 agents and produces a report explaining what failed — more useful than a crash. If Neo4j is down, entity mapping and contradiction detection are skipped but clause extraction and risk scoring still run.

### Why `errors: list[str]` instead of exceptions
One malformed PDF in a 50-document job should not abort the other 49. Agents append to `state.errors`, graph continues. Final report includes error context.

### Why full list replacement (not LangGraph append reducers)
`Annotated[list, operator.add]` appends blindly on every node execution. Full list replacement gives explicit control over deduplication — critical when retrying a failed job from a checkpoint.

---

## Data Flow Diagram

End-to-end view of every transformation from raw file to final output. Read top-to-bottom for the ingestion path, then continue into the LangGraph pipeline.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INGESTION  (run once per document set; persisted to disk)              │
└─────────────────────────────────────────────────────────────────────────┘

 Raw files (PDF / DOCX / TXT)
   │
   │ loader.py  (pymupdf page-by-page / python-docx paragraphs)
   ▼
 LoadedDocument
   { pages: list[PagedText { page_number, text }] }
   │
   │ chunker.py  (legal-bert tokenizer, 512 tok, 128 overlap)
   ▼
 list[Chunk]
   { chunk_id (UUID), doc_id, page_number, chunk_index, text, token_count }
   │
   ├─────────────────────────────────────────────────┐
   │                                                 │
   │ embedder.py
   │ (bge-m3, MPS fp16, CLS pool L2-norm dense + sparse_linear head)
   ▼
 list[EmbeddedChunk]
   { chunk_id, vector: float[1024], sparse_vector: dict[int, float] }
   │
   │ indexer.py  (batches of 100, idempotent on chunk_id)
   ▼
 Qdrant collection
   point { id: chunk_id,
           vector: { "dense": float[1024], "sparse": SparseVector },
           payload: { text, doc_id, file_path, page_number, chunk_index } }


┌─────────────────────────────────────────────────────────────────────────┐
│  LANGGRAPH PIPELINE  (per job; state flows as a single GraphState)      │
└─────────────────────────────────────────────────────────────────────────┘

 GraphState (initial)
   { job_id, documents: list[DocumentRecord], status: "pending" }
   │
   │ health_check_node
   ▼
 GraphState + { qdrant_ready: bool, neo4j_ready: bool }
   │
   ├─ qdrant_ready=False ──────────────────────────────────────┐
   │                                                           │
   │ clause_extractor_node                                     │
   │   For each (doc × 41 CUAD categories):                    │
   │     retriever.py                                          │
   │       Sparse: Qdrant SparseVector (doc_id filter) → top-20  │
   │       Dense:  Qdrant dense vector (doc_id filter) → top-20  │
   │       RRF fusion (k=60) → top-3 Chunks                      │
   │     prompts.py: chunks + category → LLM prompt           │
   │     LiteLLM (groq→groq-fallback→ollama): prompt → JSON   │
   ▼                                                           │
 GraphState + { extracted_clauses: list[ExtractedClause] }     │
   { document_id, clause_type, found: bool,                    │
     clause_text, normalized_value, confidence,                │
     source_chunk_id → Qdrant point → page_number }            │
   │                                                           │
   │ risk_scorer_node                                          │
   │   rules.py (deterministic, no LLM):                      │
   │     found=False + HIGH/MEDIUM category → RiskFlag         │
   │     clause in PRESENCE_FLAGS → RiskFlag                   │
   │     confidence < 0.4 → RiskFlag(medium)                  │
   │   agent.py (LLM, 8 curated categories only):             │
   │     clause_text + normalized_value → reasoning model     │
   │     → RiskFlag if flag=True                              │
   ▼                                                           │
 GraphState + { risk_flags: list[RiskFlag] }                   │
   { document_id, clause_type, risk_level,                     │
     reason, is_missing_clause,                                │
     source_clause_id → same Qdrant chunk_id }                 │
   │                                                           │
   ├─ neo4j_ready=False ──────────────────┐                   │
   │                                      │                   │
   │ entity_mapper_node                   │                   │
   │   extractor.py:                      │                   │
   │     clause_type membership → entity  │                   │
   │     type (Jurisdiction/Duration/     │                   │
   │     MonetaryAmount)                  │                   │
   │     org-suffix regex on clause_text  │                   │
   │     → Party names                    │                   │
   │   schema.py (Cypher MERGE):          │                   │
   ▼                                      │                   │
 Neo4j graph                              │                   │
   (:Document {doc_id})                   │                   │
     -[:HAS_CLAUSE]->                     │                   │
   (:Clause {doc_id, clause_type,         │                   │
             normalized_value,            │                   │
             confidence, found,           │                   │
             source_chunk_id})            │                   │
     -[:INVOLVES]->    (:Party)           │                   │
     -[:GOVERNED_BY]-> (:Jurisdiction)    │                   │
     -[:HAS_DURATION]->(:Duration)        │                   │
     -[:HAS_AMOUNT]->  (:MonetaryAmount)  │                   │
   │                                      │                   │
   │ GraphState + { graph_built: True }   │                   │
   │                                      │                   │
   └──────────────────────────────────────┘                   │
                        │                                     │
                        │ contradiction_detector_node          │
                        │   cypher_queries.py                 │
                        │   (scoped to job's doc_ids):        │
                        │     find_value_conflicts()           │
                        │       same clause_type, both found, │
                        │       different normalized_value    │
                        │     find_absence_conflicts()         │
                        │       same clause_type, one missing │
                        │   agent.py: LLM explanation per     │
                        │   conflict (or template fallback)   │
                        ▼                                     │
                 GraphState + { contradictions: list[Contradiction] }
                   { clause_type, doc_a, doc_b,              │
                     value_a, value_b, explanation }          │
                        │                                     │
                        └─────────────────────────────────────┘
                                         │
                                         │ report_qa_node
                                         │   formatter.py (deterministic, no LLM):
                                         │     risk table (HIGH→MED→LOW per doc)
                                         │     contradiction table
                                         │     missing clauses inventory
                                         │     processing notes / errors
                                         │   LLM (reasoning model, 1 call):
                                         │     compact summary → JSON
                                         │     { executive_summary,
                                         │       recommended_actions }
                                         │   assemble_report(): tables + narrative
                                         ▼
                                 GraphState + { final_report: str (markdown) }
                                         │
                                        END


┌─────────────────────────────────────────────────────────────────────────┐
│  Q&A  (on-demand, outside the pipeline)                                 │
└─────────────────────────────────────────────────────────────────────────┘

 user question (str)
   │
   │ qa.py — for each doc_id in job:
   │   retriever.retrieve(question, doc_id, top_k=3)
   │     → Qdrant sparse + dense (scoped to doc_id) → RRF → top chunks
   │   merge all doc results by RRF score → cross-doc context
   │
   │ LLM: question + context chunks → answer
   ▼
 { answer: str,
   citations: list[{ chunk_id, doc_id, page_number }] }
              └── chunk_id == source_chunk_id on ExtractedClause
                  == source_clause_id on RiskFlag
                  (one foreign key traces the full lineage)


┌─────────────────────────────────────────────────────────────────────────┐
│  EVAL HARNESS  (offline, separate from pipeline)                        │
└─────────────────────────────────────────────────────────────────────────┘

 chenghao/cuad_qa test split (1,244 rows)
   { context, question, answers.text, answers.answer_start }
   │
   │ cuad_eval.py — per row:
   │   1. index context into temp Qdrant collection
   │   2. retrieve(question, doc_id, top_k=3) → chunks
   │   3. Recall@K: answer_start span in any chunk? → hit/miss
   │   4. LLM: chunks + question → answer
   │   5. DeepEval: Faithfulness, AnswerRelevancy, ContextualRecall
   ▼
 eval/results/baseline_*.json
   { recall@1, recall@3, per-row breakdown }
```

---

## Ingestion Pipeline

### `ingestion/loader.py`
- **PDF:** pymupdf page-by-page, `get_text("text")` preserves reading order, skips blank pages
- **DOCX:** python-docx paragraph extraction, grouped into synthetic 50-paragraph "pages"
- **TXT:** plain text read, split into synthetic 50-line "pages" — same structure as DOCX output (added Sprint 9)
- **Output:** `LoadedDocument` with `list[PagedText]` — page_number tracked for citations
- **Why page-level tracking:** `source_chunk_id` must resolve to a specific page for PDF highlighting in Sprint 6

### `ingestion/chunker.py`
- **Strategy:** token-based using legal-bert tokenizer (not character/sentence-based)
- **Why token-based:** legal-bert has a 512-token hard limit — character-based chunks can exceed it, causing silent truncation
- **Size:** 512 tokens per chunk, 128 token overlap (25%)
- **Why 128 overlap:** prevents clause boundary splits — a clause starting at token 490 of one chunk is fully in the next chunk's overlap region
- **Output:** `list[Chunk]` — each with UUID `chunk_id`, `doc_id`, `page_number`, `chunk_index`
- **Note:** "760 > 512" warning at load time is harmless — fires when tokenizer encodes the full page before splitting

### `ingestion/embedder.py`
- **Model:** `BAAI/bge-m3`, loaded once as module-level singleton (base XLM-RoBERTa + `sparse_linear.pt` head)
- **Device:** MPS (M4 GPU) → CUDA → CPU fallback for base encoder; sparse_linear always on CPU (kernel launch overhead wins)
- **Dense pooling:** CLS token of last_hidden_state, L2-normalized → 1024-dim vector
- **Sparse encoding:** `sparse_linear` (Linear 1024→1) projected over all token positions → ReLU → max-aggregate per unique token_id → `{token_id: weight}` dict (SPLADE-style)
- **Why CLS (not mean):** bge-m3 is contrastive-trained — CLS is the meaningful sentence embedding
- **Precision:** base encoder in fp16 on MPS (halves memory: 2.24GB→1.12GB); sparse_linear kept float32 for precision
- **Batch size:** 24 — empirically optimal on M4 MPS (~140ms/chunk vs 271ms at batch 12); MPS→CPU sync is ~3s regardless of batch size, amortised by larger batches
- **FlagEmbedding not used:** FlagEmbedding ≥1.2 incompatible with transformers 5.x; implemented bge-m3 directly via `AutoModel` + `hf_hub_download('BAAI/bge-m3', 'sparse_linear.pt')`
- **Batch sparse computation:** tensors transferred to CPU in bulk (one `.detach().cpu()` call), sparse_linear applied on CPU, numpy loop for max-aggregation — 6x faster than MPS for this tiny op

### `ingestion/indexer.py`
- **Qdrant collection schema:** named vector config — `vectors_config={"dense": VectorParams(size=1024, distance=COSINE)}` + `sparse_vectors_config={"sparse": SparseVectorParams()}`
- **Qdrant upsert:** batches of 100 points, idempotent on chunk_id UUID; each point carries both `"dense"` and `"sparse"` named vectors
- **Qdrant payload:** stores full chunk text + doc_id, file_path, page_number, chunk_index, token_count
- **No BM25 pickle:** sparse retrieval is Qdrant-native — no external file to manage; sparse vectors are filtered at index level (same as dense), no post-filter set intersection needed
- **Why store text in payload:** Qdrant is single source of truth — retriever gets text immediately without a second store lookup
- **Qdrant API note:** use `client.query_points()` with `using="dense"` or `using="sparse"` — `client.search()` was removed in qdrant-client 1.7+

---

## Clause Extractor Agent (Sprint 2)

### Architecture

```
For each document:
  For each of 41 CUAD categories:
    1. retrieve(query, doc_id, top_k=3, candidate_k=20)
         → hybrid BM25 + dense, RRF fusion, scoped to doc_id
    2. build_extraction_prompt(clause_type, chunks, doc_id)
         → structured prompt asking for JSON output
    3. _call_llm(prompt)
         → LiteLLM with 3-provider fallback chain
    4. _parse_response(raw)
         → ExtractedClause or found=False on any failure
```

### `retriever.py` — Hybrid Sparse + Dense + RRF (Sprint 8)

**Two retrieval signals capture different things:**
- **Dense (bge-m3 1024-dim):** semantic similarity — "termination for convenience" matches "right to cancel without cause"
- **Sparse (bge-m3 learned weights):** exact and near-exact term matching — "Section 12.3(b)", "$2,500,000", "force-majeure". Learned weights understand legal term importance better than BM25 TF-IDF.

**RRF formula:**
```
score(chunk) = 1/(k + rank_in_sparse) + 1/(k + rank_in_dense)
```
- `k = 60` — original paper constant, dampens rank sensitivity
- Chunk ranked 1st in both lists scores highest
- Chunk only in one list contributes half the max score
- RRF is a consensus mechanism — neither signal dominates

**Why Qdrant-native sparse over BM25 pickle:**
1. Both retrievals are doc_id-filtered at the index level — no post-filter set intersection needed
2. Learned sparse weights are more informative than BM25 TF-IDF for legal text
3. No separate file to manage — sparse vectors live alongside dense in Qdrant

**Why k=60:** At k=60, rank 1 vs rank 2 differs by ~0.00027. Prevents a single strong signal from overwhelming a consistent mediocre signal.

**Why candidate_k=20 before fusion:** Fetching only top-5 from each list could miss a chunk ranked 6th in sparse but 1st in dense. Fetching top-20 from each gives RRF enough signal. After fusion, return top_k=3 to the LLM.

**Why filter by doc_id in Qdrant:** Per-document clause extraction — we want chunks from contract_a only, not semantically similar clauses from contract_b. Both dense and sparse queries carry the `doc_id` filter, applied at index level (fast).

**Sparse query API:** `SparseVector(indices=[...], values=[...])` + `using=settings.sparse_vector_name` in `query_points()`. NOT `NamedSparseVector` — that type is not accepted by `query_points()` in qdrant-client 1.17.x.

**`bm25_tokenize()` kept as legacy export:** `cuad_eval.py` imported it during Sprint 7→8 transition. Function still exists in `retriever.py` but is unused in the main retrieval path.

### `prompts.py` — CUAD 41 Categories

**Two vocabularies — why they're separate:**
- Category name: `"Governing Law"` — legal term of art, classification vocabulary
- Retrieval query: `"governing law jurisdiction choice of law state"` — BM25+dense optimised, adds synonyms

**Why JSON output (not free text):**
- Deterministically parseable
- No second LLM call to extract fields from prose
- On parse failure → `found=False` (conservative fallback)

**Two output fields — why both:**
- `clause_text`: verbatim sentence(s) from contract — used for citations in report
- `normalized_value`: short extracted fact ("Delaware", "30 days") — used for contradiction detection
- Contradiction detector compares normalized_values, not full paragraphs

**`confidence` field purpose:** Risk scorer weights decisions by confidence. `found=True` with `confidence=0.4` signals ambiguous/poorly-written clause — itself a risk.

**`_normalized_value_examples`:** Per-category examples in the prompt anchor the model to the right abstraction level. Without examples, the model may return a full sentence instead of "Delaware".

### `agent.py` — LLM Call + Fallback Chain

**LiteLLM fallback chain:**
```
Primary:    groq/llama-3.1-8b-instant       (6000 TPM, ~300ms/call)
Fallback 1: groq/llama-4-scout-17b          (separate TPM bucket)
Fallback 2: ollama/mistral-nemo             (local M4, zero rate limit)
```
LiteLLM handles the cascade automatically on `RateLimitError` — no manual retry logic needed. Why this beats sleep-and-retry: instead of waiting for the primary to recover, immediately use a provider with available capacity.

**Why `temperature=0.0`:** Extraction is not creative — we want the same answer every time for the same input. Deterministic output also makes debugging and evals reliable.

**Why `max_tokens=300`:** JSON response never exceeds 200 tokens. Lower limit = fewer tokens consumed = slower rate limit drain.

**Why catch all exceptions:** In a 50-document job, a flaky API response should not abort 49 remaining documents. Any failure → `found=False, confidence=0.0` → risk scorer treats as missing clause (conservative).

**Markdown fence stripping:** Smaller models sometimes wrap JSON in ` ```json ``` ` even when instructed not to. Strip before `json.loads()`.

**`source_chunk_id`:** Set to the top-1 RRF chunk. Used in Sprint 6 for citation tracing: risk flag → ExtractedClause → source_chunk_id → Qdrant payload → page_number → highlight in PDF.

---

## Entity Mapper Agent (Sprint 4)

### Architecture: ExtractedClause → Neo4j graph

```
For each ExtractedClause in state.extracted_clauses:
  1. MERGE Document node (one per unique document_id)
  2. MERGE Clause node (doc_id + clause_type composite key)
     SET normalized_value, confidence, found, source_chunk_id
  3. MERGE HAS_CLAUSE relationship Document → Clause
  4. extract_entities(clause) → {parties, jurisdictions, durations, amounts}
  5. MERGE entity nodes + typed relationships
```

Writes ALL clauses — found=True and found=False. Sprint 5 needs both to
distinguish "conflicting values" from "one document missing the clause."

### `extractor.py` — entity classification from normalized_value

**No external NER model used:** `normalized_value` is already structured text
("Delaware", "12 months fees", "5 years") — classifying by `clause_type`
membership in a frozenset is O(1) and 100% reliable. Party names use a
targeted suffix regex (`Inc.`, `LLC`, `Corp.`, `Ltd.` etc.) on `clause_text`.

Entity type mapping:
- `Governing Law`, `Dispute Resolution`, `Venue` → `Jurisdiction` node
- `Warranty Duration`, `Confidentiality`, `Non-Compete`, `Termination for Convenience`… → `Duration` node
- `Liability Cap`, `Liquidated Damages`, `Minimum Commitment`… → `MonetaryAmount` node
- Org-suffix regex on `clause_text` → `Party` nodes (all clause types)

### `schema.py` — Neo4j Cypher writes (MERGE, idempotent)

```
(:Document {doc_id})
    -[:HAS_CLAUSE]->
(:Clause {doc_id, clause_type, normalized_value, confidence, found, source_chunk_id})
    -[:INVOLVES]->    (:Party {name})
    -[:GOVERNED_BY]-> (:Jurisdiction {name})
    -[:HAS_DURATION]->(:Duration {value})
    -[:HAS_AMOUNT]->  (:MonetaryAmount {value})
```

**Constraints:** `doc_unique` (Document.doc_id IS UNIQUE) + composite index
on `(Clause.doc_id, Clause.clause_type)` for MERGE performance.

**NODE KEY not used:** requires Neo4j Enterprise. Community Docker image is
sufficient; MERGE semantics enforce uniqueness without the constraint.

### Sprint 4 smoke test results

```
11 Clause nodes | 2 Document nodes | parties=8 jurisdictions=2 durations=3 amounts=2
- contract-a → Governing Law → GOVERNED_BY → Delaware ✓
- contract-b → Governing Law → GOVERNED_BY → New York ✓
- Sprint 5 pre-condition: cross-doc Governing Law conflict (Delaware ≠ New York) ✓
- found=False Clause (contract-b/Termination for Convenience) persisted ✓
```

---

## Contradiction Detector Agent (Sprint 5)

### Architecture: Cypher-first, LLM-second

```
Step 1 — Structural detection (cypher_queries.py):
  a. find_value_conflicts(session, doc_ids)
       → both found=True, same clause_type, different normalized_value (toLower/trim normalised)
  b. find_absence_conflicts(session, doc_ids)
       → same clause_type, one found=True, one found=False

Step 2 — LLM explanation (settings.llm_reasoning_model):
  For each structural contradiction found:
  → call reasoning model with clause_type + value_a + value_b
  → plain-language explanation of the legal risk (1–2 sentences)
  → fallback to template explanation if LLM fails (Contradiction always populated)
```

### `cypher_queries.py` — two Cypher patterns

Both queries are scoped to `$doc_ids` — critical because the graph accumulates
documents across jobs. Without scoping, job B detects contradictions against
job A's leftover nodes. `doc_ids` comes from `state.documents`.

**Value conflict query:** `docA.doc_id < docB.doc_id` prevents duplicate (A,B)/(B,A) pairs.
`toLower(trim(...))` normalises casing/whitespace so "Delaware" ≠ "delaware" is not a false positive.

**Absence conflict query:** Returns `found_a` + `found_b` so the Python layer correctly
assigns `"(clause absent)"` to whichever side is missing, regardless of alphabetical doc_id order.

### `agent.py` — LLM explanation

**Model:** `settings.llm_reasoning_model` (default: `ollama/mistral-nemo`) — same as risk scorer.
**temperature=0.0, max_tokens=150** — one or two sentence explanation, deterministic.
**Fallback:** Template string on LLM failure — every `Contradiction` object is always complete.

**Session management:** One Neo4j session opened for both queries, closed before LLM calls.
Avoids holding a bolt connection open during potentially slow Ollama calls.

**Graceful skip:** If `state.graph_built=False` (Neo4j was down during entity mapping),
returns `contradictions=[]` without querying. Report notes what was skipped.

### Sprint 5 smoke test results

```
5 contradictions detected across contract-a / contract-b:
  VALUE   | Confidentiality          | 5 years vs 3 years
  VALUE   | Governing Law            | Delaware vs New York
  VALUE   | Liability Cap            | 12 months fees vs 6 months fees
  VALUE   | Payment Terms            | 30 days vs 45 days
  ABSENCE | Termination for Convenience | 30 days vs (clause absent)
All 5 have LLM-generated plain-language explanations ✓
```

**Bug caught:** First run returned 25 contradictions because the graph contained leftover
`doc-001`/`doc-002` nodes from Sprint 0. Fix: `IN $doc_ids` scoping in both Cypher queries.

---

## Report & Q&A Agent (Sprint 6)

### Architecture: formatter-first, LLM-second

```
Step 1 — Deterministic sections (formatter.py):
  All tables built in pure Python — no LLM involved:
  a. Risk flags table (per document, sorted HIGH → MEDIUM → LOW)
  b. Cross-document contradictions table
  c. Missing clauses inventory (is_missing_clause=True only)
  d. Processing notes / error list

Step 2 — LLM narrative (settings.llm_reasoning_model, one call):
  compact structured summary → JSON:
  {"executive_summary": "...", "recommended_actions": ["...", ...]}
  → _template_narrative() fallback if LLM fails

Step 3 — Assembly (assemble_report):
  Deterministic tables + LLM narrative → final markdown brief
  Stored in state.final_report
```

### `formatter.py` — deterministic section builder

**Why formatter-first?** A fully LLM-generated report risks misstated risk levels,
truncated table rows, or fabricated clause text when context is long. Deterministic
tables are always correct. The LLM adds value only in the two narrative slots where
prose quality matters and factual accuracy is structurally constrained by context.

**`build_narrative_prompt`** sends a compact summary (not the full state dump):
doc count, risk counts by level, top HIGH flags (labelled `MISSING CLAUSE` or
`CONTENT RISK`), all contradictions. Token count bounded to ~800 regardless of
job size. HIGH flags are labelled by type so the LLM cannot conflate a missing
clause risk with a contradiction — see prompt hardening note below.

**`_template_narrative`** fallback: produces a factually accurate (if generic)
executive summary from state counts if LLM fails. Every report always assembles.

**`assemble_report`** inserts the two narrative sections into a fixed markdown template.
LLM output cannot corrupt the structure — it only fills two named slots.

### `qa.py` — Q&A with citations

**Cross-doc retrieval:** Calls the existing per-doc `retrieve()` for each doc_id,
merges by RRF score. Avoids a second retrieval path while covering all job documents.

**Why not a single unscoped Qdrant query?** The collection accumulates across jobs.
Without doc_id scoping, Q&A would ground answers in unrelated job data. Passing
`doc_ids` from the current job keeps answers grounded in the right documents.

**Citation chain:** `chunk_id` in citations is the same `source_chunk_id` as on
`ExtractedClause` — traces to a specific Qdrant point → `page_number` payload →
exact page in original PDF. The lineage chain is complete end-to-end.

### Sprint 6 smoke test results

```
Test 1 — Formatter (no LLM):
  7 assertions pass: risk table, contradiction table, missing clauses, template narrative,
  all headers, type-labelled HIGH flags, contradiction disambiguation in prompt

Test 2 — Report synthesis (Ollama):
  2529-char markdown brief
  Exec summary: "Reviewed contracts A and B. Overall risk picture shows 5 flags, including
    1 HIGH risk uncapped liability clause in contract B. Most critical finding is the
    presence of 3 contradictions between the two contracts."
  3 recommended actions generated

Test 3 — Q&A (Qdrant + Ollama):
  Q: "What is the governing law for these contracts?"
  A: Correctly identifies Delaware (contract-a) and New York (contract-b), 6 citations
  Q: "What is the liability cap in these contracts?"
  A: Correctly identifies 12-month and 6-month caps, 6 citations
```

**Prompt hardening fix (post-Sprint 6):**
First run produced: *"High risk: missing governing law clauses in contract B."* — factually wrong.
contract-b has `Governing Law = "New York"`; the risk is the contradiction with contract-a (Delaware).
Root cause: prompt listed HIGH flags without type labels, so the LLM conflated a content risk
(Uncapped Liability) with a contradiction (Governing Law).

Two fixes applied:
1. `run_sprint6.py` fixture: removed erroneous `contract-b / Governing Law / missing=True` flag
2. `build_narrative_prompt`: each HIGH flag now labelled `(MISSING CLAUSE)` or `(CONTENT RISK)`;
   contradiction section header explicitly states *"clauses present in both docs but with conflicting values"*;
   added `IMPORTANT` disambiguation instruction before the data block.

**Production upgrade:** set `LLM_REASONING_MODEL=anthropic/claude-sonnet-4-6` in `.env`
for production-quality report narrative. Zero code change needed — LiteLLM handles routing.

---

## Risk Scorer Agent (Sprint 3)

### Architecture: rules-first, LLM-second

```
For each ExtractedClause in state.extracted_clauses:
  Pass 1 — Deterministic rules (rules.py):
    a. found=False → score_missing_clause() → RiskFlag(high/medium) or None
    b. found=True, clause in PRESENCE_FLAGS → score_presence_flag() → RiskFlag
    c. found=True, confidence < 0.4 → score_low_confidence() → RiskFlag(medium)
  Pass 2 — LLM reasoning (only if found=True, conf >= 0.4, clause in curated set):
    → call ollama/mistral-nemo with clause_text + normalized_value
    → parse JSON: {"flag": bool, "risk_level": str, "reason": str}
    → emit RiskFlag only if flag=True
```

### `rules.py` — deterministic rules layer

**`MISSING_CLAUSE_RISK`** — maps clause_type → risk level for absent clauses:
- **HIGH** (6 clauses): Governing Law, Limitation of Liability, Liability Cap, Indemnification, IP Ownership Assignment, Confidentiality
- **MEDIUM** (8 clauses): Termination for Convenience, Termination for Cause, Payment Terms, Dispute Resolution, Anti-Assignment, Change of Control, No-Solicit of Employees, Warranty Duration
- **LOW** (everything else) → no flag emitted (noise suppression — low-risk absences are informational, not actionable)

**`PRESENCE_FLAGS`** — inverted rules where finding the clause IS the risk:
- `Uncapped Liability` found → HIGH ("explicit unlimited liability exposure")
- `Joint IP Ownership` found → MEDIUM ("neither party can license without consent")
- `Liquidated Damages` found → MEDIUM ("penalty may exceed actual harm")
- `Irrevocable or Perpetual License` found → MEDIUM ("cannot be revoked even on breach")

**Confidence threshold:** `0.4` — below this, found=True clauses flag as MEDIUM ("ambiguous clause language — manual review recommended"). Only fires for clause types that are medium/high risk in the missing-clause table.

### `agent.py` — LLM reasoning

**LLM_ASSESSMENT_CATEGORIES** (8 clause types sent to reasoning model when found + confident):
`Limitation of Liability`, `Liability Cap`, `Indemnification`, `IP Ownership Assignment`, `Non-Compete`, `Governing Law`, `Termination for Convenience`, `Confidentiality`

**Model:** `settings.llm_reasoning_model` (default: `ollama/mistral-nemo`) — no fallback needed, local model has no rate limit. Swap via one `.env` line — see LLM Provider Strategy below.

**temperature=0.0, max_tokens=150** — deterministic, JSON never exceeds 100 tokens.

**Verified LLM behaviour (Sprint 3 run):**
- Fee-based liability cap ("12 months fees") → MEDIUM: "Cap based on fees may not adequately cover damages if fees are low"
- $100 liability cap → HIGH: "Unreasonably low liability cap may discourage claims and limit recovery"
- 30-day termination notice → MEDIUM: "Unilateral termination with minimal notice period may disrupt counterparty's operations"

### Sprint 3 smoke test results

```
5 flags total (HIGH=1, MEDIUM=4) | rule-based=2, llm-assessed=3
- contract-a | Liability Cap | MEDIUM (LLM) — fee-based cap may not cover damages
- contract-a | Termination for Convenience | MEDIUM (LLM) — short notice period
- contract-b | Liability Cap | MEDIUM (LLM) — fee-based cap may not cover damages
- contract-b | Termination for Convenience | MEDIUM (rule) — clause absent
- contract-b | Uncapped Liability | HIGH (rule) — presence flag
```

---

## Sample Data

Two contracts deliberately designed with contradictions to test Sprint 5:

| Clause | contract_a.txt | contract_b.txt |
|---|---|---|
| Parties | Acme Corp + TechVentures | GlobalSoft LLC + BetaCorp |
| Type | Master Service Agreement | Software License Agreement |
| Governing Law | **Delaware** | **New York** |
| Liability Cap | **12 months fees** | **6 months fees** |
| Payment Terms | **30 days** | **45 days (Net 45)** |
| Confidentiality Survival | **5 years** | **3 years** |
| Non-Solicit Employees | 1 year | 2 years |
| Termination for Convenience | 30 days notice | **missing** |

Extraction results verified:
- `contract-a`: Governing Law=Delaware, Liability Cap=12 months fees, Payment Terms=30 days, Confidentiality Survival=5 years ✓
- `contract-b`: Governing Law=New York, Liability Cap=6 months fees, Payment Terms=45 days, Confidentiality Survival=3 years ✓

---

## LLM Provider Strategy

Three providers, three roles — each chosen for a specific reason:

| Provider      |      Model        |     Role        |     Limit |
|---|---|---|---|
| Groq (primary) | llama-3.1-8b-instant | Extraction | 6000 TPM free |
| Groq (fallback 1) | llama-4-scout-17b | Extraction fallback | Separate bucket |
| Ollama (fallback 2) | mistral-nemo (7.1GB) | Extraction final fallback + large-batch reasoning | None |
| OpenRouter free | nemotron-3-super-120b:free | Reasoning — small/medium jobs (≤25 docs) | ~200 req/day |
| OpenRouter free | hermes-3-llama-3.1-405b:free | Reasoning — best quality option | ~200 req/day |

**When to use which for reasoning (`LLM_REASONING_MODEL` in `.env`):**

| Job size | Set to | Why |
|---|---|---|
| ≤25 docs | `openrouter/nvidia/nemotron-3-super-120b-a12b:free` | 120B >> mistral-nemo 7B for nuanced legal risk; ~200 req/day covers ≤200 reasoning calls |
| 50 docs | `ollama/mistral-nemo` | ~400 reasoning calls (8 cats × 50 docs) hits OpenRouter free daily limit mid-run |
| Offline | `ollama/mistral-nemo` | Only option |

**OpenRouter free tier rules:**
- Model names require `:free` suffix — without it, the call hits paid tier
- Rate limits: ~20 req/min, ~200 req/day per model (varies by model)
- `deepseek/deepseek-r1:free` is NOT available on this account's free tier
- Verified working: `nvidia/nemotron-3-super-120b-a12b:free`
- Available (may rate-limit at peak): `nousresearch/hermes-3-llama-3.1-405b:free`, `meta-llama/llama-3.3-70b-instruct:free`

**API key propagation:** `config.py` automatically pushes `GROQ_API_KEY` and `OPENROUTER_API_KEY` from `.env` into `os.environ` after settings load. LiteLLM reads from `os.environ` — no manual env var setting needed in agent code.

**Groq free tier math:** 41 categories × 3 chunks × ~500 tokens = ~61,500 tokens per document. At 6000 TPM = ~10 min per document with only one provider. Fallback chain eliminates wait time by routing to available capacity.

---

## Data Engineering Fundamentals Applied

### 1. Data Schema Layer (what flows between agents)
```
raw file → LoadedDocument → list[Chunk] → list[EmbeddedChunk] → Qdrant points
                                                                      ↓
                                                              ExtractedClause
                                                                      ↓
                                                                 RiskFlag
                                                                      ↓
                                                               Contradiction
                                                                      ↓
                                                              final_report
```
`source_chunk_id` is a **foreign key** — traces every LLM output back to a specific Qdrant point → specific page in the original PDF. That's data lineage.
`RiskFlag.source_clause_id` carries the same ID forward: risk flag → `source_clause_id` → Qdrant chunk payload → `page_number` → PDF highlight in Sprint 6.

### 2. State Machine (states, transitions, failures)
`GraphState.status` tracks pipeline position. Routing functions ARE the transition logic. `errors: list[str]` is the failure accumulation pattern — same as Spark's `badRecordsPath`.

### 3. Pipeline Design (sequential vs parallel)
Currently linear/sequential. Natural parallelisation in Sprint 4: `risk_scorer` and `entity_mapper` have no dependency on each other — can fan-out from `clause_extractor`. Will evaluate after Sprint 7 benchmarks.

### 4. Storage Tiers
```
Tier 1 — Raw (immutable):    samples/*.txt, uploaded PDFs
Tier 2 — Processed:          Qdrant vectors + BM25 pickle (rebuilds from Tier 1)
Tier 3 — Structured outputs: GraphState fields, final_report (rebuilds from Tier 2)
```
Never mutate Tier 1. If agent logic changes, re-run from Tier 2. If embeddings change, re-run from Tier 1.

### 5. Observability (current state)
- Per-agent structured logging with `[agent_name]` prefix
- `state.errors` accumulation
- `state.status` for pipeline position
- **Missing (Sprint 7):** per-node timing, embedding latency metrics, Recall@K retrieval evals

---

## Sprint Plan

| Sprint | Goal | Status |
|---|---|---|
| 0 | LangGraph state machine + 6 stub agents + health check + conditional routing | ✅ DONE |
| 1 | Ingestion pipeline: loader → chunker → embedder → Qdrant + BM25 | ✅ DONE |
| 2 | Clause Extractor: hybrid retriever (BM25+dense+RRF) + LLM extraction + fallback chain | ✅ DONE |
| 3 | Risk Scorer: deterministic rules layer + LLM reasoning + missing clause flags | ✅ DONE |
| 4 | Entity Mapper: NER → Neo4j graph schema write | ✅ DONE |
| 5 | Contradiction Detector: Cypher queries across shared graph | ✅ DONE |
| 6 | Report + Q&A: structured brief synthesis + RAG Q&A with citations | ✅ DONE |
| 7 | DeepEval evals (CUAD) + embed model search (legal-bert→bge-base+prefix = best) + re-index | ✅ DONE |
| 8 | bge-m3 hybrid: replace bge-base+BM25 with bge-m3 dense+sparse — R@3: 15%→42% (+27pp) | ✅ DONE |
| 9 | FastAPI layer: POST /jobs, GET /jobs/{id}, POST /jobs/{id}/qa, DELETE /jobs/{id} | ✅ DONE |
| 10 | Streamlit UI: upload → running → done (Report + Q&A tabs) | ✅ DONE |

---

## Key Architectural Decisions

| Decision | Why |
|---|---|
| LangGraph nodes return `dict` not model instances | Merge vs overwrite — returning a model instance replaces the whole state field |
| Health check as graph node | Route around failures rather than failing the job |
| Full list replacement in state | Explicit deduplication control vs LangGraph's blind append reducer |
| Two LLM tiers (extraction vs reasoning) | 2050 calls at scale — Haiku/fast for volume, Sonnet/strong for quality-critical outputs |
| Neo4j for contradiction detection | Cypher handles "find conflicting clauses across all docs for the same party" — Python nested loops cannot |
| Mean pooling not CLS on legal-bert | legal-bert is MLM not contrastive-trained — CLS unreliable for similarity |
| Token-based chunking | legal-bert 512-token hard limit — character-based can exceed silently |
| bge-m3 sparse + dense hybrid | Dense catches semantics; sparse (learned weights) catches exact legal terms, section numbers, dollar amounts — better than BM25 TF-IDF for legal text |
| Qdrant-native sparse (no BM25 pickle) | Both signals filtered at index level by doc_id; no post-filter set intersection; no external file to manage |
| sparse_linear on CPU not MPS | MPS kernel launch overhead dominates for Linear(1024,1); CPU + bulk tensor transfer is 6x faster for this specific op |
| bge-m3 fp16 on MPS | 570M params — fp16 halves memory (2.24GB→1.12GB) and speeds up matrix multiplications; sparse output stays float32 for precision |
| RRF k=60 | Original paper constant — consensus mechanism, prevents single-signal dominance |
| candidate_k=20 before fusion | Ensures chunks ranked 6–20 in one list but 1 in the other aren't invisible to RRF |
| Groq fallback chain | Immediate failover to available capacity beats wait-and-retry |
| LiteLLM provider abstraction | Swap Groq→Anthropic in one `.env` line, zero code changes |
| `errors: list[str]` accumulation | One bad PDF shouldn't abort 49 others |
| `source_chunk_id` on ExtractedClause | Complete data lineage for citations: risk flag → clause → chunk → page → PDF |
| `source_clause_id` on RiskFlag | Carries `source_chunk_id` forward so Sprint 6 can trace any flag directly to a Qdrant chunk and PDF page without a secondary lookup |
| BM25 as pickle (not Elasticsearch) | Correct for ≤50 docs — avoid running another service; revisit at scale |
| Rules-first, LLM-second in risk scorer | Bright-line rules (missing clause = always high risk) need no LLM; saves tokens + latency for the 2050-call scale |
| Low-risk missing clauses suppressed (no flag) | Emitting a flag for every absent CUAD category drowns the report in noise — 20+ low-risk absences are normal for any contract type |
| Presence flags for inverted risk clauses | `Uncapped Liability` found is high risk; `found=False` is safe — standard dict lookup, no LLM needed |
| `LLM_ASSESSMENT_CATEGORIES` curated to 8 | LLM reasoning only where content nuance changes the risk call; keeps per-document LLM calls bounded |
| `ollama/mistral-nemo` default for reasoning | Local, unlimited, no rate limit — safe for any batch size; upgrade to OpenRouter free for small batches |
| Cypher queries scoped to `$doc_ids` | Graph accumulates across jobs — unscoped queries return cross-job contradictions. Each job only sees its own documents. |
| Template fallback for contradiction explanations | LLM failure should not drop a contradiction from the report — every `Contradiction` object is always complete |
| Formatter-first report synthesis | Deterministic tables can't be corrupted by the LLM — only narrative slots are LLM-generated |
| One LLM call for report narrative (JSON) | Bounded token cost; structured JSON output (executive_summary + recommended_actions) is parseable and auditable |
| `_template_narrative` fallback in report | Report generation fails after all other agents succeed — a crash here loses the full pipeline output; template ensures final_report is always populated |
| Q&A cross-doc via per-doc retrieve() calls | Reuses existing hybrid retriever without a second code path; merges by RRF score; acceptable overhead for ≤50 docs |
| Type-labelled HIGH flags in narrative prompt | Without `(MISSING CLAUSE)` / `(CONTENT RISK)` labels, LLM conflates contradiction risks with absent clauses — confirmed bug on first run |
| OpenRouter free tier for reasoning (≤25 docs) | 120B–405B models >> 7B local quality for nuanced legal risk; free but ~200 req/day limit caps it at small batches |
| `os.environ` propagation in config.py | LiteLLM reads env vars directly; pydantic-settings doesn't set them — config.py bridges the gap so agents need no manual os.environ calls |
| `deepseek/deepseek-r1:free` not used | Not available on this account's OpenRouter free tier; nemotron-120B is the verified working alternative |
| FastAPI BackgroundTasks (not Celery) | Dev/demo API — no broker infra needed; swap for task queue when persistence across restarts required |
| In-memory JOB_STORE | Jobs live for server lifetime; sufficient for interactive use; replace with Redis/Postgres for persistence |
| DELETE runs in BackgroundTasks | Qdrant + Neo4j cleanup can take seconds; return 204 immediately, clean up async |
| `.txt` support in loader.py | Same 50-line synthetic page grouping as DOCX — keeps metadata structure consistent downstream |

---

## Sprint 9 — FastAPI Layer

### Endpoints

| Method | Path | Behaviour |
|---|---|---|
| POST | `/jobs` | Multipart upload (1–50 files), starts pipeline in background, returns `{job_id, status: "pending"}` immediately (HTTP 202) |
| GET | `/jobs/{job_id}` | Poll status (`pending` / `running` / `done` / `error`); `report` field populated when `done` |
| POST | `/jobs/{job_id}/qa` | Q&A on a completed job — hybrid retrieval + LLM answer + citations; returns 409 if job not `done` |
| DELETE | `/jobs/{job_id}` | Removes Qdrant points (per doc_id), Neo4j Document+Clause nodes (DETACH DELETE) + orphan entity cleanup, temp files, JOB_STORE entry; returns 204; rejects if job is `running` |

### Architecture

```
POST /jobs
  │ create_job() — save bytes to tempdir, register in JOB_STORE
  │ BackgroundTasks.add_task(run_pipeline, job_id)
  └─► return 202 immediately

run_pipeline(job_id)  [background thread]
  For each file:
    load_document() → chunk_document() → embed_chunks() → index_chunks()
  graph.invoke({job_id, documents}) → GraphState
  record.report = final_report
  record.status = "done"
```

### Files

```
legal_due_diligence/api/
├── main.py      ← FastAPI app + all 4 endpoints + shutdown hook (close Neo4j driver)
├── schemas.py   ← JobStatus enum, JobResponse, QARequest, QAResponse, Citation
└── runner.py    ← JOB_STORE dict, create_job(), run_pipeline(), delete_job(),
                   _delete_qdrant(), _delete_neo4j() (orphan cleanup included)
```

### Start the server

```bash
source .venv/bin/activate
uvicorn legal_due_diligence.api.main:app --reload --port 8000
```

### Sprint 9 smoke test results

```
5/5 test groups passed
  ✓ Schema layer (JobResponse, QAResponse round-trip)
  ✓ Job creation (JOB_STORE, tmp_dir, doc_ids)
  ✓ Pipeline execution (status=done, report non-empty, Risk + Contradiction sections present)
  ✓ Q&A (answer non-empty, 6 citations, chunk_id present)
  ✓ Cleanup (JOB_STORE entry removed, tmp_dir removed)
Elapsed: 41.7s (2 contracts, local Ollama)
```

---

## Sprint 10 — Streamlit UI

### File
`legal_due_diligence/ui/app.py`

### UI states
```
UPLOAD  → no active job; file uploader (PDF/DOCX/TXT, multi-file) + submit button
RUNNING → polls GET /jobs/{id} every 5s; spinner; auto-advances when status='done'
DONE    → two tabs:
            📄 Report — full markdown brief rendered with st.markdown()
            💬 Q&A   — question input → POST /jobs/{id}/qa → answer + expandable citations
ERROR   → shows error list from state.errors, prompts to delete and retry
```

### Sidebar
- API URL config (default `http://localhost:8000`)
- Active job ID + doc list + status
- Delete button → `DELETE /jobs/{id}` + session state reset

### How to start (both servers required)

```bash
# API — must use venv Python directly (uvicorn pyenv shim does NOT use venv packages)
/path/to/legal_dd/.venv/bin/python -m uvicorn legal_due_diligence.api.main:app --port 8000

# UI
/path/to/legal_dd/.venv/bin/python -m streamlit run legal_due_diligence/ui/app.py
```

**Critical:** `uvicorn` on this machine is a pyenv shim — it uses pyenv's system Python, not the venv. Running it directly causes `ModuleNotFoundError: No module named 'fitz'` in background tasks. Always use `python -m uvicorn` via the venv Python.

### Sprint 10 smoke test results
Tested with a PDF (research paper) + dataset file:
- Upload → Running → Done flow ✅
- Report tab renders markdown brief ✅
- Q&A tab returns grounded answer + page-level citations ✅
- Q&A correctly refused to fabricate answers for out-of-scope questions ✅
- 0 risk flags / 0 contradictions (expected — docs were not contracts) ✅

---

## Sprint 13 — Multi-Query Retrieval for Hard Categories

### Problem
Per-category Recall@3 analysis on 1,244-row eval revealed categories with near-zero recall:
- **Covenant Not To Sue** (12%, n=24) — was completely absent from CUAD_CATEGORIES; agent never extracted it
- **IP Ownership Assignment** (17%), **Volume Restriction** (18%), **Post-Termination Services** (24%), **Minimum Commitment** (25%), **Exclusivity** (30%), **Revenue/Profit Sharing** (34%), **Change of Control** (35%), **Non-Compete** (35%)

Root cause from miss analysis: vocabulary mismatch — contracts use different surface forms than the category names or primary queries (e.g. "shall not challenge/contest/attack" instead of "covenant not to sue").

### Changes

**`agents/clause_extractor/prompts.py`**
- Added `"Covenant Not To Sue"` to `CUAD_CATEGORIES` (was missing — agent never extracted this clause from any contract)
- Sharpened primary queries for 6 hard categories based on actual ground-truth language from eval misses
- Added `CUAD_ALT_QUERIES: dict[str, list[str]]` — 10 categories × 2 alt queries each, grounded in real miss analysis

**`agents/clause_extractor/retriever.py`**
- `retrieve_multi(queries, doc_id, top_k, candidate_k)` — runs one full `retrieve()` per query, sums RRF scores across queries (chunks in multiple results get a consensus boost), returns merged top-k

**`agents/clause_extractor/agent.py`**
- `_extract_category_async()` checks `CUAD_ALT_QUERIES`; if found, calls `retrieve_multi()` with primary + alt queries and passes top 5 chunks to LLM (vs 3 for single-query categories)

**`eval/cuad_eval.py`**
- Imports `CUAD_ALT_QUERIES` from `prompts.py` (single source of truth, no duplication)
- `eval_retrieve_multi()` — same sum-RRF logic adapted for the eval collection
- `--multi-query` flag: enables multi-query path for hard categories during eval

### Eval status
**Pending** — 400-row eval (`python eval/cuad_eval.py --n 400 --enrich-queries --multi-query`) not yet completed due to context limit. Run this next session and compare against:
- Baseline (no multi-query): **R@3 = 52.1%** (1,244 rows)
- Expected: improvement on Covenant Not To Sue, IP Ownership Assignment, Minimum Commitment, Exclusivity

### CUAD_ALT_QUERIES categories
`Covenant Not To Sue`, `IP Ownership Assignment`, `Minimum Commitment`, `Exclusivity`, `Revenue/Profit Sharing`, `Non-Compete`, `Change of Control`, `Post-Termination Services`, `Volume Restriction`, `Joint IP Ownership`

---

## Sprint 12 — Async Extraction Pipeline

### What changed
`agents/clause_extractor/agent.py` — full async rewrite of the extraction hot path.

**Before (Sprint 2):** sequential per doc, sequential per category
```
50 docs × 41 cats × ~300ms Groq latency = ~10 min
```

**After (Sprint 12):** all docs × all categories concurrent, bounded by semaphore
```
wall time ≈ ceil(41 / extraction_concurrency) × 300ms ≈ 1.5s per batch
50 docs  ≈ 3–10s total (Groq rate limits are now the ceiling, not latency)
```

### Design
- `_call_llm_async()` — `litellm.acompletion()` with same fallback chain
- `_extract_category_async(clause_type, query, doc_id, sem)` — retrieval via `asyncio.to_thread()` (Qdrant + bge-m3 stays sync, offloaded to thread pool); LLM under semaphore
- `_extract_document_async(doc_id, sem)` — 41 categories via `asyncio.gather()`
- `_run_extraction_async(docs)` — all unprocessed docs via `asyncio.gather()`, shared semaphore
- `clause_extractor_node()` — stays sync, calls `asyncio.run()` (safe: runs in FastAPI BackgroundTasks thread, no parent event loop)

### Config
`core/config.py` — new field:
```python
extraction_concurrency: int = Field(default=10)  # EXTRACTION_CONCURRENCY in .env
```
Global semaphore caps total concurrent LLM calls regardless of doc count. 10 is safe for Groq free tier + Ollama fallback. Lower to 5 if seeing frequent 429s.

### Interface unchanged
`clause_extractor_node(state)` signature identical — LangGraph wiring untouched.
`_call_llm()` (sync) kept for legacy callers.

---

## Sprint 11 — HyDE Investigation (Hypothetical Document Embeddings)

### Result: HyDE hurts — disabled, keeping `use_hyde=False` default

| Setup | R@1 | R@3 | Rows |
|---|---|---|---|
| bge-m3 hybrid baseline | 33.3% | **52.1%** | 1244 |
| +HyDE (50-row sanity) | 32.0% | 48.0% | 50 |
| +HyDE (400-row confirm) | 26.7% | **40.8%** | 360 |

**Why HyDE hurt:** `llama-3.1-8b-instant` generates generic boilerplate clauses that shift the dense query embedding *away* from the specific contract language in indexed chunks. bge-m3 already handles semantic variation well via its learned sparse weights — HyDE adds noise, not signal.

### Code wired up but off by default
- `retriever.py`: `retrieve(..., hyde=False)` — `_hyde_expand()` present, never called unless explicitly set
- `agent.py`: passes `hyde=settings.use_hyde` — default False
- `config.py`: `use_hyde: bool = Field(default=False)` — opt-in via `USE_HYDE=true` in `.env`
- `eval/cuad_eval.py`: `--hyde` flag for future experiments with better generator models

**Retrieval ceiling remains: R@3 = 52.1%** (bge-m3 hybrid, no reranker, no HyDE)

---

## Sprint 8 eval — Full dataset results (post-Sprint 9 investigation)

100-row stratified sample understated real performance. Full 1,244-row eval:

```
Model   : BAAI/bge-m3
Enrich  : yes (production queries)
Sample  : 1244 rows (full test set)
Recall@1: 33.3%  (414/1244)
Recall@3: 52.1%  (648/1244)   ← up from 42% reported on 100-row sample
Time    : 89.6s
```

### Per-category Recall@3 (full 1,244-row eval, worst → best)

| Category | R@3 | n |
|---|---|---|
| Most Favored Nation | 0% | 3 |
| Covenant Not To Sue | 12% | 24 |
| Joint IP Ownership | 14% | 7 |
| Non-Disparagement | 14% | 7 |
| IP Ownership Assignment | 17% | 23 |
| Volume Restriction | 18% | 17 |
| Post-Termination Services | 24% | 29 |
| No-Solicit of Customers | 29% | 7 |
| Exclusivity | 30% | 33 |
| … (middle band 33–68%) | … | … |
| Anti-Assignment | 68% | 72 |
| Insurance | 72% | 32 |
| Renewal Term | 73% | 26 |
| Agreement Date | 74% | 93 |
| Parties | 87% | 102 |
| Source Code Escrow | 100% | 1 |

**Query enrichment expanded** for low-recall categories (Sprint 9 investigation):
- `Covenant Not To Sue` — added to `_CUAD_QUERY_ENRICHMENT` (was missing entirely): `"covenant not to sue release claims waive right to bring action not assert legal claims discharge"`
- `IP Ownership Assignment`: added `"all right title interest vests assigns work product inventions"`
- `Joint IP Ownership`: added `"jointly developed co-developed each party owns shared ownership"`
- `Volume Restriction`: added `"not to exceed seats users copies units annual cap"`
- `Post-Termination Services`: added `"following termination survival obligations continue after termination data return"`
- `Non-Disparagement`: added `"derogatory defamatory comments refrain from speaking negatively"`

**Finding:** query enrichment is not the binding constraint for these categories — bge-m3 learned sparse weights already handle vocabulary variation. The remaining low-recall categories have inconsistent surface forms in CUAD passages. Further gains require chunking-level or retrieval-architecture changes (HyDE is the next untested option).

---

## Sprint 7 Plan

### Goal
Establish Recall@K baseline → swap embedding model → measure improvement → harden pipeline.

### Step 1 — DeepEval eval harness (eval/cuad_eval.py)

**Dataset:** `chenghao/cuad_qa` (Parquet, datasets 4.x compatible)
- 1,240 test rows | schema: `context`, `question`, `answers.text`, `answers.answer_start`
- Each row = one CUAD question against one contract passage

**What to measure:**

| Metric | Tool | What it validates |
|---|---|---|
| Recall@K (K=3) | manual | Does the correct chunk appear in top-3 retrieved? |
| Faithfulness | DeepEval `FaithfulnessMetric` | Does LLM answer stay grounded in retrieved chunks? |
| Answer Relevancy | DeepEval `AnswerRelevancyMetric` | Is the answer on-topic for the question? |
| Contextual Recall | DeepEval `ContextualRecallMetric` | Do retrieved chunks cover the ground-truth answer span? |

**Eval harness flow:**
```
for each test row in cuad_qa["test"]:
  1. Index context into Qdrant (temp collection, flushed after eval)
  2. retrieve(question, doc_id, top_k=3) → chunks
  3. Recall@K: ground_truth answer_start in any chunk's text span? → hit/miss
  4. Call clause extractor LLM with chunks + question
  5. DeepEval: LLMTestCase(input=question, actual_output=answer, retrieval_context=chunks)
  6. Run FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric
```

**Practical scoping:** Full 1,240-row test set = ~1,240 LLM calls. Run a 100-row stratified sample first (sample across all 41 categories). Full run once baseline is confirmed.

**Output:** `eval/results/baseline_legal_bert.json` — per-metric scores + per-row breakdown.

### Step 2 — Embedding model swap

**Swap:** `nlpaueb/legal-bert-base-uncased` → `BAAI/bge-base-en-v1.5`
- legal-bert is domain-aware but task-mismatched (MLM ≠ retrieval)
- bge-base trained with contrastive loss specifically for dense retrieval
- Also 768-dim → `embedding_dim` in config unchanged
- Change: one line in `core/config.py` + `docker compose down -v` + re-index

**Then:** re-run same eval harness → `eval/results/baseline_bge.json` → compare Recall@K delta.

### Step 3 — Hardening (post-eval)

- Per-node timing instrumentation (`time.perf_counter` in each agent)
- Embedding latency logged in `embedder.py`
- `state.errors` surfaced in report when non-empty (currently logged but not shown in brief)

### Sprint 7 file plan
```
legal_dd/
├── eval/
│   ├── cuad_eval.py          ← main eval harness ✅ BUILT
│   ├── results/
│   │   ├── baseline_legal_bert.json   ← populated after first full run
│   │   └── baseline_bge.json          ← populated after embedding swap
│   └── sample_ids.json       ← 100-row stratified sample indices (reproducible) ✅ GENERATED
└── run_sprint7.py            ← smoke test: 5-row sanity check, no full eval run ✅ PASSES
```

### Sprint 7 smoke test results (dataset + sampling, --skip-qdrant)
```
Test 1 — Dataset loading:    1244 test rows, correct schema ✓
Test 2 — Stratified sampling: 5-row sample, reproducible, file round-trip ✓
Note: 40 question groups found (not 41 — one CUAD category absent from test split)
```

### Sprint 7 baseline results (legal-bert)
```
Model   : nlpaueb/legal-bert-base-uncased
Sample  : 100 rows, stratified across 40 CUAD question groups
Recall@1: 4.0%   (4/100)
Recall@3: 9.0%   (9/100)
Output  : eval/results/baseline_legal_bert.json
```
Low recall confirms legal-bert (MLM, not retrieval-tuned) is a poor embedding choice for dense retrieval.
This is exactly what we expected — the baseline establishes the floor.

### Sprint 7 bge-base results + comparison ✅ DONE

| Metric    | legal-bert (MLM) | bge-base (contrastive) | Delta |
|-----------|-----------------|----------------------|-------|
| Recall@1  | 4.0%            | 11.0%                | +7pp  |
| Recall@3  | 9.0%            | 15.0%                | +6pp  |

bge-base is **2.75× better at Recall@1** than legal-bert on the same 100-row sample.

### Sprint 7 query prefix + top_k investigation ✅ DONE

All runs on 100-row stratified sample, `BAAI/bge-base-en-v1.5`:

| Config | Recall@1 | Recall@3 | Recall@5 |
|--------|----------|----------|----------|
| no prefix, top_k=3 (baseline) | 11% | 15% | — |
| no prefix, top_k=5 | 11% | 15% | 16% |
| prefix, top_k=3 | **12%** | 15% | — |
| prefix, top_k=5 | **12%** | 15% | 16% |

**Findings:**
- Query prefix (`"Represent this sentence for searching relevant passages: "`) gives +1pp Recall@1 — better ranking, most relevant chunk surfaces first
- top_k=5 adds only +1pp Recall@5 at cost of 67% more LLM context per query — not worth it for clause extraction (top_k=3 in agent.py stays)
- prefix is asymmetric: applied to queries only, never to indexed chunks

**Code changes made:**
- `core/config.py`: added `embedding_query_prefix` field (default = bge prefix)
- `agents/clause_extractor/retriever.py`: `_embed_query()` now prepends `settings.embedding_query_prefix`
- `eval/cuad_eval.py`: added `--query-prefix` and `--top-k` CLI flags; `embed_questions` and `run_eval` updated to accept them; auto-generated output filenames encode config

### Sprint 7 bge-large investigation ✅ DONE

Tested `bge-large-en-v1.5` (335M params, 1024-dim) vs `bge-base-en-v1.5` (109M params, 768-dim).

**Full comparison — all configs, 100-row stratified sample:**

| Config | Recall@1 | Recall@3 |
|--------|----------|----------|
| bge-base, no prefix | 11% | 15% |
| **bge-base, prefix** | **12%** | **15%** ← production config |
| bge-large, no prefix | 10% | 12% |
| bge-large, prefix | 11% | 12% |

**Verdict: bge-large is worse.** bge-large scores 3pp below bge-base at Recall@3 even with prefix.

Why: CUAD eval passages are short (200–800 words → 1–2 chunks each). bge-large's extra capacity is tuned for longer, more diverse retrieval tasks. On short legal answer spans ("30 days", "$2M"), the base model's inductive biases win. Larger ≠ better on all tasks — empirical validation matters.

**bge-m3** (sparse+dense hybrid, replaces BM25 with learned sparse weights) is the next meaningful upgrade but requires Qdrant `SparseVector` schema changes → deferred to Sprint 8.

**Final production config (config.py):**
- `embedding_model = "BAAI/bge-base-en-v1.5"` (768-dim)
- `embedding_query_prefix = "Represent this sentence for searching relevant passages: "` (default, applied to queries only)

### Sprint 7 cross-encoder reranker investigation ✅ DONE

Added `BAAI/bge-reranker-base` cross-encoder reranking after RRF fusion. Findings:

**Key discovery — eval query mismatch:** The CUAD eval uses bare category labels ("No-Solicit Of Employees") as retrieval queries, while production uses keyword-rich queries ("no solicit employees hire recruit personnel"). Without enrichment, the reranker on bare labels HURTS recall (9%) vs baseline (15%) because the cross-encoder has too little signal to compare.

**With enriched queries (production vocabulary) + reranker:**

| Config | Recall@1 | Recall@3 |
|--------|----------|----------|
| bge-base + prefix (baseline) | 12% | 15% |
| + reranker, bare labels | 9% | 9% |
| + enrich queries only | 12% | 14% |
| + enrich + reranker, ck=20 | 13% | 14% |
| **+ enrich + reranker, ck=50** | **14%** | **15%** |

**Why candidate_k=50 matters:** CUAD eval docs average 42 chunks each. candidate_k=20 covers only ~47% of chunks — the answer chunk may simply never reach the reranker. At ck=50, the full document is in the candidate pool and the reranker can correctly surface the answer.

**Result:** R@1 +2pp (12%→14%) with no R@3 regression. The reranker improves precision-at-1 — the right chunk surfaces first more often.

**Production impact:**
- `config.py`: added `reranker_model` field (default `""` = disabled)
- `retriever.py`: reranker applied after RRF when `settings.reranker_model` is set (lazy singleton load, MPS device)
- `cuad_eval.py`: `--reranker`, `--candidate-k`, `--enrich-queries` flags added
- Production queries in `prompts.py` CUAD_CATEGORIES are already keyword-rich — `--enrich-queries` makes eval representative of production

**Why reranker is disabled by default in production:**
At 50 docs × 41 categories = 2050 retrieval calls, each cross-encoder call adds ~2s for 20-50 candidate pairs. Total latency: ~1h vs ~10min without. Enable for small batches (≤5 docs) or when recall precision matters over throughput.

**bge-m3** (sparse+dense hybrid, replaces BM25 with learned sparse weights) is the next meaningful upgrade but requires Qdrant `SparseVector` schema changes → deferred to Sprint 8.

### Next steps for Sprint 7 — ALL DONE ✅
1. ~~Run `docker compose up -d` then `python run_sprint7.py`~~ ✅
2. ~~Run `python eval/cuad_eval.py` → baseline_legal_bert.json~~ ✅
3. ~~Swap embedding model, re-index, re-run → baseline_bge.json~~ ✅
4. ~~Investigate bge query prefix + top_k=5~~ ✅
5. ~~Investigate bge-large-en-v1.5~~ ✅ — bge-base + prefix is best bi-encoder config
6. ~~Investigate cross-encoder reranking~~ ✅ — R@1 +2pp with enriched queries + ck=50
7. ~~Re-index production data~~ ✅ — superseded by Sprint 8 bge-m3 re-index

### Sprint 7 competitor research → Sprint 8 backlog items

Reviewed three open-source legal contract analysis repos for recall improvement techniques:
- `arnienemeth/Copilot-studio-contract-analyses-agent-project` — Azure/no-code, no custom retrieval
- `Sreeja2002-Andela/Contract-Clause-Extraction-Analysis-Engine` — custom Python RAG on CUAD, most relevant
- `deacs11/CrewAI_Contract_Clause_Risk_Assessment` — full-text pass to GPT-4, no RAG

**Two techniques identified for Sprint 8:**

**1. Heading-merge chunking** (`chunker.py`)
Legal contracts have structure: `"12. INDEMNIFICATION\n\nEach party shall indemnify..."`.
Current token-window chunker splits at boundaries — "INDEMNIFICATION" lands at chunk tail, obligation text starts the next chunk without that keyword. BM25 query for "indemnification" scores zero on the obligation chunk.
Fix: before token-chunking, merge any paragraph `< ~80 chars` (clause heading) forward into the next paragraph so heading keyword co-occurs with obligation text.

**2. Hyphen-preserving BM25 tokenizer** (`retriever.py` + `cuad_eval.py`)
Current tokenizer: `query.lower().split()` → splits `"force-majeure"` into `["force", "majeure"]`.
A BM25 query string `"force-majeure"` becomes one token that never matches the split corpus.
Fix: use `re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())` to preserve hyphenated legal terms as single tokens (`"non-compete"`, `"most-favored-nation"`, `"work-for-hire"`, `"force-majeure"`).

Implementation order: BM25 tokenizer first (no re-index needed, testable immediately), then heading-merge (requires `docker compose down -v` + re-index).

**Results after implementing both (Sprint 8 baseline):**

| Config | R@1 | R@3 |
|--------|-----|-----|
| bge-base + prefix (Sprint 7 baseline) | 12% | 15% |
| + enriched queries (production vocab) | 12% | 14% |
| + enriched + reranker ck=50 | 14% | 15% |
| **+ BM25 hyphen fix + heading merge + enriched** | **13%** | **16%** ← best R@3 |
| + all above + reranker ck=50 | 14% | 15% |

**Key finding:** The BM25 tokenizer fix + heading merge together add +1pp Recall@3 (15%→16%) without a reranker. Adding the reranker on top cancels this gain — the cross-encoder reorders some rank-2/3 hits out of top-3. Conclusion: use BM25 fix + heading merge for best R@3 (coverage), add reranker only when R@1 precision matters (e.g. single-doc interactive use).

**Production config after Sprint 7 (bge-base baseline, superseded by Sprint 8):**
- `embedding_model = "BAAI/bge-base-en-v1.5"` (768-dim), `embedding_query_prefix = "Represent this sentence for searching relevant passages: "`
- BM25 tokenizer: hyphen-preserving (`re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())`) — in `retriever.py:bm25_tokenize()`
- Heading merge: short paragraphs (≤80 chars) prepended to next paragraph before chunking — in `chunker.py:_merge_headings()`
- `reranker_model = ""` (disabled by default); enable with `BAAI/bge-reranker-base` + candidate_k=50 for small batches

**Current production config (Sprint 8 — bge-m3):**
- `embedding_model = "BAAI/bge-m3"` (1024-dim dense + learned sparse)
- `embedding_dim = 1024`
- `sparse_vector_name = "sparse"` (Qdrant named sparse vector field)
- `embedding_query_prefix = ""` (empty — bge-m3 handles asymmetry internally)
- No BM25 pickle — sparse retrieval is Qdrant-native
- Re-index required: `docker compose down -v && docker compose up -d && python run_sprint1.py`
- `reranker_model = ""` (disabled by default; unchanged from Sprint 7)

---

## Sprint 8 Plan

### Goal
Replace bge-base-en-v1.5 (768-dim dense) + BM25 pickle with bge-m3 (1024-dim dense + learned sparse vectors stored natively in Qdrant). Eliminate the external BM25 pickle entirely.

### Why bge-m3 over bge-base + BM25

| Signal | bge-base + BM25 | bge-m3 |
|---|---|---|
| Dense | 768-dim, contrastive-trained | 1024-dim, stronger model |
| Sparse | BM25 TF-IDF weights, global pickle | Learned SPLADE-style weights, Qdrant-native |
| Doc filtering | Dense: index-level; BM25: post-filter set intersection | Both: index-level filter, no post-processing |
| File management | `bm25_index.pkl` on disk | No external file |
| Legal term weights | "notwithstanding" = same weight as "the" in TF-IDF | Learned — model knows which legal terms matter |

### Key implementation details

**Why FlagEmbedding is not used:**
FlagEmbedding ≥1.2 requires symbols removed in transformers 5.x (`is_torch_fx_available`, `GEMMA2_START_DOCSTRING`). All tested versions (1.3.5, 1.2.11, 1.1.9) failed to import. Implemented bge-m3 directly:
- Base encoder: `AutoModel.from_pretrained('BAAI/bge-m3')` (XLM-RoBERTa)
- Sparse head: `hf_hub_download(repo_id='BAAI/bge-m3', filename='sparse_linear.pt')` — Linear(1024→1) trained weights

**Qdrant named vector schema:**
```python
client.create_collection(
    collection_name=...,
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()},
)
```

**Sparse query API (qdrant-client 1.17.x):**
```python
# CORRECT — query_points() accepts SparseVector + using=
client.query_points(
    query=SparseVector(indices=[...], values=[...]),
    using="sparse",
    ...
)
# WRONG — NamedSparseVector is NOT accepted by query_points()
```

**Performance (M4 MPS, 16GB):**
- bge-m3 base encoder: fp16, 1.12GB VRAM
- sparse_linear: kept on CPU — MPS kernel launch overhead dominates for Linear(1024,1); bulk CPU transfer is 6x faster
- batch_size=24: ~140ms/chunk (vs 271ms at batch 12); MPS→CPU sync is ~3s fixed cost, amortised by larger batches

### Files changed in Sprint 8

| File | Change |
|---|---|
| `core/config.py` | `embedding_model` → `"BAAI/bge-m3"`, `embedding_dim` → 1024, added `sparse_vector_name`, cleared `embedding_query_prefix` |
| `ingestion/embedder.py` | Full rewrite: `AutoModel` + `sparse_linear.pt`; `EmbeddedChunk` now has `sparse_vector: dict[int, float]`; fp16 on MPS; batch_size=24 |
| `ingestion/indexer.py` | Full rewrite: named dense+sparse collection schema; upsert both vector types per point; removed all BM25 code |
| `agents/clause_extractor/retriever.py` | Full rewrite: `_sparse_ranks()` added; `_bm25_ranks()` / `_load_bm25()` / `_get_doc_chunk_ids()` removed; `bm25_tokenize()` kept as legacy export; `_embed_query()` returns `(dense, sparse)` tuple |
| `eval/cuad_eval.py` | Updated collection schema; `index_eval_rows()` upserts both vectors; `embed_questions()` returns `(dense, sparse)` per query; `eval_retrieve()` uses `SparseVector + using=` |
| `run_sprint1.py` | Added `using="dense"` to test query; assertion updated `768→1024`; added sparse vector assertion |

### Sprint 8 eval results ✅ DONE

```bash
python eval/cuad_eval.py --enrich-queries
```

| Config | Recall@1 | Recall@3 |
|--------|----------|----------|
| bge-base + prefix (Sprint 7 best) | 12% | 15% |
| bge-base + BM25 fix + heading merge (Sprint 7/8 best) | 13% | 16% |
| **bge-m3, enriched queries, ck=20 (Sprint 8)** | **29%** | **42%** |

**Result: bge-m3 is the largest single improvement in the project.**
- Recall@1: 12% → 29% (+17pp, 2.4× improvement)
- Recall@3: 15% → 42% (+27pp, 2.8× improvement)

**Why bge-m3 outperforms bge-base + BM25 by this margin:**
- Dense: 1024-dim vs 768-dim, stronger model trained on more data
- Sparse: learned SPLADE-style weights vs hand-crafted TF-IDF — the model knows which legal terms are discriminative without manual tokenizer tuning
- Both signals from the same model, trained end-to-end for retrieval — they are complementary in a way that bge-base + BM25 (different training objectives) was not

**Full benchmark progression (100-row stratified CUAD sample):**

| Config | R@1 | R@3 | Note |
|--------|-----|-----|------|
| legal-bert (MLM baseline) | 4% | 9% | Sprint 7 floor |
| bge-base, no prefix | 11% | 15% | Sprint 7 |
| bge-base + prefix | 12% | 15% | Sprint 7 production config |
| bge-base + BM25 fix + heading merge | 13% | 16% | Sprint 7/8 best bge-base |
| bge-m3, enriched, ck=20 | **29%** | **42%** | Sprint 8 ✅ current production |
| bge-m3, enriched, ck=50 | 28% | 43% | +1pp R@3, -1pp R@1 — within noise |

**candidate_k=50 verdict:** No meaningful improvement over ck=20 without a reranker. The 1pp R@3 gain and 1pp R@1 loss are within 100-row sampling noise (= 1 hit difference). bge-m3's retrieval quality is sufficient that the answer chunk is already in the top-20 when it's retrievable at all. **candidate_k=20 confirmed as production default.**

### Next steps for Sprint 8 — ALL DONE ✅
1. ~~Re-index production data with bge-m3~~ ✅
2. ~~Run eval → bge-m3 baseline~~ ✅ — R@1=29%, R@3=42%
3. ~~Fix run_sprint7.py for bge-m3 architecture~~ ✅
4. ~~Run candidate_k=50 eval~~ ✅ — no improvement, ck=20 confirmed
5. ~~Update sprint plan table~~ ✅

---

## Common Commands

```bash
# Environment
source /path/to/legal_dd/.venv/bin/activate

# Infrastructure
docker compose up -d
docker compose down -v && docker compose up -d   # full reset

# Smoke tests
python run_sprint0.py     # LangGraph graph runs end-to-end
python run_sprint1.py     # ingestion pipeline + Qdrant retrieval verify
python run_sprint3.py     # risk scorer (deterministic + LLM, requires ollama serve)
python run_sprint3.py --skip-llm  # deterministic rules only
python run_sprint4.py     # entity mapper (requires Neo4j running)
python run_sprint4.py --skip-neo4j  # unit/extractor tests only
python run_sprint5.py     # contradiction detector (requires Neo4j running + Ollama)
python run_sprint5.py --skip-neo4j  # stub test only
python run_sprint6.py     # report synthesis + Q&A (requires Ollama + Qdrant)
python run_sprint6.py --skip-llm   # formatter only, no LLM/Qdrant required
python run_sprint6.py --skip-qa    # formatter + report, skip Q&A

# Reset index (after docker compose down -v)
python run_sprint1.py   # no pickle to delete — sparse vectors live in Qdrant

# Check running containers
docker compose ps
```

---

## Known Issues

| Issue | Impact | Fix |
|---|---|---|
| HuggingFace unauthenticated rate limit warning | Cosmetic noise in logs | Set `HF_TOKEN` env var |
| bge-m3 first load downloads `sparse_linear.pt` from HuggingFace hub | ~1s delay on cold start, requires network | `hf_hub_download` caches automatically; subsequent loads instant |
| FlagEmbedding ≥1.2 incompatible with transformers 5.x | `ImportError: cannot import name 'is_torch_fx_available'` | Not used — bge-m3 implemented directly via `AutoModel` + `hf_hub_download` |
| `qdrant_client.query_points()` does NOT accept `NamedSparseVector` | `ValueError: Unsupported query type` | Use `SparseVector(indices=..., values=...) + using="sparse"` instead |
| Qdrant collection must be recreated after Sprint 8 re-index | Old collection has 768-dim dense only, no sparse vectors | `docker compose down -v && up -d && python run_sprint1.py` |
| `theatticusproject/cuad` dataset broken in datasets 4.x | ~~Can't use for Sprint 7 evals yet~~ **RESOLVED** | Use `chenghao/cuad_qa` — native Parquet, all 41 categories, datasets 4.x compatible: `load_dataset("chenghao/cuad_qa")` |
| Groq free tier 6000 TPM per model | Slows extraction without fallback chain | Fallback chain now in place |
| 2 chunks per sample contract (too short) | Retrieval less precise than production | Will resolve with real PDFs |
| `AutoTokenizer`/`AutoModel` Pylance red lines | Metaclass factories — `.encode()`, `.decode()`, `__call__()` not resolvable | Fixed: typed as `PreTrainedTokenizerBase` / `PreTrainedModel` in chunker.py and embedder.py |
| `page.get_text()` Pylance red line in loader.py | pymupdf stubs type as `str \| dict` — `.strip()` on non-str | Fixed: `str(page.get_text("text")).strip()` |
| `response.choices[0].message.content` red lines (retriever, report agent, qa, risk scorer) | LiteLLM stubs type content as `str \| None` | Fixed: `# type: ignore[union-attr]` on each call site |
| `graph.compile()` return type red line in graph.py | Returns `CompiledStateGraph`, not `StateGraph` — Pylance can't resolve | Fixed: `build_graph() -> Any` + `from typing import Any` |
| `_model.half()` / `_model.to()` red lines in embedder.py | `AutoModel` methods return `nn.Module`, not `AutoModel` | Fixed: `# type: ignore[assignment]` on both lines |
| `tokenizer.decode()` red line in chunker.py | Stubs type return as `str \| list[str]`; `chunks.append()` expects `str` | Fixed: `chunk_text: str = tokenizer.decode(...)  # type: ignore[assignment]` |
| `citations` red line in main.py (lines 161, 169) | List comprehension produced `list[dict]`; `QAResponse.citations` expects `list[Citation]` | Fixed: construct `Citation(...)` objects explicitly; import `Citation` from `api.schemas` |
| `from typing import Any` missing in main.py | `result: dict[str, Any]` annotation used `Any` without importing it | Fixed: added `from typing import Any` to imports |
| `graph.invoke()` arg-type red line in runner.py | Compiled graph stub doesn't accept plain `dict` — expects typed `GraphState` | Fixed: `# type: ignore[arg-type]` |





### Every Agent breakdown

**Agent1:Health Check**

Reads: GraphState — just job_id and documents to know the job exists. No domain data yet.
Writes: {"qdrant_ready": bool, "neo4j_ready": bool, "status": "checked"} — two boolean flags back into state.
Contract downstream: Clause Extractor reads qdrant_ready. Entity Mapper and Contradiction Detector read neo4j_ready. These flags gate everything that follows.
Transition: Sets the two infrastructure flags. Orchestrator reads them immediately after and routes conditionally — if qdrant_ready=False, skip to Report Agent.
If failure: Pings both services, catches connection errors, sets the flag to False. Never crashes. The job continues with reduced capability.

**Agent2:ClauseExtractor**
- Reads: state.documents — list of DocumentRecord. For each doc, it queries Qdrant dense ("dense" named vector) and Qdrant sparse ("sparse" named vector) using 41 CUAD category queries. Both vectors produced by bge-m3 in one forward pass.
- Writes: {"extracted_clauses": list[ExtractedClause], "status": "extracted"} — 41 × N objects where N is document count.
Format of each object written:
- ExtractedClause {
    document_id, clause_type, found: bool,
    clause_text, normalized_value,
    confidence: float, source_chunk_id: UUID
  }
- Contract downstream: Risk Scorer reads extracted_clauses directly. Entity Mapper also reads extracted_clauses. Both depend on this output being complete and correctly typed.
- Transition: Flat list of all extracted clauses for all documents lands in state. Pipeline moves to Risk Scorer unconditionally.
- If failure: Any single LLM call failure → that clause gets found=False, confidence=0.0. The clause still exists in the list — it just looks like a missing clause. Risk Scorer treats it conservatively as a risk. The job never stops.

**Agent3:RiskScorer**
- Reads: state.extracted_clauses — the full list[ExtractedClause] written by the Clause Extractor.
- Writes: {"risk_flags": list[RiskFlag], "status": "scored"} back into state.
Format of each object written:
- RiskFlag {
    document_id, clause_type,
    risk_level: "high"|"medium"|"low",
    reason: str,
    is_missing_clause: bool,
    source_clause_id: str | None
  }
- Contract downstream: Report Agent reads risk_flags to build the risk table. source_clause_id on each flag enables Sprint 6 citation tracing.
- Transition: Deterministic rules run first — O(1) dict lookups, no LLM. Then LLM reasoning runs only for 8 curated categories where content nuance matters. After both passes, risk_flags is written to state. Orchestrator routes to Entity - Mapper or Contradiction Detector based on neo4j_ready.
- If failure: Rules layer never fails — pure Python logic. LLM reasoning failure on a specific clause → that clause gets no LLM flag (conservative). The rule-based flags still fire normally.

**Agent4:EntityMapper**
- Reads: state.extracted_clauses — same list the Risk Scorer read. Does not read risk_flags.
- Writes: {"graph_built": True, "status": "mapped"} — just a boolean flag. The actual data goes into Neo4j, not state.
- Contract downstream: Contradiction Detector checks state.graph_built before querying Neo4j. If False, it skips querying entirely. This is the handshake between these two agents.
- Transition: Reads all extracted clauses, writes Document + Clause + entity nodes to Neo4j via MERGE (idempotent). Flips graph_built=True. Contradiction Detector runs next.
- If failure: If Neo4j write fails mid-batch, appends to state.errors, sets graph_built=False. Contradiction Detector sees False, returns empty contradictions list, report notes the skip. Partial graph writes are harmless because MERGE is idempotent — re-running doesn't duplicate.

**Agent5:ContradictionDetector**
- Reads: state.graph_built (bool) and state.documents (for doc_ids scoping). Does not read extracted_clauses from state — it queries Neo4j directly via Cypher.
- Writes: {"contradictions": list[Contradiction], "status": "detected"} back into state.
Format of each object written:
- Contradiction {
    clause_type,
    document_id_a, document_id_b,
    value_a: str, value_b: str,
    explanation: str
  }
- Contract downstream: Report Agent reads state.contradictions to build the contradiction table. The explanation field is lawyer-readable prose.
- Transition: Opens one Neo4j session, runs two Cypher queries (value conflicts + absence conflicts), closes session. Then calls LLM per contradiction for explanation. Writes completed list to state. Report Agent runs next unconditionally.
- If failure: graph_built=False → returns contradictions=[] immediately, no Neo4j query. LLM explanation failure per contradiction → template string fills explanation. Every Contradiction object is always complete — the field is never empty.

**Agent6:Report+Q&A**
- Reads: Everything — state.risk_flags, state.contradictions, state.extracted_clauses, state.documents, state.errors. This is the only agent that reads the full state.
- Writes: {"final_report": str, "status": "complete"} — a single markdown string.
- Contract downstream: Nothing. This is the terminal agent. The markdown string is what the lawyer reads.
- Transition: Formatter builds deterministic tables in pure Python first. One LLM call generates the narrative JSON {"executive_summary": "...", "recommended_actions": [...]}. Template fills two slots in fixed markdown structure. Done.
- If failure: LLM narrative failure → _template_narrative() generates a factually accurate generic summary from state counts. final_report is always populated. The pipeline never ends without a report.

**JSONIn/JSONOutSummary**
- This is the most important thing to understand about your system's data flow.
- Who receives JSON as input:
- The Clause Extractor is the primary JSON consumer. The LLM returns a JSON object per clause — {"found": bool, "clause_text": "...", "normalized_value": "...", "confidence": 0.9}. The agent parses that JSON into an ExtractedClause Pydantic model. This is the only place raw LLM JSON enters the pipeline.
- The Risk Scorer's LLM reasoning pass also receives JSON — {"flag": bool, "risk_level": "medium", "reason": "..."} — per clause assessed.
- The Report Agent receives JSON from the narrative LLM call — {"executive_summary": "...", "recommended_actions": [...]}.
- Who produces JSON as output:
- The Clause Extractor transforms raw LLM JSON → validated ExtractedClause Pydantic objects. That's the JSON-to-structured-data conversion point.
- The Risk Scorer transforms raw LLM JSON → RiskFlag Pydantic objects.
- The Contradiction Detector transforms Cypher query results (Neo4j records, not JSON) + LLM explanation text → Contradiction Pydantic objects.
- The key insight: JSON only appears at the LLM boundary — going in as a prompt instruction and coming out as a raw string. Every agent immediately validates and converts that raw string into a typed Pydantic model before writing to state. State never contains raw JSON strings. It only contains typed Python objects. That's why the system is robust to partial LLM failures — validation happens at the boundary, not deep inside the pipeline.