# Contributing to LexGraph-DD

This is a research prototype under active development. Contributions are welcome, but please read this first — the project has strong opinions about what belongs in the pipeline and a rigorous benchmarking discipline.

---

## Ground rules

- **Benchmark everything retrieval-related.** Any change to the chunker, embedder, retriever, query enrichment, or LLM chain must include before/after Recall@3 or Conditional F1 numbers. The project has a 30× cached eval harness specifically to make this low-friction.
- **One agent, one responsibility.** Do not add cross-agent logic. The LangGraph state machine passes `GraphState` between nodes — if two agents need to share logic, extract it to `core/`.
- **No silent fallbacks.** If something fails, it goes in `state.errors`. Never swallow exceptions to make the pipeline look clean.
- **No backwards-compatibility shims.** If you rename something, rename it everywhere. Unused aliases rot.
- **Temperature=0 for extraction.** Deterministic extraction = reproducible evals. Do not change this without a documented reason.

---

## Development setup

```bash
git clone https://github.com/pulkitku2004-ai/LexGraph-DD.git
cd LexGraph-DD

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add OPENAI_API_KEY (required), GROQ_API_KEY (fallback)

docker compose up -d    # Qdrant + Neo4j

python run_sprint1.py   # smoke test: ingestion + retrieval
python run_sprint9.py   # smoke test: full API lifecycle
```

---

## Running evals

```bash
# Retrieval — Recall@K on CUAD benchmark (chenghao/cuad_qa, 1244 rows)
python eval/cuad_eval.py --n 400 --enrich-queries --multi-query

# Per-category breakdown
python analyze_categories.py eval/results/<result_file>.json

# Extraction quality — Token F1 + Conditional F1 (192 rows)
python eval/e2e_eval.py --n 200 --enrich-queries --multi-query
```

**Canonical benchmarks** (do not regress):
- Retrieval: R@3 = 68.3% on 1,244 rows (Sprint 22)
- Extraction: Cond. F1 = 0.617 on 192 rows with `gpt-4o-mini` (Sprint 25)

The embedding cache in `eval/cache/` makes repeated eval runs ~2 min instead of ~50 min. The LLM response cache (keyed by model slug) is also pre-populated and loaded automatically — no flag needed.

---

## What has already been tried and rejected

Before proposing a retrieval or extraction approach, check CONTEXT.md → "What Was Evaluated". The following were benchmarked and rejected with documented regressions:

- HyDE (−3.9pp R@3)
- Cross-encoder reranker / bge-reranker-v2-m3 (−9.5pp R@3)
- MMR (no gain — parent_id dedup already handles clumping)
- Hybrid alpha tuning α≠0.5 (all worse — bge-m3 jointly trained for equal weight)
- Multi-query expansion to License Grant, Non-Disparagement, ROFR/ROFO, Change of Control, Termination for Convenience, Non-Transferable License (−3 to −23pp on affected categories)
- Verbatim copy instruction in SYSTEM_PROMPT (−4.8pp Cond. F1 with llama-3.1-8b)
- Chunk sizes 128/1024 and 512/2048 (256/2048 is optimal)

If you want to revisit one of these, bring a hypothesis for why the previous result would not apply.

---

## Submitting changes

1. Fork the repo and create a branch from `main`.
2. Make your changes. Update `CONTEXT.md` — sprint table, benchmark table, or Known Issues as appropriate.
3. Run the smoke tests (`run_sprint1.py`, `run_sprint9.py`).
4. Run evals if your change touches retrieval, extraction, or prompts.
5. Open a PR using the PR template. Include before/after benchmark numbers.

---

## What this project is not looking for

- Frontend redesigns
- New UI frameworks or admin dashboards
- Alternative orchestration frameworks (LangGraph is load-bearing)
- Changes to the Neo4j schema without a concrete contradiction detection improvement
- Anything requiring GPU infrastructure beyond a consumer M-series Mac
