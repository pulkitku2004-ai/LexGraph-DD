## Summary

<!-- What does this PR do? One paragraph. -->

## Type of change

- [ ] Bug fix
- [ ] Retrieval / extraction improvement (benchmarked)
- [ ] New agent or pipeline stage
- [ ] Refactor / cleanup
- [ ] Eval harness change
- [ ] Docs / config

## Benchmark results (required for retrieval or extraction changes)

<!-- Any change to the retriever, chunker, prompts, or LLM chain must include before/after numbers.
     Run: python eval/cuad_eval.py --n 400 --enrich-queries --multi-query
     Or:  python eval/e2e_eval.py --n 200 --enrich-queries --multi-query          -->

| Metric | Before | After | Delta |
|---|---|---|---|
| R@3 (1244 rows) | | | |
| Cond. F1 (192 rows) | | | |

If retrieval or extraction was not changed, write "N/A — not a retrieval or extraction change."

## Tests

- [ ] `python run_sprint1.py` passes (ingestion + retrieval smoke test)
- [ ] `python run_sprint9.py` passes (full API lifecycle)
- [ ] No new Pyright errors (`pyright legal_due_diligence/`)
- [ ] Docker services up during testing (`docker compose up -d`)

## CONTEXT.md updated

- [ ] Sprint table updated with new entry
- [ ] Benchmark table updated if numbers changed
- [ ] Known Issues table updated if a bug was fixed or introduced

## Notes for reviewer

<!-- Anything non-obvious about the implementation, edge cases to watch for, or follow-up work. -->
