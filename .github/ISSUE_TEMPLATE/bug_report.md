---
name: Bug report
about: Something is broken — retrieval regression, agent crash, API error
title: "[BUG] "
labels: bug
assignees: ''
---

## What happened

<!-- A clear description of the bug. -->

## Steps to reproduce

1. 
2. 
3. 

## Expected behaviour

## Actual behaviour

<!-- Include the full error message / traceback if one was raised. -->

```
paste error here
```

## Environment

| Item | Value |
|---|---|
| Python | |
| OS | |
| LLM primary | `gpt-4o-mini` / Groq / Ollama |
| Docker services running | Qdrant / Neo4j / both / neither |

## Which component

- [ ] Ingestion (loader / chunker / embedder / indexer)
- [ ] Clause extractor (retrieval or LLM extraction)
- [ ] Risk scorer
- [ ] Entity mapper / Neo4j graph
- [ ] Contradiction detector
- [ ] Report + Q&A
- [ ] FastAPI / runner
- [ ] Streamlit UI
- [ ] Eval harness (`cuad_eval.py` / `e2e_eval.py`)
- [ ] Other

## Additional context

<!-- Attach eval JSON, relevant log lines, or a minimal contract that reproduces the issue. -->
