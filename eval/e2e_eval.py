"""
Sprint 14 — End-to-End Extraction Accuracy Eval

Measures whether the LLM correctly extracts clause text AFTER retrieval,
scored against CUAD ground-truth answer spans.

Why this matters:
  cuad_eval.py measures Recall@K — did the right chunk come back?
  e2e_eval.py measures the next question — given the right chunk, does the
  LLM correctly extract the clause from it?

  These are independent failure modes:
    - Retrieval miss:  LLM never sees the right text → extraction impossible
    - Extraction miss: Right text retrieved, LLM returns found=False or wrong text

  52.1% Recall@3 means ~48% of rows never had a chance. Of the 52% that did,
  what fraction does the LLM get right? This eval answers that.

Metrics reported:
  retrieval_recall_at_3  — same as cuad_eval baseline (sanity check)
  llm_found_rate         — % rows where LLM says found=True (ground truth exists)
  extraction_substring   — % rows where ground truth is substring of extracted clause_text
  extraction_f1_mean     — mean token-level F1 (same scoring as CUAD paper / SQuAD)
  extraction_f1_median   — median token-level F1 (robust to outliers)
  conditional_f1_mean    — F1 computed only over rows where LLM said found=True
                           (isolates extraction quality from retrieval quality)

Token F1 (SQuAD-style):
  Tokenise both strings: lowercase → split on whitespace + punctuation → bag of tokens.
  precision = |common| / |predicted|
  recall    = |common| / |ground_truth|
  F1        = 2 * precision * recall / (precision + recall)

Usage:
  # Quick run — 50 rows, ~5 min (retrieval + ~50 LLM calls)
  python eval/e2e_eval.py --n 50 --enrich-queries

  # With reranker
  python eval/e2e_eval.py --n 50 --enrich-queries --reranker BAAI/bge-reranker-v2-m3

  # Per-category breakdown
  python eval/e2e_eval.py --n 200 --enrich-queries --by-category

  # Full run (1,244 rows — slow, costs ~1,244 Groq calls)
  python eval/e2e_eval.py --full --enrich-queries

Design notes:

Why import from cuad_eval rather than duplicating?
  cuad_eval.py owns the eval collection setup, indexing, and retrieval logic.
  Duplicating it would create two sources of truth. Instead, we import its
  helpers and add only what's new: the LLM extraction step and F1 scoring.

Why call the LLM synchronously (not asyncio)?
  The eval is single-threaded by design — results come in row-by-row order
  so the progress log is readable. Adding concurrency here would save time
  but make the log harder to follow and hit Groq rate limits more aggressively.
  Use --n 50 or --n 100 for fast iteration.

Why token F1 instead of exact match?
  Contracts are verbose. The LLM may extract "The agreement shall be governed
  by Delaware law" while the ground truth is "Delaware law". Exact match would
  score this 0. Token F1 scores it ~0.5 (partial credit). It matches how the
  CUAD benchmark paper measures extraction quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Path setup — mirrors cuad_eval.py ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "legal_due_diligence"))

import litellm
from agents.clause_extractor.prompts import (
    CUAD_CATEGORIES,
    SYSTEM_PROMPT,
    build_extraction_prompt,
    trim_clause_text,
)
from core.config import get_settings

# Import eval infrastructure from cuad_eval — single source of truth for
# collection setup, indexing, retrieval, and dataset loading.
from eval.cuad_eval import (  # type: ignore[import]
    CACHE_DIR,
    EVAL_COLLECTION,
    RESULTS_DIR,
    SAMPLE_IDS_PATH,
    _CUAD_QUERY_ENRICHMENT,
    embed_questions,
    eval_retrieve,
    eval_retrieve_multi,
    index_eval_rows,
    load_cuad_test,
    load_or_generate_sample_ids,
    setup_eval_collection,
    teardown_eval_collection,
)
from agents.clause_extractor.prompts import CUAD_ALT_QUERIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── LLM response cache ────────────────────────────────────────────────────────

def _llm_cache_path() -> Path:
    settings = get_settings()
    slug = settings.llm_extraction_model.split("/")[-1].lower().replace("-", "_")
    return CACHE_DIR / f"llm_responses_{slug}.pkl"


def _load_llm_cache() -> dict[str, str]:
    path = _llm_cache_path()
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_llm_cache(cache: dict[str, str]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with open(_llm_cache_path(), "wb") as f:
        pickle.dump(cache, f)


def _llm_key(user_prompt: str) -> str:
    """MD5 of system + user prompt — deterministic key for temperature=0 calls."""
    return hashlib.md5((SYSTEM_PROMPT + user_prompt).encode()).hexdigest()


# ── Token F1 (SQuAD-style) ────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 between prediction and ground_truth.
    Returns 0.0 if either string is empty after normalisation.
    """
    pred_tokens = _normalise(prediction).split()
    gt_tokens   = _normalise(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_bag = defaultdict(int)
    gt_bag   = defaultdict(int)
    for t in pred_tokens:
        pred_bag[t] += 1
    for t in gt_tokens:
        gt_bag[t] += 1

    common = sum(min(pred_bag[t], gt_bag[t]) for t in pred_bag if t in gt_bag)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def substring_match(prediction: str, ground_truth: str) -> bool:
    """True if ground_truth is a case-insensitive substring of prediction."""
    return _normalise(ground_truth) in _normalise(prediction)


# ── LLM extraction ────────────────────────────────────────────────────────────

_EXTRACTION_SENTINEL = {"found": False, "clause_text": None, "normalized_value": None, "confidence": 0.0}


def _parse_raw(raw: str, clause_type: str) -> dict[str, Any]:
    """Strip markdown fences and JSON-parse an LLM response string."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.debug("[e2e] JSON parse error for '%s': %s", clause_type, e)
        return _EXTRACTION_SENTINEL


def extract_with_llm(
    clause_type: str,
    retrieved_texts: list[str],
    doc_id: str,
    llm_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Call the extraction LLM using the same prompt as the production agent.
    Returns the parsed JSON dict, or a sentinel on failure.

    llm_cache: if provided, check before calling the LLM and store the raw
    response string on a cache miss. key = MD5(system_prompt + user_prompt).
    temperature=0 guarantees identical inputs → identical outputs, so caching
    is safe. Subsequent eval runs (e.g. after trim_clause_text changes) skip
    all LLM calls and go straight to the parse step.
    """
    settings = get_settings()
    prompt = build_extraction_prompt(clause_type, retrieved_texts, doc_id)

    # ── Cache check ────────────────────────────────────────────────────────
    if llm_cache is not None:
        key = _llm_key(prompt)
        if key in llm_cache:
            return _parse_raw(llm_cache[key], clause_type)

    # ── LLM call ──────────────────────────────────────────────────────────
    try:
        response = litellm.completion(
            model=settings.llm_extraction_model,
            fallbacks=settings.llm_extraction_fallbacks,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            api_key=settings.groq_api_key or None,
        )
        raw = (response.choices[0].message.content or "").strip()

        if llm_cache is not None:
            llm_cache[key] = raw

        return _parse_raw(raw, clause_type)

    except Exception as e:
        logger.debug("[e2e] LLM call failed for '%s': %s", clause_type, e)
        return _EXTRACTION_SENTINEL


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_e2e_eval(
    rows: list[dict],
    sample_ids: list[int],
    query_embeddings: dict[int, tuple[list[float], dict[int, float]]],
    top_k: int = 3,
    candidate_k: int = 20,
    enrich_queries: bool = False,
    multi_query: bool = False,
    llm_cache: dict[str, str] | None = None,
    apply_trim: bool = True,
) -> dict[str, Any]:
    """
    For each sampled row:
      1. Retrieve top-K chunks (hybrid RRF ± reranker ± multi-query)
      2. Call LLM extraction with the production prompt
      3. Score extracted clause_text vs ground-truth with token F1

    apply_trim: if False, skip trim_clause_text() — used for A/B comparison
    against cached responses to isolate the trimmer's effect.

    Returns full results dict with per-row details + aggregate metrics.
    """
    from ingestion.chunker import Chunk
    import uuid

    settings = get_settings()
    per_row: list[dict] = []
    _enrich_ci = {k.lower(): v for k, v in _CUAD_QUERY_ENRICHMENT.items()}
    _alts_ci = {k.lower(): v for k, v in CUAD_ALT_QUERIES.items()}

    # Aggregate counters
    retrieval_hits3 = 0
    llm_found       = 0
    substring_hits  = 0
    f1_scores: list[float] = []
    f1_when_found:  list[float] = []
    answered = skipped = 0

    for i, idx in enumerate(sample_ids):
        row          = rows[idx]
        doc_id       = f"cuad_eval_{idx}"
        question     = row["question"]
        answer_texts: list[str] = row["answers"]["text"]

        if not answer_texts or not answer_texts[0].strip():
            skipped += 1
            continue

        ground_truth = answer_texts[0]
        query_dense, query_sparse = query_embeddings[idx]
        # ── Retrieval ─────────────────────────────────────────────────────
        alt_queries = _alts_ci.get(question.lower(), []) if multi_query else []
        if alt_queries:
            alt_chunks = [
                Chunk(
                    chunk_id=str(uuid.uuid4()), doc_id="__query__", file_path="",
                    page_number=0, text=q, token_count=0, chunk_index=0,
                )
                for q in alt_queries
            ]
            from ingestion.embedder import embed_chunks
            alt_embedded = embed_chunks(alt_chunks)
            query_pairs = [(query_dense, query_sparse)] + [
                (e.vector, e.sparse_vector) for e in alt_embedded
            ]
            retrieved = eval_retrieve_multi(
                query_pairs=query_pairs,
                doc_id=doc_id,
                top_k=max(top_k, 5),
                candidate_k=candidate_k,
            )
        else:
            retrieved = eval_retrieve(
                query_dense=query_dense,
                query_sparse=query_sparse,
                doc_id=doc_id,
                top_k=top_k,
                candidate_k=candidate_k,
            )

        retrieved_texts = [text for _, text in retrieved]

        # Retrieval recall (sanity check against cuad_eval baseline)
        retrieval_hit3 = any(
            ground_truth.lower().strip() in t.lower()
            for t in retrieved_texts[:3]
        )
        if retrieval_hit3:
            retrieval_hits3 += 1

        # ── LLM Extraction ────────────────────────────────────────────────
        # Map CUAD question label → production clause_type key
        # cuad_eval uses the raw question string; the agent uses CUAD_CATEGORIES keys.
        # The question strings match the keys closely — try direct match first.
        clause_type = question if question in CUAD_CATEGORIES else (
            next((k for k in CUAD_CATEGORIES if k.lower() == question.lower()), question)
        )

        extraction = extract_with_llm(clause_type, retrieved_texts, doc_id, llm_cache)

        found        = bool(extraction.get("found", False))
        raw_text     = extraction.get("clause_text")
        clause_text  = (trim_clause_text(raw_text) if apply_trim else raw_text) or ""
        confidence   = float(extraction.get("confidence", 0.0))

        if found:
            llm_found += 1

        # ── Scoring ───────────────────────────────────────────────────────
        f1    = token_f1(clause_text, ground_truth) if clause_text else 0.0
        sub   = substring_match(clause_text, ground_truth) if clause_text else False

        f1_scores.append(f1)
        if found and clause_text:
            f1_when_found.append(f1)
        if sub:
            substring_hits += 1

        answered += 1

        per_row.append({
            "test_idx":        idx,
            "question":        question,
            "doc_id":          doc_id,
            "ground_truth":    ground_truth,
            "retrieval_hit3":  retrieval_hit3,
            "llm_found":       found,
            "llm_confidence":  confidence,
            "clause_text":     clause_text[:500] if clause_text else None,
            "token_f1":        round(f1, 4),
            "substring_match": sub,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(sample_ids):
            f1_mean = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            logger.info(
                "[e2e] %3d/%d | Ret@3=%.1f%%  Found=%.1f%%  F1=%.3f  Sub=%.1f%%  (answered=%d skip=%d)",
                i + 1, len(sample_ids),
                retrieval_hits3 / max(answered, 1) * 100,
                llm_found       / max(answered, 1) * 100,
                f1_mean,
                substring_hits  / max(answered, 1) * 100,
                answered, skipped,
            )
            if llm_cache is not None:
                _save_llm_cache(llm_cache)

    # ── Aggregate metrics ─────────────────────────────────────────────────
    def _mean(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    def _median(lst: list[float]) -> float:
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        return round((s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2), 4)

    metrics: dict[str, Any] = {
        "retrieval_recall_at_3":  round(retrieval_hits3 / answered, 4) if answered else 0.0,
        "llm_found_rate":         round(llm_found       / answered, 4) if answered else 0.0,
        "extraction_substring":   round(substring_hits  / answered, 4) if answered else 0.0,
        "extraction_f1_mean":     _mean(f1_scores),
        "extraction_f1_median":   _median(f1_scores),
        "conditional_f1_mean":    _mean(f1_when_found),   # F1 only when LLM said found=True
        "conditional_f1_median":  _median(f1_when_found),
        "answered_rows":          answered,
        "skipped_no_answer":      skipped,
        "total_sampled":          len(sample_ids),
    }

    # Per-category breakdown
    by_cat: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "rows": 0, "ret3": 0, "found": 0, "sub": 0, "f1_sum": 0.0,
    })
    for r in per_row:
        cat = r["question"]
        by_cat[cat]["rows"]   += 1
        by_cat[cat]["ret3"]   += int(r["retrieval_hit3"])
        by_cat[cat]["found"]  += int(r["llm_found"])
        by_cat[cat]["sub"]    += int(r["substring_match"])
        by_cat[cat]["f1_sum"] += r["token_f1"]

    category_metrics = {
        cat: {
            "rows":              v["rows"],
            "retrieval_recall3": round(v["ret3"]  / v["rows"], 3),
            "found_rate":        round(v["found"] / v["rows"], 3),
            "substring_rate":    round(v["sub"]   / v["rows"], 3),
            "f1_mean":           round(v["f1_sum"] / v["rows"], 3),
        }
        for cat, v in sorted(by_cat.items())
    }

    settings = get_settings()
    return {
        "eval_type":      "end_to_end_extraction",
        "model":          settings.embedding_model,
        "llm_extraction": settings.llm_extraction_model,
        "enrich_queries": enrich_queries,
        "multi_query":    multi_query,
        "apply_trim":     apply_trim,
        "top_k":          top_k,
        "candidate_k":    candidate_k,
        "dataset":        "chenghao/cuad_qa",
        "split":          "test",
        "sample_size":    len(sample_ids),
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics":        metrics,
        "by_category":    category_metrics,
        "per_row":        per_row,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end extraction eval: retrieval + LLM extraction, scored with token F1"
    )
    parser.add_argument("--n", type=int, default=50,
                        help="Rows to sample (default: 50). Ignored when --full is set.")
    parser.add_argument("--full", action="store_true",
                        help="Run the full 1,244-row test set (slow — ~1,244 LLM calls).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Chunks retrieved per row (passed to LLM extraction context).")
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Candidates fetched from each retriever before RRF fusion.")
    parser.add_argument("--enrich-queries", action="store_true",
                        help="Map CUAD category labels to keyword-rich retrieval queries.")
    parser.add_argument("--multi-query", action="store_true",
                        help="Enable multi-query retrieval for hard categories (CUAD_ALT_QUERIES).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path. Default: auto-generated.")
    parser.add_argument("--no-trim", action="store_true",
                        help="Skip trim_clause_text() — use with cached responses to A/B test the trimmer.")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep cuad_eval Qdrant collection after the run.")
    parser.add_argument("--by-category", action="store_true",
                        help="Print per-category breakdown to stdout after the run.")
    args = parser.parse_args()

    apply_trim = not args.no_trim

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    if args.output:
        output_path = Path(args.output)
    else:
        model_slug   = settings.embedding_model.split("/")[-1].lower().replace("-", "_")
        enrich_tag   = "_enriched" if args.enrich_queries else ""
        mq_tag       = "_mq"       if args.multi_query    else ""
        n_tag        = "_full"     if args.full           else f"_n{args.n}"
        trim_tag     = "_notrim"   if args.no_trim        else ""
        output_path  = RESULTS_DIR / f"e2e_{model_slug}_topk{args.top_k}{n_tag}{enrich_tag}{mq_tag}{trim_tag}.json"

    rows       = load_cuad_test()
    sample_ids = list(range(len(rows))) if args.full else load_or_generate_sample_ids(rows, args.n)

    llm_cache = _load_llm_cache()
    logger.info(
        "[e2e] %d rows | embed=%s | llm=%s | top_k=%d | enrich=%s | mq=%s | llm_cache=%d entries",
        len(sample_ids), settings.embedding_model, settings.llm_extraction_model,
        args.top_k, args.enrich_queries, args.multi_query, len(llm_cache),
    )

    setup_eval_collection()

    try:
        index_eval_rows(rows, sample_ids)
        query_embeddings, _ = embed_questions(
            rows, sample_ids,
            enrich_queries=args.enrich_queries,
        )

        t0      = time.perf_counter()
        results = run_e2e_eval(
            rows, sample_ids, query_embeddings,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            enrich_queries=args.enrich_queries,
            multi_query=args.multi_query,
            llm_cache=llm_cache,
            apply_trim=apply_trim,
        )
        elapsed = time.perf_counter() - t0

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        m = results["metrics"]
        print("\n" + "=" * 64)
        print(f"  End-to-End Extraction Eval")
        print(f"  Embed model : {results['model']}")
        print(f"  LLM         : {results['llm_extraction']}")
        print(f"  Top-K       : {results['top_k']}  (candidate_k={results['candidate_k']})")
        print(f"  Enrich      : {'yes' if results['enrich_queries'] else 'no'}")
        print(f"  Multi-query : {'yes' if results['multi_query'] else 'no'}")
        print(f"  Trim        : {'yes' if results['apply_trim'] else 'no (--no-trim)'}")
        print(f"  Sample      : {m['answered_rows']} rows ({m['skipped_no_answer']} skipped)")
        print("=" * 64)
        print(f"  Retrieval Recall@3 : {m['retrieval_recall_at_3']:.1%}  (sanity check vs cuad_eval)")
        print(f"  LLM Found Rate     : {m['llm_found_rate']:.1%}  (when ground truth exists)")
        print(f"  Substring Match    : {m['extraction_substring']:.1%}")
        print(f"  Token F1 (mean)    : {m['extraction_f1_mean']:.3f}")
        print(f"  Token F1 (median)  : {m['extraction_f1_median']:.3f}")
        print(f"  Cond. F1 (found)   : {m['conditional_f1_mean']:.3f}  (F1 when LLM said found=True)")
        print(f"  Time               : {elapsed:.1f}s")
        print(f"  Output             : {output_path}")
        print("=" * 64)

        if args.by_category:
            print("\n  Per-category breakdown (sorted by F1):")
            print(f"  {'Category':<40} {'Ret@3':>6} {'Found':>6} {'Sub':>6} {'F1':>6}  rows")
            print("  " + "-" * 74)
            by_cat = sorted(
                results["by_category"].items(),
                key=lambda x: x[1]["f1_mean"],
                reverse=True,
            )
            for cat, v in by_cat:
                print(
                    f"  {cat:<40} {v['retrieval_recall3']:>5.1%} "
                    f"{v['found_rate']:>5.1%} {v['substring_rate']:>5.1%} "
                    f"{v['f1_mean']:>5.3f}  {v['rows']}"
                )
            print()

    finally:
        _save_llm_cache(llm_cache)
        logger.info("[cache] llm cache saved — %d entries", len(llm_cache))
        if not args.no_cleanup:
            teardown_eval_collection()


if __name__ == "__main__":
    main()
