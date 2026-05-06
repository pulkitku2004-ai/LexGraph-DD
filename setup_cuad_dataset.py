"""
setup_cuad_dataset.py

Pull 100 unique contracts from the chenghao/cuad_qa HuggingFace dataset
and save them as TXT files in cuad_samples/.

Each unique title in cuad_qa maps to one contract. The context field
contains the full contract text and is identical across all rows sharing
the same title — we take one context per title.

Run:
    .venv/bin/python setup_cuad_dataset.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent / "cuad_samples"
N_CONTRACTS = 30


def safe_filename(title: str) -> str:
    # Strip path separators and collapse whitespace to underscores
    name = re.sub(r"[^\w\s-]", "", title).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:80] + ".txt"          # cap length to avoid filesystem limits


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading chenghao/cuad_qa from HuggingFace …")
    ds = load_dataset("chenghao/cuad_qa", split="train")

    seen: dict[str, str] = {}          # title → context
    for row in ds:
        title = row["title"]
        if title not in seen:
            seen[title] = row["context"]
        if len(seen) == N_CONTRACTS:
            break

    print(f"  {len(seen)} unique contracts collected")

    written = 0
    for title, context in seen.items():
        filename = safe_filename(title)
        path = OUTPUT_DIR / filename
        path.write_text(context, encoding="utf-8")
        written += 1

    print(f"  {written} files written → {OUTPUT_DIR}/")
    print()
    print("Sample filenames:")
    for p in sorted(OUTPUT_DIR.iterdir())[:5]:
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name:<85s}  {size_kb:>4d} KB")


if __name__ == "__main__":
    main()
