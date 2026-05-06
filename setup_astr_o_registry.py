"""
setup_astr_o_registry.py

One-time setup: rebuild the ASTR-O reference_registry.json to include the
LexGraph legal contracts alongside the existing aerospace sample documents.

Uses build_registry() + save_registry() so the HMAC signature is always valid.

Run:
    ASTR_O_REGISTRY_SECRET=<secret> .venv/bin/python setup_astr_o_registry.py

If ASTR_O_REGISTRY_SECRET is not set, a dev-only default is used (fine for
local integration testing — never use the default in production).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

ASTR_O_PATH = "/Users/dr_bolty/astr-o"
LEGAL_CONTRACTS_DIR = Path(__file__).parent / "samples"
CUAD_CONTRACTS_DIR  = Path(__file__).parent / "cuad_samples"
REGISTRY_OUTPUT = Path(ASTR_O_PATH) / "reference_registry.json"

sys.path.insert(0, ASTR_O_PATH)
from registry_builder import build_registry, verify_registry, save_registry  # noqa: E402


# Aerospace sample docs — same content as ASTR-O's smoke_test.py so hashes
# match what was used when the original registry was built.
AEROSPACE_DOCS = [
    (
        "Flight_Manual_v3.txt",
        "Flight manual content. Rev 3. Approved 2025-01-15.",
        "FLIGHT_MANUAL",
        "CRITICAL",
    ),
    (
        "Test_Report_2024.txt",
        "Structural test results. Pass rate: 99.7%. Date: 2024-11-01.",
        "TEST_REPORT",
        "SUPPORTING",
    ),
    (
        "Safety_Checklist.txt",
        "Pre-launch checklist. 142 items verified. Signed off by QA lead.",
        "SAFETY_CHECKLIST",
        "REFERENCE",
    ),
]

# Legal contracts from LexGraph samples/
LEGAL_DOCS = [
    ("contract_a.txt", "LEGAL_CONTRACT", "SUPPORTING"),
    ("contract_b.txt", "LEGAL_CONTRACT", "SUPPORTING"),
]


def main() -> None:
    secret = os.environ.get("ASTR_O_REGISTRY_SECRET", "lexgraph-astr-o-dev")
    if secret == "lexgraph-astr-o-dev":
        print("  [warn] ASTR_O_REGISTRY_SECRET not set — using dev default.")
        print("         Set the env var to a real secret for non-dev use.")

    with tempfile.TemporaryDirectory() as tmp:
        docs_folder = os.path.join(tmp, "docs")
        os.makedirs(docs_folder)

        # Write aerospace sample files
        for filename, content, _, _ in AEROSPACE_DOCS:
            with open(os.path.join(docs_folder, filename), "w") as f:
                f.write(content)

        # Copy sample legal contracts
        for filename, _, _ in LEGAL_DOCS:
            src = LEGAL_CONTRACTS_DIR / filename
            if not src.exists():
                print(f"  [error] {src} not found — run from the legal_dd project root")
                sys.exit(1)
            shutil.copy(src, os.path.join(docs_folder, filename))

        # Copy CUAD contracts (if cuad_samples/ exists)
        cuad_entries = []
        if CUAD_CONTRACTS_DIR.exists():
            cuad_files = sorted(CUAD_CONTRACTS_DIR.glob("*.txt"))[:30]
            for src in cuad_files:
                shutil.copy(src, os.path.join(docs_folder, src.name))
                cuad_entries.append(
                    {"filename": src.name, "source_tier": "SUPPORTING", "document_type": "LEGAL_CONTRACT"}
                )
            print(f"  including {len(cuad_entries)} CUAD contract(s) in registry")

        # Build config
        config = {
            "mission_id": "CUAD_ASTR_O_TEST_2026",
            "documents": [
                {"filename": fn, "source_tier": tier, "document_type": dtype}
                for fn, _, dtype, tier in AEROSPACE_DOCS
            ] + [
                {"filename": fn, "source_tier": tier, "document_type": dtype}
                for fn, dtype, tier in LEGAL_DOCS
            ] + cuad_entries,
        }
        config_path = os.path.join(tmp, "registry_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Build and verify
        registry = build_registry(config_path, docs_folder, secret)
        assert verify_registry(registry, secret), "Signature mismatch — should not happen"

        save_registry(registry, str(REGISTRY_OUTPUT))

    print(f"  registry written → {REGISTRY_OUTPUT}")
    print(f"  documents: {len(registry['documents'])} entries")
    for doc in registry["documents"]:
        print(f"    {doc['filename']:35s}  {doc['source_tier']}")
    print(f"  mission_id: {registry['mission_id']}")
    print(f"  signature:  {registry['signature'][:16]}...")
    print()
    print("  Next: run the test runner with the same ASTR_O_REGISTRY_SECRET set")
    print("  (or keep the dev default if you didn't set one above).")


if __name__ == "__main__":
    main()
