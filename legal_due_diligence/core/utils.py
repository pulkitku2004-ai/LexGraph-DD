"""
Shared utilities used across agent modules.
"""

from __future__ import annotations


def strip_json_fence(text: str) -> str:
    """
    Strip markdown code fence markers from an LLM response.

    Some models wrap JSON in ```json ... ``` even when instructed not to.
    Stripping fences makes downstream JSON parsers robust without needing
    a separate prompt instruction for every model or provider.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    return text
