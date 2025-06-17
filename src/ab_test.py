from __future__ import annotations

"""Simple A/B test helper for prompt usage.

Variant 'with_prompt'  – usa template selecionado
Variant 'no_prompt'    – ignora template (legacy comportamento)
"""
import os
import random
import hashlib
from typing import Optional

__all__ = ["decide_variant"]

# Default ratio (50% with prompt)
_DEFAULT_RATIO = float(os.getenv("RAG_WITH_PROMPT_RATIO", 0.5))


def _hash_to_prob(key: str) -> float:
    h = hashlib.md5(key.encode()).hexdigest()
    # take first 8 hex digits to int
    val = int(h[:8], 16)
    return (val % 10000) / 10000.0  # 0.0 – 0.9999


def decide_variant(query: str | None = None) -> str:
    """Return variant name 'with_prompt' or 'no_prompt'.

    If ENV `RAG_AB_TEST` is set to 'with' / 'no', enforce that variant.
    Otherwise use hash of query (stable) to decide based on ratio.
    """
    forced = os.getenv("RAG_AB_TEST")
    if forced in {"with", "no"}:
        return "with_prompt" if forced == "with" else "no_prompt"

    ratio = _DEFAULT_RATIO
    if query is None:
        # random fallback
        return "with_prompt" if random.random() < ratio else "no_prompt"

    prob = _hash_to_prob(query)
    return "with_prompt" if prob < ratio else "no_prompt" 