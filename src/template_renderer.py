from __future__ import annotations

"""Utility to fill placeholders in prompt templates.

Current implementation is minimal: replaces {{query}} and {{context}} tokens.
It can be extended to handle additional placeholders as needed.
"""
from typing import List

__all__ = ["render_template"]


def render_template(template: str, *, query: str, context_snippets: List[str] | None = None) -> str:
    """Fill placeholders in template.

    Parameters
    ----------
    template: str
        Raw template text containing placeholders like ``{{query}}`` or ``{{context}}``.
    query: str
        User query.
    context_snippets: list[str] | None
        List of context strings; will be concatenated with newlines.
    """
    rendered = template.replace("{{query}}", query)

    if context_snippets is None:
        context_block = ""
    else:
        context_block = "\n\n".join(context_snippets)

    rendered = rendered.replace("{{context}}", context_block)
    return rendered 