from __future__ import annotations

"""Utility to fill placeholders in prompt templates.

Current implementation is minimal: replaces {{query}} and {{context}} tokens.
It can be extended to handle additional placeholders as needed.
"""
from typing import List

__all__ = ["render_template", "TemplateRenderer"]


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


class TemplateRenderer:
    """Template renderer class for backward compatibility with tests."""
    
    def __init__(self):
        pass
    
    def render(self, template: str, *, query: str, context_snippets: List[str] | None = None) -> str:
        """Render template with given parameters."""
        return render_template(template, query=query, context_snippets=context_snippets)
    
    def render_template(self, template: str, **kwargs) -> str:
        """Render template with kwargs."""
        query = kwargs.get('query', '')
        context_snippets = kwargs.get('context_snippets')
        return render_template(template, query=query, context_snippets=context_snippets) 