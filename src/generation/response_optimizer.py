from __future__ import annotations

from typing import List, Dict, Any
import re

class ResponseOptimizer:
    """Otimizador simples: injeta citações numeradas com base nas fontes."""

    def add_citations(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Anexa [n] ao final de sentenças que contêm trecho de fonte."""
        if not sources:
            return answer
        citations = []
        for i, src in enumerate(sources, 1):
            title = src.get("metadata", {}).get("source", f"Fonte {i}")
            citations.append(f"[{i}] {title}")
        return answer + "\n\n" + "\n".join(citations) 