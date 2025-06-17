from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseStaticAnalyzer(ABC):
    """Interface comum para analisadores de código em diversas linguagens."""

    language: str  # nome curto da linguagem (python, javascript, ...)

    # ------------------------------------------------------------------
    # Métodos obrigatórios
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_symbols(self, code: str) -> List[Dict[str, Any]]:
        """Extrai símbolos (funções, classes, variáveis globais).

        Retorna lista de dicionários como::
            {"name": "Foo", "type": "class", "line": 42}
        """

    @abstractmethod
    def extract_relations(self, code: str) -> List[Dict[str, Any]]:
        """Extrai relações (importa, estende, chama, etc.).

        Estrutura livre; cada dicionário deve ter chaves *source*, *target*, *relation_type*.
        """

    # ------------------------------------------------------------------
    # APIs utilitárias
    # ------------------------------------------------------------------

    def analyze_content(self, code: str) -> Dict[str, Any]:
        """Retorna estrutura completa (símbolos + relações)."""
        return {
            "language": self.language,
            "symbols": self.extract_symbols(code),
            "relations": self.extract_relations(code),
        }

    # ------------------------------------------------------------------
    # Opcional: Docstrings / Comentários
    # ------------------------------------------------------------------

    def extract_docstrings(self, code: str) -> List[Dict[str, Any]]:  # pragma: no cover
        """Retorna lista de docstrings/comentários.

        Cada item: {"type": "module|class|function", "name": str, "doc": str}. 
        Implementação padrão retorna lista vazia; subclasses podem sobrescrever.
        """
        return [] 