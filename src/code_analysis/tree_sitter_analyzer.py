from __future__ import annotations

from typing import List, Dict, Any

from tree_sitter import Parser
from tree_sitter_languages import get_language

from .base_analyzer import BaseStaticAnalyzer

# ------------------------------------------------------------
# Consultas simples para extrair funções e classes por linguagem
# ------------------------------------------------------------
_QUERIES: dict[str, str] = {
    "javascript": """
        (function_declaration name: (identifier) @func_name)
        (class_declaration name: (identifier) @class_name)
    """,
    "typescript": """
        (function_declaration name: (identifier) @func_name)
        (method_definition name: (property_identifier) @func_name)
        (class_declaration name: (identifier) @class_name)
    """,
    "csharp": """
        (method_declaration name: (identifier) @func_name)
        (class_declaration name: (identifier) @class_name)
    """,
    "java": """
        (method_declaration name: (identifier) @func_name)
        (class_declaration name: (identifier) @class_name)
        (interface_declaration name: (identifier) @class_name)
    """,
    "go": """
        (function_declaration name: (identifier) @func_name)
        (type_spec name: (type_identifier) @class_name)
    """,
    "ruby": """
        (method name: (identifier) @func_name)
        (class name: (constant) @class_name)
    """,
}


class TreeSitterAnalyzer(BaseStaticAnalyzer):
    """Analisador genérico usando *tree-sitter* para linguagens suportadas."""

    def __init__(self, language: str):
        self.language = language.lower()
        try:
            lang_obj = get_language(self.language)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Gramatica tree-sitter para {self.language} não encontrada: {exc}") from exc
        self.parser = Parser()
        self.parser.set_language(lang_obj)

        self._query = _QUERIES.get(self.language, "")

    # ------------------------------------------------------------------
    def extract_symbols(self, code: str) -> List[Dict[str, Any]]:
        if not self._query:
            return []
        tree = self.parser.parse(code.encode())
        code_bytes = code.encode()
        query = self.parser.language.query(self._query)  # type: ignore[attr-defined]
        captures = query.captures(tree.root_node)
        symbols: List[Dict[str, Any]] = []
        for node, capture_name in captures:
            if capture_name in ("func_name", "class_name"):
                symbol_type = "function" if "func" in capture_name else "class"
                name = code_bytes[node.start_byte : node.end_byte].decode("utf8", errors="ignore")
                symbols.append(
                    {
                        "name": name,
                        "type": symbol_type,
                        "line": node.start_point[0] + 1,
                    }
                )
        return symbols

    def extract_relations(self, code: str) -> List[Dict[str, Any]]:  # pragma: no cover
        """Extrai relações básicas (imports) para algumas linguagens."""
        relations: List[Dict[str, Any]] = []
        tree = self.parser.parse(code.encode())
        code_bytes = code.encode()

        if self.language in {"javascript", "typescript"}:
            query_src = "(import_statement (string) @module_path)"
            q = self.parser.language.query(query_src)  # type: ignore[attr-defined]
            for node, _ in q.captures(tree.root_node):
                module_path = code_bytes[node.start_byte + 1 : node.end_byte - 1].decode("utf8", errors="ignore")  # strip quotes
                relations.append({"source": "module", "target": module_path, "relation_type": "imports"})

        elif self.language == "java":
            query_src = "(import_declaration (scoped_identifier) @imp)"
            q = self.parser.language.query(query_src)  # type: ignore[attr-defined]
            for node, _ in q.captures(tree.root_node):
                imp = code_bytes[node.start_byte : node.end_byte].decode("utf8", errors="ignore")
                relations.append({"source": "class", "target": imp, "relation_type": "imports"})

        elif self.language == "go":
            query_src = "(import_spec (interpreted_string_literal) @path)"
            q = self.parser.language.query(query_src)  # type: ignore[attr-defined]
            for node, _ in q.captures(tree.root_node):
                path = code_bytes[node.start_byte + 1 : node.end_byte - 1].decode("utf8", errors="ignore")
                relations.append({"source": "package", "target": path, "relation_type": "imports"})

        return relations 