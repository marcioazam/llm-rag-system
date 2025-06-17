from __future__ import annotations

import ast
from typing import List, Dict, Any

from .base_analyzer import BaseStaticAnalyzer


class PythonAnalyzer(BaseStaticAnalyzer):
    """Analisador estático simples para Python usando `ast`."""

    language = "python"

    # ------------------------------------------------------------------
    # Implementações obrigatórias
    # ------------------------------------------------------------------

    def extract_symbols(self, code: str) -> List[Dict[str, Any]]:
        symbols: List[Dict[str, Any]] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:  # pragma: no cover
            return symbols

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbols.append(
                    {
                        "name": node.name,
                        "type": "function",
                        "line": getattr(node, "lineno", None),
                    }
                )
            elif isinstance(node, ast.AsyncFunctionDef):
                symbols.append(
                    {
                        "name": node.name,
                        "type": "async_function",
                        "line": getattr(node, "lineno", None),
                    }
                )
            elif isinstance(node, ast.ClassDef):
                symbols.append(
                    {
                        "name": node.name,
                        "type": "class",
                        "line": getattr(node, "lineno", None),
                    }
                )
        return symbols

    def extract_relations(self, code: str) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:  # pragma: no cover
            return relations

        # Imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    relations.append(
                        {
                            "source": "module",
                            "target": alias.name,
                            "relation_type": "imports",
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    target = f"{module}.{alias.name}" if module else alias.name
                    relations.append(
                        {
                            "source": "module",
                            "target": target,
                            "relation_type": "imports",
                        }
                    )
        return relations

    # -------------------------------------------------------------
    # Docstrings
    # -------------------------------------------------------------

    def extract_docstrings(self, code: str) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:  # pragma: no cover
            return docs

        module_doc = ast.get_docstring(tree)
        if module_doc:
            docs.append({"type": "module", "name": "__module__", "doc": module_doc})

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                if doc:
                    ntype = "class" if isinstance(node, ast.ClassDef) else "function"
                    docs.append({"type": ntype, "name": node.name, "doc": doc})
        return docs 