from __future__ import annotations

"""Analisador estático simples que percorre arquivos Python, extrai informações
básicas de estrutura (imports, classes, funções) e as injeta no Neo4j via
:class:`Neo4jStore`.

Uso rápido:

```python
from graphdb.neo4j_store import Neo4jStore
from graphdb.code_analyzer import CodeAnalyzer

import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

store = Neo4jStore(password=os.getenv("NEO4J_PASSWORD", ""))
CodeAnalyzer(store).analyze_project("/caminho/para/projeto")
store.close()
```
"""

import ast
import os
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .neo4j_store import Neo4jStore
from .graph_models import NodeType, RelationType, GraphRelation


class CodeAnalyzer:
    """Analisa o código-fonte de um projeto Python e popular o grafo Neo4j.

    A extração contempla:
    * Arquivos (`NodeType.CODE_FILE`)
    * Classes (`NodeType.CLASS`)
    * Funções (`NodeType.FUNCTION`)
    * Relações de *import* (`RelationType.IMPORTS`)
    * Relação *CONTAINS* (arquivo → classe/função)
    * Relação *EXTENDS* (classe → super classe)
    """

    def __init__(self, graph_store: Neo4jStore) -> None:
        self.graph_store = graph_store

    # ------------------------------------------------------------------
    # APIs públicas
    # ------------------------------------------------------------------

    def analyze_project(self, project_path: str) -> None:
        """Percorre recursivamente o diretório analisando todos os arquivos `*.py` em paralelo."""

        paths = [
            str(p)
            for p in Path(project_path).rglob("*.py")
            if not any(part.startswith(".") for part in p.parts)
        ]

        max_workers = os.cpu_count() or 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.analyze_python_file, p) for p in paths]
            for _ in as_completed(futures):
                pass  # resultados não são necessários; falhas registradas internamente

    def analyze_python_file(self, file_path: str) -> None:
        """Analisa um único arquivo Python e cria nós/relacionamentos."""

        try:
            with open(file_path, "r", encoding="utf-8") as fp:
                content = fp.read()
        except (UnicodeDecodeError, FileNotFoundError):  # pragma: no cover
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:  # pragma: no cover
            # Arquivo pode conter código inválido para AST -> ignora
            return

        # ------------------------------------------------------------------
        # Nó do arquivo
        # ------------------------------------------------------------------
        file_id = f"file::{file_path}"
        self.graph_store.add_code_element(
            {
                "id": file_id,
                "name": os.path.basename(file_path),
                "type": NodeType.CODE_FILE.value,
                "file_path": file_path,
                "content": content[:1000],  # armazenar amostra curta
                "metadata": {},
            }
        )

        # ------------------------------------------------------------------
        # Percorrer AST
        # ------------------------------------------------------------------
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._add_import(file_id, alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = f"{module}.{alias.name}" if module else alias.name
                    self._add_import(file_id, name)

            elif isinstance(node, ast.ClassDef):
                self._add_class(file_id, node)

            elif isinstance(node, ast.FunctionDef):
                # Apenas funções de nível superior (não dentro de classes)
                if isinstance(node.parent, ast.Module):
                    self._add_function(file_id, node)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _add_import(self, file_id: str, import_name: str) -> None:
        """Cria (se necessário) um nó para o módulo importado e a relação IMPORTS."""

        import_id = f"import::{import_name}"

        # Nós de importação são tratados como Concept
        self.graph_store.add_code_element(
            {
                "id": import_id,
                "name": import_name,
                "type": "Import",  # tipo textual para diferenciar
                "file_path": "",
                "content": "",
                "metadata": {},
            }
        )

        self.graph_store.add_relationship(
            GraphRelation(
                source_id=file_id,
                target_id=import_id,
                type=RelationType.IMPORTS.value,
            )
        )

    def _add_class(self, file_id: str, node: ast.ClassDef) -> None:
        class_id = f"class::{node.name}@{file_id}"

        # Criar nó da classe
        self.graph_store.add_code_element(
            {
                "id": class_id,
                "name": node.name,
                "type": NodeType.CLASS.value,
                "file_path": file_id.split("::", 1)[-1],
                "content": "",
                "metadata": {},
            }
        )

        # Relacionamento arquivo CONTAINS classe
        self.graph_store.add_relationship(
            GraphRelation(
                source_id=file_id,
                target_id=class_id,
                type=RelationType.CONTAINS.value,
            )
        )

        # Herança (EXTENDS)
        for base in node.bases:
            base_name = self._resolve_name(base)
            if not base_name:
                continue

            base_id = f"class::{base_name}"
            # Pode ser externo, criar como placeholder se não existir
            self.graph_store.add_code_element(
                {
                    "id": base_id,
                    "name": base_name,
                    "type": NodeType.CLASS.value,
                    "file_path": "",
                    "content": "",
                    "metadata": {},
                }
            )
            self.graph_store.add_relationship(
                GraphRelation(
                    source_id=class_id,
                    target_id=base_id,
                    type=RelationType.EXTENDS.value,
                )
            )

    def _add_function(self, file_id: str, node: ast.FunctionDef) -> None:
        func_id = f"func::{node.name}@{file_id}"

        self.graph_store.add_code_element(
            {
                "id": func_id,
                "name": node.name,
                "type": NodeType.FUNCTION.value,
                "file_path": file_id.split("::", 1)[-1],
                "content": "",
                "metadata": {},
            }
        )

        self.graph_store.add_relationship(
            GraphRelation(
                source_id=file_id,
                target_id=func_id,
                type=RelationType.CONTAINS.value,
            )
        )

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_name(node: ast.AST) -> str | None:
        """Resolve nome simples a partir de um nó AST (Name ou Attribute)."""

        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value  # type: ignore[assignment]
            if isinstance(node, ast.Name):
                parts.append(node.id)
                return ".".join(reversed(parts))
        return None


# ------------------------------------------------------------
# Patch de AST para acessar nó pai (facilita verificação nível)
# ------------------------------------------------------------

def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


# Monkeypatch ast.parse para já anexar pais
_original_parse = ast.parse


def _patched_parse(source, filename="<unknown>", mode="exec", **kw):  # type: ignore[override]
    tree = _original_parse(source, filename=filename, mode=mode, **kw)
    _attach_parents(tree)
    return tree


ast.parse = _patched_parse  # type: ignore[assignment] 