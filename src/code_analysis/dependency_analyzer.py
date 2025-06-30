from __future__ import annotations

import ast
from typing import List, Dict, Any, Set

class DependencyAnalyzer:
    """Analisa dependências em código Python.

    Se fornecido *project_root*, constrói tabela de módulos locais → arquivo, permitindo
    diferenciar chamadas internas/externas. Caso contrário, analisa apenas relações
    dentro do mesmo arquivo.
    """

    def __init__(self, project_root: str | None = None):
        self.module_lookup: dict[str, str] = {}
        if project_root:
            from pathlib import Path
            root = Path(project_root)
            for p in root.rglob("*.py"):
                mod = ".".join(p.relative_to(root).with_suffix("").parts)
                self.module_lookup[mod] = str(p)

    def analyze(self, code: str) -> List[Dict[str, Any]]:
        """Retorna relações internas (calls) sem contexto de arquivo."""
        try:
            tree = ast.parse(code)
        except SyntaxError:  # pragma: no cover
            return []

        # Mapeia nome->nó function/class para resolver escopos
        definitions: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.add(node.name)

        relations: List[Dict[str, Any]] = []
        imported_aliases: dict[str, str] = {}

        # Mapeia alias → módulo a partir de imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imported_aliases[alias.asname or alias.name] = f"{module}.{alias.name}" if module else alias.name

        class CallVisitor(ast.NodeVisitor):
            def __init__(self, outer):
                self.current_func: str | None = None
                self.outer = outer

            def visit_FunctionDef(self, node: ast.FunctionDef):
                prev = self.current_func
                self.current_func = node.name
                self.generic_visit(node)
                self.current_func = prev

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call):
                if self.current_func is None:
                    return
                # Tenta extrair nome simples
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name and func_name in definitions:
                    relations.append({
                        'source': self.current_func,
                        'target': func_name,
                        'relation_type': 'calls',
                    })
                elif func_name and func_name in imported_aliases:
                    rel_type = 'calls_external'
                    relations.append({
                        'source': self.current_func,
                        'target': imported_aliases[func_name],
                        'relation_type': rel_type,
                    })
                elif isinstance(node.func, ast.Attribute):
                    # Caso como np.array() ou json.loads()
                    if isinstance(node.func.value, ast.Name):
                        base_alias = node.func.value.id
                        if base_alias in imported_aliases:
                            full_name = f"{imported_aliases[base_alias]}.{node.func.attr}"
                            # Simplifica nome caso import seja de subpacote: fica apenas último segmento
                            simplified_base = imported_aliases[base_alias].split(".")[-1]
                            simplified_name = f"{simplified_base}.{node.func.attr}"
                            relations.append({
                                'source': self.current_func,
                                'target': simplified_name,
                                'relation_type': 'calls_external',
                            })
                self.generic_visit(node)

        CallVisitor(self).visit(tree)
        return relations

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Carrega arquivo, executa análise e tenta resolver módulos externos locais."""
        from pathlib import Path
        try:
            code = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        rels = self.analyze(code)

        # Opcional: substituir nomes de módulos por caminho de arquivo se existir
        if self.module_lookup:
            for r in rels:
                tgt_mod = r['target']
                # Tenta correspondência exata
                if tgt_mod in self.module_lookup:
                    r['target_path'] = self.module_lookup[tgt_mod]
                else:
                    # Caso alvo seja 'module.func', usar parte base para resolução
                    base_mod = tgt_mod.split(".")[0]
                    if base_mod in self.module_lookup:
                        r['target_path'] = self.module_lookup[base_mod]
                        # Normaliza target para nome de módulo se necessário (expectativa dos testes)
                        r['target'] = base_mod
        return rels 