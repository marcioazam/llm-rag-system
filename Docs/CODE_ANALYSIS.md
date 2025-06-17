# Subsystem: Code Analysis

Este documento descreve os componentes responsáveis por detectar linguagem, extrair símbolos e dependências e enriquecer os metadados dos documentos indexados.

## 1. Detecção de Linguagem

Arquivo: `src/code_analysis/language_detector.py`

* Estratégias combinadas:
  * MIME-Type via **python-magic** (quando disponível).
  * Heurística por extensão (`.py`, `.js`, `.cs`, etc.).
  * Inspeção de conteúdo com **Pygments**.

Resultado retornado: string curta (`"python"`, `"javascript"`, ...).

## 2. Analisadores Estáticos

| Linguagem | Implementação | Arquivo | Capacidades |
|-----------|---------------|---------|-------------|
| Python    | AST nativo    | `python_analyzer.py` | Símbolos, imports, docstrings, relações calls |
| JS/TS/JSX | tree-sitter   | `tree_sitter_analyzer.py` | Símbolos, imports (ESM/CommonJS) |
| C#        | tree-sitter   | idem | Símbolos |
| Java      | tree-sitter   | idem | Símbolos, imports |
| Go        | tree-sitter   | idem | Símbolos, imports |
| Ruby      | tree-sitter   | idem | Símbolos |

Todos herdam de `BaseStaticAnalyzer` que define:
* `extract_symbols`
* `extract_relations`
* `extract_docstrings` (opcional)

## 3. Dependency Analyzer

Arquivo: `dependency_analyzer.py`

* Percorre AST Python e cria lista `{source, target, relation_type="calls"}`.
* Usado pelo script `watch_and_index.py` para popular grafo Neo4j (arestas CALLS).

## 4. Orquestrador

`CodeContextDetector` escolhe o analisador adequado e devolve:
```json
{
  "language": "python",
  "symbols": [...],
  "relations": [...],
  "docstrings": [...]
}
```
Esses dados são:
1. Adicionados a `metadata` do chunk.
2. Persistidos no SQLite (`chunks.db`).
3. Usados nos prompts (`ContextInjector`).

## 5. Expansão de Linguagens

Para adicionar nova linguagem:
1. Instale grammar `tree-sitter-<lang>`.
2. Acrescente consulta em `_QUERIES` (capturar funções/classes).
3. Registre em `CodeContextDetector`.

## 6. Integração com Neo4j

* `code_analyzer.CodeAnalyzer` (existente) já popula nós/arestas para Python.
* `watch_and_index.py` insere arestas CALLS em tempo-real.

---
Última atualização: {{date}} 