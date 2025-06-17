# Developer Tools

Este documento apresenta as ferramentas auxiliares incluídas no projeto para acelerar o ciclo de desenvolvimento.

## 1. CodeGenerator

Arquivo: `src/devtools/code_generator.py`

* Depende de um `llm_client` compatível com `.generate(model, prompt)` (Ollama, OpenAI…).
* Entrada:
  * `task` – descrição curta.
  * `context` – trecho de código relevante.
  * `language`, `style` – metadata.
* Saída: string com código gerado.

Exemplo:
```python
from src.devtools.code_generator import CodeGenerator
from ollama import Client

gen = CodeGenerator(Client(host="http://localhost:11434"), "codellama:7b-instruct")
print(gen.generate("criar função soma", "", "python"))
```

## 2. FormatterService

Arquivo: `src/devtools/formatter.py`

* Python → **Black** (precisa estar no PATH).
* JS/TS → **Prettier**.

Uso:
```python
from src.devtools.formatter import FormatterService
fmt = FormatterService()
print(fmt.format(code_str, "python"))
```

## 3. SnippetManager

Arquivo: `src/devtools/snippet_manager.py`

* SQLite (`~/.cursor_snippets.db`).
* Operações: `save_snippet`, `search`.

## 4. FileWatcher

* Documentado em `Docs/INCREMENTAL_INDEXING.md`.

## 5. Próximos

* `debug_helper.py` – LLM explica stack-traces.
* `perf_analyzer.py` – mede tempo/cProfile do snippet.

## 6. AutoDocumenter

Arquivo: `src/devtools/auto_documenter.py`

* `generate_docstring(code, style="google")` – devolve versão documentada do trecho.
* `generate_readme(project_path)` – cria README.md a partir da estrutura do projeto.

Exemplo rápido:
```python
from src.devtools.auto_documenter import AutoDocumenter
from ollama import Client

doc = AutoDocumenter(Client(host="http://localhost:11434"))
print(doc.generate_docstring(open('app.py').read()))
```

---
Última atualização: {{date}} 