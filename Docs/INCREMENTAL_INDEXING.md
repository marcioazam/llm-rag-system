# Incremental Indexing & Real-Time Updates

Este documento explica como o sistema mantém o índice vetorial, o SQLite de metadados e o grafo Neo4j sempre em sincronia com o código-fonte durante o desenvolvimento.

## 1. File Watcher

Arquivo: `src/devtools/file_watcher.py`

* Baseado em **watchdog**.
* Observa padrões: `*.py`, `*.js`, `*.ts`, `*.tsx`, `*.java`, `*.go`, `*.rb`, `*.cs`.
* Emite callback `on_change(path)` para cada arquivo criado/modificado.

## 2. Script CLI

Arquivo: `scripts/watch_and_index.py`

```
python scripts/watch_and_index.py src/
```

1. Instancia `RAGPipeline()` uma única vez.
2. Cria `DependencyAnalyzer` (call-graph Python).
3. Para cada mudança:
   1. `SmartDocumentLoader.load()` → metadados + conteúdo.
   2. `RAGPipeline.add_documents([doc])` →
      * Chunking + embeddings → Chroma.
      * Texto salvo `persist_dir/chunks/<uuid>.txt`.
      * Metadados upsert `SQLiteMetadataStore`.
   3. Se `graph_store` ativo, roda `DependencyAnalyzer` e insere arestas `CALLS` em Neo4j.

Ctrl+C → encerra observer com `signal.pause()`.

## 3. Benefícios

* **Latência baixa**: apenas arquivo alterado é reprocessado.
* **Consistência**: metadados, vetor e grafo atualizados em uma única chamada.
* **Escalável**: para projetos grandes, o watcher evita reindexação completa.

## 4. Customização

* Alterar padrões de arquivo: edite `patterns` no `FileWatcher`.
* Para incluir linguagens extras: basta adicionar extensão.
* Para desabilitar update Neo4j: inicie `RAGPipeline` sem `use_graph_store`.

---
Última atualização: {{date}} 