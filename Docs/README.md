# Sistema RAG Multimodelo – Documentação Geral

> Esta documentação foi escrita em Português e descreve **passo a passo** como o projeto está organizado,   
> como instalar dependências, executar o pipeline, adicionar novos documentos e utilizar o grafo Neo4j   
> para enriquecimento de contexto.

---

## 1. Visão Geral

O projeto implementa um **sistema RAG (Retrieval-Augmented Generation)** focado em:

1. Indexação de documentos em um banco vetorial (ChromaDB).
2. Recuperação híbrida com reranking opcional.
3. Geração de respostas via modelos LLM (Ollama / Llama-based).
4. Roteamento inteligente de múltiplos modelos (simples ou avançado).
5. Enriquecimento de contexto com **grafo Neo4j** (opcional).

O diagrama abaixo resume o fluxo alto-nível:

```mermaid
flowchart LR
    A[Usuário] -->|Pergunta| B(RAGPipeline)
    B --> C[Retriever (ChromaDB)]
    C -->|Chunks relevantes| D{Usa Graph?}
    D -- Não --> E[Contexto]
    D -- Sim --> F[Neo4jStore]
    F -->|Nós e relações| E
    E --> G[Model Router]
    G --> H[LLMs]
    H --> I[Resposta]
    I --> A
```

---

## 2. Estrutura de Pastas

| Diretório | Descrição | 
|-----------|-----------|
| `src` | Código-fonte principal do sistema |
| `src/chunking` | Algoritmos de chunking semântico e recursivo |
| `src/embeddings` | Serviço de embeddings |
| `src/vectordb` | Implementação ChromaDB |
| `src/graphdb` | Integração Neo4j + modelos de grafo |
| `src/retrieval` | Retriever híbrido + reranking |
| `src/models` | Model Router simples e avançado |
| `tests` | Casos de teste |
| `Docs` | **Documentação** (pasta atual) |
| `data` | Persistência local do ChromaDB |
| `config` | Arquivos YAML de configuração |

---

## 3. Instalação

1. **Clone** o repositório.
2. Crie ambiente virtual (opcional):
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Linux/macOS
   # No Windows
   .venv\Scripts\activate
   ```
3. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. (Opcional) Configure Neo4j localmente:
   * Instale Neo4j Desktop ou use container Docker.
   * Defina URI, usuário e senha no seu arquivo `config.yaml` ou passe via dict.

---

## 4. Execução Rápida

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    config_path="config/config.yaml"  # ou None para config interna default
)

resposta = pipeline.query("O que é RAG?", k=5)
print(resposta["answer"])
```

### Habilitando o Grafo Neo4j

```python
pipeline.config.update({
    "use_graph_store": True,
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "sua_senha"
})

pipeline._initialize_components()  # reinicializa com grafo
```

---

## 5. Fluxo Detalhado Passo a Passo

1. **Chunking** – `SemanticChunker` ou `RecursiveChunker` divide documentos em pedaços.
2. **Embedding** – `EmbeddingService` gera vetores para cada chunk.
3. **Indexação** – `ChromaVectorStore` armazena embeddings + metadados.
4. **Consulta (query)**
   1. `HybridRetriever` busca K chunks mais similares (opção de reranking Cross-Encoder).
   2. Se `use_graph_store=True`:
       * IDs de embedding → `Neo4jStore.find_by_embedding_ids`.
       * Contexto expandido (`find_related_context`).
   3. Contexto final agregado.
5. **Model Router** – Decide estratégia:
   * `ModelRouter` (simples) OU `AdvancedModelRouter` (multi-modelo).
6. **LLM (Ollama)** – Gera resposta usando prompt com contexto.
7. **Resposta** – Retornada com fontes, modelos usados, etc.

---

## 6. Componentes Principais

| Componente | Arquivo | Papel |
|------------|---------|-------|
| `RAGPipeline` | `src/rag_pipeline.py` | Orquestrador geral |
| `ChromaVectorStore` | `src/vectordb/chroma_store.py` | Banco vetorial persistente |
| `Neo4jStore` | `src/graphdb/neo4j_store.py` | Persistência em grafo |
| `Graph Models` | `src/graphdb/graph_models.py` | Tipos de nós/relacionamento |
| `HybridRetriever` | `src/retrieval/retriever.py` | Busca + reranking |
| `ModelRouter` | `src/models` | Seleção de modelos |

---

## 7. Como Adicionar Novos Documentos

1. Prepare uma lista de dicionários: `[{"content": "texto", "metadata": {"title": "Doc 1"}}]`.
2. Execute:
   ```python
   pipeline.add_documents(docs, chunking_strategy="recursive")
   ```
3. Chunks e embeddings serão persistidos automaticamente.

---

## 8. Neo4j – Operações Básicas

| Método | Descrição |
|--------|-----------|
| `add_document_node` | Cria/atualiza nó de documento |
| `add_code_element` | Persiste elemento de código |
| `add_relationship` | Cria relação entre nós |
| `find_by_embedding_ids` | Recupera nós pelo `embedding_id` |
| `find_related_context` | Expande contexto a partir de um nó |

---

## 9. Testes & Benchmarks

* Execute `pytest tests/` (caso existam) ou scripts de benchmark incluídos.
* Use `pipeline.benchmark_models()` para comparar respostas.

---

## 10. Próximos Passos / Ideias

* Implementar cache de resultados do grafo.
* UI Web para visualizar grafo e chunks.
* Integração com outras bases vetoriais (Milvus, Weaviate).
* Deploy via Docker-Compose (Ollama + Neo4j + API FastAPI).

---

&copy; 2024 – Sistema RAG Multimodelo 