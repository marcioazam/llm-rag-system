# Guia de Configuração do Neo4j

Este guia mostra como instalar e configurar o Neo4j para uso com o `Neo4jStore`.

---

## 1. Opções de Instalação

### 1.1 Desktop (GUI)

1. Baixe o **Neo4j Desktop** em <https://neo4j.com/download/>.
2. Crie um novo **Local DBMS**.
3. Defina um nome e senha (anote, será usada no `config.yaml`).
4. Inicie o DBMS e verifique a porta **bolt** (padrão `7687`).

### 1.2 Docker

```bash
docker run -d --name neo4j \
  -p7474:7474 -p7687:7687 \
  -e NEO4J_AUTH=neo4j/senha_supersecreta \
  -e NEO4J_server_memory_heap_initial__size=1G \
  -e NEO4J_server_memory_heap_max__size=1G \
  neo4j:5.18
```

* A interface web será exposta em `http://localhost:7474`.
* O Bolt URI será `bolt://localhost:7687`.

---

## 2. Índices e Constraints

O `Neo4jStore` cria automaticamente índices e constraints:

```cypher
CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.id);
CREATE INDEX IF NOT EXISTS FOR (n:CodeElement) ON (n.id);
CREATE INDEX IF NOT EXISTS FOR (n:CodeElement) ON (n.name);
CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.embedding_id);
CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.name);
```

Caso precise recriar manualmente:

```cypher
DROP INDEX document_id IF EXISTS;
-- repita para outros índices...
```

---

## 3. Teste de Conexão

```python
from graphdb.neo4j_store import Neo4jStore

store = Neo4jStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="senha_supersecreta"
)
print("Conectado com sucesso!")
store.close()
```

---

## 4. Problemas Comuns

| Problema | Solução |
|----------|---------|
| `ServiceUnavailable: WebSocket connection failure` | Verifique se o container/DBMS está em execução e a porta 7687 exposta. |
| `Neo.ClientError.Security.Unauthorized` | Senha incorreta. Atualize `neo4j_password` na config. |
| Timeout em queries complexas | Crie índices adicionais ou aumente parâmetros de timeout. |

---

## 5. Produção

* Use volumes Docker para persistir dados: `-v ./neo4j_data:/data`.
* Configure usuários e roles específicos via `neo4j-admin`.  
* Habilite TLS se expor fora da rede local. 