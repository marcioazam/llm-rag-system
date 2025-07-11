from typing import Dict, List, Any, Optional
import logging
from time import perf_counter
try:
    from neo4j import GraphDatabase  # type: ignore
except ImportError:  # pragma: no cover
    GraphDatabase = None  # type: ignore

from .graph_models import GraphNode, GraphRelation, NodeType, RelationType
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
GRAPH_QUERY_COUNT = Counter("neo4j_queries_total", "Total de queries executadas", ["status"])
GRAPH_QUERY_LATENCY = Histogram("neo4j_query_latency_seconds", "Latência das queries Neo4j")

class Neo4jStore:
    """Camada de persistência em grafo usando Neo4j.

    A classe encapsula as operações básicas necessárias para armazenar documentos, elementos
    de código e relações, além de consultas específicas para fornecer contexto adicional
    ao pipeline de RAG.
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = "neo4j",  # Permite selecionar banco em Neo4j 4+
    ) -> None:
        import os
        
        # Usar variáveis de ambiente como padrão
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD")
        
        if not password:
            raise ValueError("Neo4j password must be provided via NEO4J_PASSWORD environment variable or password parameter")
        if GraphDatabase is None:
            raise ImportError(
                "A dependência 'neo4j' não está instalada. Adicione-a em requirements.txt e execute `pip install -r requirements.txt`."
            )

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.logger = logger
        self._init_constraints()

    # ---------------------------------------------------------------------
    # Configuração inicial
    # ---------------------------------------------------------------------

    def _init_constraints(self) -> None:
        """Cria índices e constraints necessárias para performance.

        Caso já existam, Neo4j levanta uma exceção que é capturada para evitar
        quebra de execução.
        """

        constraints: List[str] = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:CodeElement) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:CodeElement) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.embedding_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n) ON (n.type, n.id)",
        ]

        with self.driver.session(database=self.database) as session:
            for statement in constraints:
                try:
                    session.run(statement)
                except Exception as exc:  # pragma: no cover
                    self.logger.debug("Constraint já existe ou falhou a criação: %s", exc)

    # ---------------------------------------------------------------------
    # Operações de CRUD para nós
    # ---------------------------------------------------------------------

    def add_document_node(
        self, doc_id: str, metadata: Dict[str, Any], embedding_id: Optional[str] = None
    ) -> None:
        """Adiciona (ou atualiza) um nó de documento no grafo."""

        query = (
            "MERGE (d:Document {id: $id}) "
            "SET d += $metadata "
            "SET d.embedding_id = $embedding_id "
            "SET d.updated_at = timestamp() "
        )

        params = {"id": doc_id, "metadata": metadata, "embedding_id": embedding_id}

        with self.driver.session(database=self.database) as session:
            session.run(query, **params)

    def add_code_element(self, element: Dict[str, Any]) -> None:
        """Adiciona (ou atualiza) um elemento de código (classe, função, etc)."""

        query = (
            "MERGE (e:CodeElement {id: $id}) "
            "SET e.name = $name "
            "SET e.type = $type "
            "SET e.file_path = $file_path "
            "SET e.content = $content "
            "SET e.metadata = $metadata "
            "SET e.updated_at = timestamp() "
        )

        self._safe_run(query, **element)

    def add_node(self, node) -> None:
        """Adiciona um nó genérico ao grafo."""
        # Extrair informações do nó
        node_id = node.id
        node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
        properties = getattr(node, 'properties', {}) or {}
        
        query = (
            f"MERGE (n:{node_type} {{id: $id}}) "
            "SET n += $properties "
            "SET n.updated_at = timestamp()"
        )
        
        params = {
            "id": node_id,
            "properties": properties
        }
        
        self._safe_run(query, **params)

    # ---------------------------------------------------------------------
    # Operações de relacionamento
    # ---------------------------------------------------------------------

    def add_relationship(self, relation: GraphRelation) -> None:
        """Cria (ou atualiza) uma relação entre dois nós."""

        query = (
            f"MATCH (a {{id: $source_id}}) MATCH (b {{id: $target_id}}) "
            f"MERGE (a)-[r:{relation.type}]->(b) "
            "SET r += $properties "
        )

        params = {
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "properties": relation.properties or {},
        }

        self._safe_run(query, **params)

    def add_relation(self, relation: GraphRelation) -> None:
        """Alias para add_relationship para compatibilidade com testes."""
        self.add_relationship(relation)

    def delete_relation(self, source_id: str, target_id: str, relation_type) -> None:
        """Remove uma relação específica entre dois nós."""
        query = (
            f"MATCH (a {{id: $from_id}})-[r:{relation_type}]->(b {{id: $to_id}}) "
            "DELETE r"
        )
        
        params = {
            "from_id": source_id,
            "to_id": target_id
        }
        
        self._safe_run(query, **params)

    def search_nodes_by_content(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Busca nós por conteúdo."""
        query = (
            "MATCH (n) "
            "WHERE n.content CONTAINS $search_term OR n.name CONTAINS $search_term "
            "RETURN n "
            f"LIMIT {limit}"
        )
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, search_term=search_term)
            return [dict(record["n"]) for record in result]

    def get_document_context(self, document_id: str) -> List[Dict[str, Any]]:
        """Recupera o contexto de um documento."""
        query = (
            "MATCH (doc {id: $doc_id})-[:CONTAINS]->(chunk) "
            "RETURN chunk"
        )
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, doc_id=document_id)
            return [dict(record["chunk"]) for record in result]

    def delete_node(self, node_id: str) -> None:
        """Remove um nó e todas suas relações."""
        query = (
            "MATCH (n {id: $node_id}) "
            "DETACH DELETE n"
        )
        
        self._safe_run(query, node_id=node_id)

    def clear_all(self) -> None:
        """Remove todos os nós e relações do banco."""
        query = "MATCH (n) DETACH DELETE n"
        self._safe_run(query)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do banco."""
        with self.driver.session(database=self.database) as session:
            # Contar nós
            nodes_result = session.run("MATCH (n) RETURN count(n) as total_nodes")
            nodes_record = nodes_result.single()
            total_nodes = nodes_record["total_nodes"] if nodes_record else 0
            
            # Contar relações
            rels_result = session.run("MATCH ()-[r]->() RETURN count(r) as total_relations")
            rels_record = rels_result.single()
            total_relations = rels_record["total_relations"] if rels_record else 0
            
            return {
                "total_nodes": total_nodes,
                "total_relations": total_relations
            }

    def find_related_nodes(self, node_id: str, relation_type, depth: int = 1) -> List[Dict[str, Any]]:
        """Encontra nós relacionados a um nó específico."""
        query = (
            f"MATCH (n {{id: $node_id}})-[:{relation_type}*1..{depth}]-(related) "
            "RETURN DISTINCT related"
        )
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, node_id=node_id)
            return [dict(record["related"]) for record in result]

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Recupera um nó específico pelo ID."""
        query = "MATCH (n {id: $node_id}) RETURN n"
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            return dict(record["n"]) if record else None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def batch_add_nodes(self, nodes: List) -> None:
        """Adiciona múltiplos nós em lote."""
        for node in nodes:
            self.add_node(node)

    def begin_transaction(self):
        """Inicia uma transação explícita."""
        session = self.driver.session(database=self.database)
        return session.begin_transaction()

    # ---------------------------------------------------------------------
    # Consultas para contexto e análise
    # ---------------------------------------------------------------------

    def find_related_context(self, node_id: str, max_depth: int = 2) -> Optional[Dict[str, Any]]:
        """Recupera um nó e seu contexto relacionado até determinada profundidade."""

        query = (
            "MATCH (n {id: $id}) "
            "OPTIONAL MATCH path = (n)-[*1..$depth]-(related) "
            "RETURN n as node, "
            "       collect(DISTINCT related) as related_nodes, "
            "       collect(DISTINCT relationships(path)) as relationships "
        )

        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=node_id, depth=max_depth)
            record = result.single()

            if not record:
                return None

            return {
                "node": dict(record["node"]),
                "related": [dict(r) for r in record["related_nodes"]],
                "relationships": record["relationships"],
            }

    def find_by_embedding_ids(self, embedding_ids: List[str]) -> List[Dict[str, Any]]:
        """Busca nós que possuem embedding_id em uma lista de IDs."""

        query = (
            "MATCH (n) WHERE n.embedding_id IN $ids "
            "OPTIONAL MATCH (n)-[r]-(connected) "
            "RETURN n, collect(DISTINCT {node: connected, relationship: type(r)}) as connections "
        )

        with self.driver.session(database=self.database) as session:
            result = session.run(query, ids=embedding_ids)
            records = []
            for record in result:
                records.append({
                    "node": dict(record["n"]),
                    "connections": record["connections"],
                })
            return records

    def analyze_code_dependencies(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analisa dependências de um arquivo de código específico."""

        query = (
            "MATCH (f:CodeElement {file_path: $path}) "
            "OPTIONAL MATCH (f)-[:IMPORTS]->(imported) "
            "OPTIONAL MATCH (f)<-[:IMPORTS]-(importing) "
            "OPTIONAL MATCH (f)-[:CONTAINS]->(contained) "
            "RETURN f as file, "
            "       collect(DISTINCT imported) as imports, "
            "       collect(DISTINCT importing) as imported_by, "
            "       collect(DISTINCT contained) as contains "
        )

        with self.driver.session(database=self.database) as session:
            result = session.run(query, path=file_path)
            record = result.single()
            if not record:
                return None

            return {
                "file": dict(record["file"]),
                "imports": [dict(n) for n in record["imports"]],
                "imported_by": [dict(n) for n in record["imported_by"]],
                "contains": [dict(n) for n in record["contains"]],
            }

    # ---------------------------------------------------------------------
    # PageRank expansion (opcional, requer GDS instalado)
    # ---------------------------------------------------------------------

    def expand_with_pagerank(self, seed_ids: List[str], depth: int = 2, top_k: int = 20):
        """Expande contexto usando Personalized PageRank a partir de seeds.

        Retorna nós ordenados por score. Requer plugin GDS na instância Neo4j.
        """

        query = (
            "MATCH (n) WHERE n.id IN $seed_ids "
            "WITH collect(n) AS seeds "
            "CALL gds.graph.project.cypher(\"tmp\", "
            "  'MATCH (m)-[r*1..{depth}]-(n) WHERE m.id IN $seed_ids RETURN DISTINCT m AS source, n AS target', "
            "  'MATCH (m)-[rel]->(n) RETURN id(m) AS source, id(n) AS target, type(rel) AS type') "
            "YIELD graphName "
            "CALL gds.pageRank.stream(graphName, {maxIterations:20, dampingFactor:0.85}) "
            "YIELD nodeId, score "
            "WITH gds.util.asNode(graphName, nodeId) AS node, score "
            "RETURN node, score ORDER BY score DESC LIMIT $top_k "
        ).replace("{depth}", str(depth))

        try:
            with self.driver.session(database=self.database) as session:
                results = session.run(query, seed_ids=seed_ids, top_k=top_k)
                return [
                    {"node": dict(rec["node"]), "score": rec["score"]}
                    for rec in results
                ]
        except Exception as exc:
            self.logger.debug("PageRank não disponível ou erro: %s", exc)
            return []

    # ---------------------------------------------------------------------
    # Housekeeping
    # ---------------------------------------------------------------------

    def close(self) -> None:
        """Fecha a conexão com o banco Neo4j."""

        try:
            self.driver.close()
            self.logger.debug("Conexão com Neo4j fechada com sucesso.")
        except Exception as exc:  # pragma: no cover
            self.logger.error("Erro ao fechar a conexão com Neo4j: %s", exc)

    # Helper para executar queries ignorando falhas (ex.: autenticação em ambiente de testes)
    def _safe_run(self, query: str, **params):
        start = perf_counter()
        status = "ok"
        try:
            with self.driver.session(database=self.database) as session:
                session.run(query, **params)
        except Exception as exc:  # pragma: no cover
            status = "error"
            self.logger.warning("Falha ao executar query Neo4j: %s", exc)
        finally:
            GRAPH_QUERY_COUNT.labels(status=status).inc()
            GRAPH_QUERY_LATENCY.observe(perf_counter() - start)

    # ------------------------------------------------------------------
    # Operações em lote
    # ------------------------------------------------------------------

    def add_code_elements_bulk(self, elements: List[Dict[str, Any]]) -> None:
        """Adicionar vários nós CodeElement em uma única transação UNWIND."""

        if not elements:
            return

        query = (
            "UNWIND $batch AS elem "
            "MERGE (e:CodeElement {id: elem.id}) "
            "SET e.name = elem.name, "
            "    e.type = elem.type, "
            "    e.file_path = elem.file_path, "
            "    e.content = elem.content, "
            "    e.embedding_id = elem.embedding_id, "
            "    e.metadata = elem.metadata, "
            "    e.updated_at = timestamp()"
        )

        self._safe_run(query, batch=elements)

    def add_relationships_bulk(self, relations: List[GraphRelation]):
        """Criar vários relacionamentos em lote com UNWIND."""

        if not relations:
            return

        rel_batch = [
            {
                "source_id": r.source_id,
                "target_id": r.target_id,
                "type": r.type,
                "properties": r.properties or {},
            }
            for r in relations
        ]

        query = (
            "UNWIND $rels AS rel "
            "MATCH (a {id: rel.source_id}) "
            "MATCH (b {id: rel.target_id}) "
            "MERGE (a)-[r:`__RELTYPE__`]->(b) "
            "SET r += rel.properties"
        )

        # substitui placeholder pelo type dinâmico usando APOC não; precisamos UNWIND por type groups
        # Simplificação: faz loop por tipo
        from collections import defaultdict

        by_type = defaultdict(list)
        for item in rel_batch:
            by_type[item["type"]].append(item)

        for rel_type, batch in by_type.items():
            q = query.replace("__RELTYPE__", rel_type)
            self._safe_run(q, rels=batch)