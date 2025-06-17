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
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "arrozefeijao13",
        database: str = "neo4j",  # Permite selecionar banco em Neo4j 4+
    ) -> None:
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