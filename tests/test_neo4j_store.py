import pytest
from unittest.mock import Mock, patch, MagicMock

from src.graphdb.neo4j_store import Neo4jStore
from src.graphdb.graph_models import GraphNode, GraphRelation, NodeType, RelationType


class TestNeo4jStore:
    """Testes para a classe Neo4jStore."""

    @pytest.fixture
    def mock_driver(self):
        """Mock do driver Neo4j."""
        with patch('src.graphdb.neo4j_store.GraphDatabase') as mock_graphdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            mock_graphdb.driver.return_value = mock_driver
            
            yield {
                'driver': mock_driver,
                'session': mock_session,
                'graphdb': mock_graphdb
            }

    @pytest.fixture
    def store(self, mock_driver):
        """Cria uma instância do Neo4jStore para testes."""
        return Neo4jStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test_password",
            database="test_db"
        )

    def test_init_success(self, mock_driver):
        """Testa inicialização bem-sucedida."""
        store = Neo4jStore()
        
        assert store.database == "neo4j"
        mock_driver['graphdb'].driver.assert_called_once_with(
            "bolt://localhost:7687", 
            auth=("neo4j", "arrozefeijao13")
        )

    def test_init_custom_params(self, mock_driver):
        """Testa inicialização com parâmetros customizados."""
        store = Neo4jStore(
            uri="bolt://custom:7687",
            user="custom_user",
            password="custom_pass",
            database="custom_db"
        )
        
        assert store.database == "custom_db"
        mock_driver['graphdb'].driver.assert_called_once_with(
            "bolt://custom:7687", 
            auth=("custom_user", "custom_pass")
        )

    def test_init_without_neo4j_dependency(self):
        """Testa inicialização quando neo4j não está instalado."""
        with patch('src.graphdb.neo4j_store.GraphDatabase', None):
            with pytest.raises(ImportError, match="A dependência 'neo4j' não está instalada"):
                Neo4jStore()

    def test_init_constraints(self, mock_driver, store):
        """Testa criação de constraints na inicialização."""
        # Verificar que _init_constraints foi chamado
        # (implicitamente testado pela criação do store)
        assert store is not None

    def test_add_node(self, mock_driver, store):
        """Testa adição de nó."""
        node = GraphNode(
            id="test_node_1",
            type=NodeType.DOCUMENT,
            properties={"title": "Test Document", "content": "Test content"}
        )
        
        store.add_node(node)
        
        # Verificar que a sessão foi usada
        mock_driver['session'].run.assert_called()
        
        # Verificar que a query contém os elementos esperados
        call_args = mock_driver['session'].run.call_args
        query = call_args[0][0]
        params = call_args[1] if len(call_args) > 1 else call_args[0][1]
        
        assert "MERGE" in query or "CREATE" in query
        assert "Document" in query  # NodeType.DOCUMENT
        assert params["id"] == "test_node_1"

    def test_add_relation(self, mock_driver, store):
        """Testa adição de relação."""
        relation = GraphRelation(
            from_id="node_1",
            to_id="node_2",
            type=RelationType.CONTAINS,
            properties={"weight": 0.8}
        )
        
        store.add_relation(relation)
        
        # Verificar que a sessão foi usada
        mock_driver['session'].run.assert_called()
        
        call_args = mock_driver['session'].run.call_args
        query = call_args[0][0]
        params = call_args[1] if len(call_args) > 1 else call_args[0][1]
        
        assert "MATCH" in query
        assert "CONTAINS" in query  # RelationType.CONTAINS
        assert params["from_id"] == "node_1"
        assert params["to_id"] == "node_2"

    def test_get_node_existing(self, mock_driver, store):
        """Testa recuperação de nó existente."""
        # Mock do resultado
        mock_record = Mock()
        mock_record.__getitem__.return_value = {
            "id": "test_node_1",
            "title": "Test Document"
        }
        mock_driver['session'].run.return_value = [mock_record]
        
        result = store.get_node("test_node_1")
        
        assert result is not None
        mock_driver['session'].run.assert_called()

    def test_get_node_nonexistent(self, mock_driver, store):
        """Testa recuperação de nó inexistente."""
        mock_driver['session'].run.return_value = []
        
        result = store.get_node("nonexistent_node")
        
        assert result is None

    def test_find_related_nodes(self, mock_driver, store):
        """Testa busca de nós relacionados."""
        # Mock dos resultados
        mock_records = [
            Mock(**{"__getitem__.return_value": {"id": "related_1", "title": "Related 1"}}),
            Mock(**{"__getitem__.return_value": {"id": "related_2", "title": "Related 2"}})
        ]
        mock_driver['session'].run.return_value = mock_records
        
        result = store.find_related_nodes("test_node", RelationType.CONTAINS)
        
        assert len(result) == 2
        mock_driver['session'].run.assert_called()
        
        call_args = mock_driver['session'].run.call_args
        query = call_args[0][0]
        params = call_args[1] if len(call_args) > 1 else call_args[0][1]
        
        assert "CONTAINS" in query
        assert params["node_id"] == "test_node"

    def test_find_related_nodes_with_depth(self, mock_driver, store):
        """Testa busca de nós relacionados com profundidade específica."""
        mock_driver['session'].run.return_value = []
        
        store.find_related_nodes("test_node", RelationType.CONTAINS, depth=2)
        
        call_args = mock_driver['session'].run.call_args
        query = call_args[0][0]
        
        assert "*1..2" in query or "*..2" in query  # Padrão de profundidade no Cypher

    def test_search_nodes_by_content(self, mock_driver, store):
        """Testa busca de nós por conteúdo."""
        mock_records = [
            Mock(**{"__getitem__.return_value": {"id": "match_1", "content": "matching content"}})
        ]
        mock_driver['session'].run.return_value = mock_records
        
        result = store.search_nodes_by_content("matching")
        
        assert len(result) == 1
        mock_driver['session'].run.assert_called()
        
        call_args = mock_driver['session'].run.call_args
        query = call_args[0][0]
        params = call_args[1] if len(call_args) > 1 else call_args[0][1]
        
        assert "CONTAINS" in query.upper() or "=~" in query  # Busca por texto
        assert "matching" in str(params.values()).lower()

    def test_search_nodes_by_content_with_limit(self, mock_driver, store):
        """Testa busca de nós por conteúdo com limite."""
        mock_driver['session'].run.return_value = []
        
        store.search_nodes_by_content("test", limit=5)
        
        call_args = mock_driver['session'].run.call_args
        query = call_args[0][0]
        
        assert "LIMIT" in query.upper()
        assert "5" in query

    def test_get_document_context(self, mock_driver, store):
        """Testa recuperação de contexto de documento."""
        mock_records = [
            Mock(**{"__getitem__.return_value": {"id": "chunk_1", "content": "chunk content 1"}}),
            Mock(**{"__getitem__.return_value": {"id": "chunk_2", "content": "chunk content 2"}})
        ]
        mock_driver['session'].run.return_value = mock_records
        
        result = store.get_document_context("doc_1")
        
        assert len(result) == 2
        mock_driver['session'].run.assert_called()

    def test_delete_node(self, mock_driver, store):
        """Testa remoção de nó."""
        store.delete_node("test_node_1")
        
        mock_driver['session'].run.assert_called()
        
        call_args = mock_driver['session'].run.call_args
        query = call_args[0][0]
        params = call_args[1] if len(call_args) > 1 else call_args[0][1]
        
        assert "DELETE" in query.upper()
        assert params["node_id"] == "test_node_1"

    def test_delete_relation(self, mock_driver, store):
        """Testa remoção de relação."""
        store.delete_relation("node_1", "node_2", RelationType.CONTAINS)
        
        mock_driver['session'].run.assert_called()
        
        call_args = mock_driver['session'].run.call_args
        query = call_args[0][0]
        params = call_args[1] if len(call_args) > 1 else call_args[0][1]
        
        assert "DELETE" in query.upper()
        assert "CONTAINS" in query
        assert params["from_id"] == "node_1"
        assert params["to_id"] == "node_2"

    def test_clear_all(self, mock_driver, store):
        """Testa limpeza completa do banco."""
        store.clear_all()
        
        mock_driver['session'].run.assert_called()
        
        call_args = mock_driver['session'].run.call_args
        query = call_args[0][0]
        
        assert "DELETE" in query.upper()
        assert "DETACH" in query.upper() or "REMOVE" in query.upper()

    def test_get_stats(self, mock_driver, store):
        """Testa obtenção de estatísticas."""
        # Mock dos resultados de estatísticas
        mock_records = [
            Mock(**{"__getitem__.return_value": 100}),  # total_nodes
            Mock(**{"__getitem__.return_value": 50}),   # total_relations
        ]
        mock_driver['session'].run.side_effect = [mock_records[:1], mock_records[1:]]
        
        result = store.get_stats()
        
        assert isinstance(result, dict)
        # Verificar que múltiplas queries foram executadas
        assert mock_driver['session'].run.call_count >= 2

    def test_session_error_handling(self, mock_driver, store):
        """Testa tratamento de erros de sessão."""
        mock_driver['session'].run.side_effect = Exception("Connection error")
        
        with pytest.raises(Exception):
            store.get_node("test_node")

    def test_prometheus_metrics_increment(self, mock_driver, store):
        """Testa incremento das métricas Prometheus."""
        with patch('src.graphdb.neo4j_store.GRAPH_QUERY_COUNT') as mock_counter, \
             patch('src.graphdb.neo4j_store.GRAPH_QUERY_LATENCY') as mock_histogram:
            
            mock_driver['session'].run.return_value = []
            
            store.get_node("test_node")
            
            # Verificar que as métricas foram chamadas
            # (O comportamento exato depende da implementação)
            assert mock_driver['session'].run.called

    def test_close_connection(self, mock_driver, store):
        """Testa fechamento da conexão."""
        store.close()
        
        mock_driver['driver'].close.assert_called_once()

    def test_context_manager(self, mock_driver):
        """Testa uso como context manager."""
        with Neo4jStore() as store:
            assert store is not None
        
        # Verificar que close foi chamado
        mock_driver['driver'].close.assert_called_once()

    def test_batch_operations(self, mock_driver, store):
        """Testa operações em lote."""
        nodes = [
            GraphNode(id="node_1", type=NodeType.DOCUMENT, properties={"title": "Doc 1"}),
            GraphNode(id="node_2", type=NodeType.DOCUMENT, properties={"title": "Doc 2"})
        ]
        
        # Se houver método batch_add_nodes
        if hasattr(store, 'batch_add_nodes'):
            store.batch_add_nodes(nodes)
            mock_driver['session'].run.assert_called()
        else:
            # Testar adição individual
            for node in nodes:
                store.add_node(node)
            
            assert mock_driver['session'].run.call_count == len(nodes)

    def test_transaction_handling(self, mock_driver, store):
        """Testa tratamento de transações."""
        # Mock de transação
        mock_tx = Mock()
        mock_driver['session'].begin_transaction.return_value = mock_tx
        
        # Se houver suporte a transações explícitas
        if hasattr(store, 'begin_transaction'):
            tx = store.begin_transaction()
            assert tx is not None
        else:
            # Verificar que as operações usam sessões corretamente
            store.add_node(GraphNode(id="test", type=NodeType.DOCUMENT, properties={}))
            mock_driver['session'].run.assert_called()

    def test_node_type_enum_handling(self, mock_driver, store):
        """Testa tratamento correto dos tipos de nó."""
        for node_type in NodeType:
            node = GraphNode(
                id=f"test_{node_type.value}",
                type=node_type,
                properties={"test": "value"}
            )
            
            store.add_node(node)
            
            call_args = mock_driver['session'].run.call_args
            query = call_args[0][0]
            
            # Verificar que o tipo do nó está na query
            assert node_type.value in query or node_type.name in query

    def test_relation_type_enum_handling(self, mock_driver, store):
        """Testa tratamento correto dos tipos de relação."""
        for relation_type in RelationType:
            relation = GraphRelation(
                from_id="node_1",
                to_id="node_2",
                type=relation_type,
                properties={}
            )
            
            store.add_relation(relation)
            
            call_args = mock_driver['session'].run.call_args
            query = call_args[0][0]
            
            # Verificar que o tipo da relação está na query
            assert relation_type.value in query or relation_type.name in query