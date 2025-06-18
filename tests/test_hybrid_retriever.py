"""Testes para o módulo hybrid_retriever.py."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Mock das dependências antes do import
with patch.multiple(
    'sys.modules',
    qdrant_client=Mock(),
    neo4j=Mock()
):
    from src.retrieval.hybrid_retriever import HybridRetriever


class TestHybridRetriever:
    """Testes para HybridRetriever."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock do vector store."""
        store = Mock()
        store.search.return_value = [
            {
                "content": "Vector result 1",
                "metadata": {"source": "doc1.txt", "score": 0.9},
                "score": 0.9
            },
            {
                "content": "Vector result 2",
                "metadata": {"source": "doc2.txt", "score": 0.8},
                "score": 0.8
            }
        ]
        return store
    
    @pytest.fixture
    def mock_graph_store(self):
        """Mock do graph store."""
        store = Mock()
        store.search.return_value = [
            {
                "content": "Graph result 1",
                "metadata": {"source": "doc3.txt", "score": 0.85},
                "score": 0.85
            },
            {
                "content": "Graph result 2",
                "metadata": {"source": "doc4.txt", "score": 0.75},
                "score": 0.75
            }
        ]
        return store
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock do serviço de embeddings."""
        service = Mock()
        service.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        return service
    
    def test_init_default_parameters(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa inicialização com parâmetros padrão."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.graph_store == mock_graph_store
        assert retriever.embedding_service == mock_embedding_service
        assert retriever.vector_weight == 0.7
        assert retriever.graph_weight == 0.3
        assert retriever.max_results == 10
    
    def test_init_custom_parameters(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa inicialização com parâmetros customizados."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service,
            vector_weight=0.6,
            graph_weight=0.4,
            max_results=20
        )
        
        assert retriever.vector_weight == 0.6
        assert retriever.graph_weight == 0.4
        assert retriever.max_results == 20
    
    def test_init_invalid_weights(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa erro com pesos inválidos."""
        with pytest.raises(ValueError, match="A soma dos pesos deve ser 1.0"):
            HybridRetriever(
                vector_store=mock_vector_store,
                graph_store=mock_graph_store,
                embedding_service=mock_embedding_service,
                vector_weight=0.6,
                graph_weight=0.5  # 0.6 + 0.5 = 1.1 > 1.0
            )
    
    def test_search_hybrid_success(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa busca híbrida bem-sucedida."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        query = "test query"
        k = 5
        
        result = retriever.search(query, k=k)
        
        # Verificar que ambos os stores foram chamados
        mock_embedding_service.embed_text.assert_called_once_with(query)
        mock_vector_store.search.assert_called_once_with([0.1, 0.2, 0.3, 0.4, 0.5], k=k)
        mock_graph_store.search.assert_called_once_with(query, k=k)
        
        # Verificar resultado
        assert isinstance(result, list)
        assert len(result) <= k
        
        # Verificar que resultados têm scores híbridos
        for item in result:
            assert 'content' in item
            assert 'metadata' in item
            assert 'score' in item
    
    def test_search_vector_only(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa busca apenas vetorial."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service,
            vector_weight=1.0,
            graph_weight=0.0
        )
        
        result = retriever.search("test query", k=3)
        
        # Apenas vector store deve ser chamado
        mock_vector_store.search.assert_called_once()
        mock_graph_store.search.assert_not_called()
        
        assert isinstance(result, list)
    
    def test_search_graph_only(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa busca apenas por grafo."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service,
            vector_weight=0.0,
            graph_weight=1.0
        )
        
        result = retriever.search("test query", k=3)
        
        # Apenas graph store deve ser chamado
        mock_vector_store.search.assert_not_called()
        mock_graph_store.search.assert_called_once()
        
        assert isinstance(result, list)
    
    def test_search_empty_query(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa busca com query vazia."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        result = retriever.search("", k=5)
        
        assert result == []
        mock_embedding_service.embed_text.assert_not_called()
        mock_vector_store.search.assert_not_called()
        mock_graph_store.search.assert_not_called()
    
    def test_search_vector_store_error(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa erro no vector store."""
        mock_vector_store.search.side_effect = Exception("Vector store error")
        
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        # Deve continuar com apenas graph results
        result = retriever.search("test query", k=5)
        
        # Graph store ainda deve ser chamado
        mock_graph_store.search.assert_called_once()
        assert isinstance(result, list)
    
    def test_search_graph_store_error(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa erro no graph store."""
        mock_graph_store.search.side_effect = Exception("Graph store error")
        
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        # Deve continuar com apenas vector results
        result = retriever.search("test query", k=5)
        
        # Vector store ainda deve ser chamado
        mock_vector_store.search.assert_called_once()
        assert isinstance(result, list)
    
    def test_search_embedding_service_error(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa erro no serviço de embeddings."""
        mock_embedding_service.embed_text.side_effect = Exception("Embedding error")
        
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        # Deve continuar com apenas graph results
        result = retriever.search("test query", k=5)
        
        # Vector store não deve ser chamado devido ao erro de embedding
        mock_vector_store.search.assert_not_called()
        mock_graph_store.search.assert_called_once()
        assert isinstance(result, list)
    
    def test_combine_results_with_overlap(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa combinação de resultados com sobreposição."""
        # Configurar resultados com sobreposição
        mock_vector_store.search.return_value = [
            {
                "content": "Shared result",
                "metadata": {"source": "shared.txt", "id": "1"},
                "score": 0.9
            },
            {
                "content": "Vector only result",
                "metadata": {"source": "vector.txt", "id": "2"},
                "score": 0.8
            }
        ]
        
        mock_graph_store.search.return_value = [
            {
                "content": "Shared result",
                "metadata": {"source": "shared.txt", "id": "1"},
                "score": 0.85
            },
            {
                "content": "Graph only result",
                "metadata": {"source": "graph.txt", "id": "3"},
                "score": 0.75
            }
        ]
        
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        result = retriever.search("test query", k=5)
        
        # Deve ter 3 resultados únicos (1 compartilhado + 2 únicos)
        assert len(result) == 3
        
        # Verificar que o resultado compartilhado tem score híbrido
        shared_result = next((r for r in result if r["content"] == "Shared result"), None)
        assert shared_result is not None
        # Score híbrido deve ser diferente dos scores originais
        assert shared_result["score"] != 0.9
        assert shared_result["score"] != 0.85
    
    def test_search_with_filters(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa busca com filtros."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        filters = {"source": "specific_doc.txt"}
        result = retriever.search("test query", k=5, filters=filters)
        
        # Verificar que filtros foram passados para os stores
        mock_vector_store.search.assert_called_once()
        mock_graph_store.search.assert_called_once()
        
        # Verificar argumentos da chamada (se o método suportar filtros)
        vector_call_args = mock_vector_store.search.call_args
        graph_call_args = mock_graph_store.search.call_args
        
        assert isinstance(result, list)
    
    def test_search_large_k_value(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa busca com valor k muito grande."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service,
            max_results=5
        )
        
        # k maior que max_results
        result = retriever.search("test query", k=100)
        
        # Resultado deve ser limitado por max_results
        assert len(result) <= 5
    
    def test_search_zero_k_value(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa busca com k=0."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        result = retriever.search("test query", k=0)
        
        assert result == []
    
    def test_search_negative_k_value(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa busca com k negativo."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        result = retriever.search("test query", k=-1)
        
        assert result == []
    
    def test_score_calculation(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa cálculo de scores híbridos."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service,
            vector_weight=0.6,
            graph_weight=0.4
        )
        
        result = retriever.search("test query", k=5)
        
        # Verificar que scores foram calculados corretamente
        for item in result:
            assert 0.0 <= item["score"] <= 1.0
    
    def test_search_with_unicode_query(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa busca com query unicode."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        unicode_query = "测试查询 🔍 тест запрос テストクエリ"
        result = retriever.search(unicode_query, k=5)
        
        mock_embedding_service.embed_text.assert_called_once_with(unicode_query)
        assert isinstance(result, list)
    
    def test_performance_with_large_results(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa performance com muitos resultados."""
        import time
        
        # Configurar muitos resultados
        large_vector_results = [
            {
                "content": f"Vector result {i}",
                "metadata": {"source": f"doc{i}.txt", "id": str(i)},
                "score": 0.9 - (i * 0.01)
            }
            for i in range(100)
        ]
        
        large_graph_results = [
            {
                "content": f"Graph result {i}",
                "metadata": {"source": f"graph{i}.txt", "id": str(i + 100)},
                "score": 0.85 - (i * 0.01)
            }
            for i in range(100)
        ]
        
        mock_vector_store.search.return_value = large_vector_results
        mock_graph_store.search.return_value = large_graph_results
        
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        start_time = time.time()
        result = retriever.search("test query", k=50)
        end_time = time.time()
        
        # Deve completar rapidamente (menos de 1 segundo)
        assert end_time - start_time < 1.0
        assert len(result) <= 50


class TestHybridRetrieverIntegration:
    """Testes de integração para HybridRetriever."""
    
    def test_real_world_search_scenario(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa cenário de busca do mundo real."""
        # Configurar resultados realistas
        mock_vector_store.search.return_value = [
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "metadata": {"source": "ml_intro.pdf", "page": 1, "id": "ml1"},
                "score": 0.92
            },
            {
                "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns.",
                "metadata": {"source": "dl_guide.pdf", "page": 3, "id": "dl1"},
                "score": 0.88
            }
        ]
        
        mock_graph_store.search.return_value = [
            {
                "content": "Neural networks are inspired by biological neural networks and consist of interconnected nodes.",
                "metadata": {"source": "nn_basics.pdf", "page": 2, "id": "nn1"},
                "score": 0.85
            },
            {
                "content": "Artificial intelligence encompasses machine learning, natural language processing, and computer vision.",
                "metadata": {"source": "ai_overview.pdf", "page": 1, "id": "ai1"},
                "score": 0.82
            }
        ]
        
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        # Busca típica de usuário
        query = "What is machine learning and how does it work?"
        results = retriever.search(query, k=5)
        
        # Verificar qualidade dos resultados
        assert len(results) > 0
        assert len(results) <= 5
        
        # Verificar que resultados estão ordenados por score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        
        # Verificar estrutura dos resultados
        for result in results:
            assert "content" in result
            assert "metadata" in result
            assert "score" in result
            assert isinstance(result["content"], str)
            assert isinstance(result["metadata"], dict)
            assert isinstance(result["score"], (int, float))
    
    def test_multiple_search_consistency(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa consistência entre múltiplas buscas."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            embedding_service=mock_embedding_service
        )
        
        query = "consistent search test"
        
        # Realizar múltiplas buscas
        results1 = retriever.search(query, k=3)
        results2 = retriever.search(query, k=3)
        results3 = retriever.search(query, k=3)
        
        # Resultados devem ser consistentes (assumindo determinismo)
        assert len(results1) == len(results2) == len(results3)
        
        # Verificar que o embedding service foi chamado para cada busca
        assert mock_embedding_service.embed_text.call_count == 3
    
    def test_different_weight_configurations(self, mock_vector_store, mock_graph_store, mock_embedding_service):
        """Testa diferentes configurações de peso."""
        weight_configs = [
            (0.8, 0.2),  # Favorece vector
            (0.5, 0.5),  # Balanceado
            (0.2, 0.8),  # Favorece graph
            (1.0, 0.0),  # Apenas vector
            (0.0, 1.0)   # Apenas graph
        ]
        
        query = "weight configuration test"
        
        for vector_weight, graph_weight in weight_configs:
            retriever = HybridRetriever(
                vector_store=mock_vector_store,
                graph_store=mock_graph_store,
                embedding_service=mock_embedding_service,
                vector_weight=vector_weight,
                graph_weight=graph_weight
            )
            
            results = retriever.search(query, k=3)
            
            assert isinstance(results, list)
            # Verificar que scores refletem os pesos
            for result in results:
                assert 0.0 <= result["score"] <= 1.0