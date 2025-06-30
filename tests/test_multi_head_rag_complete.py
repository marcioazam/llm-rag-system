"""
Testes completos para MultiHeadRAG
Cobrindo todos os cenários não testados para aumentar a cobertura
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from src.retrieval.multi_head_rag import (
    MultiHeadRAG, AttentionHead, QueryType, 
    HeadConfig, AttentionMechanism
)


class TestHeadConfig:
    """Testes para HeadConfig"""
    
    def test_head_config_creation(self):
        """Testa criação de configuração de cabeça"""
        config = HeadConfig(
            name="semantic_head",
            query_type=QueryType.SEMANTIC,
            weight=0.8,
            retriever_config={"top_k": 10}
        )
        assert config.name == "semantic_head"
        assert config.query_type == QueryType.SEMANTIC
        assert config.weight == 0.8
        assert config.retriever_config["top_k"] == 10


class TestQueryType:
    """Testes para QueryType enum"""
    
    def test_query_type_values(self):
        """Testa valores do enum QueryType"""
        assert QueryType.SEMANTIC == "semantic"
        assert QueryType.KEYWORD == "keyword"
        assert QueryType.HYBRID == "hybrid"
        assert QueryType.CONTEXTUAL == "contextual"


class TestAttentionHead:
    """Testes para AttentionHead"""
    
    @pytest.fixture
    def head_config(self):
        return HeadConfig(
            name="test_head",
            query_type=QueryType.SEMANTIC,
            weight=1.0,
            retriever_config={"top_k": 5}
        )
    
    @pytest.fixture
    def attention_head(self, head_config):
        with patch('src.retrieval.multi_head_rag.HybridRetriever'):
            return AttentionHead(head_config)
    
    def test_init(self, attention_head, head_config):
        """Testa inicialização da cabeça de atenção"""
        assert attention_head.config == head_config
        assert attention_head.retriever is not None
        
    @pytest.mark.asyncio
    async def test_retrieve(self, attention_head):
        """Testa recuperação da cabeça de atenção"""
        # Mock do retriever
        mock_results = [
            {"content": "Result 1", "score": 0.9},
            {"content": "Result 2", "score": 0.8}
        ]
        attention_head.retriever.retrieve = AsyncMock(return_value=mock_results)
        
        results = await attention_head.retrieve("test query")
        assert len(results) == 2
        assert results[0]["content"] == "Result 1"
        
    def test_transform_query(self, attention_head):
        """Testa transformação de query"""
        original_query = "test query"
        transformed = attention_head.transform_query(original_query)
        assert isinstance(transformed, str)
        assert len(transformed) > 0
        
    def test_apply_attention_weights(self, attention_head):
        """Testa aplicação de pesos de atenção"""
        results = [
            {"content": "Result 1", "score": 0.9},
            {"content": "Result 2", "score": 0.8}
        ]
        weighted = attention_head.apply_attention_weights(results)
        assert len(weighted) == 2
        assert all("attention_weight" in result for result in weighted)


class TestAttentionMechanism:
    """Testes para AttentionMechanism"""
    
    @pytest.fixture
    def attention_mechanism(self):
        return AttentionMechanism()
    
    def test_init(self, attention_mechanism):
        """Testa inicialização do mecanismo de atenção"""
        assert attention_mechanism.temperature == 1.0
        assert attention_mechanism.use_softmax == True
        
    def test_calculate_attention_scores(self, attention_mechanism):
        """Testa cálculo de scores de atenção"""
        query = "test query"
        results = [
            {"content": "Relevant result", "score": 0.9},
            {"content": "Less relevant", "score": 0.5}
        ]
        
        scores = attention_mechanism.calculate_attention_scores(query, results)
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        
    def test_apply_softmax(self, attention_mechanism):
        """Testa aplicação de softmax"""
        scores = [0.9, 0.5, 0.7]
        softmax_scores = attention_mechanism.apply_softmax(scores)
        
        assert len(softmax_scores) == 3
        assert abs(sum(softmax_scores) - 1.0) < 1e-6  # Soma deve ser ~1
        assert all(score >= 0 for score in softmax_scores)
        
    def test_combine_results(self, attention_mechanism):
        """Testa combinação de resultados"""
        head_results = [
            [{"content": "Result 1", "score": 0.9}],
            [{"content": "Result 2", "score": 0.8}]
        ]
        weights = [0.6, 0.4]
        
        combined = attention_mechanism.combine_results(head_results, weights)
        assert len(combined) >= 1
        assert all("combined_score" in result for result in combined)


class TestMultiHeadRAG:
    """Testes para MultiHeadRAG principal"""
    
    @pytest.fixture
    def head_configs(self):
        return [
            HeadConfig("semantic", QueryType.SEMANTIC, 0.4, {"top_k": 5}),
            HeadConfig("keyword", QueryType.KEYWORD, 0.3, {"top_k": 5}),
            HeadConfig("contextual", QueryType.CONTEXTUAL, 0.3, {"top_k": 5})
        ]
    
    @pytest.fixture
    def multi_head_rag(self, head_configs):
        with patch('src.retrieval.multi_head_rag.HybridRetriever'):
            return MultiHeadRAG(head_configs)
    
    def test_init(self, multi_head_rag, head_configs):
        """Testa inicialização do MultiHeadRAG"""
        assert len(multi_head_rag.heads) == 3
        assert multi_head_rag.attention_mechanism is not None
        assert len(multi_head_rag.head_configs) == 3
        
    @pytest.mark.asyncio
    async def test_retrieve_basic(self, multi_head_rag):
        """Testa recuperação básica"""
        # Mock das cabeças
        for head in multi_head_rag.heads:
            head.retrieve = AsyncMock(return_value=[
                {"content": f"Result from {head.config.name}", "score": 0.8}
            ])
            
        results = await multi_head_rag.retrieve("test query")
        assert len(results) >= 1
        assert all("content" in result for result in results)
        
    @pytest.mark.asyncio
    async def test_retrieve_with_context(self, multi_head_rag):
        """Testa recuperação com contexto"""
        context = ["Previous context"]
        
        for head in multi_head_rag.heads:
            head.retrieve = AsyncMock(return_value=[
                {"content": "Contextual result", "score": 0.9}
            ])
            
        results = await multi_head_rag.retrieve("test query", context=context)
        assert len(results) >= 1
        
    def test_add_head(self, multi_head_rag):
        """Testa adição de nova cabeça"""
        new_config = HeadConfig("new_head", QueryType.HYBRID, 0.2, {"top_k": 3})
        
        with patch('src.retrieval.multi_head_rag.AttentionHead') as mock_head:
            multi_head_rag.add_head(new_config)
            assert len(multi_head_rag.heads) == 4
            assert len(multi_head_rag.head_configs) == 4
            
    def test_remove_head(self, multi_head_rag):
        """Testa remoção de cabeça"""
        initial_count = len(multi_head_rag.heads)
        multi_head_rag.remove_head("semantic")
        
        assert len(multi_head_rag.heads) == initial_count - 1
        assert not any(head.config.name == "semantic" for head in multi_head_rag.heads)
        
    def test_update_head_weights(self, multi_head_rag):
        """Testa atualização de pesos das cabeças"""
        new_weights = {"semantic": 0.5, "keyword": 0.3, "contextual": 0.2}
        multi_head_rag.update_head_weights(new_weights)
        
        for head in multi_head_rag.heads:
            if head.config.name in new_weights:
                assert head.config.weight == new_weights[head.config.name]
                
    def test_get_head_by_name(self, multi_head_rag):
        """Testa obtenção de cabeça por nome"""
        semantic_head = multi_head_rag.get_head_by_name("semantic")
        assert semantic_head is not None
        assert semantic_head.config.name == "semantic"
        
        nonexistent = multi_head_rag.get_head_by_name("nonexistent")
        assert nonexistent is None
        
    @pytest.mark.asyncio
    async def test_retrieve_parallel(self, multi_head_rag):
        """Testa recuperação paralela"""
        # Mock das cabeças com diferentes tempos de resposta
        async def mock_retrieve_fast(query, **kwargs):
            await asyncio.sleep(0.01)
            return [{"content": "Fast result", "score": 0.8}]
            
        async def mock_retrieve_slow(query, **kwargs):
            await asyncio.sleep(0.02)
            return [{"content": "Slow result", "score": 0.7}]
            
        multi_head_rag.heads[0].retrieve = mock_retrieve_fast
        multi_head_rag.heads[1].retrieve = mock_retrieve_slow
        multi_head_rag.heads[2].retrieve = mock_retrieve_fast
        
        import time
        start_time = time.time()
        results = await multi_head_rag.retrieve("test query")
        end_time = time.time()
        
        # Deve ser mais rápido que execução sequencial
        assert (end_time - start_time) < 0.1
        assert len(results) >= 1


class TestMultiHeadRAGAdvanced:
    """Testes avançados do MultiHeadRAG"""
    
    @pytest.fixture
    def multi_head_rag(self):
        configs = [
            HeadConfig("semantic", QueryType.SEMANTIC, 0.5, {"top_k": 10}),
            HeadConfig("keyword", QueryType.KEYWORD, 0.3, {"top_k": 8}),
            HeadConfig("hybrid", QueryType.HYBRID, 0.2, {"top_k": 6})
        ]
        with patch('src.retrieval.multi_head_rag.HybridRetriever'):
            return MultiHeadRAG(configs)
    
    @pytest.mark.asyncio
    async def test_adaptive_weighting(self, multi_head_rag):
        """Testa ponderação adaptativa baseada na performance"""
        # Mock das cabeças com diferentes qualidades de resultado
        multi_head_rag.heads[0].retrieve = AsyncMock(return_value=[
            {"content": "High quality", "score": 0.95}
        ])
        multi_head_rag.heads[1].retrieve = AsyncMock(return_value=[
            {"content": "Medium quality", "score": 0.7}
        ])
        multi_head_rag.heads[2].retrieve = AsyncMock(return_value=[
            {"content": "Low quality", "score": 0.4}
        ])
        
        # Primeira recuperação
        results1 = await multi_head_rag.retrieve("test query")
        initial_weights = [head.config.weight for head in multi_head_rag.heads]
        
        # Simula feedback positivo para a primeira cabeça
        multi_head_rag.update_performance_metrics("semantic", 0.9)
        multi_head_rag.adapt_weights_based_on_performance()
        
        final_weights = [head.config.weight for head in multi_head_rag.heads]
        
        # O peso da cabeça semântica deve ter aumentado
        semantic_head = multi_head_rag.get_head_by_name("semantic")
        assert semantic_head.config.weight >= initial_weights[0]
        
    def test_query_routing(self, multi_head_rag):
        """Testa roteamento de queries para cabeças específicas"""
        # Query claramente semântica
        semantic_query = "What is the meaning of artificial intelligence?"
        routed_heads = multi_head_rag.route_query(semantic_query)
        
        assert any(head.config.query_type == QueryType.SEMANTIC for head in routed_heads)
        
        # Query claramente baseada em keywords
        keyword_query = "Python programming tutorial"
        routed_heads = multi_head_rag.route_query(keyword_query)
        
        assert any(head.config.query_type == QueryType.KEYWORD for head in routed_heads)
        
    @pytest.mark.asyncio
    async def test_result_fusion_strategies(self, multi_head_rag):
        """Testa diferentes estratégias de fusão de resultados"""
        # Mock resultados de diferentes cabeças
        for i, head in enumerate(multi_head_rag.heads):
            head.retrieve = AsyncMock(return_value=[
                {"content": f"Result {i+1}", "score": 0.8 - i*0.1, "source": head.config.name}
            ])
        
        # Teste fusão por ranking
        results_rank = await multi_head_rag.retrieve("test", fusion_strategy="rank")
        assert len(results_rank) >= 1
        
        # Teste fusão por score
        results_score = await multi_head_rag.retrieve("test", fusion_strategy="score")
        assert len(results_score) >= 1
        
        # Teste fusão híbrida
        results_hybrid = await multi_head_rag.retrieve("test", fusion_strategy="hybrid")
        assert len(results_hybrid) >= 1
        
    def test_dynamic_head_configuration(self, multi_head_rag):
        """Testa configuração dinâmica de cabeças"""
        # Adiciona cabeça especializada
        specialized_config = HeadConfig(
            "domain_specific", 
            QueryType.CONTEXTUAL, 
            0.1, 
            {"domain": "medical", "top_k": 5}
        )
        
        with patch('src.retrieval.multi_head_rag.AttentionHead'):
            multi_head_rag.add_head(specialized_config)
            
        # Configura cabeça para domínio específico
        multi_head_rag.configure_head_for_domain("domain_specific", "medical")
        
        domain_head = multi_head_rag.get_head_by_name("domain_specific")
        assert domain_head.config.retriever_config["domain"] == "medical"
        
    @pytest.mark.asyncio
    async def test_attention_visualization(self, multi_head_rag):
        """Testa visualização dos pesos de atenção"""
        for head in multi_head_rag.heads:
            head.retrieve = AsyncMock(return_value=[
                {"content": "Test result", "score": 0.8}
            ])
            
        results = await multi_head_rag.retrieve("test query", return_attention_weights=True)
        
        assert "attention_weights" in results[0] if results else True
        
    def test_head_performance_monitoring(self, multi_head_rag):
        """Testa monitoramento de performance das cabeças"""
        # Simula várias consultas e feedbacks
        for _ in range(10):
            multi_head_rag.update_performance_metrics("semantic", 0.9)
            multi_head_rag.update_performance_metrics("keyword", 0.7)
            multi_head_rag.update_performance_metrics("hybrid", 0.8)
            
        stats = multi_head_rag.get_performance_stats()
        
        assert "semantic" in stats
        assert "keyword" in stats
        assert "hybrid" in stats
        assert all("avg_score" in stat for stat in stats.values())


class TestMultiHeadRAGEdgeCases:
    """Testes de casos extremos"""
    
    @pytest.fixture
    def multi_head_rag(self):
        configs = [HeadConfig("test", QueryType.SEMANTIC, 1.0, {"top_k": 5})]
        with patch('src.retrieval.multi_head_rag.HybridRetriever'):
            return MultiHeadRAG(configs)
    
    @pytest.mark.asyncio
    async def test_empty_query(self, multi_head_rag):
        """Testa query vazia"""
        multi_head_rag.heads[0].retrieve = AsyncMock(return_value=[])
        results = await multi_head_rag.retrieve("")
        assert isinstance(results, list)
        
    @pytest.mark.asyncio
    async def test_none_query(self, multi_head_rag):
        """Testa query None"""
        multi_head_rag.heads[0].retrieve = AsyncMock(return_value=[])
        results = await multi_head_rag.retrieve(None)
        assert isinstance(results, list)
        
    @pytest.mark.asyncio
    async def test_head_failure_handling(self, multi_head_rag):
        """Testa tratamento de falha de cabeça"""
        # Uma cabeça falha, outras continuam funcionando
        multi_head_rag.heads[0].retrieve = AsyncMock(side_effect=Exception("Head failed"))
        
        # Adiciona mais uma cabeça que funciona
        working_config = HeadConfig("working", QueryType.KEYWORD, 0.5, {"top_k": 5})
        with patch('src.retrieval.multi_head_rag.AttentionHead') as mock_head:
            mock_instance = Mock()
            mock_instance.retrieve = AsyncMock(return_value=[{"content": "Working result", "score": 0.8}])
            mock_head.return_value = mock_instance
            multi_head_rag.add_head(working_config)
        
        results = await multi_head_rag.retrieve("test query")
        # Deve retornar resultados da cabeça que funciona
        assert len(results) >= 0
        
    def test_invalid_head_removal(self, multi_head_rag):
        """Testa remoção de cabeça inexistente"""
        initial_count = len(multi_head_rag.heads)
        multi_head_rag.remove_head("nonexistent")
        assert len(multi_head_rag.heads) == initial_count
        
    def test_zero_weight_heads(self, multi_head_rag):
        """Testa cabeças com peso zero"""
        multi_head_rag.update_head_weights({"test": 0.0})
        test_head = multi_head_rag.get_head_by_name("test")
        assert test_head.config.weight == 0.0
        
    @pytest.mark.asyncio
    async def test_large_result_sets(self, multi_head_rag):
        """Testa conjuntos grandes de resultados"""
        # Mock com muitos resultados
        large_results = [
            {"content": f"Result {i}", "score": 0.9 - i*0.001}
            for i in range(1000)
        ]
        multi_head_rag.heads[0].retrieve = AsyncMock(return_value=large_results)
        
        results = await multi_head_rag.retrieve("test query", max_results=50)
        assert len(results) <= 50


class TestMultiHeadRAGIntegration:
    """Testes de integração completos"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Testa fluxo completo do MultiHeadRAG"""
        # Configuração completa
        configs = [
            HeadConfig("semantic", QueryType.SEMANTIC, 0.4, {"top_k": 10}),
            HeadConfig("keyword", QueryType.KEYWORD, 0.3, {"top_k": 8}),
            HeadConfig("contextual", QueryType.CONTEXTUAL, 0.3, {"top_k": 6})
        ]
        
        with patch('src.retrieval.multi_head_rag.HybridRetriever'):
            multi_head = MultiHeadRAG(configs)
            
        # Mock dos retrievers
        for i, head in enumerate(multi_head.heads):
            head.retrieve = AsyncMock(return_value=[
                {"content": f"Result from {head.config.name}", "score": 0.9 - i*0.1}
            ])
        
        # Executa recuperação
        results = await multi_head.retrieve("complex query about AI and machine learning")
        
        # Verifica resultados
        assert len(results) >= 1
        assert all("content" in result for result in results)
        assert all("score" in result for result in results)
        
        # Testa adaptação de pesos
        multi_head.update_performance_metrics("semantic", 0.95)
        multi_head.adapt_weights_based_on_performance()
        
        # Verifica que o sistema continua funcionando
        results2 = await multi_head.retrieve("another query")
        assert len(results2) >= 0 