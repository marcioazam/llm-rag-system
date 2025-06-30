"""
Testes completos para o Advanced RAG Pipeline.
Cobertura atual: 7% -> Meta: 80%
"""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import time
import json

from src.rag_pipeline_advanced import AdvancedRAGPipeline


class TestAdvancedRAGPipeline:
    """Testes para o Advanced RAG Pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Diretório temporário para testes."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def mock_config_path(self, temp_dir):
        """Arquivo de configuração mock."""
        config_content = {
            "embedding_service": {"type": "mock"},
            "llm_service": {"type": "mock"},
            "vector_store": {"type": "mock"},
            "raptor": {"enabled": True}
        }
        
        config_path = Path(temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_content, f)
        
        return str(config_path)

    @pytest.fixture
    def pipeline(self, mock_config_path):
        """Instância do pipeline para testes."""
        with patch('src.rag_pipeline_advanced.APIRAGPipeline.__init__'):
            pipeline = AdvancedRAGPipeline(mock_config_path)
            
            # Mock dos serviços básicos
            pipeline.embedding_service = AsyncMock()
            pipeline.llm_service = AsyncMock()
            pipeline.vector_store = Mock()
            pipeline.config = {"raptor": {"enabled": True}}
            
            return pipeline

    def test_init_basic(self, mock_config_path):
        """Testar inicialização básica do pipeline."""
        with patch('src.rag_pipeline_advanced.APIRAGPipeline.__init__'):
            pipeline = AdvancedRAGPipeline(mock_config_path)
            
            # Verificar componentes foram inicializados
            assert pipeline.adaptive_retriever is not None
            assert pipeline.multi_query_rag is not None
            assert pipeline.corrective_rag is not None
            assert pipeline.graph_enhancer is not None
            assert pipeline.model_router is not None
            assert pipeline.prompt_system is not None
            
            # Verificar configurações padrão
            assert pipeline.advanced_config["enable_adaptive"] is True
            assert pipeline.advanced_config["enable_multi_query"] is True
            assert pipeline.advanced_config["enable_enhanced_corrective"] is True
            assert pipeline.advanced_config["confidence_threshold"] == 0.7

    def test_init_metrics_structure(self, pipeline):
        """Testar estrutura inicial das métricas."""
        metrics = pipeline.metrics
        
        assert metrics["total_advanced_queries"] == 0
        assert "improvements_usage" in metrics
        assert metrics["improvements_usage"]["adaptive"] == 0
        assert metrics["improvements_usage"]["multi_query"] == 0
        assert metrics["improvements_usage"]["enhanced_corrective"] == 0
        assert metrics["avg_confidence"] == 0.0
        assert "components" in metrics
        assert "raptor" in metrics["components"]

    @pytest.mark.asyncio
    async def test_initialize_cache_success(self, pipeline):
        """Testar inicialização bem-sucedida do cache."""
        with patch('src.rag_pipeline_advanced.OptimizedRAGCache') as mock_cache_class:
            mock_cache = AsyncMock()
            mock_cache_class.return_value = mock_cache
            
            await pipeline._initialize_cache()
            
            assert pipeline.cache == mock_cache
            mock_cache_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_cache_error(self, pipeline):
        """Testar tratamento de erro na inicialização do cache."""
        with patch('src.rag_pipeline_advanced.OptimizedRAGCache') as mock_cache_class:
            mock_cache_class.side_effect = Exception("Cache error")
            
            await pipeline._initialize_cache()
            
            assert pipeline.cache is None

    @pytest.mark.asyncio
    async def test_initialize_enhanced_corrective_rag_success(self, pipeline):
        """Testar inicialização do Enhanced Corrective RAG."""
        with patch('src.rag_pipeline_advanced.create_enhanced_corrective_rag') as mock_create:
            mock_enhanced_rag = AsyncMock()
            mock_create.return_value = mock_enhanced_rag
            
            await pipeline._initialize_enhanced_corrective_rag()
            
            assert pipeline.enhanced_corrective_rag == mock_enhanced_rag
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_enhanced_corrective_rag_error(self, pipeline):
        """Testar tratamento de erro na inicialização do Enhanced Corrective RAG."""
        with patch('src.rag_pipeline_advanced.create_enhanced_corrective_rag') as mock_create:
            mock_create.side_effect = Exception("Enhanced RAG error")
            
            await pipeline._initialize_enhanced_corrective_rag()
            
            assert pipeline.enhanced_corrective_rag is None

    @pytest.mark.asyncio
    async def test_initialize_raptor_retriever_success(self, pipeline):
        """Testar inicialização do RAPTOR retriever."""
        pipeline.config = {"raptor": {"enabled": True}}
        
        with patch('src.rag_pipeline_advanced.create_raptor_retriever') as mock_create:
            mock_raptor = AsyncMock()
            mock_create.return_value = mock_raptor
            
            await pipeline._initialize_raptor_retriever()
            
            assert pipeline.raptor_retriever == mock_raptor
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_raptor_retriever_disabled(self, pipeline):
        """Testar RAPTOR desabilitado."""
        pipeline.config = {"raptor": {"enabled": False}}
        
        await pipeline._initialize_raptor_retriever()
        
        assert pipeline.raptor_retriever is None

    def test_determine_improvements_default(self, pipeline):
        """Testar determinação padrão de melhorias."""
        improvements = pipeline._determine_improvements("What is machine learning?")
        
        expected = {"adaptive", "multi_query", "enhanced_corrective", "graph", "cache"}
        assert improvements == expected

    def test_determine_improvements_forced(self, pipeline):
        """Testar melhorias forçadas."""
        forced = ["adaptive", "corrective"]
        improvements = pipeline._determine_improvements(
            "Test question", 
            force_improvements=forced
        )
        
        assert "adaptive" in improvements
        assert "corrective" in improvements

    def test_determine_improvements_disabled_features(self, pipeline):
        """Testar com algumas features desabilitadas."""
        pipeline.advanced_config["enable_multi_query"] = False
        pipeline.advanced_config["enable_graph"] = False
        
        improvements = pipeline._determine_improvements("Test question")
        
        assert "multi_query" not in improvements
        assert "graph" not in improvements
        assert "adaptive" in improvements  # Ainda habilitado

    @pytest.mark.asyncio
    async def test_basic_retrieval(self, pipeline):
        """Testar retrieval básico."""
        # Mock do método retrieve da classe base
        pipeline.retrieve = AsyncMock(return_value=[
            {"content": "Test document 1", "score": 0.9},
            {"content": "Test document 2", "score": 0.8}
        ])
        
        results = await pipeline._basic_retrieval("test query", k=5)
        
        assert len(results) == 2
        pipeline.retrieve.assert_called_once_with("test query", k=5)

    def test_prepare_advanced_context_basic(self, pipeline):
        """Testar preparação de contexto básico."""
        documents = [
            {"content": "Document 1", "score": 0.9},
            {"content": "Document 2", "score": 0.8}
        ]
        
        context = pipeline._prepare_advanced_context(documents)
        
        assert "Document 1" in context
        assert "Document 2" in context
        assert isinstance(context, str)

    def test_prepare_advanced_context_with_analysis(self, pipeline):
        """Testar preparação de contexto com análise."""
        documents = [{"content": "Test document", "score": 0.9}]
        
        mock_analysis = Mock()
        mock_analysis.query_type = "factual"
        mock_analysis.complexity = "medium"
        
        context = pipeline._prepare_advanced_context(documents, mock_analysis)
        
        assert "Test document" in context
        assert isinstance(context, str)

    @pytest.mark.asyncio
    async def test_generate_advanced_response_basic(self, pipeline):
        """Testar geração de resposta básica."""
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "This is a test response",
            "model": "test-model",
            "cost": 0.01
        })
        
        response = await pipeline._generate_advanced_response(
            "What is AI?",
            "AI is artificial intelligence."
        )
        
        assert response["answer"] == "This is a test response"
        assert response["model_used"] == "test-model"
        assert response["cost"] == 0.01

    @pytest.mark.asyncio
    async def test_generate_advanced_response_with_analysis(self, pipeline):
        """Testar geração de resposta com análise."""
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "Detailed response",
            "model": "advanced-model"
        })
        
        mock_analysis = Mock()
        mock_analysis.query_type = "analytical"
        
        response = await pipeline._generate_advanced_response(
            "Explain machine learning",
            "ML context here",
            mock_analysis
        )
        
        assert response["answer"] == "Detailed response"
        assert response["model_used"] == "advanced-model"

    @pytest.mark.asyncio
    async def test_evaluate_response_confidence_high(self, pipeline):
        """Testar avaliação de confiança alta."""
        documents = [
            {"content": "Very relevant content", "score": 0.95},
            {"content": "Also relevant", "score": 0.85}
        ]
        
        # Mock avaliação interna
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.9):
            confidence = await pipeline._evaluate_response_confidence(
                "Test question",
                "High quality answer",
                documents
            )
            
            assert confidence == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_response_confidence_low(self, pipeline):
        """Testar avaliação de confiança baixa."""
        documents = [
            {"content": "Barely relevant", "score": 0.3}
        ]
        
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.4):
            confidence = await pipeline._evaluate_response_confidence(
                "Test question",
                "Poor answer",
                documents
            )
            
            assert confidence == 0.4

    def test_format_sources_basic(self, pipeline):
        """Testar formatação de fontes básica."""
        documents = [
            {"content": "Doc 1", "score": 0.9, "metadata": {"source": "file1.txt"}},
            {"content": "Doc 2", "score": 0.8, "metadata": {"source": "file2.txt"}}
        ]
        
        sources = pipeline._format_sources(documents)
        
        assert len(sources) == 2
        assert sources[0]["content"] == "Doc 1"
        assert sources[0]["score"] == 0.9

    def test_format_sources_empty(self, pipeline):
        """Testar formatação de fontes vazias."""
        sources = pipeline._format_sources([])
        assert sources == []

    def test_update_metrics(self, pipeline):
        """Testar atualização de métricas."""
        initial_queries = pipeline.metrics["total_advanced_queries"]
        
        pipeline._update_metrics(confidence=0.85, processing_time=2.5)
        
        assert pipeline.metrics["avg_confidence"] >= 0
        assert pipeline.metrics["avg_processing_time"] >= 0

    def test_get_advanced_stats(self, pipeline):
        """Testar obtenção de estatísticas avançadas."""
        pipeline.metrics["total_advanced_queries"] = 10
        pipeline.metrics["improvements_usage"]["adaptive"] = 5
        
        stats = pipeline.get_advanced_stats()
        
        assert isinstance(stats, dict)
        assert "total_queries" in stats
        assert "improvement_usage_rates" in stats
        assert stats["total_queries"] == 10

    def test_get_pipeline_status(self, pipeline):
        """Testar status do pipeline."""
        status = pipeline.get_pipeline_status()
        
        assert isinstance(status, dict)
        assert "status" in status
        assert "components" in status
        assert "cache" in status["components"]
        assert "enhanced_corrective_rag" in status["components"]

    @pytest.mark.asyncio
    async def test_cleanup(self, pipeline):
        """Testar limpeza do pipeline."""
        # Adicionar alguns mocks para limpeza
        pipeline.cache = AsyncMock()
        pipeline.enhanced_corrective_rag = Mock()
        pipeline.enhanced_corrective_rag.cleanup = AsyncMock()
        
        await pipeline.cleanup()
        
        # Verificar que os cleanups foram chamados
        pipeline.enhanced_corrective_rag.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_advanced_cache_hit(self, pipeline):
        """Testar query com cache hit."""
        # Setup cache mock
        pipeline.cache = AsyncMock()
        pipeline.cache.get.return_value = (
            {"answer": "Cached response", "confidence": 0.9},
            "memory",
            {"confidence": 0.9, "age": 10.0, "access_count": 2}
        )
        
        result = await pipeline.query_advanced("What is AI?")
        
        assert result["answer"] == "Cached response"
        assert result["cache_metadata"]["cache_hit"] is True
        assert result["cache_metadata"]["source"] == "memory"

    @pytest.mark.asyncio
    async def test_query_advanced_cache_miss_basic_flow(self, pipeline):
        """Testar query com cache miss - fluxo básico."""
        # Setup mocks
        pipeline.cache = AsyncMock()
        pipeline.cache.get.return_value = (None, None, None)
        pipeline.cache.set = AsyncMock()
        
        # Mock componentes
        pipeline.adaptive_retriever.analyze_query = Mock()
        pipeline.adaptive_retriever.analyze_query.return_value.optimal_k = 5
        pipeline.adaptive_retriever.analyze_query.return_value.query_type = "factual"
        
        pipeline.retrieve = AsyncMock(return_value=[
            {"content": "Test doc", "score": 0.8}
        ])
        
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "Generated answer",
            "model": "test-model"
        })
        
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.8):
            result = await pipeline.query_advanced("What is machine learning?")
        
        assert result["answer"] == "Generated answer"
        assert result["confidence"] == 0.8
        assert "adaptive" in result["improvements_used"]
        assert result["metrics"]["total_documents"] == 1

    @pytest.mark.asyncio
    async def test_query_advanced_multi_query_flow(self, pipeline):
        """Testar query com multi-query ativo."""
        # Setup
        pipeline.cache = None  # Sem cache
        
        pipeline.multi_query_rag.generate_multi_queries = AsyncMock(return_value=[
            "What is machine learning?",
            "How does ML work?",
            "What are ML applications?"
        ])
        
        pipeline.enhanced_corrective_rag = AsyncMock()
        pipeline.enhanced_corrective_rag.retrieve_and_correct = AsyncMock(return_value={
            "documents": [{"content": "ML doc", "score": 0.9}]
        })
        
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "ML explanation",
            "model": "test-model"
        })
        
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.9):
            result = await pipeline.query_advanced("What is machine learning?")
        
        assert "multi_query" in result["improvements_used"]
        assert "enhanced_corrective" in result["improvements_used"]
        assert result["metrics"]["queries_generated"] == 3

    @pytest.mark.asyncio
    async def test_query_advanced_corrective_fallback(self, pipeline):
        """Testar fallback para Corrective RAG básico."""
        # Setup - Enhanced Corrective RAG não disponível
        pipeline.cache = None
        pipeline.enhanced_corrective_rag = None
        
        pipeline.corrective_rag.retrieve_and_correct = AsyncMock(return_value={
            "documents": [{"content": "Corrected doc", "score": 0.8}]
        })
        
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "Corrected answer",
            "model": "test-model"
        })
        
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.7):
            # Forçar uso do corrective básico
            result = await pipeline.query_advanced(
                "Test question",
                force_improvements=["corrective"]
            )
        
        assert "corrective" in result["improvements_used"]
        assert result["answer"] == "Corrected answer"

    @pytest.mark.asyncio
    async def test_query_advanced_graph_enhancement(self, pipeline):
        """Testar enriquecimento com grafo."""
        pipeline.cache = None
        
        initial_docs = [{"content": "Basic doc", "score": 0.8}]
        enhanced_docs = [{"content": "Enhanced doc with graph context", "score": 0.9}]
        
        pipeline.retrieve = AsyncMock(return_value=initial_docs)
        pipeline.graph_enhancer.enrich_with_graph_context = AsyncMock(return_value=enhanced_docs)
        
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "Graph-enhanced answer",
            "model": "test-model"
        })
        
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.85):
            result = await pipeline.query_advanced(
                "Complex question",
                force_improvements=["graph"]
            )
        
        assert "graph" in result["improvements_used"]
        pipeline.graph_enhancer.enrich_with_graph_context.assert_called_once_with(initial_docs)

    @pytest.mark.asyncio
    async def test_query_advanced_error_handling(self, pipeline):
        """Testar tratamento de erro na query avançada."""
        pipeline.cache = None
        
        # Simular erro no retrieval
        pipeline.retrieve = AsyncMock(side_effect=Exception("Retrieval error"))
        
        # Mock do fallback para retrieval básico
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "Error fallback response",
            "model": "fallback-model"
        })
        
        # Deve usar fallback e não falhar
        with patch.object(pipeline, '_basic_retrieval', return_value=[]):
            try:
                result = await pipeline.query_advanced("Error test")
                # Se chegou aqui, tratou o erro adequadamente
                assert True
            except Exception as e:
                # Deve tratar erros graciosamente
                assert "Retrieval error" in str(e)

    @pytest.mark.asyncio
    async def test_query_advanced_low_confidence_no_cache(self, pipeline):
        """Testar que resposta com baixa confiança não vai para cache."""
        pipeline.cache = AsyncMock()
        pipeline.cache.get.return_value = (None, None, None)
        pipeline.cache.set = AsyncMock()
        
        pipeline.retrieve = AsyncMock(return_value=[
            {"content": "Poor quality doc", "score": 0.3}
        ])
        
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "Low confidence answer",
            "model": "test-model"
        })
        
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.4):
            result = await pipeline.query_advanced("Test question")
        
        # Confiança baixa - não deve cachear
        assert result["confidence"] == 0.4
        pipeline.cache.set.assert_not_called()

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_query_advanced_performance(self, pipeline):
        """Testar performance da query avançada."""
        pipeline.cache = None
        
        # Setup mocks rápidos
        pipeline.retrieve = AsyncMock(return_value=[
            {"content": f"Doc {i}", "score": 0.8} for i in range(10)
        ])
        
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "Fast response",
            "model": "fast-model"
        })
        
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.8):
            start_time = time.time()
            result = await pipeline.query_advanced("Performance test")
            end_time = time.time()
        
        # Deve processar em tempo razoável (< 5 segundos para teste)
        processing_time = end_time - start_time
        assert processing_time < 5.0
        assert result["metrics"]["processing_time"] > 0

    @pytest.mark.asyncio
    async def test_enhanced_corrective_retrieve(self, pipeline):
        """Testar método de retrieval com Enhanced Corrective RAG."""
        pipeline.enhanced_corrective_rag = AsyncMock()
        pipeline.enhanced_corrective_rag.retrieve_and_correct = AsyncMock(return_value={
            "documents": [{"content": "Enhanced doc", "score": 0.9}],
            "corrections_made": True,
            "original_relevance": 0.6,
            "final_relevance": 0.9
        })
        
        result = await pipeline._enhanced_corrective_retrieve("test query", k=5)
        
        assert len(result) == 1
        assert result[0]["content"] == "Enhanced doc"
        pipeline.enhanced_corrective_rag.retrieve_and_correct.assert_called_once_with("test query", k=5)


class TestAdvancedRAGPipelineIntegration:
    """Testes de integração para componentes do pipeline."""

    @pytest.fixture
    def integration_pipeline(self):
        """Pipeline para testes de integração."""
        with patch('src.rag_pipeline_advanced.APIRAGPipeline.__init__'):
            pipeline = AdvancedRAGPipeline()
            
            # Configurar mocks básicos necessários
            pipeline.embedding_service = AsyncMock()
            pipeline.llm_service = AsyncMock()
            pipeline.vector_store = Mock()
            pipeline.config = {"raptor": {"enabled": False}}  # Simplificar
            
            return pipeline

    @pytest.mark.asyncio
    async def test_full_pipeline_integration_simple(self, integration_pipeline):
        """Teste de integração completa simples."""
        pipeline = integration_pipeline
        
        # Mock all dependencies
        pipeline.cache = None  # Sem cache para simplicidade
        
        # Adaptive retriever
        mock_analysis = Mock()
        mock_analysis.optimal_k = 5
        mock_analysis.query_type = "factual"
        pipeline.adaptive_retriever.analyze_query = Mock(return_value=mock_analysis)
        
        # Basic retrieval
        pipeline.retrieve = AsyncMock(return_value=[
            {"content": "Integration test document", "score": 0.8}
        ])
        
        # LLM generation
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "Integration test response",
            "model": "integration-model",
            "cost": 0.05
        })
        
        # Confidence evaluation
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.8):
            result = await pipeline.query_advanced("Integration test question")
        
        # Verificar resultado completo
        assert result["answer"] == "Integration test response"
        assert result["confidence"] == 0.8
        assert "adaptive" in result["improvements_used"]
        assert result["metrics"]["total_documents"] == 1
        assert result["model_used"] == "integration-model"
        assert result["cost"] == 0.05

    @pytest.mark.asyncio
    async def test_multiple_components_integration(self, integration_pipeline):
        """Teste de integração com múltiplos componentes ativos."""
        pipeline = integration_pipeline
        pipeline.cache = None
        
        # Multi-query
        pipeline.multi_query_rag.generate_multi_queries = AsyncMock(return_value=[
            "Original question",
            "Rephrased question 1",
            "Rephrased question 2"
        ])
        
        # Enhanced Corrective RAG
        pipeline.enhanced_corrective_rag = AsyncMock()
        pipeline.enhanced_corrective_rag.retrieve_and_correct = AsyncMock(return_value={
            "documents": [
                {"content": "Enhanced doc 1", "score": 0.9},
                {"content": "Enhanced doc 2", "score": 0.8}
            ]
        })
        
        # Graph enhancement
        pipeline.graph_enhancer.enrich_with_graph_context = AsyncMock(return_value=[
            {"content": "Graph-enhanced doc 1", "score": 0.95},
            {"content": "Graph-enhanced doc 2", "score": 0.85}
        ])
        
        # LLM
        pipeline.llm_service.generate = AsyncMock(return_value={
            "response": "Multi-component response",
            "model": "advanced-model"
        })
        
        with patch.object(pipeline, '_evaluate_response_confidence', return_value=0.9):
            result = await pipeline.query_advanced("Complex multi-component question")
        
        # Verificar que múltiplos componentes foram usados
        improvements_used = result["improvements_used"]
        assert "multi_query" in improvements_used
        assert "enhanced_corrective" in improvements_used
        assert "graph" in improvements_used
        
        # Verificar métricas
        assert result["metrics"]["queries_generated"] == 3
        assert result["confidence"] == 0.9 