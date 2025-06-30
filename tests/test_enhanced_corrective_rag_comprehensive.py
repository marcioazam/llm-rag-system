"""
Testes completos para o Enhanced Corrective RAG.
Objetivo: Testes de integração e cobertura abrangente
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time
import json
from datetime import datetime, timedelta

from src.retrieval.enhanced_corrective_rag import (
    EnhancedCorrectiveRAG,
    T5RetrievalEvaluator,
    QueryDecomposer,
    QueryComplexity,
    CorrectionStrategy,
    QueryComponent,
    EvaluationResult,
    EnhancedDocumentWithScore,
    create_enhanced_corrective_rag
)


class TestEvaluationResult:
    """Testes para a classe EvaluationResult."""

    def test_evaluation_result_creation(self):
        """Testar criação de resultado de avaliação."""
        result = EvaluationResult(
            relevance_score=0.85,
            confidence=0.9,
            explanation="High relevance document",
            categories=["technical", "python"],
            semantic_similarity=0.8,
            factual_accuracy=0.95,
            completeness=0.7
        )
        
        assert result.relevance_score == 0.85
        assert result.confidence == 0.9
        assert result.explanation == "High relevance document"
        assert "technical" in result.categories
        assert "python" in result.categories
        assert result.semantic_similarity == 0.8
        assert result.factual_accuracy == 0.95
        assert result.completeness == 0.7

    def test_evaluation_result_defaults(self):
        """Testar valores padrão."""
        result = EvaluationResult(
            relevance_score=0.5,
            confidence=0.6,
            explanation="Medium relevance"
        )
        
        assert result.categories == []
        assert result.semantic_similarity == 0.0
        assert result.factual_accuracy == 0.0
        assert result.completeness == 0.0


class TestQueryComponent:
    """Testes para a classe QueryComponent."""

    def test_query_component_creation(self):
        """Testar criação de componente de query."""
        component = QueryComponent(
            text="Python programming",
            aspect="language",
            importance=0.8,
            dependencies=["syntax", "libraries"],
            metadata={"type": "programming_language"}
        )
        
        assert component.text == "Python programming"
        assert component.aspect == "language"
        assert component.importance == 0.8
        assert component.dependencies == ["syntax", "libraries"]
        assert component.metadata["type"] == "programming_language"

    def test_query_component_defaults(self):
        """Testar valores padrão."""
        component = QueryComponent(
            text="Test query",
            aspect="general",
            importance=0.5
        )
        
        assert component.dependencies == []
        assert component.metadata == {}


class TestEnhancedDocumentWithScore:
    """Testes para documentos enhanced."""

    def test_enhanced_document_creation(self):
        """Testar criação de documento enhanced."""
        eval_result = EvaluationResult(
            relevance_score=0.8,
            confidence=0.9,
            explanation="Good match"
        )
        
        doc = EnhancedDocumentWithScore(
            content="Python is a programming language",
            metadata={"source": "wiki", "topic": "programming"},
            relevance_score=0.8,
            evaluation_result=eval_result,
            validation_status="validated",
            correction_applied=True,
            source_component="language_aspect",
            rerank_score=0.85
        )
        
        assert doc.content == "Python is a programming language"
        assert doc.metadata["source"] == "wiki"
        assert doc.relevance_score == 0.8
        assert doc.evaluation_result.relevance_score == 0.8
        assert doc.validation_status == "validated"
        assert doc.correction_applied is True
        assert doc.source_component == "language_aspect"
        assert doc.rerank_score == 0.85

    def test_enhanced_document_defaults(self):
        """Testar valores padrão."""
        eval_result = EvaluationResult(
            relevance_score=0.5,
            confidence=0.7,
            explanation="Default"
        )
        
        doc = EnhancedDocumentWithScore(
            content="Test content",
            metadata={},
            relevance_score=0.5,
            evaluation_result=eval_result
        )
        
        assert doc.validation_status == "pending"
        assert doc.correction_applied is False
        assert doc.source_component is None
        assert doc.rerank_score == 0.0



    @pytest.fixture
    def basic_router_config(self):
        """Configuração básica para APIModelRouter"""
        return {
            "providers": {
                "openai": {
                    "models": {
                        "gpt4o_mini": {
                            "name": "gpt-4o-mini",
                            "max_tokens": 4096,
                            "temperature": 0.7,
                            "responsibilities": ["primary_reasoning", "code_generation"],
                            "context_window": 128000,
                            "cost_per_1k_tokens": 0.0015,
                            "priority": 1
                        }
                    }
                }
            },
            "routing": {
                "strategy": "cost_performance_optimized",
                "fallback_chain": ["openai.gpt4o_mini"]
            }
        }

class TestT5RetrievalEvaluator:
    """Testes para o avaliador T5."""

    @pytest.fixture
    def mock_cache(self):
        """Cache mock para testes."""
        cache = AsyncMock()
        cache.get.return_value = None
        cache.set.return_value = None
        return cache

    @pytest.fixture
    def evaluator(self, mock_cache):
        """Avaliador configurado para testes."""
        with patch('src.retrieval.enhanced_corrective_rag.CircuitBreaker'):
            evaluator = T5RetrievalEvaluator(
                model_router=Mock(),
                cache=mock_cache,
                config={}
            )
            return evaluator

    def test_init_basic(self, evaluator):
        """Testar inicialização básica."""
        assert evaluator.model_router is not None
        assert evaluator.cache is not None
        assert isinstance(evaluator.config, dict)
        
        # Verificar configurações da API
        assert 'openai' in evaluator.api_config
        assert 'anthropic' in evaluator.api_config
        assert 'huggingface' in evaluator.api_config
        
        # Verificar provider chain
        assert evaluator.provider_chain == ['openai', 'anthropic', 'huggingface']
        
        # Verificar circuit breakers
        assert len(evaluator.circuit_breakers) == 3
        
        # Verificar métricas iniciais
        assert evaluator.evaluation_stats['total_evaluations'] == 0
        assert evaluator.evaluation_stats['cache_hits'] == 0

    def test_create_evaluation_prompt(self, evaluator):
        """Testar criação de prompt de avaliação."""
        query = "What is Python?"
        document = "Python is a programming language"
        context = "Programming languages discussion"
        
        prompt = evaluator._create_evaluation_prompt(query, document, context)
        
        assert isinstance(prompt, str)
        assert "What is Python?" in prompt
        assert "Python is a programming language" in prompt
        assert "Programming languages discussion" in prompt
        assert "relevance" in prompt.lower()

    def test_create_evaluation_prompt_no_context(self, evaluator):
        """Testar criação de prompt sem contexto."""
        query = "What is Python?"
        document = "Python is a programming language"
        
        prompt = evaluator._create_evaluation_prompt(query, document)
        
        assert isinstance(prompt, str)
        assert "What is Python?" in prompt
        assert "Python is a programming language" in prompt

    def test_regex_fallback_parse(self, evaluator):
        """Testar parsing de fallback com regex."""
        response = """
        Relevance Score: 0.85
        Confidence: 0.9
        Categories: technical, programming
        Explanation: This document is highly relevant
        """
        
        result = evaluator._regex_fallback_parse(response, "openai")
        
        assert isinstance(result, EvaluationResult)
        assert result.relevance_score == 0.85
        assert result.confidence == 0.9
        assert "technical" in result.categories
        assert "programming" in result.categories
        assert "highly relevant" in result.explanation.lower()

    def test_fallback_evaluation(self, evaluator):
        """Testar avaliação de fallback."""
        query = "Python programming"
        document = "Python is a programming language used for development"
        
        result = evaluator._fallback_evaluation(query, document)
        
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.relevance_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.explanation, str)

    def test_update_response_time(self, evaluator):
        """Testar atualização do tempo de resposta."""
        initial_avg = evaluator.evaluation_stats['avg_response_time']
        
        evaluator._update_response_time(2.5)
        
        # Primeira atualização deve definir o tempo
        assert evaluator.evaluation_stats['avg_response_time'] > 0

    def test_update_success_rate(self, evaluator):
        """Testar atualização da taxa de sucesso."""
        initial_rate = evaluator.evaluation_stats['provider_success_rate']['openai']
        
        evaluator._update_success_rate('openai', False)
        
        # Taxa de sucesso deve diminuir
        assert evaluator.evaluation_stats['provider_success_rate']['openai'] < initial_rate

    @pytest.mark.asyncio
    async def test_evaluate_relevance_cache_hit(self, evaluator):
        """Testar avaliação com cache hit."""
        cached_result = {
            'relevance_score': 0.9,
            'confidence': 0.85,
            'explanation': 'Cached result',
            'categories': ['cached'],
            'semantic_similarity': 0.8,
            'factual_accuracy': 0.9,
            'completeness': 0.7
        }
        evaluator.cache.get.return_value = cached_result
        
        result = await evaluator.evaluate_relevance(
            "Test query",
            "Test document"
        )
        
        assert result.relevance_score == 0.9
        assert result.confidence == 0.85
        assert result.explanation == 'Cached result'
        assert evaluator.evaluation_stats['cache_hits'] == 1

    @pytest.mark.asyncio
    async def test_evaluate_relevance_cache_miss_fallback(self, evaluator):
        """Testar avaliação com cache miss e fallback."""
        evaluator.cache.get.return_value = None
        
        # Mock todos os circuit breakers como fechados
        for cb in evaluator.circuit_breakers.values():
            cb.can_execute.return_value = False
        
        result = await evaluator.evaluate_relevance(
            "Test query",
            "Test document"
        )
        
        # Deve usar fallback evaluation
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.relevance_score <= 1.0


class TestQueryDecomposer:
    """Testes para o decompositor de queries."""

    @pytest.fixture
    def mock_model_router(self):
        """Model router mock."""
        router = AsyncMock()
        router.route_request.return_value = {
            "response": "Medium complexity query about Python programming",
            "model": "test-model"
        }
        return router

    @pytest.fixture
    def decomposer(self, mock_model_router):
        """Decompositor configurado para testes."""
        return QueryDecomposer(mock_model_router)

    @pytest.mark.asyncio
    async def test_analyze_complexity_simple(self, decomposer):
        """Testar análise de complexidade simples."""
        simple_query = "What is Python?"
        
        complexity = await decomposer.analyze_complexity(simple_query)
        
        assert complexity in [QueryComplexity.SIMPLE, QueryComplexity.MEDIUM]

    @pytest.mark.asyncio
    async def test_analyze_complexity_complex(self, decomposer):
        """Testar análise de complexidade complexa."""
        complex_query = "Compare Python and Java in terms of performance, ease of use, and community support for machine learning applications"
        
        complexity = await decomposer.analyze_complexity(complex_query)
        
        assert complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_ASPECT]

    @pytest.mark.asyncio
    async def test_decompose_query_simple(self, decomposer, mock_model_router):
        """Testar decomposição de query simples."""
        mock_model_router.route_request.return_value = {
            "response": json.dumps({
                "components": [
                    {
                        "text": "Python definition",
                        "aspect": "definition",
                        "importance": 1.0,
                        "dependencies": []
                    }
                ]
            }),
            "model": "test-model"
        }
        
        components = await decomposer.decompose_query("What is Python?")
        
        assert len(components) >= 1
        assert all(isinstance(c, QueryComponent) for c in components)
        assert components[0].aspect == "definition"

    @pytest.mark.asyncio
    async def test_decompose_query_error_handling(self, decomposer, mock_model_router):
        """Testar tratamento de erro na decomposição."""
        mock_model_router.route_request.side_effect = Exception("API Error")
        
        components = await decomposer.decompose_query("Test query")
        
        # Deve retornar fallback com um componente
        assert len(components) == 1
        assert components[0].text == "Test query"
        assert components[0].aspect == "general"

    @pytest.mark.asyncio
    async def test_recompose_results_basic(self, decomposer):
        """Testar recomposição básica de resultados."""
        eval_result = EvaluationResult(0.8, 0.9, "Good match")
        
        component_results = {
            "definition": [
                EnhancedDocumentWithScore(
                    "Python definition content",
                    {"source": "wiki"},
                    0.9,
                    eval_result,
                    source_component="definition"
                )
            ],
            "usage": [
                EnhancedDocumentWithScore(
                    "Python usage content",
                    {"source": "tutorial"},
                    0.8,
                    eval_result,
                    source_component="usage"
                )
            ]
        }
        
        result = await decomposer.recompose_results(
            "What is Python and how to use it?",
            component_results
        )
        
        assert len(result) == 2
        assert all(isinstance(doc, EnhancedDocumentWithScore) for doc in result)
        # Documentos devem estar ordenados por relevância
        assert result[0].relevance_score >= result[1].relevance_score

    @pytest.mark.asyncio
    async def test_contextual_rerank(self, decomposer):
        """Testar reranking contextual."""
        eval_result = EvaluationResult(0.8, 0.9, "Good match")
        
        docs = [
            EnhancedDocumentWithScore(
                "Low relevance content",
                {"source": "wiki"},
                0.6,
                eval_result
            ),
            EnhancedDocumentWithScore(
                "High relevance content",
                {"source": "tutorial"},
                0.9,
                eval_result
            )
        ]
        
        reranked = await decomposer._contextual_rerank("Test query", docs)
        
        assert len(reranked) == 2
        # Deve estar ordenado por relevância
        assert reranked[0].relevance_score >= reranked[1].relevance_score


class TestEnhancedCorrectiveRAG:
    """Testes para o Enhanced Corrective RAG principal."""

    @pytest.fixture
    def mock_retriever(self):
        """Retriever mock."""
        retriever = AsyncMock()
        retriever.retrieve.return_value = [
            Mock(
                content="Test document content",
                metadata={"source": "test"},
                combined_score=0.8
            )
        ]
        return retriever

    @pytest.fixture
    def enhanced_rag(self, mock_retriever):
        """Enhanced Corrective RAG configurado."""
        with patch('src.retrieval.enhanced_corrective_rag.T5RetrievalEvaluator'), \
             patch('src.retrieval.enhanced_corrective_rag.QueryDecomposer'):
            
            rag = EnhancedCorrectiveRAG(
                retriever=mock_retriever,
                relevance_threshold=0.75,
                max_reformulation_attempts=3,
                enable_decomposition=True
            )
            
            return rag

    def test_init_basic(self, enhanced_rag):
        """Testar inicialização básica."""
        assert enhanced_rag.retriever is not None
        assert enhanced_rag.relevance_threshold == 0.75
        assert enhanced_rag.max_reformulation_attempts == 3
        assert enhanced_rag.enable_decomposition is True
        
        # Verificar métricas iniciais
        stats = enhanced_rag.correction_stats
        assert stats['total_queries'] == 0
        assert stats['corrections_applied'] == 0
        assert stats['decompositions_used'] == 0

    @pytest.mark.asyncio
    async def test_retrieve_and_correct_simple_query(self, enhanced_rag):
        """Testar retrieve and correct para query simples."""
        # Mock do decomposer para retornar complexidade simples
        enhanced_rag.query_decomposer.analyze_complexity.return_value = QueryComplexity.SIMPLE
        
        # Mock do t5_evaluator
        eval_result = EvaluationResult(
            relevance_score=0.9,
            confidence=0.8,
            explanation="High relevance"
        )
        enhanced_rag.t5_evaluator.evaluate_relevance.return_value = eval_result
        
        result = await enhanced_rag.retrieve_and_correct("What is Python?", k=5)
        
        assert isinstance(result, dict)
        assert "documents" in result
        assert "correction_applied" in result
        assert "strategy_used" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_retrieve_and_correct_complex_query(self, enhanced_rag):
        """Testar retrieve and correct para query complexa."""
        # Mock do decomposer para retornar complexidade complexa
        enhanced_rag.query_decomposer.analyze_complexity.return_value = QueryComplexity.COMPLEX
        
        # Mock da decomposição
        components = [
            QueryComponent("Python definition", "definition", 0.8),
            QueryComponent("Python usage", "usage", 0.6)
        ]
        enhanced_rag.query_decomposer.decompose_query.return_value = components
        
        # Mock do t5_evaluator
        eval_result = EvaluationResult(0.9, 0.8, "High relevance")
        enhanced_rag.t5_evaluator.evaluate_relevance.return_value = eval_result
        
        # Mock da recomposição
        recomposed_docs = [
            EnhancedDocumentWithScore(
                "Recomposed content",
                {"source": "recomposed"},
                0.9,
                eval_result
            )
        ]
        enhanced_rag.query_decomposer.recompose_results.return_value = recomposed_docs
        
        result = await enhanced_rag.retrieve_and_correct(
            "What is Python and how to use it?",
            k=5,
            use_decomposition=True
        )
        
        assert isinstance(result, dict)
        assert "documents" in result
        assert result["strategy_used"] == "decomposition"
        assert result["metadata"]["complexity"] == "complex"

    @pytest.mark.asyncio
    async def test_retrieve_for_component(self, enhanced_rag):
        """Testar retrieval para componente específico."""
        component = QueryComponent(
            text="Python programming",
            aspect="language",
            importance=0.8
        )
        
        # Mock do t5_evaluator
        eval_result = EvaluationResult(0.8, 0.9, "Good match")
        enhanced_rag.t5_evaluator.evaluate_relevance.return_value = eval_result
        
        docs = await enhanced_rag._retrieve_for_component(component, k=3)
        
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(doc, EnhancedDocumentWithScore) for doc in docs)

    @pytest.mark.asyncio
    async def test_traditional_enhanced_retrieve(self, enhanced_rag):
        """Testar retrieve tradicional enhanced."""
        # Mock do t5_evaluator para baixa relevância inicialmente
        low_eval = EvaluationResult(0.6, 0.7, "Low relevance")
        high_eval = EvaluationResult(0.9, 0.8, "High relevance")
        
        enhanced_rag.t5_evaluator.evaluate_relevance.side_effect = [low_eval, high_eval]
        
        result = await enhanced_rag._traditional_enhanced_retrieve("Test query", k=5)
        
        assert isinstance(result, dict)
        assert "documents" in result
        assert "correction_applied" in result

    @pytest.mark.asyncio
    async def test_enhanced_reformulate_query(self, enhanced_rag):
        """Testar reformulação enhanced de query."""
        original_query = "What is Python?"
        current_query = "What is Python programming language?"
        
        low_relevance_docs = [
            EnhancedDocumentWithScore(
                "Irrelevant content",
                {"source": "test"},
                0.3,
                EvaluationResult(0.3, 0.5, "Low relevance")
            )
        ]
        
        # Mock do model router
        enhanced_rag.model_router.route_request.return_value = {
            "response": "What is Python programming language and its applications?",
            "model": "test-model"
        }
        
        reformulated = await enhanced_rag._enhanced_reformulate_query(
            original_query,
            current_query,
            low_relevance_docs
        )
        
        assert isinstance(reformulated, str)
        assert len(reformulated) > 0
        assert reformulated != original_query

    def test_get_correction_stats(self, enhanced_rag):
        """Testar obtenção de estatísticas de correção."""
        # Simular algumas estatísticas
        enhanced_rag.correction_stats['total_queries'] = 10
        enhanced_rag.correction_stats['corrections_applied'] = 3
        enhanced_rag.correction_stats['decompositions_used'] = 2
        
        stats = enhanced_rag.get_correction_stats()
        
        assert isinstance(stats, dict)
        assert stats['total_queries'] == 10
        assert stats['corrections_applied'] == 3
        assert stats['correction_rate'] == 0.3
        assert stats['decomposition_rate'] == 0.2

    @pytest.mark.asyncio
    async def test_evaluate_system_performance(self, enhanced_rag):
        """Testar avaliação de performance do sistema."""
        test_queries = [
            "What is Python?",
            "How to use Python for machine learning?",
            "Python vs Java comparison"
        ]
        
        # Mock das respostas
        mock_result = {
            "documents": [
                EnhancedDocumentWithScore(
                    "Test content",
                    {"source": "test"},
                    0.8,
                    EvaluationResult(0.8, 0.9, "Good")
                )
            ],
            "correction_applied": True,
            "strategy_used": "traditional",
            "metadata": {"processing_time": 0.5}
        }
        
        with patch.object(enhanced_rag, 'retrieve_and_correct', return_value=mock_result):
            performance = await enhanced_rag.evaluate_system_performance(test_queries)
        
        assert isinstance(performance, dict)
        assert "avg_response_time" in performance
        assert "avg_relevance_score" in performance
        assert "correction_rate" in performance
        assert "total_queries_processed" in performance
        assert performance["total_queries_processed"] == 3

    @pytest.mark.asyncio
    async def test_error_handling_retrieve_and_correct(self, enhanced_rag):
        """Testar tratamento de erro no retrieve and correct."""
        # Simular erro no retriever
        enhanced_rag.retriever.retrieve.side_effect = Exception("Retrieval error")
        
        result = await enhanced_rag.retrieve_and_correct("Test query", k=5)
        
        # Deve retornar resultado vazio mas válido
        assert isinstance(result, dict)
        assert "documents" in result
        assert result["documents"] == []
        assert "error" in result


class TestCreateEnhancedCorrectiveRAG:
    """Testes para a função factory."""

    def test_create_enhanced_corrective_rag_default(self):
        """Testar criação com configuração padrão."""
        with patch('src.retrieval.enhanced_corrective_rag.HybridRetriever'), \
             patch('src.retrieval.enhanced_corrective_rag.T5RetrievalEvaluator'), \
             patch('src.retrieval.enhanced_corrective_rag.QueryDecomposer'):
            
            rag = create_enhanced_corrective_rag()
            
            assert isinstance(rag, EnhancedCorrectiveRAG)
            assert rag.relevance_threshold == 0.75
            assert rag.max_reformulation_attempts == 3
            assert rag.enable_decomposition is True

    def test_create_enhanced_corrective_rag_custom_config(self):
        """Testar criação com configuração customizada."""
        config = {
            "relevance_threshold": 0.8,
            "max_reformulation_attempts": 5,
            "enable_decomposition": False,
            "api_providers": ["openai", "anthropic"]
        }
        
        with patch('src.retrieval.enhanced_corrective_rag.HybridRetriever'), \
             patch('src.retrieval.enhanced_corrective_rag.T5RetrievalEvaluator'), \
             patch('src.retrieval.enhanced_corrective_rag.QueryDecomposer'):
            
            rag = create_enhanced_corrective_rag(config)
            
            assert isinstance(rag, EnhancedCorrectiveRAG)
            assert rag.relevance_threshold == 0.8
            assert rag.max_reformulation_attempts == 5
            assert rag.enable_decomposition is False


@pytest.mark.integration
class TestEnhancedCorrectiveRAGIntegration:
    """Testes de integração end-to-end."""

    @pytest.fixture
    def integration_rag(self):
        """RAG configurado para testes de integração."""
        with patch('src.retrieval.enhanced_corrective_rag.HybridRetriever') as mock_retriever, \
             patch('src.retrieval.enhanced_corrective_rag.T5RetrievalEvaluator') as mock_evaluator, \
             patch('src.retrieval.enhanced_corrective_rag.QueryDecomposer') as mock_decomposer:
            
            # Configurar mocks
            mock_retriever_instance = AsyncMock()
            mock_retriever.return_value = mock_retriever_instance
            
            mock_evaluator_instance = AsyncMock()
            mock_evaluator.return_value = mock_evaluator_instance
            
            mock_decomposer_instance = AsyncMock()
            mock_decomposer.return_value = mock_decomposer_instance
            
            rag = EnhancedCorrectiveRAG(
                retriever=mock_retriever_instance,
                relevance_threshold=0.75,
                max_reformulation_attempts=3,
                enable_decomposition=True
            )
            
            rag.evaluator = mock_evaluator_instance
            rag.query_decomposer = mock_decomposer_instance
            
            return rag

    @pytest.mark.asyncio
    async def test_end_to_end_simple_query(self, integration_rag):
        """Teste end-to-end para query simples."""
        # Configurar mocks
        integration_rag.query_decomposer.analyze_complexity.return_value = QueryComplexity.SIMPLE
        
        integration_rag.retriever.retrieve.return_value = [
            Mock(
                content="Python is a programming language",
                metadata={"source": "wiki", "topic": "programming"},
                combined_score=0.8
            )
        ]
        
        integration_rag.evaluator.evaluate_relevance.return_value = EvaluationResult(
            relevance_score=0.9,
            confidence=0.85,
            explanation="Highly relevant document about Python",
            categories=["programming", "language"],
            semantic_similarity=0.8,
            factual_accuracy=0.9,
            completeness=0.8
        )
        
        result = await integration_rag.retrieve_and_correct(
            "What is Python programming language?",
            k=5
        )
        
        # Verificar resultado
        assert isinstance(result, dict)
        assert "documents" in result
        assert len(result["documents"]) > 0
        assert result["strategy_used"] == "traditional"
        assert "metadata" in result
        
        # Verificar documento retornado
        doc = result["documents"][0]
        assert isinstance(doc, EnhancedDocumentWithScore)
        assert doc.content == "Python is a programming language"
        assert doc.evaluation_result.relevance_score == 0.9

    @pytest.mark.asyncio
    async def test_end_to_end_complex_query_with_decomposition(self, integration_rag):
        """Teste end-to-end para query complexa com decomposição."""
        # Configurar mocks para query complexa
        integration_rag.query_decomposer.analyze_complexity.return_value = QueryComplexity.COMPLEX
        
        # Mock da decomposição
        components = [
            QueryComponent("Python definition", "definition", 0.8),
            QueryComponent("Python performance", "performance", 0.7),
            QueryComponent("Python vs Java", "comparison", 0.6)
        ]
        integration_rag.query_decomposer.decompose_query.return_value = components
        
        # Mock do retrieval para componentes
        integration_rag.retriever.retrieve.return_value = [
            Mock(
                content="Component-specific content",
                metadata={"source": "docs", "component": "definition"},
                combined_score=0.85
            )
        ]
        
        # Mock da avaliação
        integration_rag.evaluator.evaluate_relevance.return_value = EvaluationResult(
            relevance_score=0.88,
            confidence=0.9,
            explanation="Component-specific relevant content",
            categories=["technical", "comparison"]
        )
        
        # Mock da recomposição
        recomposed_docs = [
            EnhancedDocumentWithScore(
                content="Recomposed comprehensive answer",
                metadata={"source": "recomposed", "components": ["definition", "performance", "comparison"]},
                relevance_score=0.92,
                evaluation_result=EvaluationResult(0.92, 0.95, "Comprehensive answer"),
                source_component="recomposed"
            )
        ]
        integration_rag.query_decomposer.recompose_results.return_value = recomposed_docs
        
        result = await integration_rag.retrieve_and_correct(
            "Compare Python and Java in terms of performance and ease of use",
            k=10,
            use_decomposition=True
        )
        
        # Verificar resultado
        assert isinstance(result, dict)
        assert result["strategy_used"] == "decomposition"
        assert result["metadata"]["complexity"] == "complex"
        assert result["metadata"]["components_processed"] == 3
        
        # Verificar documento recomposto
        doc = result["documents"][0]
        assert doc.content == "Recomposed comprehensive answer"
        assert doc.source_component == "recomposed"
        assert doc.relevance_score == 0.92

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_multiple_queries(self, integration_rag):
        """Teste de performance para múltiplas queries."""
        queries = [
            "What is Python?",
            "How to use Python for data science?",
            "Python vs Java performance comparison",
            "Best Python frameworks for web development",
            "Python machine learning libraries"
        ]
        
        # Configurar mocks rápidos
        integration_rag.query_decomposer.analyze_complexity.return_value = QueryComplexity.MEDIUM
        integration_rag.retriever.retrieve.return_value = [
            Mock(content="Fast response", metadata={}, combined_score=0.8)
        ]
        integration_rag.evaluator.evaluate_relevance.return_value = EvaluationResult(
            0.8, 0.9, "Fast evaluation"
        )
        
        start_time = time.time()
        
        tasks = [
            integration_rag.retrieve_and_correct(query, k=5)
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verificar performance
        assert len(results) == 5
        assert total_time < 10.0  # Deve processar 5 queries em menos de 10 segundos
        
        # Verificar que todos os resultados são válidos
        for result in results:
            assert isinstance(result, dict)
            assert "documents" in result
            assert "strategy_used" in result 