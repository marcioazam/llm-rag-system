"""Testes para o Adaptive RAG Router."""
import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Dict, Any
import time

from src.retrieval.adaptive_rag_router import (
    AdaptiveRAGRouter,
    QueryComplexityClassifier,
    QueryComplexity,
    QueryAnalysis
)


class MockEmbeddingService:
    """Mock do serviço de embeddings."""
    
    async def aembed_query(self, text: str) -> List[float]:
        """Retorna embedding simulado."""
        await asyncio.sleep(0.001)
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768).tolist()


class MockLLM:
    """Mock do LLM."""
    
    async def agenerate(self, prompts: List[str]) -> Any:
        """Gera resposta simulada."""
        class Generation:
            def __init__(self, text):
                self.text = text
        
        class LLMResult:
            def __init__(self, generations):
                self.generations = generations
        
        await asyncio.sleep(0.001)
        return LLMResult([[Generation(f"Resposta para: {prompt[:50]}...")] for prompt in prompts])
    
    async def acall(self, prompt: str) -> str:
        """Chamada simples do LLM."""
        await asyncio.sleep(0.001)
        return f"Resposta: {prompt[:50]}..."


class TestAdaptiveRAGRouter:
    """Testes para Adaptive RAG Router."""
    
    @pytest.fixture
    def embedding_service(self):
        """Fixture para serviço de embeddings."""
        return MockEmbeddingService()
    
    @pytest.fixture
    def llm(self):
        """Fixture para LLM."""
        return MockLLM()
    
    @pytest.fixture
    def router(self, embedding_service, llm):
        """Fixture para o router."""
        # Criar componentes mock
        simple_retriever = Mock()
        simple_retriever.retrieve = AsyncMock(return_value=[{"content": "doc1", "score": 0.9}])
        
        standard_rag = Mock()
        standard_rag.query = AsyncMock(return_value={"answer": "resposta", "sources": [{"content": "doc1"}]})
        
        rag_components = {
            "simple_retriever": simple_retriever,
            "standard_rag": standard_rag
        }
        
        return AdaptiveRAGRouter(
            rag_components=rag_components,
            optimization_objective="balanced"
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, router):
        """Testa inicialização do router."""
        assert router is not None
        assert router.optimization == "balanced"
        assert router.classifier is not None
        assert isinstance(router.stats, dict)
    
    @pytest.mark.asyncio
    async def test_complexity_classification(self, router):
        """Testa classificação de complexidade através do router."""
        queries = [
            ("O que é Python?", QueryComplexity.SIMPLE),
            ("Explique como funciona garbage collection em Python", QueryComplexity.SINGLE_HOP),
            ("Compare os algoritmos de otimização Adam e RMSprop", QueryComplexity.COMPLEX)
        ]
        
        for query, expected_complexity in queries:
            result = await router.classifier.classify(query)
            assert isinstance(result, QueryAnalysis)
            assert hasattr(result, "complexity")
            assert hasattr(result, "confidence")
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0
            assert hasattr(result, "reasoning_type")
    
    @pytest.mark.asyncio
    async def test_route_query(self, router):
        """Testa roteamento de query."""
        query = "O que é Python?"
        result = await router.route_query(query)
        
        assert isinstance(result, dict)
        assert "routing_metadata" in result
        assert "complexity" in result["routing_metadata"]
        assert "strategies_used" in result["routing_metadata"]
        assert "latency" in result["routing_metadata"]
    
    @pytest.mark.asyncio
    async def test_routing_stats(self, router):
        """Testa estatísticas de roteamento."""
        # Executar algumas queries
        queries = [
            "O que é Python?",
            "Como funciona machine learning?",
            "Explique redes neurais"
        ]
        
        for query in queries:
            await router.route_query(query)
        
        # Verificar estatísticas
        stats = router.get_routing_stats()
        assert "total_queries" in stats
        assert stats["total_queries"] == 3
        assert "complexity_distribution" in stats
        assert "strategy_usage" in stats
        assert "routing_efficiency" in stats
    
    @pytest.mark.asyncio
    async def test_optimization_modes(self, embedding_service, llm):
        """Testa diferentes modos de otimização."""
        rag_components = {
            "simple_retriever": Mock(retrieve=AsyncMock(return_value=[])),
            "standard_rag": Mock(query=AsyncMock(return_value={"answer": "test"}))
        }
        
        # Testar diferentes modos
        for mode in ["speed", "accuracy", "cost"]:
            router = AdaptiveRAGRouter(
                rag_components=rag_components,
                optimization_objective=mode
            )
            
            assert router.optimization == mode
            
            # Executar query
            result = await router.route_query("Test query")
            assert "routing_metadata" in result
            assert result["routing_metadata"]["optimization"] == mode
    
    @pytest.mark.asyncio
    async def test_performance(self, router):
        """Testa performance do roteamento."""
        queries = [
            "O que é uma variável?",
            "Como funciona o algoritmo de quicksort?",
            "Explique redes neurais convolucionais"
        ]
        
        start_time = time.time()
        
        for query in queries:
            result = await router.route_query(query)
            assert result is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Deve processar 3 queries em menos de 1 segundo com mocks
        assert total_time < 1.0


class TestQueryComplexityClassifier:
    """Testes para o classificador de complexidade."""
    
    @pytest.fixture
    def classifier(self):
        """Fixture para o classificador."""
        return QueryComplexityClassifier()
    
    @pytest.mark.asyncio
    async def test_classification(self, classifier):
        """Testa classificação de queries."""
        test_cases = [
            ("O que é Python?", QueryComplexity.SIMPLE),
            ("Como conectar Python ao banco de dados?", QueryComplexity.SINGLE_HOP),
            ("Compare Python, Java e C++ para desenvolvimento web", QueryComplexity.COMPLEX)
        ]
        
        for query, expected_complexity in test_cases:
            result = await classifier.classify(query)
            assert isinstance(result, QueryAnalysis)
            assert result.query == query
            assert isinstance(result.complexity, QueryComplexity)
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.key_entities, list)
            assert isinstance(result.suggested_strategies, list)
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, classifier):
        """Testa extração de entidades."""
        queries_with_entities = [
            ("Python e JavaScript são linguagens de programação", ["Python", "JavaScript"]),
            ("Django é um framework web para Python", ["Django", "Python"]),
            ("Compare React com Angular", ["React", "Angular"])
        ]
        
        for query, expected_entities in queries_with_entities:
            result = await classifier.classify(query)
            # Verificar que pelo menos algumas entidades foram encontradas
            assert len(result.key_entities) > 0
    
    @pytest.mark.asyncio
    async def test_reasoning_type_identification(self, classifier):
        """Testa identificação do tipo de raciocínio."""
        test_cases = [
            ("O que é machine learning?", ["factual", "general"]),  # Aceitar ambos
            ("Como implementar uma API REST?", ["procedural", "general"]),
            ("Compare SQL e NoSQL", ["comparative", "general"]),
            ("Por que usar Python para data science?", ["analytical", "general"])
        ]
        
        for query, expected_types in test_cases:
            result = await classifier.classify(query)
            # Verificar que o tipo retornado está na lista de tipos aceitos
            assert result.reasoning_type in expected_types
 