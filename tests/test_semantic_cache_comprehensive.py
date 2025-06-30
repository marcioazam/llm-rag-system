"""
Testes abrangentes para o sistema de Semantic Cache.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Mock do Redis para evitar dependência externa
class MockRedis:
    def __init__(self):
        self.data = {}
        self.expiry = {}
    
    async def get(self, key: str):
        if key in self.expiry and datetime.now() > self.expiry[key]:
            del self.data[key]
            del self.expiry[key]
            return None
        return self.data.get(key)
    
    async def set(self, key: str, value: str, ex: int = None):
        self.data[key] = value
        if ex:
            self.expiry[key] = datetime.now() + timedelta(seconds=ex)
        return True
    
    async def delete(self, key: str):
        self.data.pop(key, None)
        self.expiry.pop(key, None)
        return True
    
    async def exists(self, key: str):
        return key in self.data
    
    async def keys(self, pattern: str = "*"):
        return list(self.data.keys())
    
    async def flushdb(self):
        self.data.clear()
        self.expiry.clear()
        return True

# Mock do EmbeddingService
class MockEmbeddingService:
    def __init__(self):
        self.call_count = 0
    
    async def embed_query(self, text: str) -> List[float]:
        self.call_count += 1
        # Retorna embedding baseado no hash do texto para consistência
        return [float(hash(text) % 100) / 100.0 for _ in range(384)]
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.call_count += len(texts)
        return [await self.embed_query(text) for text in texts]

# Importação com mock do Redis
with patch('redis.asyncio.Redis', MockRedis):
    from src.cache.semantic_cache import SemanticCache


class TestSemanticCache:
    """Test suite para SemanticCache."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock do serviço de embedding."""
        return MockEmbeddingService()
    
    @pytest.fixture
    def mock_redis(self):
        """Mock do Redis."""
        return MockRedis()
    
    @pytest_asyncio.fixture
    async def semantic_cache(self, mock_embedding_service, mock_redis):
        """Instância do SemanticCache."""
        with patch('redis.asyncio.Redis', return_value=mock_redis):
            cache = SemanticCache(
                embedding_service=mock_embedding_service,
                similarity_threshold=0.85,
                enable_redis=False,  # Disable Redis for tests
                max_memory_entries=100
            )
            return cache
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_embedding_service):
        """Test de inicialização do cache."""
        with patch('redis.asyncio.Redis', MockRedis):
            cache = SemanticCache(
                embedding_service=mock_embedding_service,
                similarity_threshold=0.9,
                max_memory_entries=500,
                enable_redis=False
            )
            
            assert cache.similarity_threshold == 0.9
            assert cache.max_memory_entries == 500
            assert cache.embedding_service == mock_embedding_service
    
    @pytest.mark.asyncio
    async def test_cache_miss_and_store(self, semantic_cache):
        """Test de cache miss e armazenamento."""
        query = "What is machine learning?"
        response = {
            "answer": "Machine learning is a subset of AI...",
            "sources": ["doc1.pdf", "doc2.pdf"],
            "model": "gpt-4"
        }
        
        # Primeira busca - deve ser cache miss
        result = await semantic_cache.get_semantic(query)
        assert result is None
        
        # Armazenar no cache
        await semantic_cache.set_semantic(query, response)
        
        # Segunda busca - deve ser cache hit
        cached_result = await semantic_cache.get_semantic(query)
        assert cached_result is not None
        assert cached_result["answer"] == response["answer"]
        assert cached_result["sources"] == response["sources"]
        assert cached_result["model"] == response["model"]
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_hit(self, semantic_cache):
        """Test de cache hit por similaridade semântica."""
        original_query = "What is artificial intelligence?"
        similar_query = "What is AI?"
        
        response = {
            "answer": "AI is the simulation of human intelligence...",
            "sources": ["ai_basics.pdf"]
        }
        
        # Armazenar resposta para query original
        await semantic_cache.set_semantic(original_query, response)
        
        # Mock da similaridade para ser alta
        with patch.object(semantic_cache, '_calculate_similarity', return_value=0.9):
            # Buscar com query similar - deve encontrar
            result = await semantic_cache.get_semantic(similar_query)
            assert result is not None
            assert result["answer"] == response["answer"]
    
    @pytest.mark.asyncio
    async def test_similarity_threshold_miss(self, semantic_cache):
        """Test de cache miss quando similaridade está abaixo do threshold."""
        query1 = "What is machine learning?"
        query2 = "How to cook pasta?"
        
        response1 = {"answer": "ML is...", "sources": []}
        
        # Armazenar primeira resposta
        await semantic_cache.set_semantic(query1, response1)
        
        # Mock da similaridade para ser baixa
        with patch.object(semantic_cache, '_calculate_similarity', return_value=0.3):
            # Buscar query não relacionada - deve ser miss
            result = await semantic_cache.get_semantic(query2)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, mock_embedding_service):
        """Test de expiração do cache."""
        with patch('redis.asyncio.Redis', MockRedis):
            # Cache configurado para teste
            cache = SemanticCache(
                embedding_service=mock_embedding_service,
                enable_redis=False,
                max_memory_entries=10
            )
            
            query = "Test query"
            response = {"answer": "Test response"}
            
            # Armazenar no cache
            await cache.set_semantic(query, response)
            
            # Verificar que está no cache
            result = await cache.get_semantic(query)
            assert result is not None
            
            # Aguardar expiração
            await asyncio.sleep(2)
            
            # Verificar que expirou
            result = await cache.get_semantic(query)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, semantic_cache):
        """Test de estatísticas do cache."""
        # Estado inicial
        stats = await semantic_cache.get_stats()
        initial_hits = stats.get("hits", 0)
        initial_misses = stats.get("misses", 0)
        
        query = "Test query for stats"
        response = {"answer": "Test response"}
        
        # Cache miss
        result = await semantic_cache.get_semantic(query)
        assert result is None
        
        # Verificar incremento de miss
        stats = await semantic_cache.get_stats()
        assert stats["misses"] == initial_misses + 1
        
        # Armazenar e buscar novamente
        await semantic_cache.set_semantic(query, response)
        result = await semantic_cache.get_semantic(query)
        assert result is not None
        
        # Verificar incremento de hit
        stats = await semantic_cache.get_stats()
        assert stats["hits"] == initial_hits + 1
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, semantic_cache):
        """Test de invalidação do cache."""
        query = "Test invalidation"
        response = {"answer": "Test response"}
        
        # Armazenar no cache
        await semantic_cache.set_semantic(query, response)
        
        # Verificar que está no cache
        result = await semantic_cache.get_semantic(query)
        assert result is not None
        
        # Invalidar cache
        await semantic_cache.invalidate(query)
        
        # Verificar que foi removido
        result = await semantic_cache.get_semantic(query)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, semantic_cache):
        """Test de limpeza completa do cache."""
        queries = ["Query 1", "Query 2", "Query 3"]
        response = {"answer": "Response"}
        
        # Armazenar múltiplas entradas
        for query in queries:
            await semantic_cache.set_semantic(query, response)
        
        # Verificar que estão no cache
        for query in queries:
            result = await semantic_cache.get_semantic(query)
            assert result is not None
        
        # Limpar cache
        await semantic_cache.clear()
        
        # Verificar que foram removidas
        for query in queries:
            result = await semantic_cache.get_semantic(query)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self, semantic_cache):
        """Test de cache de embeddings."""
        query = "Test embedding cache"
        
        # Primeira busca - deve gerar embedding
        initial_call_count = semantic_cache.embedding_service.call_count
        await semantic_cache.get_semantic(query)
        
        assert semantic_cache.embedding_service.call_count > initial_call_count
        
        # Segunda busca da mesma query - deve usar embedding em cache
        current_call_count = semantic_cache.embedding_service.call_count
        await semantic_cache.get_semantic(query)
        
        # Call count não deve aumentar (embedding em cache)
        assert semantic_cache.embedding_service.call_count == current_call_count
    
    @pytest.mark.asyncio
    async def test_complex_response_caching(self, semantic_cache):
        """Test de cache com resposta complexa."""
        query = "Complex query test"
        complex_response = {
            "answer": "Complex answer with multiple parts",
            "sources": [
                {"filename": "doc1.pdf", "score": 0.95, "content": "..."},
                {"filename": "doc2.txt", "score": 0.87, "content": "..."}
            ],
            "model": "gpt-4",
            "metadata": {
                "tokens_used": 150,
                "response_time": 2.5,
                "strategy": "hybrid"
            },
            "context": ["Context 1", "Context 2"]
        }
        
        # Armazenar resposta complexa
        await semantic_cache.set_semantic(query, complex_response)
        
        # Recuperar e verificar integridade
        result = await semantic_cache.get_semantic(query)
        assert result is not None
        assert result["answer"] == complex_response["answer"]
        assert len(result["sources"]) == 2
        assert result["sources"][0]["filename"] == "doc1.pdf"
        assert result["metadata"]["tokens_used"] == 150
        assert result["context"] == ["Context 1", "Context 2"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, semantic_cache):
        """Test de tratamento de erros."""
        # Test com query inválida
        with pytest.raises((ValueError, TypeError)):
            await semantic_cache.get_semantic(None)
        
        with pytest.raises((ValueError, TypeError)):
            await semantic_cache.get_semantic("")
        
        # Test com response inválida
        with pytest.raises((ValueError, TypeError)):
            await semantic_cache.set_semantic("valid query", None)
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, semantic_cache):
        """Test de métricas de performance."""
        query = "Performance test query"
        response = {"answer": "Performance response"}
        
        # Medir tempo de cache miss
        start_time = time.time()
        await semantic_cache.get_semantic(query)
        miss_time = time.time() - start_time
        
        # Armazenar no cache
        await semantic_cache.set_semantic(query, response)
        
        # Medir tempo de cache hit
        start_time = time.time()
        result = await semantic_cache.get_semantic(query)
        hit_time = time.time() - start_time
        
        # Cache hit deve ser mais rápido que miss
        assert hit_time < miss_time
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, semantic_cache):
        """Test de acesso concorrente ao cache."""
        query = "Concurrent test"
        response = {"answer": "Concurrent response"}
        
        # Armazenar no cache
        await semantic_cache.set_semantic(query, response)
        
        # Função para buscar no cache
        async def cache_lookup():
            return await semantic_cache.get_semantic(query)
        
        # Executar múltiplas buscas concorrentes
        tasks = [cache_lookup() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Todas devem retornar o mesmo resultado
        for result in results:
            assert result is not None
            assert result["answer"] == response["answer"]
    
    @pytest.mark.asyncio
    async def test_cache_size_limit(self, mock_embedding_service):
        """Test de limite de tamanho do cache."""
        with patch('redis.asyncio.Redis', MockRedis):
            # Cache com limite pequeno
            cache = SemanticCache(
                embedding_service=mock_embedding_service,
                max_memory_entries=3,
                enable_redis=False
            )
            
            response = {"answer": "Test response"}
            
            # Adicionar mais entradas que o limite
            for i in range(5):
                await cache.set_semantic(f"Query {i}", response)
            
            # Verificar que apenas as mais recentes estão no cache
            stats = await cache.get_stats()
            assert stats.get("total_entries", 0) <= 3


class TestSemanticCacheIntegration:
    """Testes de integração do Semantic Cache."""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """Mock do RAG Pipeline."""
        pipeline = Mock()
        pipeline.query = AsyncMock(return_value={
            "answer": "Mocked RAG response",
            "sources": ["source1.pdf"],
            "model": "test-model"
        })
        return pipeline
    
    @pytest.mark.asyncio
    async def test_cache_integration_with_rag(self, mock_embedding_service, mock_rag_pipeline):
        """Test de integração com RAG Pipeline."""
        with patch('redis.asyncio.Redis', MockRedis):
            cache = SemanticCache(
                embedding_service=mock_embedding_service,
                enable_redis=False
            )
            
            query = "Integration test query"
            
            # Primeira chamada - deve ir para o RAG
            result1 = await cache.get_semantic(query)
            if result1 is None:
                rag_response = await mock_rag_pipeline.query(query)
                await cache.set_semantic(query, rag_response)
                result1 = rag_response
            
            # Segunda chamada - deve vir do cache
            result2 = await cache.get_semantic(query)
            
            assert result1["answer"] == result2["answer"]
            assert mock_rag_pipeline.query.call_count == 1  # RAG chamado apenas uma vez
    
    @pytest.mark.asyncio
    async def test_cache_warmup(self, semantic_cache):
        """Test de aquecimento do cache."""
        common_queries = [
            "What is Python?",
            "How to use FastAPI?",
            "What is machine learning?",
            "How to deploy with Docker?"
        ]
        
        responses = [
            {"answer": f"Answer for query {i}", "sources": []}
            for i in range(len(common_queries))
        ]
        
        # Aquecer o cache
        for query, response in zip(common_queries, responses):
            await semantic_cache.set_semantic(query, response)
        
        # Verificar que todas estão no cache
        for i, query in enumerate(common_queries):
            result = await semantic_cache.get_semantic(query)
            assert result is not None
            assert result["answer"] == responses[i]["answer"]
        
        # Verificar estatísticas
        stats = await semantic_cache.get_stats()
        assert stats.get("total_entries", 0) >= len(common_queries)


if __name__ == "__main__":
    # Executar testes específicos
    pytest.main([__file__, "-v", "--tb=short"]) 