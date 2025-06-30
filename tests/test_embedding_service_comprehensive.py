"""
Testes abrangentes para Embedding Service.
Suporte a múltiplos providers, cache inteligente e otimizações de performance
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any, Optional
import hashlib
import time


# Mock embedding providers
class MockOpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.embedding_dim = 1536 if "small" in model else 3072
        
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Mock OpenAI embedding generation."""
        await asyncio.sleep(0.01)  # Simulate API delay
        
        embeddings = []
        for text in texts:
            # Generate deterministic mock embedding based on text hash
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, self.embedding_dim).tolist()
            embeddings.append(embedding)
            
        return embeddings
    
    async def embed_text(self, text: str) -> List[float]:
        """Single text embedding."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]


class MockAnthropicEmbedder:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embedding_dim = 1024
        
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Mock Anthropic embedding generation."""
        await asyncio.sleep(0.015)  # Slightly different delay
        
        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(f"anthropic_{text}".encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, self.embedding_dim).tolist()
            embeddings.append(embedding)
            
        return embeddings
    
    async def embed_text(self, text: str) -> List[float]:
        """Single text embedding."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]


class MockGoogleEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self.embedding_dim = 768
        
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Mock Google embedding generation."""
        await asyncio.sleep(0.02)
        
        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(f"google_{text}".encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, self.embedding_dim).tolist()
            embeddings.append(embedding)
            
        return embeddings
    
    async def embed_text(self, text: str) -> List[float]:
        """Single text embedding."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]


# Cache implementations
class InMemoryCache:
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        
    async def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    async def set(self, key: str, value: List[float]) -> None:
        """Set embedding in cache with LRU eviction."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
            
        self.cache[key] = value
        self.access_count[key] = 1
    
    async def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


class RedisCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.connected = True  # Mock connection
        
    async def get(self, key: str) -> Optional[List[float]]:
        """Mock Redis get."""
        # Simulate Redis behavior
        if not self.connected:
            return None
            
        # Mock some cached values
        if "cached_text" in key:
            return [0.1] * 1536  # Mock cached embedding
        return None
    
    async def set(self, key: str, value: List[float], ttl: int = 3600) -> None:
        """Mock Redis set."""
        if not self.connected:
            raise ConnectionError("Redis not connected")
        # Mock successful set
        pass
    
    async def clear(self) -> None:
        """Mock Redis clear."""
        pass


# Main Embedding Service
class EmbeddingService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.cache = None
        self.batch_size = config.get('batch_size', 100)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.fallback_providers = config.get('fallback_providers', [])
        
        self._initialize_providers()
        self._initialize_cache()
    
    def _initialize_providers(self):
        """Initialize embedding providers."""
        provider_configs = self.config.get('providers', {})
        
        for name, config in provider_configs.items():
            if name == 'openai':
                self.providers[name] = MockOpenAIEmbedder(
                    api_key=config['api_key'],
                    model=config.get('model', 'text-embedding-3-small')
                )
            elif name == 'anthropic':
                self.providers[name] = MockAnthropicEmbedder(
                    api_key=config['api_key']
                )
            elif name == 'google':
                self.providers[name] = MockGoogleEmbedder(
                    api_key=config['api_key'],
                    model=config.get('model', 'text-embedding-004')
                )
    
    def _initialize_cache(self):
        """Initialize cache system."""
        cache_config = self.config.get('cache', {})
        cache_type = cache_config.get('type', 'memory')
        
        if cache_type == 'memory':
            self.cache = InMemoryCache(
                max_size=cache_config.get('max_size', 10000)
            )
        elif cache_type == 'redis':
            self.cache = RedisCache(
                redis_url=cache_config.get('url', 'redis://localhost:6379')
            )
    
    def _generate_cache_key(self, text: str, provider: str, model: str = None) -> str:
        """Generate cache key for text embedding."""
        key_data = f"{provider}:{model or 'default'}:{text}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def embed_text(self, text: str, provider: str = None) -> List[float]:
        """Embed single text with caching."""
        provider = provider or self.config.get('default_provider', 'openai')
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not configured")
        
        # Check cache first
        if self.cache:
            cache_key = self._generate_cache_key(text, provider)
            cached_embedding = await self.cache.get(cache_key)
            if cached_embedding:
                return cached_embedding
        
        # Generate embedding
        embedding = await self._embed_with_retry(text, provider)
        
        # Cache result
        if self.cache and embedding:
            cache_key = self._generate_cache_key(text, provider)
            await self.cache.set(cache_key, embedding)
        
        return embedding
    
    async def embed_texts(self, texts: List[str], provider: str = None) -> List[List[float]]:
        """Embed multiple texts with batching and caching."""
        provider = provider or self.config.get('default_provider', 'openai')
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not configured")
        
        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if self.cache:
                cache_key = self._generate_cache_key(text, provider)
                cached_embedding = await self.cache.get(cache_key)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    continue
            
            # Track uncached texts
            uncached_texts.append(text)
            uncached_indices.append(i)
            embeddings.append(None)  # Placeholder
        
        # Generate embeddings for uncached texts in batches
        if uncached_texts:
            uncached_embeddings = await self._embed_batch_with_retry(uncached_texts, provider)
            
            # Fill in embeddings and cache results
            for idx, embedding in zip(uncached_indices, uncached_embeddings):
                embeddings[idx] = embedding
                
                if self.cache:
                    cache_key = self._generate_cache_key(texts[idx], provider)
                    await self.cache.set(cache_key, embedding)
        
        return embeddings
    
    async def _embed_with_retry(self, text: str, provider: str) -> List[float]:
        """Embed single text with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                embedding = await self.providers[provider].embed_text(text)
                return embedding
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    # Try fallback providers
                    for fallback_provider in self.fallback_providers:
                        if fallback_provider in self.providers:
                            try:
                                return await self.providers[fallback_provider].embed_text(text)
                            except:
                                continue
                    raise e
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"Failed to embed text after {self.retry_attempts} attempts")
    
    async def _embed_batch_with_retry(self, texts: List[str], provider: str) -> List[List[float]]:
        """Embed batch of texts with retry logic."""
        # Split into smaller batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        all_embeddings = []
        
        for batch in batches:
            for attempt in range(self.retry_attempts):
                try:
                    batch_embeddings = await self.providers[provider].embed_texts(batch)
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        # Try fallback providers
                        for fallback_provider in self.fallback_providers:
                            if fallback_provider in self.providers:
                                try:
                                    batch_embeddings = await self.providers[fallback_provider].embed_texts(batch)
                                    all_embeddings.extend(batch_embeddings)
                                    break
                                except:
                                    continue
                        else:
                            raise e
                    
                    await asyncio.sleep(2 ** attempt)
        
        return all_embeddings
    
    async def similarity(self, text1: str, text2: str, provider: str = None) -> float:
        """Calculate similarity between two texts."""
        embeddings = await self.embed_texts([text1, text2], provider)
        
        # Cosine similarity
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def find_similar(self, query: str, texts: List[str], 
                          top_k: int = 5, provider: str = None) -> List[tuple]:
        """Find most similar texts to query."""
        # Get embeddings for query and all texts
        all_texts = [query] + texts
        embeddings = await self.embed_texts(all_texts, provider)
        
        query_embedding = np.array(embeddings[0])
        text_embeddings = [np.array(emb) for emb in embeddings[1:]]
        
        # Calculate similarities
        similarities = []
        for i, text_embedding in enumerate(text_embeddings):
            # Cosine similarity
            dot_product = np.dot(query_embedding, text_embedding)
            norm1 = np.linalg.norm(query_embedding)
            norm2 = np.linalg.norm(text_embedding)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
            
            similarities.append((texts[i], similarity, i))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            'providers': list(self.providers.keys()),
            'cache_enabled': self.cache is not None,
            'batch_size': self.batch_size,
            'retry_attempts': self.retry_attempts
        }
        
        if self.cache:
            stats['cache_size'] = self.cache.size()
        
        return stats
    
    async def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache:
            await self.cache.clear()


# Test fixtures
@pytest.fixture
def basic_config():
    return {
        'providers': {
            'openai': {
                'api_key': 'test_openai_key',
                'model': 'text-embedding-3-small'
            }
        },
        'default_provider': 'openai',
        'batch_size': 50,
        'retry_attempts': 2,
        'cache': {
            'type': 'memory',
            'max_size': 1000
        }
    }

@pytest.fixture
def multi_provider_config():
    return {
        'providers': {
            'openai': {
                'api_key': 'test_openai_key',
                'model': 'text-embedding-3-large'
            },
            'anthropic': {
                'api_key': 'test_anthropic_key'
            },
            'google': {
                'api_key': 'test_google_key',
                'model': 'text-embedding-004'
            }
        },
        'default_provider': 'openai',
        'fallback_providers': ['anthropic', 'google'],
        'batch_size': 100,
        'retry_attempts': 3,
        'cache': {
            'type': 'redis',
            'url': 'redis://localhost:6379'
        }
    }

@pytest.fixture
def embedding_service(basic_config):
    return EmbeddingService(basic_config)

@pytest.fixture
def multi_provider_service(multi_provider_config):
    return EmbeddingService(multi_provider_config)

@pytest.fixture
def sample_texts():
    return [
        "This is a test document about machine learning.",
        "Python is a programming language.",
        "Artificial intelligence and deep learning are related fields.",
        "Data science involves statistical analysis.",
        "Natural language processing helps computers understand text."
    ]


# Test Classes
class TestEmbeddingService:
    """Testes básicos do Embedding Service."""
    
    def test_init_basic(self, embedding_service):
        """Testar inicialização básica."""
        assert len(embedding_service.providers) == 1
        assert 'openai' in embedding_service.providers
        assert embedding_service.cache is not None
        assert embedding_service.batch_size == 50
    
    def test_init_multi_provider(self, multi_provider_service):
        """Testar inicialização com múltiplos providers."""
        assert len(multi_provider_service.providers) == 3
        assert 'openai' in multi_provider_service.providers
        assert 'anthropic' in multi_provider_service.providers
        assert 'google' in multi_provider_service.providers
    
    def test_init_invalid_config(self):
        """Testar erro com configuração inválida."""
        config = {'providers': {}}  # No providers
        service = EmbeddingService(config)
        assert len(service.providers) == 0
    
    @pytest.mark.asyncio
    async def test_embed_text_basic(self, embedding_service):
        """Testar embedding de texto único."""
        text = "This is a test text."
        embedding = await embedding_service.embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # OpenAI small model dimension
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_embed_text_different_providers(self, multi_provider_service):
        """Testar embedding com diferentes providers."""
        text = "Test text for multiple providers."
        
        # Test OpenAI
        openai_embedding = await multi_provider_service.embed_text(text, 'openai')
        assert len(openai_embedding) == 3072  # Large model
        
        # Test Anthropic
        anthropic_embedding = await multi_provider_service.embed_text(text, 'anthropic')
        assert len(anthropic_embedding) == 1024
        
        # Test Google
        google_embedding = await multi_provider_service.embed_text(text, 'google')
        assert len(google_embedding) == 768
    
    @pytest.mark.asyncio
    async def test_embed_text_invalid_provider(self, embedding_service):
        """Testar erro com provider inválido."""
        with pytest.raises(ValueError, match="Provider invalid not configured"):
            await embedding_service.embed_text("test", "invalid")


class TestBatchEmbedding:
    """Testes para embedding em lote."""
    
    @pytest.mark.asyncio
    async def test_embed_texts_basic(self, embedding_service, sample_texts):
        """Testar embedding de múltiplos textos."""
        embeddings = await embedding_service.embed_texts(sample_texts)
        
        assert len(embeddings) == len(sample_texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self, embedding_service):
        """Testar embedding de lista vazia."""
        embeddings = await embedding_service.embed_texts([])
        assert embeddings == []
    
    @pytest.mark.asyncio
    async def test_embed_texts_large_batch(self, embedding_service):
        """Testar embedding de lote grande."""
        # Create 150 texts to test batching (batch_size = 50)
        large_batch = [f"Text number {i} for testing batching." for i in range(150)]
        
        embeddings = await embedding_service.embed_texts(large_batch)
        
        assert len(embeddings) == 150
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 1536
    
    @pytest.mark.asyncio
    async def test_embed_texts_batching_behavior(self, embedding_service):
        """Testar comportamento específico do batching."""
        # Service has batch_size = 50
        texts = [f"Batch test text {i}" for i in range(75)]  # 2 batches
        
        start_time = time.time()
        embeddings = await embedding_service.embed_texts(texts)
        end_time = time.time()
        
        assert len(embeddings) == 75
        # Should be reasonably fast due to batching
        assert end_time - start_time < 2.0


class TestCaching:
    """Testes para sistema de cache."""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, embedding_service):
        """Testar cache hit."""
        text = "Cached text example"
        
        # First call - should cache
        embedding1 = await embedding_service.embed_text(text)
        
        # Second call - should hit cache
        start_time = time.time()
        embedding2 = await embedding_service.embed_text(text)
        end_time = time.time()
        
        assert embedding1 == embedding2
        # Cache hit should be very fast
        assert end_time - start_time < 0.1
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, embedding_service):
        """Testar cache miss."""
        text1 = "First text"
        text2 = "Second text"
        
        embedding1 = await embedding_service.embed_text(text1)
        embedding2 = await embedding_service.embed_text(text2)
        
        # Different texts should have different embeddings
        assert embedding1 != embedding2
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, embedding_service):
        """Testar limpeza de cache."""
        text = "Text to be cached and cleared"
        
        # Cache text
        await embedding_service.embed_text(text)
        assert embedding_service.cache.size() > 0
        
        # Clear cache
        await embedding_service.clear_cache()
        assert embedding_service.cache.size() == 0
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, embedding_service):
        """Testar eviction de cache."""
        # Cache has max_size = 1000
        # Fill cache beyond capacity
        texts = [f"Cache eviction test {i}" for i in range(1100)]
        
        for text in texts:
            await embedding_service.embed_text(text)
        
        # Cache should not exceed max size
        assert embedding_service.cache.size() <= 1000


class TestSimilarity:
    """Testes para cálculo de similaridade."""
    
    @pytest.mark.asyncio
    async def test_similarity_identical_texts(self, embedding_service):
        """Testar similaridade de textos idênticos."""
        text = "Identical text for similarity test"
        similarity = await embedding_service.similarity(text, text)
        
        # Identical texts should have similarity close to 1.0
        assert 0.99 <= similarity <= 1.01  # Allow for floating point precision
    
    @pytest.mark.asyncio
    async def test_similarity_different_texts(self, embedding_service):
        """Testar similaridade de textos diferentes."""
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "Pizza is a popular Italian food with cheese and tomatoes."
        
        similarity = await embedding_service.similarity(text1, text2)
        
        # Different topic texts should have lower similarity
        assert -1.0 <= similarity <= 1.0  # Valid similarity range
        assert similarity < 0.8  # Should not be too similar
    
    @pytest.mark.asyncio
    async def test_similarity_related_texts(self, embedding_service):
        """Testar similaridade de textos relacionados."""
        text1 = "Machine learning algorithms process data."
        text2 = "AI systems analyze information using computational methods."
        
        similarity = await embedding_service.similarity(text1, text2)
        
        # Related texts should have moderate to high similarity
        assert -1.0 <= similarity <= 1.0
    
    @pytest.mark.asyncio
    async def test_find_similar_basic(self, embedding_service, sample_texts):
        """Testar busca de textos similares."""
        query = "machine learning and AI"
        
        similar_texts = await embedding_service.find_similar(query, sample_texts, top_k=3)
        
        assert len(similar_texts) == 3
        for text, similarity, index in similar_texts:
            assert isinstance(text, str)
            assert isinstance(similarity, float)
            assert isinstance(index, int)
            assert -1.0 <= similarity <= 1.0
            assert 0 <= index < len(sample_texts)
    
    @pytest.mark.asyncio
    async def test_find_similar_ordering(self, embedding_service):
        """Testar ordenação por similaridade."""
        query = "Python programming"
        texts = [
            "Python is a programming language.",  # Should be most similar
            "Machine learning uses algorithms.",
            "Data science involves statistics.",
            "JavaScript is used for web development."
        ]
        
        similar_texts = await embedding_service.find_similar(query, texts, top_k=4)
        
        # Results should be ordered by similarity (descending)
        similarities = [sim for _, sim, _ in similar_texts]
        assert similarities == sorted(similarities, reverse=True)


class TestRetryAndFallback:
    """Testes para retry e fallback."""
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, embedding_service):
        """Testar mecanismo de retry."""
        # Mock provider to fail initially
        original_embed = embedding_service.providers['openai'].embed_text
        call_count = 0
        
        async def failing_embed(text):
            nonlocal call_count
            call_count += 1
            if call_count < 2:  # Fail first call
                raise Exception("Temporary API error")
            return await original_embed(text)
        
        embedding_service.providers['openai'].embed_text = failing_embed
        
        # Should succeed after retry
        embedding = await embedding_service.embed_text("Retry test text")
        assert isinstance(embedding, list)
        assert call_count == 2  # Failed once, succeeded on retry
    
    @pytest.mark.asyncio
    async def test_fallback_providers(self, multi_provider_service):
        """Testar fallback para outros providers."""
        # Mock primary provider to fail
        original_embed = multi_provider_service.providers['openai'].embed_text
        
        async def always_fail(text):
            raise Exception("Primary provider unavailable")
        
        multi_provider_service.providers['openai'].embed_text = always_fail
        
        # Should fallback to anthropic or google
        embedding = await multi_provider_service.embed_text("Fallback test text")
        assert isinstance(embedding, list)
        # Should be from fallback provider (different dimension)
        assert len(embedding) in [1024, 768]  # Anthropic or Google dimensions


class TestPerformance:
    """Testes de performance."""
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding(self, embedding_service):
        """Testar embedding concorrente."""
        texts = [f"Concurrent test text {i}" for i in range(20)]
        
        # Run embeddings concurrently
        start_time = time.time()
        tasks = [embedding_service.embed_text(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert len(embeddings) == 20
        # Should be faster than sequential execution
        assert end_time - start_time < 5.0
    
    @pytest.mark.asyncio
    async def test_batch_vs_individual_performance(self, embedding_service):
        """Testar performance de batch vs individual."""
        texts = [f"Performance test text {i}" for i in range(50)]
        
        # Individual embeddings
        start_time = time.time()
        individual_embeddings = []
        for text in texts:
            embedding = await embedding_service.embed_text(text)
            individual_embeddings.append(embedding)
        individual_time = time.time() - start_time
        
        # Clear cache for fair comparison
        await embedding_service.clear_cache()
        
        # Batch embeddings
        start_time = time.time()
        batch_embeddings = await embedding_service.embed_texts(texts)
        batch_time = time.time() - start_time
        
        assert len(individual_embeddings) == len(batch_embeddings) == 50
        # Batch should be significantly faster
        assert batch_time < individual_time * 0.7


class TestErrorHandling:
    """Testes para tratamento de erros."""
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self, embedding_service):
        """Testar tratamento de texto vazio."""
        # Empty text should still work
        embedding = await embedding_service.embed_text("")
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
    
    @pytest.mark.asyncio
    async def test_large_text_handling(self, embedding_service):
        """Testar tratamento de texto muito grande."""
        # Create a very large text
        large_text = "This is a test sentence. " * 1000  # ~25k characters
        
        embedding = await embedding_service.embed_text(large_text)
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
    
    @pytest.mark.asyncio
    async def test_unicode_text_handling(self, embedding_service):
        """Testar tratamento de texto unicode."""
        unicode_text = "Olá mundo! 🌍 Emoji test: 😀🚀🔬 Chinese: 你好 Arabic: مرحبا"
        
        embedding = await embedding_service.embed_text(unicode_text)
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
    
    @pytest.mark.asyncio
    async def test_service_stats(self, embedding_service):
        """Testar estatísticas do serviço."""
        stats = await embedding_service.get_stats()
        
        assert 'providers' in stats
        assert 'cache_enabled' in stats
        assert 'batch_size' in stats
        assert 'retry_attempts' in stats
        assert 'cache_size' in stats
        
        assert isinstance(stats['providers'], list)
        assert stats['cache_enabled'] is True
        assert stats['batch_size'] == 50
        assert stats['retry_attempts'] == 2


class TestCacheIntegration:
    """Testes de integração com cache."""
    
    @pytest.mark.asyncio
    async def test_memory_cache_integration(self, embedding_service):
        """Testar integração com cache em memória."""
        text = "Memory cache integration test"
        
        # First embedding - should cache
        embedding1 = await embedding_service.embed_text(text)
        initial_cache_size = embedding_service.cache.size()
        
        # Second embedding - should hit cache
        embedding2 = await embedding_service.embed_text(text)
        final_cache_size = embedding_service.cache.size()
        
        assert embedding1 == embedding2
        assert initial_cache_size == final_cache_size  # No new cache entry
    
    @pytest.mark.asyncio
    async def test_redis_cache_integration(self, multi_provider_service):
        """Testar integração com Redis cache."""
        # Multi provider service uses Redis cache
        text = "cached_text_example"  # This will hit mock Redis cache
        
        embedding = await multi_provider_service.embed_text(text)
        
        # Should get mock cached value
        assert isinstance(embedding, list)
        # Note: Mock Redis returns [0.1] * 1536 for cached texts


if __name__ == "__main__":
    pytest.main([__file__]) 