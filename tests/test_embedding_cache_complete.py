"""
Testes completos para EmbeddingCache
Cobrindo todos os cenários não testados para aumentar a cobertura
"""
import pytest
import tempfile
import os
import json
import pickle
import time
from unittest.mock import Mock, patch, MagicMock
from src.embeddings.embedding_cache import (
    EmbeddingCache, CacheEntry, CacheStrategy, 
    LRUCache, TTLCache, SizeLimitedCache
)


class TestCacheEntry:
    """Testes para CacheEntry"""
    
    def test_cache_entry_creation(self):
        """Testa criação de entrada de cache"""
        embedding = [0.1, 0.2, 0.3]
        entry = CacheEntry(
            text="test text",
            embedding=embedding,
            metadata={"model": "test-model"}
        )
        assert entry.text == "test text"
        assert entry.embedding == embedding
        assert entry.metadata["model"] == "test-model"
        assert entry.timestamp is not None
        assert entry.access_count == 0
        
    def test_cache_entry_access(self):
        """Testa acesso à entrada do cache"""
        entry = CacheEntry("test", [0.1, 0.2])
        initial_count = entry.access_count
        initial_time = entry.last_accessed
        
        time.sleep(0.01)  # Pequena pausa
        entry.mark_accessed()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time
        
    def test_cache_entry_size(self):
        """Testa cálculo do tamanho da entrada"""
        embedding = [0.1] * 1000  # 1000 dimensões
        entry = CacheEntry("test text", embedding)
        size = entry.get_size()
        assert size > 0
        assert isinstance(size, int)
        
    def test_cache_entry_serialization(self):
        """Testa serialização da entrada"""
        entry = CacheEntry("test", [0.1, 0.2], {"key": "value"})
        serialized = entry.to_dict()
        
        assert serialized["text"] == "test"
        assert serialized["embedding"] == [0.1, 0.2]
        assert serialized["metadata"]["key"] == "value"
        
        # Testa desserialização
        new_entry = CacheEntry.from_dict(serialized)
        assert new_entry.text == entry.text
        assert new_entry.embedding == entry.embedding
        assert new_entry.metadata == entry.metadata


class TestCacheStrategy:
    """Testes para CacheStrategy enum"""
    
    def test_cache_strategy_values(self):
        """Testa valores do enum CacheStrategy"""
        assert CacheStrategy.LRU == "lru"
        assert CacheStrategy.TTL == "ttl"
        assert CacheStrategy.SIZE_LIMITED == "size_limited"
        assert CacheStrategy.FIFO == "fifo"


class TestLRUCache:
    """Testes para LRUCache"""
    
    @pytest.fixture
    def lru_cache(self):
        return LRUCache(max_size=3)
    
    def test_init(self, lru_cache):
        """Testa inicialização"""
        assert lru_cache.max_size == 3
        assert len(lru_cache.cache) == 0
        assert len(lru_cache.access_order) == 0
        
    def test_put_and_get(self, lru_cache):
        """Testa inserção e recuperação"""
        entry = CacheEntry("test", [0.1, 0.2])
        lru_cache.put("key1", entry)
        
        retrieved = lru_cache.get("key1")
        assert retrieved is not None
        assert retrieved.text == "test"
        
    def test_lru_eviction(self, lru_cache):
        """Testa remoção LRU"""
        # Adiciona 4 entradas (limite é 3)
        for i in range(4):
            entry = CacheEntry(f"test{i}", [0.1 * i])
            lru_cache.put(f"key{i}", entry)
        
        # A primeira entrada deve ter sido removida
        assert lru_cache.get("key0") is None
        assert lru_cache.get("key1") is not None
        assert lru_cache.get("key2") is not None
        assert lru_cache.get("key3") is not None
        
    def test_access_order_update(self, lru_cache):
        """Testa atualização da ordem de acesso"""
        # Adiciona 3 entradas
        for i in range(3):
            entry = CacheEntry(f"test{i}", [0.1 * i])
            lru_cache.put(f"key{i}", entry)
        
        # Acessa a primeira entrada
        lru_cache.get("key0")
        
        # Adiciona nova entrada
        entry = CacheEntry("test3", [0.3])
        lru_cache.put("key3", entry)
        
        # key1 deve ter sido removida (era a menos recentemente usada)
        assert lru_cache.get("key1") is None
        assert lru_cache.get("key0") is not None  # Foi acessada recentemente
        
    def test_contains(self, lru_cache):
        """Testa verificação de existência"""
        entry = CacheEntry("test", [0.1])
        lru_cache.put("key1", entry)
        
        assert "key1" in lru_cache
        assert "key2" not in lru_cache
        
    def test_clear(self, lru_cache):
        """Testa limpeza do cache"""
        entry = CacheEntry("test", [0.1])
        lru_cache.put("key1", entry)
        
        lru_cache.clear()
        assert len(lru_cache.cache) == 0
        assert len(lru_cache.access_order) == 0
        
    def test_size(self, lru_cache):
        """Testa obtenção do tamanho"""
        assert lru_cache.size() == 0
        
        entry = CacheEntry("test", [0.1])
        lru_cache.put("key1", entry)
        assert lru_cache.size() == 1


class TestTTLCache:
    """Testes para TTLCache"""
    
    @pytest.fixture
    def ttl_cache(self):
        return TTLCache(ttl_seconds=0.1)  # TTL muito baixo para testes
    
    def test_init(self, ttl_cache):
        """Testa inicialização"""
        assert ttl_cache.ttl_seconds == 0.1
        assert len(ttl_cache.cache) == 0
        
    def test_put_and_get_valid(self, ttl_cache):
        """Testa inserção e recuperação dentro do TTL"""
        entry = CacheEntry("test", [0.1, 0.2])
        ttl_cache.put("key1", entry)
        
        retrieved = ttl_cache.get("key1")
        assert retrieved is not None
        assert retrieved.text == "test"
        
    def test_ttl_expiration(self, ttl_cache):
        """Testa expiração por TTL"""
        entry = CacheEntry("test", [0.1, 0.2])
        ttl_cache.put("key1", entry)
        
        # Espera expirar
        time.sleep(0.15)
        
        retrieved = ttl_cache.get("key1")
        assert retrieved is None
        
    def test_cleanup_expired(self, ttl_cache):
        """Testa limpeza de entradas expiradas"""
        # Adiciona várias entradas
        for i in range(3):
            entry = CacheEntry(f"test{i}", [0.1 * i])
            ttl_cache.put(f"key{i}", entry)
        
        # Espera expirar
        time.sleep(0.15)
        
        # Força limpeza
        ttl_cache.cleanup_expired()
        
        assert ttl_cache.size() == 0
        
    def test_is_expired(self, ttl_cache):
        """Testa verificação de expiração"""
        entry = CacheEntry("test", [0.1])
        ttl_cache.put("key1", entry)
        
        # Não deve estar expirada imediatamente
        assert not ttl_cache.is_expired("key1")
        
        # Espera expirar
        time.sleep(0.15)
        assert ttl_cache.is_expired("key1")


class TestSizeLimitedCache:
    """Testes para SizeLimitedCache"""
    
    @pytest.fixture
    def size_cache(self):
        return SizeLimitedCache(max_size_bytes=1000)
    
    def test_init(self, size_cache):
        """Testa inicialização"""
        assert size_cache.max_size_bytes == 1000
        assert size_cache.current_size_bytes == 0
        assert len(size_cache.cache) == 0
        
    def test_put_within_limit(self, size_cache):
        """Testa inserção dentro do limite"""
        entry = CacheEntry("small", [0.1, 0.2])  # Entrada pequena
        size_cache.put("key1", entry)
        
        assert size_cache.get("key1") is not None
        assert size_cache.current_size_bytes > 0
        
    def test_size_limit_eviction(self, size_cache):
        """Testa remoção por limite de tamanho"""
        # Cria entrada grande que excede o limite
        large_embedding = [0.1] * 500  # Embedding grande
        large_entry = CacheEntry("large", large_embedding)
        
        # Adiciona entrada pequena primeiro
        small_entry = CacheEntry("small", [0.1, 0.2])
        size_cache.put("small_key", small_entry)
        
        # Adiciona entrada grande
        size_cache.put("large_key", large_entry)
        
        # Verifica se o cache não excede o limite
        assert size_cache.current_size_bytes <= size_cache.max_size_bytes
        
    def test_calculate_entry_size(self, size_cache):
        """Testa cálculo do tamanho da entrada"""
        entry = CacheEntry("test", [0.1, 0.2, 0.3])
        size = size_cache.calculate_entry_size(entry)
        assert size > 0
        assert isinstance(size, int)
        
    def test_evict_oldest(self, size_cache):
        """Testa remoção da entrada mais antiga"""
        # Adiciona várias entradas pequenas
        for i in range(3):
            entry = CacheEntry(f"test{i}", [0.1 * i])
            size_cache.put(f"key{i}", entry)
        
        initial_size = size_cache.size()
        size_cache.evict_oldest()
        
        assert size_cache.size() == initial_size - 1


class TestEmbeddingCache:
    """Testes para EmbeddingCache principal"""
    
    @pytest.fixture
    def embedding_cache(self):
        return EmbeddingCache(strategy=CacheStrategy.LRU, max_size=5)
    
    def test_init_lru(self):
        """Testa inicialização com estratégia LRU"""
        cache = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=10)
        assert isinstance(cache.cache, LRUCache)
        assert cache.cache.max_size == 10
        
    def test_init_ttl(self):
        """Testa inicialização com estratégia TTL"""
        cache = EmbeddingCache(strategy=CacheStrategy.TTL, ttl_seconds=60)
        assert isinstance(cache.cache, TTLCache)
        assert cache.cache.ttl_seconds == 60
        
    def test_init_size_limited(self):
        """Testa inicialização com estratégia de tamanho limitado"""
        cache = EmbeddingCache(strategy=CacheStrategy.SIZE_LIMITED, max_size_bytes=2048)
        assert isinstance(cache.cache, SizeLimitedCache)
        assert cache.cache.max_size_bytes == 2048
        
    def test_put_and_get(self, embedding_cache):
        """Testa inserção e recuperação"""
        text = "test text"
        embedding = [0.1, 0.2, 0.3]
        metadata = {"model": "test-model"}
        
        embedding_cache.put(text, embedding, metadata)
        
        retrieved = embedding_cache.get(text)
        assert retrieved is not None
        assert retrieved.embedding == embedding
        assert retrieved.metadata == metadata
        
    def test_get_nonexistent(self, embedding_cache):
        """Testa recuperação de entrada inexistente"""
        result = embedding_cache.get("nonexistent")
        assert result is None
        
    def test_contains(self, embedding_cache):
        """Testa verificação de existência"""
        text = "test text"
        embedding = [0.1, 0.2]
        
        assert not embedding_cache.contains(text)
        
        embedding_cache.put(text, embedding)
        assert embedding_cache.contains(text)
        
    def test_remove(self, embedding_cache):
        """Testa remoção de entrada"""
        text = "test text"
        embedding = [0.1, 0.2]
        
        embedding_cache.put(text, embedding)
        assert embedding_cache.contains(text)
        
        embedding_cache.remove(text)
        assert not embedding_cache.contains(text)
        
    def test_clear(self, embedding_cache):
        """Testa limpeza do cache"""
        embedding_cache.put("text1", [0.1])
        embedding_cache.put("text2", [0.2])
        
        assert embedding_cache.size() == 2
        
        embedding_cache.clear()
        assert embedding_cache.size() == 0
        
    def test_size(self, embedding_cache):
        """Testa obtenção do tamanho"""
        assert embedding_cache.size() == 0
        
        embedding_cache.put("text1", [0.1])
        assert embedding_cache.size() == 1
        
        embedding_cache.put("text2", [0.2])
        assert embedding_cache.size() == 2
        
    def test_get_stats(self, embedding_cache):
        """Testa obtenção de estatísticas"""
        # Adiciona algumas entradas
        embedding_cache.put("text1", [0.1])
        embedding_cache.put("text2", [0.2])
        
        # Faz alguns acessos
        embedding_cache.get("text1")
        embedding_cache.get("text1")  # Hit
        embedding_cache.get("nonexistent")  # Miss
        
        stats = embedding_cache.get_stats()
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        
        assert stats["size"] == 2
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


class TestEmbeddingCachePersistence:
    """Testes de persistência do cache"""
    
    @pytest.fixture
    def embedding_cache(self):
        return EmbeddingCache(strategy=CacheStrategy.LRU, max_size=5)
    
    def test_save_to_file(self, embedding_cache):
        """Testa salvamento em arquivo"""
        # Adiciona algumas entradas
        embedding_cache.put("text1", [0.1, 0.2])
        embedding_cache.put("text2", [0.3, 0.4])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
            
        try:
            embedding_cache.save_to_file(filepath)
            assert os.path.exists(filepath)
            
            # Verifica conteúdo do arquivo
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert "entries" in data
                assert len(data["entries"]) == 2
                
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
                
    def test_load_from_file(self, embedding_cache):
        """Testa carregamento de arquivo"""
        # Adiciona dados e salva
        embedding_cache.put("text1", [0.1, 0.2], {"model": "test"})
        embedding_cache.put("text2", [0.3, 0.4])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
            
        try:
            embedding_cache.save_to_file(filepath)
            
            # Cria novo cache e carrega
            new_cache = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=5)
            new_cache.load_from_file(filepath)
            
            # Verifica se os dados foram carregados
            assert new_cache.size() == 2
            assert new_cache.contains("text1")
            assert new_cache.contains("text2")
            
            retrieved = new_cache.get("text1")
            assert retrieved.embedding == [0.1, 0.2]
            assert retrieved.metadata["model"] == "test"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
                
    def test_save_to_pickle(self, embedding_cache):
        """Testa salvamento em formato pickle"""
        embedding_cache.put("text1", [0.1, 0.2])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
            
        try:
            embedding_cache.save_to_pickle(filepath)
            assert os.path.exists(filepath)
            
            # Verifica se pode ser carregado
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                assert isinstance(data, dict)
                
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
                
    def test_load_from_pickle(self, embedding_cache):
        """Testa carregamento de formato pickle"""
        embedding_cache.put("text1", [0.1, 0.2])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
            
        try:
            embedding_cache.save_to_pickle(filepath)
            
            # Cria novo cache e carrega
            new_cache = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=5)
            new_cache.load_from_pickle(filepath)
            
            assert new_cache.size() == 1
            assert new_cache.contains("text1")
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestEmbeddingCacheAdvanced:
    """Testes avançados do EmbeddingCache"""
    
    def test_batch_operations(self):
        """Testa operações em lote"""
        cache = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=10)
        
        # Batch put
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        cache.batch_put(texts, embeddings, metadatas)
        
        assert cache.size() == 3
        for text in texts:
            assert cache.contains(text)
            
        # Batch get
        results = cache.batch_get(texts)
        assert len(results) == 3
        assert all(result is not None for result in results)
        
    def test_cache_warming(self):
        """Testa aquecimento do cache"""
        cache = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=5)
        
        # Dados para aquecimento
        warm_data = [
            ("important_text1", [0.1, 0.2], {"priority": "high"}),
            ("important_text2", [0.3, 0.4], {"priority": "high"}),
            ("normal_text", [0.5, 0.6], {"priority": "normal"})
        ]
        
        cache.warm_cache(warm_data)
        
        assert cache.size() == 3
        for text, embedding, metadata in warm_data:
            entry = cache.get(text)
            assert entry is not None
            assert entry.embedding == embedding
            assert entry.metadata == metadata
            
    def test_cache_metrics_detailed(self):
        """Testa métricas detalhadas do cache"""
        cache = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=3)
        
        # Adiciona entradas
        cache.put("text1", [0.1])
        cache.put("text2", [0.2])
        cache.put("text3", [0.3])
        
        # Faz acessos variados
        cache.get("text1")  # Hit
        cache.get("text1")  # Hit
        cache.get("text2")  # Hit
        cache.get("nonexistent1")  # Miss
        cache.get("nonexistent2")  # Miss
        
        metrics = cache.get_detailed_metrics()
        
        assert "total_entries" in metrics
        assert "hit_rate" in metrics
        assert "miss_rate" in metrics
        assert "average_access_count" in metrics
        assert "most_accessed_entries" in metrics
        
        assert metrics["total_entries"] == 3
        assert metrics["hit_rate"] > 0
        assert metrics["miss_rate"] > 0
        
    def test_cache_eviction_callbacks(self):
        """Testa callbacks de remoção"""
        evicted_entries = []
        
        def eviction_callback(key, entry):
            evicted_entries.append((key, entry))
            
        cache = EmbeddingCache(
            strategy=CacheStrategy.LRU, 
            max_size=2,
            eviction_callback=eviction_callback
        )
        
        # Adiciona 3 entradas (limite é 2)
        cache.put("text1", [0.1])
        cache.put("text2", [0.2])
        cache.put("text3", [0.3])  # Deve causar remoção
        
        assert len(evicted_entries) == 1
        assert evicted_entries[0][0] == "text1"  # Primeira entrada removida
        
    def test_memory_usage_monitoring(self):
        """Testa monitoramento de uso de memória"""
        cache = EmbeddingCache(strategy=CacheStrategy.SIZE_LIMITED, max_size_bytes=1024)
        
        # Adiciona entradas de tamanhos conhecidos
        small_embedding = [0.1] * 10
        large_embedding = [0.1] * 100
        
        cache.put("small", small_embedding)
        cache.put("large", large_embedding)
        
        memory_info = cache.get_memory_usage()
        
        assert "current_size_bytes" in memory_info
        assert "max_size_bytes" in memory_info
        assert "utilization_percentage" in memory_info
        assert "entry_count" in memory_info
        
        assert memory_info["current_size_bytes"] > 0
        assert memory_info["entry_count"] == 2


class TestEmbeddingCacheIntegration:
    """Testes de integração do EmbeddingCache"""
    
    def test_with_embedding_service_mock(self):
        """Testa integração com serviço de embedding (mock)"""
        cache = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=5)
        
        # Mock do serviço de embedding
        mock_service = Mock()
        mock_service.embed_text.return_value = [0.1, 0.2, 0.3]
        
        def get_embedding_with_cache(text):
            # Verifica cache primeiro
            cached = cache.get(text)
            if cached:
                return cached.embedding
                
            # Se não estiver em cache, calcula e armazena
            embedding = mock_service.embed_text(text)
            cache.put(text, embedding)
            return embedding
        
        # Primeira chamada - deve chamar o serviço
        result1 = get_embedding_with_cache("test text")
        assert result1 == [0.1, 0.2, 0.3]
        assert mock_service.embed_text.call_count == 1
        
        # Segunda chamada - deve usar cache
        result2 = get_embedding_with_cache("test text")
        assert result2 == [0.1, 0.2, 0.3]
        assert mock_service.embed_text.call_count == 1  # Não chamou novamente
        
    def test_concurrent_access(self):
        """Testa acesso concorrente (simulado)"""
        import threading
        import time
        
        cache = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=10)
        results = []
        
        def worker(worker_id):
            for i in range(5):
                text = f"worker_{worker_id}_text_{i}"
                embedding = [worker_id * 0.1, i * 0.1]
                cache.put(text, embedding)
                
                # Pequena pausa para simular processamento
                time.sleep(0.001)
                
                retrieved = cache.get(text)
                results.append(retrieved is not None)
        
        # Cria e executa threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Espera todas terminarem
        for thread in threads:
            thread.join()
        
        # Verifica resultados
        assert len(results) == 15  # 3 workers * 5 operações
        assert all(results)  # Todas as operações devem ter sucesso
        
    def test_cache_persistence_workflow(self):
        """Testa fluxo completo de persistência"""
        # Cache inicial
        cache1 = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=5)
        cache1.put("persistent_text1", [0.1, 0.2], {"source": "test"})
        cache1.put("persistent_text2", [0.3, 0.4])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
            
        try:
            # Salva cache
            cache1.save_to_file(filepath)
            
            # Simula reinicialização da aplicação
            cache2 = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=5)
            cache2.load_from_file(filepath)
            
            # Verifica se os dados persistiram
            assert cache2.size() == 2
            
            entry1 = cache2.get("persistent_text1")
            assert entry1 is not None
            assert entry1.embedding == [0.1, 0.2]
            assert entry1.metadata["source"] == "test"
            
            entry2 = cache2.get("persistent_text2")
            assert entry2 is not None
            assert entry2.embedding == [0.3, 0.4]
            
            # Adiciona nova entrada e salva novamente
            cache2.put("new_text", [0.5, 0.6])
            cache2.save_to_file(filepath)
            
            # Carrega em terceiro cache
            cache3 = EmbeddingCache(strategy=CacheStrategy.LRU, max_size=5)
            cache3.load_from_file(filepath)
            
            assert cache3.size() == 3
            assert cache3.contains("new_text")
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath) 