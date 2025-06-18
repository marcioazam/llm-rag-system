"""
Sistema de Cache Multi-layer para RAG
Implementa semantic cache, prefix cache, e KV cache
Otimiza latência e resource usage
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import OrderedDict
import redis
import pickle
import aioredis

from ..embeddings.api_embedding_service import APIEmbeddingService

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Entrada de cache com metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0
    ttl: Optional[int] = None
    cache_type: str = "generic"
    metadata: Dict[str, Any] = None

class SemanticCache:
    """
    Cache semântico com threshold de similaridade
    Reduz chamadas de API para queries similares
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.embedding_service = APIEmbeddingService()
        
        # Cache em memória: embedding -> resultado
        self.cache: Dict[str, CacheEntry] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Configurações
        self.max_cache_size = 10000
        self.ttl = 3600  # 1 hora
        
        logger.info(f"Semantic cache inicializado (threshold: {similarity_threshold})")
    
    async def get(self, query: str) -> Optional[Any]:
        """
        Busca no cache semântico
        Retorna resultado se encontrar query similar
        """
        # Gerar embedding da query
        query_embedding = await self.embedding_service.embed_text(query)
        query_embedding = np.array(query_embedding)
        
        # Buscar queries similares
        best_match = None
        best_similarity = 0.0
        
        for cached_query, cached_embedding in self.embeddings.items():
            # Calcular similaridade coseno
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_query
        
        if best_match:
            # Atualizar estatísticas
            cache_entry = self.cache[best_match]
            cache_entry.access_count += 1
            cache_entry.last_access = time.time()
            
            logger.debug(f"Semantic cache hit: {best_similarity:.3f} similarity")
            return cache_entry.value
        
        return None
    
    async def set(self, query: str, value: Any, metadata: Dict[str, Any] = None):
        """Adiciona entrada ao cache semântico"""
        # Gerar embedding
        embedding = await self.embedding_service.embed_text(query)
        embedding = np.array(embedding)
        
        # Criar entrada de cache
        cache_entry = CacheEntry(
            key=query,
            value=value,
            timestamp=time.time(),
            ttl=self.ttl,
            cache_type="semantic",
            metadata=metadata or {}
        )
        
        # Adicionar ao cache
        self.cache[query] = cache_entry
        self.embeddings[query] = embedding
        
        # Limpar cache se necessário
        if len(self.cache) > self.max_cache_size:
            self._evict_lru()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similaridade coseno entre vetores"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _evict_lru(self):
        """Remove entrada menos recentemente usada"""
        if not self.cache:
            return
        
        # Encontrar LRU
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_access or self.cache[k].timestamp
        )
        
        # Remover
        del self.cache[lru_key]
        del self.embeddings[lru_key]

class PrefixCache:
    """
    Cache de prefixo para code completion patterns
    Otimizado para queries que começam com mesmo prefixo
    """
    
    def __init__(self, min_prefix_length: int = 10):
        self.min_prefix_length = min_prefix_length
        self.trie = {}  # Trie structure para prefixos
        self.cache_data = {}  # Hash -> dados
        
        # Configurações
        self.max_entries = 5000
        self.ttl = 1800  # 30 minutos
    
    def get(self, query: str) -> Optional[Any]:
        """Busca por prefixo no cache"""
        if len(query) < self.min_prefix_length:
            return None
        
        # Navegar pela trie
        node = self.trie
        longest_match = None
        
        for i, char in enumerate(query):
            if char not in node:
                break
            
            node = node[char]
            
            # Verificar se há cache neste nível
            if '_cache_key' in node and i >= self.min_prefix_length - 1:
                cache_key = node['_cache_key']
                if cache_key in self.cache_data:
                    entry = self.cache_data[cache_key]
                    
                    # Verificar TTL
                    if self._is_valid(entry):
                        longest_match = entry
        
        if longest_match:
            # Atualizar estatísticas
            longest_match.access_count += 1
            longest_match.last_access = time.time()
            
            logger.debug(f"Prefix cache hit for: {query[:30]}...")
            return longest_match.value
        
        return None
    
    def set(self, query: str, value: Any, metadata: Dict[str, Any] = None):
        """Adiciona ao cache de prefixo"""
        if len(query) < self.min_prefix_length:
            return
        
        # Gerar cache key
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # Criar entrada
        entry = CacheEntry(
            key=cache_key,
            value=value,
            timestamp=time.time(),
            ttl=self.ttl,
            cache_type="prefix",
            metadata=metadata or {}
        )
        
        # Adicionar à trie
        node = self.trie
        for char in query:
            if char not in node:
                node[char] = {}
            node = node[char]
        
        node['_cache_key'] = cache_key
        self.cache_data[cache_key] = entry
        
        # Limpar se necessário
        if len(self.cache_data) > self.max_entries:
            self._evict_old_entries()
    
    def _is_valid(self, entry: CacheEntry) -> bool:
        """Verifica se entrada ainda é válida"""
        if entry.ttl:
            age = time.time() - entry.timestamp
            return age < entry.ttl
        return True
    
    def _evict_old_entries(self):
        """Remove entradas antigas"""
        current_time = time.time()
        
        # Remover expiradas
        expired_keys = []
        for key, entry in self.cache_data.items():
            if entry.ttl and (current_time - entry.timestamp) > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache_data[key]

class KVCache:
    """
    Key-Value cache para resultados de graph traversal
    Usa Redis para persistência e compartilhamento
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.local_cache = OrderedDict()  # Cache local LRU
        self.max_local_size = 1000
        
        # Configurações
        self.default_ttl = 3600  # 1 hora
        self.key_prefix = "rag:kv:"
    
    async def connect(self):
        """Conecta ao Redis"""
        try:
            self.redis_client = await aioredis.create_redis_pool(self.redis_url)
            logger.info("Conectado ao Redis para KV cache")
        except Exception as e:
            logger.warning(f"Erro ao conectar ao Redis: {e}. Usando apenas cache local.")
    
    async def get(self, key: str) -> Optional[Any]:
        """Busca no cache KV"""
        full_key = f"{self.key_prefix}{key}"
        
        # Verificar cache local primeiro
        if full_key in self.local_cache:
            # Move para final (LRU)
            self.local_cache.move_to_end(full_key)
            return self.local_cache[full_key]
        
        # Verificar Redis se disponível
        if self.redis_client:
            try:
                data = await self.redis_client.get(full_key)
                if data:
                    value = pickle.loads(data)
                    
                    # Adicionar ao cache local
                    self._add_to_local_cache(full_key, value)
                    
                    return value
            except Exception as e:
                logger.error(f"Erro ao buscar no Redis: {e}")
        
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ):
        """Adiciona ao cache KV"""
        full_key = f"{self.key_prefix}{key}"
        ttl = ttl or self.default_ttl
        
        # Adicionar ao cache local
        self._add_to_local_cache(full_key, value)
        
        # Adicionar ao Redis se disponível
        if self.redis_client:
            try:
                # Serializar com pickle
                data = pickle.dumps(value)
                await self.redis_client.setex(full_key, ttl, data)
                
                # Armazenar metadata se fornecida
                if metadata:
                    meta_key = f"{full_key}:meta"
                    meta_data = json.dumps(metadata)
                    await self.redis_client.setex(meta_key, ttl, meta_data)
                    
            except Exception as e:
                logger.error(f"Erro ao salvar no Redis: {e}")
    
    def _add_to_local_cache(self, key: str, value: Any):
        """Adiciona ao cache local com LRU"""
        # Remover se já existe
        if key in self.local_cache:
            del self.local_cache[key]
        
        # Adicionar no final
        self.local_cache[key] = value
        
        # Limitar tamanho
        if len(self.local_cache) > self.max_local_size:
            # Remove o mais antigo (primeiro)
            self.local_cache.popitem(last=False)
    
    async def delete(self, key: str):
        """Remove do cache"""
        full_key = f"{self.key_prefix}{key}"
        
        # Remover do cache local
        if full_key in self.local_cache:
            del self.local_cache[full_key]
        
        # Remover do Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(full_key)
                await self.redis_client.delete(f"{full_key}:meta")
            except Exception as e:
                logger.error(f"Erro ao deletar do Redis: {e}")
    
    async def close(self):
        """Fecha conexão com Redis"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()

class MultiLayerCache:
    """
    Sistema de cache multi-layer integrado
    Combina semantic, prefix e KV cache
    """
    
    def __init__(
        self,
        semantic_threshold: float = 0.95,
        enable_redis: bool = True,
        redis_url: str = "redis://localhost:6379"
    ):
        # Inicializar camadas
        self.semantic_cache = SemanticCache(semantic_threshold)
        self.prefix_cache = PrefixCache()
        self.kv_cache = KVCache(redis_url) if enable_redis else None
        
        # Estatísticas
        self.stats = {
            'semantic_hits': 0,
            'prefix_hits': 0,
            'kv_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        logger.info("Sistema de cache multi-layer inicializado")
    
    async def initialize(self):
        """Inicializa conexões necessárias"""
        if self.kv_cache:
            await self.kv_cache.connect()
    
    async def get(
        self, 
        key: str,
        cache_type: str = "auto"
    ) -> Tuple[Optional[Any], str]:
        """
        Busca em múltiplas camadas de cache
        
        Returns:
            Tuple de (valor, tipo_cache_hit)
        """
        self.stats['total_requests'] += 1
        
        # 1. Tentar KV cache para keys exatas
        if cache_type in ["auto", "kv"] and self.kv_cache:
            result = await self.kv_cache.get(key)
            if result is not None:
                self.stats['kv_hits'] += 1
                return result, "kv"
        
        # 2. Tentar prefix cache
        if cache_type in ["auto", "prefix"]:
            result = self.prefix_cache.get(key)
            if result is not None:
                self.stats['prefix_hits'] += 1
                return result, "prefix"
        
        # 3. Tentar semantic cache
        if cache_type in ["auto", "semantic"]:
            result = await self.semantic_cache.get(key)
            if result is not None:
                self.stats['semantic_hits'] += 1
                return result, "semantic"
        
        self.stats['misses'] += 1
        return None, "miss"
    
    async def set(
        self,
        key: str,
        value: Any,
        cache_types: List[str] = ["semantic", "prefix", "kv"],
        ttl: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ):
        """Adiciona a múltiplas camadas de cache"""
        # Semantic cache
        if "semantic" in cache_types:
            await self.semantic_cache.set(key, value, metadata)
        
        # Prefix cache
        if "prefix" in cache_types:
            self.prefix_cache.set(key, value, metadata)
        
        # KV cache
        if "kv" in cache_types and self.kv_cache:
            await self.kv_cache.set(key, value, ttl, metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de cache"""
        total_hits = (
            self.stats['semantic_hits'] +
            self.stats['prefix_hits'] +
            self.stats['kv_hits']
        )
        
        hit_rate = (
            total_hits / self.stats['total_requests']
            if self.stats['total_requests'] > 0
            else 0
        )
        
        return {
            **self.stats,
            'total_hits': total_hits,
            'hit_rate': hit_rate,
            'cache_sizes': {
                'semantic': len(self.semantic_cache.cache),
                'prefix': len(self.prefix_cache.cache_data),
                'kv_local': len(self.kv_cache.local_cache) if self.kv_cache else 0
            }
        }
    
    def clear_all(self):
        """Limpa todos os caches"""
        self.semantic_cache.cache.clear()
        self.semantic_cache.embeddings.clear()
        
        self.prefix_cache.trie.clear()
        self.prefix_cache.cache_data.clear()
        
        if self.kv_cache:
            self.kv_cache.local_cache.clear()
        
        logger.info("Todos os caches foram limpos")
    
    async def close(self):
        """Fecha conexões"""
        if self.kv_cache:
            await self.kv_cache.close()

# Factory function
async def create_multi_layer_cache(
    semantic_threshold: float = 0.95,
    enable_redis: bool = True
) -> MultiLayerCache:
    """Cria e inicializa sistema de cache multi-layer"""
    cache = MultiLayerCache(semantic_threshold, enable_redis)
    await cache.initialize()
    return cache 