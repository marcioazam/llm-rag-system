"""
Sistema de Cache Multicamada para Enhanced Corrective RAG.
Implementa L1 (Memória), L2 (Redis), L3 (SQLite) com inteligência adaptativa.
"""

import asyncio
import json
import sqlite3
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadados do cache."""
    key: str
    level: str  # L1, L2, L3
    created_at: float
    accessed_at: float
    access_count: int
    ttl: int
    size_bytes: int
    hit_rate: float = 0.0


class MemoryCache:
    """Cache L1 - Memória com LRU."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_times[key] = time.time()
            self.stats['hits'] += 1
            return self.cache[key]
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        # Verificar se precisa fazer eviction
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            'value': value,
            'created_at': time.time(),
            'ttl': ttl
        }
        self.access_times[key] = time.time()
    
    def _evict_lru(self):
        """Remove o item menos recentemente usado."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.stats['evictions'] += 1
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()


class RedisCache:
    """Cache L2 - Redis com persistência."""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 prefix: str = 'enhanced_rag:'):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.redis = None
        self.connected = False
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
    
    async def connect(self):
        """Conecta ao Redis."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis não disponível - pulando conexão")
            return False
        
        try:
            self.redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Testar conexão
            await self.redis.ping()
            self.connected = True
            logger.info(f"✅ Conectado ao Redis: {self.host}:{self.port}/{self.db}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao conectar Redis: {e}")
            self.connected = False
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        if not self.connected:
            return None
        
        try:
            full_key = f"{self.prefix}{key}"
            data = await self.redis.get(full_key)
            
            if data:
                self.stats['hits'] += 1
                return json.loads(data)
            else:
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.warning(f"Erro ao ler do Redis: {e}")
            self.stats['errors'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        if not self.connected:
            return False
        
        try:
            full_key = f"{self.prefix}{key}"
            data = json.dumps(value, default=str)
            
            await self.redis.setex(full_key, ttl, data)
            return True
            
        except Exception as e:
            logger.warning(f"Erro ao escrever no Redis: {e}")
            self.stats['errors'] += 1
            return False
    
    async def delete(self, key: str):
        if not self.connected:
            return False
        
        try:
            full_key = f"{self.prefix}{key}"
            await self.redis.delete(full_key)
            return True
        except Exception as e:
            logger.warning(f"Erro ao deletar do Redis: {e}")
            return False
    
    async def clear_pattern(self, pattern: str):
        """Limpa chaves que fazem match com o pattern."""
        if not self.connected:
            return 0
        
        try:
            full_pattern = f"{self.prefix}{pattern}"
            keys = await self.redis.keys(full_pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Erro ao limpar pattern do Redis: {e}")
            return 0


class SQLiteCache:
    """Cache L3 - SQLite para persistência longa."""
    
    def __init__(self, db_path: str = "cache/enhanced_rag_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        self._initialize_db()
    
    def _initialize_db(self):
        """Inicializa o banco SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        accessed_at REAL NOT NULL,
                        access_count INTEGER DEFAULT 1,
                        ttl INTEGER NOT NULL,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_accessed_at 
                    ON cache_entries(accessed_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at 
                    ON cache_entries(created_at)
                """)
                
                conn.commit()
                logger.info(f"✅ SQLite cache inicializado: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar SQLite cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, ttl, created_at FROM cache_entries WHERE key = ?",
                    (key,)
                )
                result = cursor.fetchone()
                
                if result:
                    value_str, ttl, created_at = result
                    
                    # Verificar TTL
                    if time.time() - created_at > ttl:
                        # Expirado - remover
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                        self.stats['misses'] += 1
                        return None
                    
                    # Atualizar estatísticas de acesso
                    conn.execute(
                        "UPDATE cache_entries SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                        (time.time(), key)
                    )
                    conn.commit()
                    
                    self.stats['hits'] += 1
                    return json.loads(value_str)
                else:
                    self.stats['misses'] += 1
                    return None
                    
        except Exception as e:
            logger.warning(f"Erro ao ler do SQLite: {e}")
            self.stats['errors'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600, metadata: Optional[Dict] = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                value_str = json.dumps(value, default=str)
                metadata_str = json.dumps(metadata or {}, default=str)
                current_time = time.time()
                
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_at, accessed_at, ttl, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (key, value_str, current_time, current_time, ttl, metadata_str))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.warning(f"Erro ao escrever no SQLite: {e}")
            self.stats['errors'] += 1
            return False
    
    def cleanup_expired(self) -> int:
        """Remove entradas expiradas."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                current_time = time.time()
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE (? - created_at) > ttl",
                    (current_time,)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Removidas {deleted_count} entradas expiradas do cache SQLite")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Erro ao limpar cache SQLite: {e}")
            return 0


class MultiLayerCache:
    """
    Sistema de cache multicamada com inteligência adaptativa.
    L1: Memória (rápido, volátil)
    L2: Redis (médio, persistente, distribuído)
    L3: SQLite (lento, persistente local, backup)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        
        # Configurações
        self.config = {
            'enable_l1': config.get('enable_l1', True),
            'enable_l2': config.get('enable_l2', True),
            'enable_l3': config.get('enable_l3', True),
            'l1_max_size': config.get('l1_max_size', 1000),
            'redis_host': config.get('redis_host', os.getenv('REDIS_HOST', 'localhost')),
            'redis_port': config.get('redis_port', int(os.getenv('REDIS_PORT', '6379'))),
            'redis_db': config.get('redis_db', int(os.getenv('REDIS_DB', '1'))),
            'redis_password': config.get('redis_password', os.getenv('REDIS_PASSWORD')),
            'sqlite_path': config.get('sqlite_path', 'cache/enhanced_rag_cache.db'),
            'default_ttl': config.get('default_ttl', 3600),
            'promotion_threshold': config.get('promotion_threshold', 3),  # Acesso para promover para L1
        }
        
        # Inicializar camadas
        self.l1 = MemoryCache(self.config['l1_max_size']) if self.config['enable_l1'] else None
        self.l2 = RedisCache(
            host=self.config['redis_host'],
            port=self.config['redis_port'],
            db=self.config['redis_db'],
            password=self.config['redis_password']
        ) if self.config['enable_l2'] else None
        self.l3 = SQLiteCache(self.config['sqlite_path']) if self.config['enable_l3'] else None
        
        # Estatísticas globais
        self.global_stats = {
            'total_requests': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0,
            'promotions': 0,
            'demotions': 0
        }
        
        # Conectar Redis assincronamente
        self.redis_connected = False
    
    async def initialize(self):
        """Inicializa conexões assíncronas."""
        if self.l2:
            self.redis_connected = await self.l2.connect()
    
    async def get(self, key: str) -> Tuple[Optional[Any], str, Dict[str, Any]]:
        """
        Busca valor no cache multicamada.
        
        Returns:
            Tuple[value, source_level, metadata]
        """
        self.global_stats['total_requests'] += 1
        
        # L1: Memória
        if self.l1:
            value = self.l1.get(key)
            if value is not None:
                self.global_stats['l1_hits'] += 1
                return value['value'], 'L1', {
                    'hit_level': 'L1',
                    'created_at': value['created_at'],
                    'age': time.time() - value['created_at'],
                    'access_count': 1  # L1 não rastreia count detalhado
                }
        
        # L2: Redis
        if self.l2 and self.redis_connected:
            value = await self.l2.get(key)
            if value is not None:
                self.global_stats['l2_hits'] += 1
                
                # Promover para L1 se configurado
                if self.l1:
                    self.l1.set(key, value, self.config['default_ttl'])
                    self.global_stats['promotions'] += 1
                
                return value, 'L2', {
                    'hit_level': 'L2',
                    'promoted_to_l1': self.l1 is not None,
                    'age': time.time() - value.get('created_at', time.time()),
                    'access_count': value.get('access_count', 1)
                }
        
        # L3: SQLite
        if self.l3:
            value = self.l3.get(key)
            if value is not None:
                self.global_stats['l3_hits'] += 1
                
                # Promover para L2 e L1
                if self.l2 and self.redis_connected:
                    await self.l2.set(key, value, self.config['default_ttl'])
                
                if self.l1:
                    self.l1.set(key, value, self.config['default_ttl'])
                    self.global_stats['promotions'] += 1
                
                return value, 'L3', {
                    'hit_level': 'L3',
                    'promoted_to_l2': self.l2 is not None and self.redis_connected,
                    'promoted_to_l1': self.l1 is not None,
                    'age': time.time() - value.get('created_at', time.time()),
                    'access_count': value.get('access_count', 1)
                }
        
        # Miss em todas as camadas
        self.global_stats['misses'] += 1
        return None, 'MISS', {'hit_level': 'MISS'}
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict] = None):
        """
        Armazena valor em todas as camadas configuradas.
        """
        ttl = ttl or self.config['default_ttl']
        metadata = metadata or {}
        
        # Preparar valor com metadata
        cached_value = {
            'value': value,
            'created_at': time.time(),
            'metadata': metadata
        }
        
        # L1: Memória
        if self.l1:
            self.l1.set(key, cached_value, ttl)
        
        # L2: Redis
        if self.l2 and self.redis_connected:
            await self.l2.set(key, cached_value, ttl)
        
        # L3: SQLite
        if self.l3:
            self.l3.set(key, cached_value, ttl, metadata)
        
        logger.debug(f"Cached key '{key}' em camadas disponíveis")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas do cache."""
        stats = {
            'global': self.global_stats.copy(),
            'l1': self.l1.stats if self.l1 else None,
            'l2': self.l2.stats if self.l2 else None,
            'l3': self.l3.stats if self.l3 else None,
            'hit_rates': {}
        }
        
        # Calcular hit rates
        total_requests = self.global_stats['total_requests']
        if total_requests > 0:
            stats['hit_rates'] = {
                'l1_hit_rate': self.global_stats['l1_hits'] / total_requests,
                'l2_hit_rate': self.global_stats['l2_hits'] / total_requests,
                'l3_hit_rate': self.global_stats['l3_hits'] / total_requests,
                'overall_hit_rate': (
                    self.global_stats['l1_hits'] + 
                    self.global_stats['l2_hits'] + 
                    self.global_stats['l3_hits']
                ) / total_requests
            }
        
        return stats
    
    async def clear_pattern(self, pattern: str):
        """Limpa chaves que fazem match com o pattern em todas as camadas."""
        cleared_count = 0
        
        # L1: Memória (implementação simples)
        if self.l1:
            keys_to_remove = [k for k in self.l1.cache.keys() if pattern in k]
            for key in keys_to_remove:
                if key in self.l1.cache:
                    del self.l1.cache[key]
                if key in self.l1.access_times:
                    del self.l1.access_times[key]
            cleared_count += len(keys_to_remove)
        
        # L2: Redis
        if self.l2 and self.redis_connected:
            redis_cleared = await self.l2.clear_pattern(pattern)
            cleared_count += redis_cleared
        
        # L3: SQLite (implementação básica)
        if self.l3:
            try:
                with sqlite3.connect(self.l3.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM cache_entries WHERE key LIKE ?",
                        (f"%{pattern}%",)
                    )
                    cleared_count += cursor.rowcount
                    conn.commit()
            except Exception as e:
                logger.warning(f"Erro ao limpar pattern do SQLite: {e}")
        
        logger.info(f"Limpas {cleared_count} entradas com pattern '{pattern}'")
        return cleared_count
    
    async def cleanup(self):
        """Limpeza periódica de entradas expiradas."""
        cleaned_count = 0
        
        if self.l3:
            cleaned_count += self.l3.cleanup_expired()
        
        return cleaned_count


# Factory function
def create_multi_layer_cache(config: Optional[Dict] = None) -> MultiLayerCache:
    """Cria instância do cache multicamada."""
    return MultiLayerCache(config) 