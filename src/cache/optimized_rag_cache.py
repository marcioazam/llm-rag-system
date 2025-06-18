"""
Cache Otimizado para RAG - Estratégia Híbrida Recomendada
Combina memória local + SQLite + Redis opcional
"""

import sqlite3
import json
import time
import hashlib
import logging
from typing import Optional, Any, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
from collections import OrderedDict

# Importar configurações
try:
    from src.config.cache_config import get_cache_config, get_redis_settings
except ImportError:
    # Fallback para quando executado standalone
    def get_cache_config():
        return {
            "db_path": "storage/rag_cache.db",
            "max_memory_entries": 1000,
            "enable_redis": False,
            "redis_url": "redis://localhost:6379"
        }
    
    def get_redis_settings():
        return {
            "url": "redis://localhost:6379",
            "max_connections": 10,
            "socket_timeout": 5.0
        }

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Entrada de cache otimizada para RAG"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0
    confidence: float = 0.0
    tokens_saved: int = 0
    processing_time_saved: float = 0.0

class OptimizedRAGCache:
    """
    Cache híbrido otimizado especificamente para sistemas RAG
    
    Estratégia:
    - L1: Memória (respostas recentes, alta frequência)
    - L2: SQLite (persistência local, consultas semânticas)
    - L3: Redis (compartilhamento entre instâncias) - OPCIONAL
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        max_memory_entries: Optional[int] = None,
        enable_redis: Optional[bool] = None,
        redis_url: Optional[str] = None
    ):
        # Carregar configurações do ambiente se não fornecidas
        config = get_cache_config()
        
        self.db_path = Path(db_path or config["db_path"])
        self.max_memory_entries = max_memory_entries or config["max_memory_entries"]
        self.enable_redis = enable_redis if enable_redis is not None else config["enable_redis"]
        self.redis_url = redis_url or config["redis_url"]
        
        # L1: Cache em memória (LRU)
        self.memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.memory_lock = threading.Lock()
        
        # L2: SQLite para persistência
        self._init_sqlite()
        
        # L3: Redis (opcional)
        self.redis_client = None
        if self.enable_redis:
            self._init_redis(self.redis_url)
        
        # Estatísticas otimizadas para RAG
        self.stats = {
            "total_requests": 0,
            "l1_hits": 0,  # Memória
            "l2_hits": 0,  # SQLite
            "l3_hits": 0,  # Redis
            "misses": 0,
            "tokens_saved": 0,
            "processing_time_saved": 0.0,
            "cost_savings": 0.0
        }
        
        logger.info("OptimizedRAGCache inicializado com estratégia híbrida")
    
    def _init_sqlite(self):
        """Inicializa banco SQLite otimizado para RAG"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(
            str(self.db_path), 
            check_same_thread=False,
            timeout=30.0
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        
        # Schema otimizado para RAG
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                response_data TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                tokens_saved INTEGER DEFAULT 0,
                processing_time_saved REAL DEFAULT 0.0,
                cost_savings REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 1,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                expires_at REAL
            )
        """)
        
        # Índices para performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON rag_cache(query_hash)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON rag_cache(last_accessed)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON rag_cache(confidence)")
        
        self.conn.commit()
        logger.info("SQLite cache inicializado com otimizações")
    
    def _init_redis(self, redis_url: str):
        """Inicializa Redis opcional"""
        try:
            import redis
            
            # Usar configurações avançadas se disponíveis
            redis_settings = get_redis_settings()
            
            self.redis_client = redis.from_url(
                redis_url, 
                decode_responses=True,
                max_connections=redis_settings.get("max_connections", 10),
                socket_timeout=redis_settings.get("socket_timeout", 5.0),
                socket_connect_timeout=redis_settings.get("socket_connect_timeout", 5.0),
                retry_on_timeout=redis_settings.get("retry_on_timeout", True)
            )
            
            # Testar conexão
            self.redis_client.ping()
            logger.info(f"Redis conectado para cache L3: {redis_url}")
            
        except ImportError:
            logger.error("Redis não instalado. Execute: pip install redis")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"Redis não disponível: {e}. Usando apenas L1+L2")
            self.redis_client = None
    
    def _get_query_hash(self, query: str) -> str:
        """Gera hash consistente para query"""
        # Normalizar query para melhor cache hit
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    async def get(self, query: str) -> Tuple[Optional[Dict], str, Dict]:
        """
        Busca em cache com estratégia híbrida
        
        Returns:
            (resultado, fonte_cache, metadados)
        """
        self.stats["total_requests"] += 1
        query_hash = self._get_query_hash(query)
        
        # L1: Verificar memória primeiro (mais rápido)
        with self.memory_lock:
            if query_hash in self.memory_cache:
                entry = self.memory_cache[query_hash]
                # Mover para final (LRU)
                self.memory_cache.move_to_end(query_hash)
                entry.access_count += 1
                entry.last_access = time.time()
                
                self.stats["l1_hits"] += 1
                self.stats["tokens_saved"] += entry.tokens_saved
                self.stats["processing_time_saved"] += entry.processing_time_saved
                
                return entry.value, "memory", {
                    "confidence": entry.confidence,
                    "access_count": entry.access_count,
                    "age": time.time() - entry.timestamp
                }
        
        # L2: Verificar SQLite
        sqlite_result = self._get_from_sqlite(query_hash, query)
        if sqlite_result:
            result, metadata = sqlite_result
            
            # Adicionar ao cache L1 para próximas consultas
            self._add_to_memory(query_hash, result, metadata["confidence"])
            
            self.stats["l2_hits"] += 1
            self.stats["tokens_saved"] += metadata.get("tokens_saved", 0)
            
            return result, "sqlite", metadata
        
        # L3: Verificar Redis (se disponível)
        if self.redis_client:
            redis_result = self._get_from_redis(query_hash)
            if redis_result:
                result, metadata = redis_result
                
                # Adicionar aos caches L1 e L2
                self._add_to_memory(query_hash, result, metadata["confidence"])
                self._save_to_sqlite(query_hash, query, result, metadata)
                
                self.stats["l3_hits"] += 1
                return result, "redis", metadata
        
        # Cache miss
        self.stats["misses"] += 1
        return None, "miss", {}
    
    async def set(
        self, 
        query: str, 
        result: Dict, 
        confidence: float = 0.0,
        tokens_saved: int = 0,
        processing_time_saved: float = 0.0,
        cost_savings: float = 0.0,
        ttl_hours: int = 24
    ):
        """Adiciona resultado ao cache híbrido"""
        query_hash = self._get_query_hash(query)
        
        # Calcular expiração
        expires_at = time.time() + (ttl_hours * 3600)
        
        metadata = {
            "confidence": confidence,
            "tokens_saved": tokens_saved,
            "processing_time_saved": processing_time_saved,
            "cost_savings": cost_savings,
            "expires_at": expires_at
        }
        
        # L1: Memória (sempre)
        self._add_to_memory(query_hash, result, confidence, tokens_saved, processing_time_saved)
        
        # L2: SQLite (para persistência)
        self._save_to_sqlite(query_hash, query, result, metadata)
        
        # L3: Redis (se disponível e confiança alta)
        if self.redis_client and confidence > 0.7:
            self._save_to_redis(query_hash, result, metadata, ttl_hours * 3600)
        
        # Atualizar estatísticas
        self.stats["tokens_saved"] += tokens_saved
        self.stats["processing_time_saved"] += processing_time_saved
        self.stats["cost_savings"] += cost_savings
    
    def _add_to_memory(
        self, 
        query_hash: str, 
        result: Dict, 
        confidence: float,
        tokens_saved: int = 0,
        processing_time_saved: float = 0.0
    ):
        """Adiciona ao cache L1 (memória)"""
        with self.memory_lock:
            entry = CacheEntry(
                key=query_hash,
                value=result,
                timestamp=time.time(),
                confidence=confidence,
                tokens_saved=tokens_saved,
                processing_time_saved=processing_time_saved
            )
            
            # Remover se já existe
            if query_hash in self.memory_cache:
                del self.memory_cache[query_hash]
            
            # Adicionar no final
            self.memory_cache[query_hash] = entry
            
            # Limitar tamanho (LRU eviction)
            while len(self.memory_cache) > self.max_memory_entries:
                self.memory_cache.popitem(last=False)
    
    def _get_from_sqlite(self, query_hash: str, query_text: str) -> Optional[Tuple[Dict, Dict]]:
        """Busca no cache L2 (SQLite)"""
        try:
            cursor = self.conn.execute("""
                SELECT response_data, confidence, tokens_saved, processing_time_saved, 
                       cost_savings, access_count, created_at, last_accessed
                FROM rag_cache 
                WHERE query_hash = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (query_hash, time.time()))
            
            row = cursor.fetchone()
            if row:
                # Atualizar estatísticas de acesso
                self.conn.execute("""
                    UPDATE rag_cache 
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE query_hash = ?
                """, (time.time(), query_hash))
                self.conn.commit()
                
                result = json.loads(row[0])
                metadata = {
                    "confidence": row[1],
                    "tokens_saved": row[2],
                    "processing_time_saved": row[3],
                    "cost_savings": row[4],
                    "access_count": row[5] + 1,
                    "age": time.time() - row[6]
                }
                
                return result, metadata
                
        except Exception as e:
            logger.error(f"Erro ao buscar no SQLite: {e}")
        
        return None
    
    def _save_to_sqlite(self, query_hash: str, query_text: str, result: Dict, metadata: Dict):
        """Salva no cache L2 (SQLite)"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO rag_cache 
                (query_hash, query_text, response_data, confidence, tokens_saved, 
                 processing_time_saved, cost_savings, created_at, last_accessed, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_hash,
                query_text,
                json.dumps(result, ensure_ascii=False),
                metadata.get("confidence", 0.0),
                metadata.get("tokens_saved", 0),
                metadata.get("processing_time_saved", 0.0),
                metadata.get("cost_savings", 0.0),
                time.time(),
                time.time(),
                metadata.get("expires_at")
            ))
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Erro ao salvar no SQLite: {e}")
    
    def _get_from_redis(self, query_hash: str) -> Optional[Tuple[Dict, Dict]]:
        """Busca no cache L3 (Redis)"""
        if not self.redis_client:
            return None
            
        try:
            data = self.redis_client.get(f"rag:{query_hash}")
            if data:
                cached = json.loads(data)
                return cached["result"], cached["metadata"]
        except Exception as e:
            logger.error(f"Erro ao buscar no Redis: {e}")
        
        return None
    
    def _save_to_redis(self, query_hash: str, result: Dict, metadata: Dict, ttl: int):
        """Salva no cache L3 (Redis)"""
        if not self.redis_client:
            return
            
        try:
            data = {
                "result": result,
                "metadata": metadata,
                "timestamp": time.time()
            }
            
            self.redis_client.setex(
                f"rag:{query_hash}",
                ttl,
                json.dumps(data, ensure_ascii=False)
            )
            
        except Exception as e:
            logger.error(f"Erro ao salvar no Redis: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas otimizadas para RAG"""
        total_hits = (
            self.stats["l1_hits"] + 
            self.stats["l2_hits"] + 
            self.stats["l3_hits"]
        )
        
        hit_rate = (
            total_hits / self.stats["total_requests"] 
            if self.stats["total_requests"] > 0 else 0
        )
        
        # Estatísticas do SQLite
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM rag_cache")
            sqlite_entries = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT AVG(confidence) FROM rag_cache")
            avg_confidence = cursor.fetchone()[0] or 0.0
        except:
            sqlite_entries = 0
            avg_confidence = 0.0
        
        return {
            **self.stats,
            "total_hits": total_hits,
            "hit_rate": hit_rate,
            "avg_confidence": avg_confidence,
            "cache_sizes": {
                "memory": len(self.memory_cache),
                "sqlite": sqlite_entries,
                "redis": "connected" if self.redis_client else "disabled"
            },
            "efficiency": {
                "tokens_saved_per_request": (
                    self.stats["tokens_saved"] / max(1, self.stats["total_requests"])
                ),
                "time_saved_per_request": (
                    self.stats["processing_time_saved"] / max(1, self.stats["total_requests"])
                ),
                "cost_savings_total": self.stats["cost_savings"]
            }
        }
    
    def clear_all(self):
        """Limpa todos os caches"""
        with self.memory_lock:
            self.memory_cache.clear()
        
        try:
            self.conn.execute("DELETE FROM rag_cache")
            self.conn.commit()
        except Exception as e:
            logger.error(f"Erro ao limpar SQLite: {e}")
        
        if self.redis_client:
            try:
                # Limpar apenas chaves do RAG
                keys = self.redis_client.keys("rag:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Erro ao limpar Redis: {e}")
        
        logger.info("Cache híbrido limpo completamente")
    
    def close(self):
        """Fecha conexões"""
        try:
            self.conn.close()
        except:
            pass
        
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
