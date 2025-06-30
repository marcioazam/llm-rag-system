"""
Sistema de Cache Semântico para RAG - Implementação Avançada
Cache baseado em similaridade semântica com adaptação de respostas e warming preditivo
"""

import asyncio
import numpy as np
import logging
import json
import time
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from collections import defaultdict
import threading

# Importações para embeddings
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheEntry:
    """Entrada do cache semântico com metadados avançados"""
    query_text: str
    query_embedding: List[float]
    response_data: Dict[str, Any]
    confidence_score: float
    similarity_threshold: float
    access_count: int
    created_at: float
    last_accessed: float
    adaptation_history: List[Dict] = None
    tokens_saved: int = 0
    processing_time_saved: float = 0.0
    cost_savings: float = 0.0
    source_model: str = ""
    response_quality_score: float = 0.0
    
    def __post_init__(self):
        if self.adaptation_history is None:
            self.adaptation_history = []


@dataclass
class AdaptationRule:
    """Regra de adaptação de resposta cached"""
    rule_id: str
    pattern: str
    adaptation_strategy: str
    confidence_threshold: float
    adaptation_template: str
    usage_count: int = 0
    success_rate: float = 0.0


class SemanticEmbeddingService:
    """Serviço de embeddings para cache semântico"""
    
    def __init__(self, 
                 provider: str = "openai",
                 model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.client = None
        self._init_embedding_client()
    
    def _init_embedding_client(self):
        """Inicializa cliente de embeddings"""
        try:
            if self.provider == "openai" and OPENAI_AVAILABLE:
                self.client = AsyncOpenAI(api_key=self.api_key)
                logger.info(f"Serviço de embeddings inicializado: {self.provider}/{self.model}")
            else:
                logger.warning("Serviço de embeddings não disponível - usando fallback")
        except Exception as e:
            logger.error(f"Erro ao inicializar embeddings: {e}")
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Gera embedding para texto"""
        try:
            if not self.client:
                # Fallback: usar hash como embedding simplificado
                return self._generate_hash_embedding(text)
            
            # Limpar e preparar texto
            clean_text = text.strip().lower()[:8000]  # Limite do modelo
            
            response = await self.client.embeddings.create(
                input=clean_text,
                model=self.model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.warning(f"Erro ao gerar embedding: {e}")
            return self._generate_hash_embedding(text)
    
    def _generate_hash_embedding(self, text: str, dimension: int = 1536) -> List[float]:
        """Fallback: gera embedding baseado em hash"""
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Converter para vetor de floats
        embedding = []
        for i in range(0, min(len(hash_bytes), dimension // 8)):
            chunk = hash_bytes[i:i+8]
            value = int.from_bytes(chunk.ljust(8, b'\x00'), 'big')
            embedding.extend([
                (value >> (8*j)) & 0xFF / 255.0 - 0.5 
                for j in range(8)
            ])
        
        # Preencher até dimensão desejada
        while len(embedding) < dimension:
            embedding.append(0.0)
        
        return embedding[:dimension]


class SemanticCache:
    """
    Cache Semântico Avançado para RAG
    
    Funcionalidades:
    - Busca por similaridade semântica
    - Adaptação automática de respostas
    - Predictive cache warming
    - Análise de qualidade de cache
    """
    
    def __init__(self,
                 db_path: str = "storage/semantic_cache.db",
                 similarity_threshold: float = 0.85,
                 adaptation_threshold: float = 0.75,
                 max_memory_entries: int = 1000,
                 embedding_service: Optional[SemanticEmbeddingService] = None,
                 enable_redis: bool = True,
                 redis_url: str = "redis://localhost:6379"):
        
        self.db_path = Path(db_path)
        self.similarity_threshold = similarity_threshold
        self.adaptation_threshold = adaptation_threshold
        self.max_memory_entries = max_memory_entries
        self.enable_redis = enable_redis
        self.redis_url = redis_url
        
        # Serviços
        self.embedding_service = embedding_service or SemanticEmbeddingService()
        self.redis_client = None
        
        # Cache em memória para embeddings
        self.memory_cache: Dict[str, SemanticCacheEntry] = {}
        self.memory_lock = threading.Lock()
        
        # Regras de adaptação
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        
        # Estatísticas
        self.stats = {
            "total_queries": 0,
            "semantic_hits": 0,
            "exact_hits": 0,
            "adaptations": 0,
            "warming_predictions": 0,
            "average_similarity": 0.0,
            "tokens_saved": 0,
            "processing_time_saved": 0.0,
            "cost_savings": 0.0
        }
        
        # Inicializar componentes
        self._init_database()
        if self.enable_redis:
            asyncio.create_task(self._init_redis())
        self._load_adaptation_rules()
        
        logger.info("SemanticCache inicializado com funcionalidades avançadas")
    
    def _init_database(self):
        """Inicializa database SQLite otimizado para cache semântico"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Tabela principal de cache semântico
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE NOT NULL,
                    query_text TEXT NOT NULL,
                    query_embedding TEXT NOT NULL,  -- JSON array
                    response_data TEXT NOT NULL,    -- JSON
                    confidence_score REAL DEFAULT 0.0,
                    similarity_threshold REAL DEFAULT 0.85,
                    access_count INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    adaptation_history TEXT DEFAULT '[]',
                    tokens_saved INTEGER DEFAULT 0,
                    processing_time_saved REAL DEFAULT 0.0,
                    cost_savings REAL DEFAULT 0.0,
                    source_model TEXT DEFAULT '',
                    response_quality_score REAL DEFAULT 0.0
                )
            ''')
            
            # Tabela de regras de adaptação
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS adaptation_rules (
                    rule_id TEXT PRIMARY KEY,
                    pattern TEXT NOT NULL,
                    adaptation_strategy TEXT NOT NULL,
                    confidence_threshold REAL DEFAULT 0.75,
                    adaptation_template TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Tabela de histórico de warming preditivo
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictive_warming (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_query TEXT NOT NULL,
                    prediction_confidence REAL NOT NULL,
                    base_query_hash TEXT NOT NULL,
                    semantic_distance REAL NOT NULL,
                    predicted_at REAL NOT NULL,
                    warmed BOOLEAN DEFAULT FALSE,
                    warming_success BOOLEAN DEFAULT FALSE,
                    actual_query_matched TEXT,
                    match_confidence REAL DEFAULT 0.0
                )
            ''')
            
            # Índices para performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_hash ON semantic_cache(query_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON semantic_cache(confidence_score DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_count ON semantic_cache(access_count DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON semantic_cache(last_accessed DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictive_warmed ON predictive_warming(warmed, prediction_confidence DESC)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database semântico inicializado: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar database semântico: {e}")
    
    async def _init_redis(self):
        """Inicializa Redis para cache distribuído"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis não disponível - modo apenas local")
            return
        
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Para embeddings binários
                socket_timeout=5.0
            )
            await self.redis_client.ping()
            logger.info("Redis conectado para cache semântico distribuído")
        except Exception as e:
            logger.warning(f"Redis não disponível: {e}")
            self.redis_client = None
    
    def _load_adaptation_rules(self):
        """Carrega regras de adaptação do database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT rule_id, pattern, adaptation_strategy, confidence_threshold,
                       adaptation_template, usage_count, success_rate
                FROM adaptation_rules
            ''')
            
            for row in cursor.fetchall():
                rule_id, pattern, strategy, threshold, template, usage, success = row
                self.adaptation_rules[rule_id] = AdaptationRule(
                    rule_id=rule_id,
                    pattern=pattern,
                    adaptation_strategy=strategy,
                    confidence_threshold=threshold,
                    adaptation_template=template,
                    usage_count=usage,
                    success_rate=success
                )
            
            conn.close()
            logger.info(f"Carregadas {len(self.adaptation_rules)} regras de adaptação")
            
        except Exception as e:
            logger.warning(f"Erro ao carregar regras de adaptação: {e}")
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calcula similaridade cosseno entre embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.warning(f"Erro ao calcular similaridade: {e}")
            return 0.0
    
    async def get_semantic(self, query: str, 
                          similarity_threshold: Optional[float] = None) -> Tuple[Optional[Dict], float, Dict]:
        """
        Busca semântica no cache
        
        Returns:
            (resultado_adaptado, similaridade, metadados)
        """
        self.stats["total_queries"] += 1
        threshold = similarity_threshold or self.similarity_threshold
        
        # Gerar embedding da query
        query_embedding = await self.embedding_service.get_embedding(query)
        if not query_embedding:
            return None, 0.0, {}
        
        # Buscar primeira em memória
        best_match = await self._search_memory_semantic(query_embedding, threshold)
        if best_match:
            entry, similarity = best_match
            self.stats["semantic_hits"] += 1
            
            # Adaptar resposta se necessário
            adapted_response = await self._adapt_response(entry, query, similarity)
            
            return adapted_response, similarity, {
                "source": "memory",
                "access_count": entry.access_count,
                "adaptation_applied": adapted_response != entry.response_data
            }
        
        # Buscar no database
        best_match = await self._search_database_semantic(query_embedding, threshold)
        if best_match:
            entry, similarity = best_match
            self.stats["semantic_hits"] += 1
            
            # Carregar na memória
            with self.memory_lock:
                self.memory_cache[entry.query_text] = entry
                if len(self.memory_cache) > self.max_memory_entries:
                    # Remove entrada mais antiga
                    oldest_key = min(self.memory_cache.keys(), 
                                   key=lambda k: self.memory_cache[k].last_accessed)
                    del self.memory_cache[oldest_key]
            
            # Adaptar resposta
            adapted_response = await self._adapt_response(entry, query, similarity)
            
            return adapted_response, similarity, {
                "source": "database",
                "access_count": entry.access_count,
                "adaptation_applied": adapted_response != entry.response_data
            }
        
        return None, 0.0, {}
    
    async def _search_memory_semantic(self, query_embedding: List[float], 
                                    threshold: float) -> Optional[Tuple[SemanticCacheEntry, float]]:
        """Busca semântica na memória"""
        best_match = None
        best_similarity = 0.0
        
        with self.memory_lock:
            for entry in self.memory_cache.values():
                similarity = self._calculate_similarity(query_embedding, entry.query_embedding)
                
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
                    
                    # Atualizar acesso
                    entry.access_count += 1
                    entry.last_accessed = time.time()
        
        return (best_match, best_similarity) if best_match else None
    
    async def _search_database_semantic(self, query_embedding: List[float], 
                                      threshold: float) -> Optional[Tuple[SemanticCacheEntry, float]]:
        """Busca semântica no database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Buscar todas as entradas (otimização futura: usar índices vetoriais)
            cursor.execute('''
                SELECT query_text, query_embedding, response_data, confidence_score,
                       similarity_threshold, access_count, created_at, last_accessed,
                       adaptation_history, tokens_saved, processing_time_saved,
                       cost_savings, source_model, response_quality_score
                FROM semantic_cache
                ORDER BY access_count DESC, last_accessed DESC
                LIMIT 1000
            ''')
            
            best_match = None
            best_similarity = 0.0
            
            for row in cursor.fetchall():
                stored_embedding = json.loads(row[1])
                similarity = self._calculate_similarity(query_embedding, stored_embedding)
                
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    
                    # Criar entrada
                    best_match = SemanticCacheEntry(
                        query_text=row[0],
                        query_embedding=stored_embedding,
                        response_data=json.loads(row[2]),
                        confidence_score=row[3],
                        similarity_threshold=row[4],
                        access_count=row[5] + 1,  # Incrementar
                        created_at=row[6],
                        last_accessed=time.time(),  # Atualizar
                        adaptation_history=json.loads(row[8]),
                        tokens_saved=row[9],
                        processing_time_saved=row[10],
                        cost_savings=row[11],
                        source_model=row[12],
                        response_quality_score=row[13]
                    )
            
            conn.close()
            
            # Atualizar acesso no database se encontrou match
            if best_match:
                await self._update_access_stats(best_match)
            
            return (best_match, best_similarity) if best_match else None
            
        except Exception as e:
            logger.error(f"Erro na busca semântica database: {e}")
            return None
    
    async def _adapt_response(self, entry: SemanticCacheEntry, 
                            new_query: str, similarity: float) -> Dict[str, Any]:
        """Adapta resposta cached para nova query similar"""
        
        # Se similaridade é muito alta, retorna resposta original
        if similarity >= 0.95:
            return entry.response_data
        
        # Verificar se deve adaptar baseado no threshold
        if similarity < self.adaptation_threshold:
            return entry.response_data
        
        try:
            # Aplicar regras de adaptação
            for rule in self.adaptation_rules.values():
                if similarity >= rule.confidence_threshold:
                    adapted = await self._apply_adaptation_rule(
                        rule, entry.response_data, new_query, entry.query_text
                    )
                    
                    if adapted:
                        # Registrar adaptação
                        adaptation_record = {
                            "rule_id": rule.rule_id,
                            "original_query": entry.query_text,
                            "new_query": new_query,
                            "similarity": similarity,
                            "adapted_at": time.time()
                        }
                        
                        entry.adaptation_history.append(adaptation_record)
                        self.stats["adaptations"] += 1
                        
                        return adapted
            
            # Se não há regras aplicáveis, retorna original
            return entry.response_data
            
        except Exception as e:
            logger.warning(f"Erro na adaptação de resposta: {e}")
            return entry.response_data
    
    async def _apply_adaptation_rule(self, rule: AdaptationRule, 
                                   original_response: Dict, new_query: str, 
                                   original_query: str) -> Optional[Dict]:
        """Aplica regra específica de adaptação"""
        try:
            # Estratégias básicas de adaptação
            if rule.adaptation_strategy == "template_substitution":
                # Substituir partes da resposta usando template
                adapted = dict(original_response)
                
                # Aplicar template (implementação simplificada)
                if "answer" in adapted:
                    adapted["answer"] = rule.adaptation_template.format(
                        original_answer=adapted["answer"],
                        new_query=new_query,
                        original_query=original_query
                    )
                
                # Adicionar nota de adaptação
                adapted["_adapted"] = True
                adapted["_adaptation_rule"] = rule.rule_id
                
                return adapted
            
            elif rule.adaptation_strategy == "contextual_enhancement":
                # Enriquecer resposta com contexto da nova query
                adapted = dict(original_response)
                
                if "answer" in adapted:
                    enhanced_answer = f"{adapted['answer']}\n\n[Contexto adaptado para: {new_query}]"
                    adapted["answer"] = enhanced_answer
                
                adapted["_adapted"] = True
                return adapted
            
            # Adicionar mais estratégias conforme necessário
            
        except Exception as e:
            logger.warning(f"Erro ao aplicar regra de adaptação {rule.rule_id}: {e}")
        
        return None
    
    async def set_semantic(self, query: str, response: Dict[str, Any],
                         confidence_score: float = 0.0,
                         tokens_saved: int = 0,
                         processing_time_saved: float = 0.0,
                         cost_savings: float = 0.0,
                         source_model: str = "",
                         response_quality_score: float = 0.0):
        """Salva no cache semântico"""
        
        try:
            # Gerar embedding
            query_embedding = await self.embedding_service.get_embedding(query)
            if not query_embedding:
                logger.warning("Não foi possível gerar embedding - entrada não salva")
                return
            
            # Criar entrada
            entry = SemanticCacheEntry(
                query_text=query,
                query_embedding=query_embedding,
                response_data=response,
                confidence_score=confidence_score,
                similarity_threshold=self.similarity_threshold,
                access_count=1,
                created_at=time.time(),
                last_accessed=time.time(),
                tokens_saved=tokens_saved,
                processing_time_saved=processing_time_saved,
                cost_savings=cost_savings,
                source_model=source_model,
                response_quality_score=response_quality_score
            )
            
            # Salvar na memória
            with self.memory_lock:
                self.memory_cache[query] = entry
                
                # Limpar se necessário
                if len(self.memory_cache) > self.max_memory_entries:
                    oldest_key = min(self.memory_cache.keys(),
                                   key=lambda k: self.memory_cache[k].last_accessed)
                    del self.memory_cache[oldest_key]
            
            # Salvar no database
            await self._save_to_database(entry)
            
            # Gerar predições para warming
            await self._generate_warming_predictions(entry)
            
            logger.debug(f"Entrada salva no cache semântico: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Erro ao salvar no cache semântico: {e}")
    
    async def _save_to_database(self, entry: SemanticCacheEntry):
        """Salva entrada no database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            query_hash = hashlib.sha256(entry.query_text.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO semantic_cache 
                (query_hash, query_text, query_embedding, response_data,
                 confidence_score, similarity_threshold, access_count, 
                 created_at, last_accessed, adaptation_history,
                 tokens_saved, processing_time_saved, cost_savings,
                 source_model, response_quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                query_hash,
                entry.query_text,
                json.dumps(entry.query_embedding),
                json.dumps(entry.response_data, default=str),
                entry.confidence_score,
                entry.similarity_threshold,
                entry.access_count,
                entry.created_at,
                entry.last_accessed,
                json.dumps(entry.adaptation_history),
                entry.tokens_saved,
                entry.processing_time_saved,
                entry.cost_savings,
                entry.source_model,
                entry.response_quality_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro ao salvar no database: {e}")
    
    async def _update_access_stats(self, entry: SemanticCacheEntry):
        """Atualiza estatísticas de acesso no database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            query_hash = hashlib.sha256(entry.query_text.encode()).hexdigest()
            
            cursor.execute('''
                UPDATE semantic_cache 
                SET access_count = ?, last_accessed = ?
                WHERE query_hash = ?
            ''', (entry.access_count, entry.last_accessed, query_hash))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Erro ao atualizar stats de acesso: {e}")
    
    async def _generate_warming_predictions(self, entry: SemanticCacheEntry):
        """Gera predições para warming preditivo baseado na nova entrada"""
        try:
            # Implementação simplificada de predição
            # Em produção, usaria ML mais sofisticado
            
            base_query = entry.query_text.lower()
            predictions = []
            
            # Gerar variações sintáticas comuns
            variations = [
                f"Como {base_query}",
                f"Qual a melhor forma de {base_query}",
                f"Exemplo de {base_query}",
                f"Tutorial sobre {base_query}",
                f"{base_query} em Python",
                f"{base_query} passo a passo"
            ]
            
            # Calcular confiança baseada em padrões históricos
            for variation in variations:
                if len(variation) > 10 and variation != base_query:
                    prediction_confidence = 0.7  # Valor base
                    
                    # Ajustar baseado na qualidade da resposta original
                    prediction_confidence *= entry.response_quality_score
                    
                    if prediction_confidence > 0.5:
                        predictions.append({
                            "predicted_query": variation,
                            "prediction_confidence": prediction_confidence,
                            "base_query_hash": hashlib.sha256(base_query.encode()).hexdigest(),
                            "semantic_distance": 0.8,  # Estimativa
                            "predicted_at": time.time()
                        })
            
            # Salvar predições no database
            if predictions:
                await self._save_warming_predictions(predictions)
                self.stats["warming_predictions"] += len(predictions)
            
        except Exception as e:
            logger.warning(f"Erro ao gerar predições de warming: {e}")
    
    async def _save_warming_predictions(self, predictions: List[Dict]):
        """Salva predições de warming no database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            for pred in predictions:
                cursor.execute('''
                    INSERT OR IGNORE INTO predictive_warming
                    (predicted_query, prediction_confidence, base_query_hash,
                     semantic_distance, predicted_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    pred["predicted_query"],
                    pred["prediction_confidence"],
                    pred["base_query_hash"],
                    pred["semantic_distance"],
                    pred["predicted_at"]
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erro ao salvar predições: {e}")
    
    async def execute_predictive_warming(self, 
                                       pipeline_instance=None,
                                       max_predictions: int = 10,
                                       min_confidence: float = 0.6) -> Dict[str, Any]:
        """Executa warming preditivo baseado nas predições"""
        
        if not pipeline_instance:
            logger.warning("Pipeline não fornecido para warming preditivo")
            return {"warmed": 0, "errors": 0}
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Buscar predições não processadas com alta confiança
            cursor.execute('''
                SELECT id, predicted_query, prediction_confidence
                FROM predictive_warming
                WHERE warmed = FALSE 
                  AND prediction_confidence >= ?
                ORDER BY prediction_confidence DESC
                LIMIT ?
            ''', (min_confidence, max_predictions))
            
            predictions = cursor.fetchall()
            conn.close()
            
            warming_results = {
                "warmed": 0,
                "errors": 0,
                "total_predictions": len(predictions),
                "results": []
            }
            
            # Processar cada predição
            for pred_id, predicted_query, confidence in predictions:
                try:
                    # Executar query no pipeline
                    start_time = time.time()
                    result = await pipeline_instance.query(predicted_query)
                    processing_time = time.time() - start_time
                    
                    # Salvar no cache semântico
                    await self.set_semantic(
                        query=predicted_query,
                        response=result,
                        confidence_score=confidence,
                        processing_time_saved=processing_time,
                        source_model="predictive_warming"
                    )
                    
                    # Marcar como processado
                    await self._mark_prediction_warmed(pred_id, True, predicted_query, confidence)
                    
                    warming_results["warmed"] += 1
                    warming_results["results"].append({
                        "query": predicted_query,
                        "confidence": confidence,
                        "success": True,
                        "processing_time": processing_time
                    })
                    
                    logger.debug(f"Warming preditivo executado: {predicted_query[:50]}...")
                    
                except Exception as e:
                    logger.warning(f"Erro no warming de '{predicted_query}': {e}")
                    await self._mark_prediction_warmed(pred_id, False, None, 0.0)
                    warming_results["errors"] += 1
                    warming_results["results"].append({
                        "query": predicted_query,
                        "confidence": confidence,
                        "success": False,
                        "error": str(e)
                    })
            
            logger.info(f"Warming preditivo concluído: {warming_results['warmed']} queries processadas")
            return warming_results
            
        except Exception as e:
            logger.error(f"Erro no warming preditivo: {e}")
            return {"warmed": 0, "errors": 1, "error": str(e)}
    
    async def _mark_prediction_warmed(self, pred_id: int, success: bool, 
                                    actual_query: Optional[str], match_confidence: float):
        """Marca predição como processada"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE predictive_warming
                SET warmed = TRUE, warming_success = ?, 
                    actual_query_matched = ?, match_confidence = ?
                WHERE id = ?
            ''', (success, actual_query, match_confidence, pred_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Erro ao marcar predição: {e}")
    
    def add_adaptation_rule(self, rule: AdaptationRule):
        """Adiciona nova regra de adaptação"""
        try:
            self.adaptation_rules[rule.rule_id] = rule
            
            # Salvar no database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO adaptation_rules
                (rule_id, pattern, adaptation_strategy, confidence_threshold,
                 adaptation_template, usage_count, success_rate, 
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id, rule.pattern, rule.adaptation_strategy,
                rule.confidence_threshold, rule.adaptation_template,
                rule.usage_count, rule.success_rate,
                time.time(), time.time()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Regra de adaptação adicionada: {rule.rule_id}")
            
        except Exception as e:
            logger.error(f"Erro ao adicionar regra de adaptação: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache semântico"""
        
        # Calcular hit rate semântico
        total_queries = self.stats["total_queries"]
        semantic_hit_rate = (self.stats["semantic_hits"] / total_queries) if total_queries > 0 else 0.0
        
        # Estatísticas da memória
        memory_stats = {
            "entries": len(self.memory_cache),
            "max_entries": self.max_memory_entries,
            "utilization": len(self.memory_cache) / self.max_memory_entries
        }
        
        # Estatísticas de adaptação
        adaptation_rate = (self.stats["adaptations"] / total_queries) if total_queries > 0 else 0.0
        
        return {
            "semantic_cache_stats": dict(self.stats),
            "semantic_hit_rate": semantic_hit_rate,
            "adaptation_rate": adaptation_rate,
            "memory_cache": memory_stats,
            "adaptation_rules": len(self.adaptation_rules),
            "similarity_threshold": self.similarity_threshold,
            "adaptation_threshold": self.adaptation_threshold,
            "embedding_service": {
                "provider": self.embedding_service.provider,
                "model": self.embedding_service.model
            }
        }
    
    async def analyze_cache_effectiveness(self) -> Dict[str, Any]:
        """Analisa efetividade do cache semântico"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Estatísticas gerais
            cursor.execute('SELECT COUNT(*) FROM semantic_cache')
            total_entries = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(access_count) FROM semantic_cache')
            avg_access_count = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT AVG(confidence_score) FROM semantic_cache')
            avg_confidence = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(tokens_saved) FROM semantic_cache')
            total_tokens_saved = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(cost_savings) FROM semantic_cache')
            total_cost_savings = cursor.fetchone()[0] or 0
            
            # Estatísticas de warming preditivo
            cursor.execute('SELECT COUNT(*) FROM predictive_warming WHERE warmed = TRUE')
            total_warmed = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM predictive_warming WHERE warming_success = TRUE')
            successful_warming = cursor.fetchone()[0]
            
            warming_success_rate = (successful_warming / total_warmed) if total_warmed > 0 else 0
            
            # Top queries por acesso
            cursor.execute('''
                SELECT query_text, access_count, confidence_score, tokens_saved
                FROM semantic_cache 
                ORDER BY access_count DESC 
                LIMIT 10
            ''')
            top_queries = cursor.fetchall()
            
            conn.close()
            
            return {
                "cache_size": total_entries,
                "average_access_count": round(avg_access_count, 2),
                "average_confidence": round(avg_confidence, 3),
                "total_tokens_saved": total_tokens_saved,
                "total_cost_savings": round(total_cost_savings, 2),
                "predictive_warming": {
                    "total_warmed": total_warmed,
                    "successful_warming": successful_warming,
                    "success_rate": round(warming_success_rate, 3)
                },
                "top_queries": [
                    {
                        "query": q[0][:100] + "..." if len(q[0]) > 100 else q[0],
                        "access_count": q[1],
                        "confidence": q[2],
                        "tokens_saved": q[3]
                    }
                    for q in top_queries
                ],
                "effectiveness_score": round(
                    (semantic_hit_rate * 0.4 + 
                     avg_confidence * 0.3 + 
                     warming_success_rate * 0.3), 3
                ) if 'semantic_hit_rate' in locals() else 0.0
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de efetividade: {e}")
            return {}
    
    async def cleanup_expired_entries(self, max_age_days: int = 30) -> int:
        """Remove entradas antigas do cache"""
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Remover entradas antigas com baixo acesso
            cursor.execute('''
                DELETE FROM semantic_cache 
                WHERE created_at < ? AND access_count < 3
            ''', (cutoff_time,))
            
            removed_count = cursor.rowcount
            
            # Cleanup de predições antigas
            cursor.execute('''
                DELETE FROM predictive_warming 
                WHERE predicted_at < ? AND warmed = TRUE
            ''', (cutoff_time,))
            
            conn.commit()
            conn.close()
            
            # Limpar cache de memória de entradas antigas
            with self.memory_lock:
                old_keys = [
                    key for key, entry in self.memory_cache.items()
                    if entry.created_at < cutoff_time and entry.access_count < 3
                ]
                for key in old_keys:
                    del self.memory_cache[key]
            
            logger.info(f"Cleanup do cache: {removed_count} entradas removidas")
            return removed_count
            
        except Exception as e:
            logger.error(f"Erro no cleanup: {e}")
            return 0
    
    async def close(self):
        """Fecha conexões e limpa recursos"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("SemanticCache fechado")


# Regras de adaptação padrão
DEFAULT_ADAPTATION_RULES = [
    AdaptationRule(
        rule_id="generic_contextual",
        pattern=".*",
        adaptation_strategy="contextual_enhancement",
        confidence_threshold=0.75,
        adaptation_template="[Resposta adaptada para contexto similar]"
    ),
    AdaptationRule(
        rule_id="code_examples",
        pattern=".*código|.*exemplo|.*implementar.*",
        adaptation_strategy="template_substitution",
        confidence_threshold=0.80,
        adaptation_template="Baseado na consulta '{new_query}', aqui está uma adaptação: {original_answer}"
    ),
    AdaptationRule(
        rule_id="tutorial_requests",
        pattern=".*como.*|.*tutorial.*|.*passo.*",
        adaptation_strategy="contextual_enhancement",
        confidence_threshold=0.70,
        adaptation_template="Adaptado para: {new_query}"
    )
]


def create_semantic_cache(config: Optional[Dict] = None) -> SemanticCache:
    """Factory function para criar instância do cache semântico"""
    
    if config is None:
        config = {}
    
    # Configurações padrão
    default_config = {
        "db_path": "storage/semantic_cache.db",
        "similarity_threshold": 0.85,
        "adaptation_threshold": 0.75,
        "max_memory_entries": 1000,
        "enable_redis": True,
        "redis_url": "redis://localhost:6379",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small"
    }
    
    # Merge configurações
    final_config = {**default_config, **config}
    
    # Criar serviço de embeddings
    embedding_service = SemanticEmbeddingService(
        provider=final_config["embedding_provider"],
        model=final_config["embedding_model"]
    )
    
    # Criar cache
    cache = SemanticCache(
        db_path=final_config["db_path"],
        similarity_threshold=final_config["similarity_threshold"],
        adaptation_threshold=final_config["adaptation_threshold"],
        max_memory_entries=final_config["max_memory_entries"],
        embedding_service=embedding_service,
        enable_redis=final_config["enable_redis"],
        redis_url=final_config["redis_url"]
    )
    
    # Adicionar regras padrão
    for rule in DEFAULT_ADAPTATION_RULES:
        cache.add_adaptation_rule(rule)
    
    return cache