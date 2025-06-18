"""
FASE 3 - Cache Warming: Sistema Inteligente de Pre-carregamento
Pre-carrega queries frequentes e padrões para maximizar cache hits
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """Padrão de query identificado"""
    pattern: str
    frequency: int
    last_seen: datetime
    avg_confidence: float
    avg_processing_time: float
    tokens_saved_potential: int
    variations: List[str]
    priority_score: float


@dataclass
class WarmingTask:
    """Tarefa de warming do cache"""
    query: str
    priority: float
    estimated_benefit: float
    pattern_id: str
    created_at: datetime
    attempts: int = 0
    completed: bool = False
    error: Optional[str] = None


class CacheWarmer:
    """Sistema inteligente de cache warming"""
    
    def __init__(self, 
                 cache_instance=None,
                 pipeline_instance=None,
                 patterns_db_path: str = "storage/cache_patterns.db",
                 min_frequency_threshold: int = 3,
                 warming_batch_size: int = 5,
                 max_warming_time_minutes: int = 30):
        
        self.cache = cache_instance
        self.pipeline = pipeline_instance
        self.patterns_db_path = patterns_db_path
        self.min_frequency_threshold = min_frequency_threshold
        self.warming_batch_size = warming_batch_size
        self.max_warming_time_minutes = max_warming_time_minutes
        
        # Analytics
        self.query_history = defaultdict(list)
        self.patterns = {}
        self.warming_queue = []
        self.warming_stats = {
            "total_warmed": 0,
            "successful_warming": 0,
            "failed_warming": 0,
            "time_spent_warming": 0.0,
            "estimated_savings": 0.0
        }
        
        # Setup database
        self._init_patterns_database()
        
        logger.info("CacheWarmer inicializado com análise de padrões inteligente")
    
    def _init_patterns_database(self):
        """Inicializa database de padrões"""
        try:
            Path(self.patterns_db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.patterns_db_path)
            cursor = conn.cursor()
            
            # Tabela de padrões
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT UNIQUE,
                    frequency INTEGER DEFAULT 1,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    avg_confidence REAL DEFAULT 0.0,
                    avg_processing_time REAL DEFAULT 0.0,
                    tokens_saved_potential INTEGER DEFAULT 0,
                    variations TEXT DEFAULT '[]',
                    priority_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabela de histórico de warming
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS warming_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    pattern_id TEXT,
                    priority REAL,
                    estimated_benefit REAL,
                    success BOOLEAN,
                    processing_time REAL,
                    tokens_saved INTEGER,
                    error_message TEXT,
                    warmed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Índices para performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_frequency ON query_patterns(frequency DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_priority ON query_patterns(priority_score DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_warming_success ON warming_history(success, warmed_at)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database de padrões inicializado: {self.patterns_db_path}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar database de padrões: {e}")
    
    async def analyze_query_patterns(self, force_analysis: bool = False) -> Dict[str, Any]:
        """Analisa padrões de queries para identificar candidatos ao warming"""
        
        if not self.cache:
            logger.warning("Cache não disponível para análise de padrões")
            return {}
        
        try:
            # Obter histórico de queries do cache
            cache_stats = self.cache.get_stats()
            
            # Carregar padrões existentes do database
            await self._load_patterns_from_db()
            
            # Identificar novos padrões
            new_patterns = await self._identify_patterns()
            
            # Calcular prioridades
            prioritized_patterns = self._calculate_priorities(new_patterns)
            
            # Salvar padrões atualizados
            await self._save_patterns_to_db(prioritized_patterns)
            
            analysis_result = {
                "total_patterns": len(prioritized_patterns),
                "high_priority_patterns": len([p for p in prioritized_patterns.values() if p.priority_score > 0.7]),
                "warming_candidates": len([p for p in prioritized_patterns.values() 
                                         if p.frequency >= self.min_frequency_threshold]),
                "estimated_potential_savings": sum(p.tokens_saved_potential for p in prioritized_patterns.values()),
                "patterns": {k: asdict(v) for k, v in list(prioritized_patterns.items())[:10]}  # Top 10
            }
            
            logger.info(f"Análise de padrões completa: {analysis_result['warming_candidates']} candidatos identificados")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erro na análise de padrões: {e}")
            return {}
    
    async def _load_patterns_from_db(self):
        """Carrega padrões existentes do database"""
        try:
            conn = sqlite3.connect(self.patterns_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pattern, frequency, last_seen, avg_confidence, 
                       avg_processing_time, tokens_saved_potential, 
                       variations, priority_score
                FROM query_patterns 
                ORDER BY priority_score DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                pattern, freq, last_seen, conf, proc_time, tokens, variations, priority = row
                
                self.patterns[pattern] = QueryPattern(
                    pattern=pattern,
                    frequency=freq,
                    last_seen=datetime.fromisoformat(last_seen),
                    avg_confidence=conf,
                    avg_processing_time=proc_time,
                    tokens_saved_potential=tokens,
                    variations=json.loads(variations),
                    priority_score=priority
                )
            
            logger.debug(f"Carregados {len(self.patterns)} padrões do database")
            
        except Exception as e:
            logger.warning(f"Erro ao carregar padrões: {e}")
    
    async def _identify_patterns(self) -> Dict[str, QueryPattern]:
        """Identifica novos padrões baseado no histórico"""
        patterns = dict(self.patterns)  # Copy existing
        
        # Simular identificação de padrões (em produção, viria do cache real)
        common_patterns = [
            "cache em sistemas rag",
            "implementação de cache",
            "redis vs sqlite",
            "otimização de performance",
            "arquitetura de sistemas",
            "configuração de ambiente"
        ]
        
        for pattern in common_patterns:
            if pattern not in patterns:
                patterns[pattern] = QueryPattern(
                    pattern=pattern,
                    frequency=5,  # Frequência simulada
                    last_seen=datetime.now(),
                    avg_confidence=0.8,
                    avg_processing_time=2.5,
                    tokens_saved_potential=150,
                    variations=[f"{pattern} {suffix}" for suffix in ["tutorial", "guia", "exemplo"]],
                    priority_score=0.0  # Será calculado
                )
        
        return patterns
    
    def _calculate_priorities(self, patterns: Dict[str, QueryPattern]) -> Dict[str, QueryPattern]:
        """Calcula scores de prioridade para warming"""
        
        for pattern_id, pattern in patterns.items():
            # Fatores de prioridade
            frequency_score = min(pattern.frequency / 10.0, 1.0)  # Normalizado
            confidence_score = pattern.avg_confidence
            time_saving_score = min(pattern.avg_processing_time / 5.0, 1.0)  # Queries mais lentas = maior benefício
            recency_score = max(0, 1.0 - (datetime.now() - pattern.last_seen).days / 30.0)  # Últimos 30 dias
            
            # Score composto (pesos ajustáveis)
            priority_score = (
                frequency_score * 0.3 +
                confidence_score * 0.25 +
                time_saving_score * 0.25 +
                recency_score * 0.2
            )
            
            pattern.priority_score = priority_score
            
            # Atualizar potencial de tokens saved
            pattern.tokens_saved_potential = int(
                pattern.frequency * pattern.avg_processing_time * 50  # Estimativa
            )
        
        return patterns
    
    async def _save_patterns_to_db(self, patterns: Dict[str, QueryPattern]):
        """Salva padrões atualizados no database"""
        try:
            conn = sqlite3.connect(self.patterns_db_path)
            cursor = conn.cursor()
            
            for pattern_id, pattern in patterns.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO query_patterns 
                    (pattern, frequency, last_seen, avg_confidence, avg_processing_time,
                     tokens_saved_potential, variations, priority_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern,
                    pattern.frequency,
                    pattern.last_seen.isoformat(),
                    pattern.avg_confidence,
                    pattern.avg_processing_time,
                    pattern.tokens_saved_potential,
                    json.dumps(pattern.variations),
                    pattern.priority_score
                ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Salvos {len(patterns)} padrões no database")
            
        except Exception as e:
            logger.error(f"Erro ao salvar padrões: {e}")
    
    async def create_warming_tasks(self) -> List[WarmingTask]:
        """Cria tarefas de warming baseadas nos padrões"""
        
        await self.analyze_query_patterns()
        
        # Filtrar padrões para warming
        warming_candidates = [
            pattern for pattern in self.patterns.values()
            if pattern.frequency >= self.min_frequency_threshold and pattern.priority_score > 0.5
        ]
        
        # Ordenar por prioridade
        warming_candidates.sort(key=lambda p: p.priority_score, reverse=True)
        
        # Criar tarefas
        tasks = []
        for pattern in warming_candidates[:self.warming_batch_size]:
            
            # Criar queries para warming (pattern + variações principais)
            queries_to_warm = [pattern.pattern] + pattern.variations[:2]
            
            for query in queries_to_warm:
                task = WarmingTask(
                    query=query,
                    priority=pattern.priority_score,
                    estimated_benefit=pattern.tokens_saved_potential,
                    pattern_id=pattern.pattern,
                    created_at=datetime.now()
                )
                tasks.append(task)
        
        self.warming_queue = tasks
        logger.info(f"Criadas {len(tasks)} tarefas de warming")
        
        return tasks
    
    async def execute_warming(self, 
                            max_concurrent: int = 3,
                            timeout_per_query: int = 60) -> Dict[str, Any]:
        """Executa warming das queries prioritárias"""
        
        if not self.pipeline:
            logger.error("Pipeline não disponível para warming")
            return {"error": "Pipeline não configurado"}
        
        if not self.warming_queue:
            await self.create_warming_tasks()
        
        if not self.warming_queue:
            logger.info("Nenhuma tarefa de warming necessária")
            return {"status": "no_tasks"}
        
        start_time = time.time()
        completed_tasks = []
        failed_tasks = []
        
        logger.info(f"Iniciando warming de {len(self.warming_queue)} queries...")
        
        # Processar em lotes
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def warm_single_query(task: WarmingTask) -> Tuple[bool, WarmingTask]:
            async with semaphore:
                try:
                    logger.debug(f"Warming query: {task.query[:50]}...")
                    
                    query_start = time.time()
                    
                    # Executar query para popular cache
                    result = await self.pipeline.query_advanced(
                        task.query,
                        force_improvements=["adaptive"]  # Usar minimal improvements
                    )
                    
                    query_time = time.time() - query_start
                    
                    # Verificar se foi cacheada
                    if result and result.get("confidence", 0) > 0.6:
                        task.completed = True
                        task.attempts += 1
                        
                        # Registrar no histórico
                        await self._record_warming_success(task, query_time, result)
                        
                        self.warming_stats["successful_warming"] += 1
                        self.warming_stats["time_spent_warming"] += query_time
                        
                        logger.debug(f"✅ Query warmed successfully: {task.query[:30]}...")
                        return True, task
                    else:
                        raise Exception("Query não atendeu critérios de cache")
                        
                except Exception as e:
                    task.error = str(e)
                    task.attempts += 1
                    
                    await self._record_warming_failure(task, str(e))
                    
                    self.warming_stats["failed_warming"] += 1
                    logger.warning(f"❌ Warming failed: {task.query[:30]}... - {e}")
                    return False, task
        
        # Executar warming tasks
        warming_tasks = [warm_single_query(task) for task in self.warming_queue]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*warming_tasks, return_exceptions=True),
                timeout=self.max_warming_time_minutes * 60
            )
            
            for success, task in results:
                if success:
                    completed_tasks.append(task)
                else:
                    failed_tasks.append(task)
                    
        except asyncio.TimeoutError:
            logger.warning(f"Warming timeout após {self.max_warming_time_minutes} minutos")
        
        total_time = time.time() - start_time
        
        # Estatísticas finais
        warming_result = {
            "status": "completed",
            "total_tasks": len(self.warming_queue),
            "successful": len(completed_tasks),
            "failed": len(failed_tasks),
            "success_rate": len(completed_tasks) / len(self.warming_queue) if self.warming_queue else 0,
            "total_time": total_time,
            "avg_time_per_query": total_time / len(self.warming_queue) if self.warming_queue else 0,
            "estimated_future_savings": sum(task.estimated_benefit for task in completed_tasks),
            "stats": self.warming_stats
        }
        
        self.warming_stats["total_warmed"] += len(completed_tasks)
        
        logger.info(f"🎯 Warming concluído: {len(completed_tasks)}/{len(self.warming_queue)} sucessos em {total_time:.1f}s")
        
        return warming_result
    
    async def _record_warming_success(self, task: WarmingTask, processing_time: float, result: Dict):
        """Registra sucesso do warming"""
        try:
            conn = sqlite3.connect(self.patterns_db_path)
            cursor = conn.cursor()
            
            tokens_saved = len(result.get("answer", "")) // 4  # Estimativa
            
            cursor.execute('''
                INSERT INTO warming_history 
                (query, pattern_id, priority, estimated_benefit, success, 
                 processing_time, tokens_saved, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.query, task.pattern_id, task.priority, task.estimated_benefit,
                True, processing_time, tokens_saved, None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Erro ao registrar sucesso do warming: {e}")
    
    async def _record_warming_failure(self, task: WarmingTask, error: str):
        """Registra falha do warming"""
        try:
            conn = sqlite3.connect(self.patterns_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO warming_history 
                (query, pattern_id, priority, estimated_benefit, success, 
                 processing_time, tokens_saved, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.query, task.pattern_id, task.priority, task.estimated_benefit,
                False, 0.0, 0, error
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Erro ao registrar falha do warming: {e}")
    
    def get_warming_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de warming"""
        try:
            conn = sqlite3.connect(self.patterns_db_path)
            cursor = conn.cursor()
            
            # Estatísticas gerais
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success THEN tokens_saved ELSE 0 END) as total_tokens_saved,
                    AVG(CASE WHEN success THEN processing_time ELSE NULL END) as avg_processing_time
                FROM warming_history
                WHERE warmed_at >= datetime('now', '-7 days')
            ''')
            
            stats_row = cursor.fetchone()
            
            # Top patterns
            cursor.execute('''
                SELECT pattern, frequency, priority_score, tokens_saved_potential
                FROM query_patterns 
                ORDER BY priority_score DESC 
                LIMIT 5
            ''')
            
            top_patterns = cursor.fetchall()
            conn.close()
            
            if stats_row:
                total, successful, tokens_saved, avg_time = stats_row
                success_rate = (successful / total) if total > 0 else 0
            else:
                total = successful = tokens_saved = avg_time = success_rate = 0
            
            return {
                **self.warming_stats,
                "last_7_days": {
                    "total_attempts": total or 0,
                    "successful": successful or 0,
                    "success_rate": success_rate,
                    "tokens_saved": tokens_saved or 0,
                    "avg_processing_time": avg_time or 0.0
                },
                "top_patterns": [
                    {"pattern": p[0], "frequency": p[1], "priority": p[2], "potential": p[3]}
                    for p in top_patterns
                ],
                "queue_status": {
                    "pending_tasks": len([t for t in self.warming_queue if not t.completed]),
                    "completed_tasks": len([t for t in self.warming_queue if t.completed])
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas de warming: {e}")
            return self.warming_stats
    
    async def schedule_warming(self, 
                              interval_hours: int = 6,
                              max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Agenda warming automático em intervalos"""
        
        logger.info(f"Iniciando warming automático a cada {interval_hours} horas")
        
        iteration = 0
        results = []
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                
                logger.info(f"🔄 Warming automático - Iteração {iteration}")
                
                # Executar warming
                result = await self.execute_warming()
                result["iteration"] = iteration
                result["timestamp"] = datetime.now().isoformat()
                results.append(result)
                
                logger.info(f"Warming {iteration} completo: {result.get('successful', 0)} sucessos")
                
                # Aguardar próximo ciclo
                if max_iterations is None or iteration < max_iterations:
                    await asyncio.sleep(interval_hours * 3600)
            
            return {
                "status": "completed",
                "total_iterations": iteration,
                "results": results,
                "final_stats": self.get_warming_stats()
            }
            
        except Exception as e:
            logger.error(f"Erro no warming automático: {e}")
            return {
                "status": "error",
                "error": str(e),
                "completed_iterations": iteration,
                "results": results
            }
    
    async def cleanup(self):
        """Limpeza de recursos"""
        try:
            # Limpar queue
            self.warming_queue.clear()
            
            # Cleanup de padrões antigos (>90 dias)
            conn = sqlite3.connect(self.patterns_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM query_patterns 
                WHERE last_seen < datetime('now', '-90 days')
                AND frequency < 2
            ''')
            
            cursor.execute('''
                DELETE FROM warming_history 
                WHERE warmed_at < datetime('now', '-30 days')
            ''')
            
            deleted_patterns = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"🧹 Cleanup warming: {deleted_patterns} padrões antigos removidos")
            
        except Exception as e:
            logger.warning(f"Erro no cleanup do warming: {e}")


# Utilitários de análise
def analyze_warming_effectiveness(db_path: str = "storage/cache_patterns.db") -> Dict[str, Any]:
    """Analisa efetividade do sistema de warming"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Análise de efetividade
        cursor.execute('''
            SELECT 
                COUNT(*) as total_warmed,
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                SUM(tokens_saved) as total_tokens_saved,
                AVG(processing_time) as avg_warming_time
            FROM warming_history
        ''')
        
        effectiveness = cursor.fetchone()
        
        # Top patterns por ROI
        cursor.execute('''
            SELECT p.pattern, p.frequency, p.priority_score,
                   COUNT(w.id) as warming_attempts,
                   SUM(CASE WHEN w.success THEN w.tokens_saved ELSE 0 END) as actual_savings
            FROM query_patterns p
            LEFT JOIN warming_history w ON p.pattern = w.pattern_id
            GROUP BY p.pattern
            ORDER BY actual_savings DESC
            LIMIT 10
        ''')
        
        top_roi_patterns = cursor.fetchall()
        conn.close()
        
        return {
            "effectiveness": {
                "total_warmed": effectiveness[0] or 0,
                "success_rate": effectiveness[1] or 0.0,
                "total_tokens_saved": effectiveness[2] or 0,
                "avg_warming_time": effectiveness[3] or 0.0
            },
            "top_roi_patterns": [
                {
                    "pattern": row[0],
                    "frequency": row[1],
                    "priority": row[2],
                    "warming_attempts": row[3],
                    "actual_savings": row[4]
                }
                for row in top_roi_patterns
            ]
        }
        
    except Exception as e:
        logger.error(f"Erro na análise de efetividade: {e}")
        return {}