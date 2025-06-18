"""
FASE 3 - Cache Tuning: Auto-ajuste Inteligente de Parâmetros
Sistema que monitora e ajusta automaticamente configurações de cache para máxima eficiência
"""

import asyncio
import logging
import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from collections import defaultdict, deque
import statistics
from enum import Enum

logger = logging.getLogger(__name__)


class TuningStrategy(Enum):
    AGGRESSIVE = "aggressive"      # Ajustes rápidos e significativos
    CONSERVATIVE = "conservative"  # Ajustes lentos e pequenos
    BALANCED = "balanced"         # Meio termo
    ADAPTIVE = "adaptive"         # Muda estratégia baseado na performance


class TuningAction(Enum):
    INCREASE_TTL = "increase_ttl"
    DECREASE_TTL = "decrease_ttl"
    INCREASE_MEMORY = "increase_memory"
    DECREASE_MEMORY = "decrease_memory"
    ADJUST_EVICTION = "adjust_eviction"
    ENABLE_COMPRESSION = "enable_compression"
    DISABLE_COMPRESSION = "disable_compression"
    REBALANCE_LAYERS = "rebalance_layers"


@dataclass
class TuningRule:
    """Regra de tuning baseada em condições"""
    id: str
    name: str
    condition: str  # Python expression
    action: TuningAction
    target_param: str
    adjustment_value: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    cooldown_minutes: int = 30
    priority: int = 1  # 1=highest, 10=lowest


@dataclass
class TuningEvent:
    """Evento de tuning aplicado"""
    timestamp: datetime
    rule_id: str
    action: TuningAction
    target_param: str
    old_value: float
    new_value: float
    reason: str
    performance_before: Dict[str, float]
    performance_after: Optional[Dict[str, float]] = None
    effectiveness_score: Optional[float] = None


@dataclass
class PerformanceProfile:
    """Perfil de performance para análise"""
    timestamp: datetime
    hit_rate: float
    response_time: float
    memory_usage: float
    throughput: float
    error_rate: float
    cost_per_request: float
    user_satisfaction_score: float  # 0-1


class CacheTuner:
    """Sistema inteligente de auto-tuning para cache"""
    
    def __init__(self,
                 cache_instance=None,
                 analytics_instance=None,
                 tuning_db_path: str = "storage/cache_tuning.db",
                 strategy: TuningStrategy = TuningStrategy.BALANCED,
                 tuning_interval_minutes: int = 15,
                 learning_window_hours: int = 24):
        
        self.cache = cache_instance
        self.analytics = analytics_instance
        self.tuning_db_path = tuning_db_path
        self.strategy = strategy
        self.tuning_interval_minutes = tuning_interval_minutes
        self.learning_window_hours = learning_window_hours
        
        # Estado atual
        self.current_config = {}
        self.performance_history = deque(maxlen=1000)
        self.tuning_history = []
        self.active_rules = {}
        self.rule_cooldowns = {}
        
        # Métricas de aprendizado
        self.baseline_performance = None
        self.best_performance = None
        self.tuning_effectiveness = defaultdict(list)
        
        # Setup database e regras
        self._init_tuning_database()
        self._load_default_rules()
        
        logger.info(f"CacheTuner inicializado com estratégia {strategy.value}")
    
    def _init_tuning_database(self):
        """Inicializa database de tuning"""
        try:
            Path(self.tuning_db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.tuning_db_path)
            cursor = conn.cursor()
            
            # Tabela de regras de tuning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tuning_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    condition_expr TEXT,
                    action TEXT,
                    target_param TEXT,
                    adjustment_value REAL,
                    min_value REAL,
                    max_value REAL,
                    cooldown_minutes INTEGER,
                    priority INTEGER,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabela de eventos de tuning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tuning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rule_id TEXT,
                    action TEXT,
                    target_param TEXT,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    performance_before TEXT,
                    performance_after TEXT,
                    effectiveness_score REAL
                )
            ''')
            
            # Tabela de configurações
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS config_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config_snapshot TEXT,
                    performance_snapshot TEXT,
                    tuning_strategy TEXT
                )
            ''')
            
            # Índices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON tuning_events(timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_rule ON tuning_events(rule_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_config_timestamp ON config_history(timestamp DESC)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database de tuning inicializado: {self.tuning_db_path}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar database de tuning: {e}")
    
    def _load_default_rules(self):
        """Carrega regras padrão de tuning"""
        default_rules = [
            # Regras para Hit Rate
            TuningRule(
                id="low_hit_rate_increase_ttl",
                name="Aumentar TTL quando hit rate baixo",
                condition="hit_rate < 0.4 and response_time < 3.0",
                action=TuningAction.INCREASE_TTL,
                target_param="default_ttl",
                adjustment_value=1.5,  # Multiplier
                min_value=300,   # 5 min
                max_value=86400, # 24 hours
                cooldown_minutes=30,
                priority=1
            ),
            
            TuningRule(
                id="high_memory_decrease_ttl",
                name="Diminuir TTL quando memória alta",
                condition="memory_usage > 0.8 and hit_rate > 0.6",
                action=TuningAction.DECREASE_TTL,
                target_param="default_ttl",
                adjustment_value=0.8,  # Multiplier
                min_value=300,
                max_value=86400,
                cooldown_minutes=45,
                priority=2
            ),
            
            # Regras para Response Time
            TuningRule(
                id="slow_response_increase_memory",
                name="Aumentar cache de memória quando resposta lenta",
                condition="response_time > 2.0 and memory_usage < 0.6",
                action=TuningAction.INCREASE_MEMORY,
                target_param="max_memory_entries",
                adjustment_value=1.3,  # Multiplier
                min_value=100,
                max_value=2000,
                cooldown_minutes=60,
                priority=1
            ),
            
            # Regras para Throughput
            TuningRule(
                id="low_throughput_rebalance",
                name="Rebalancear layers quando throughput baixo",
                condition="throughput < 1.0 and error_rate < 0.05",
                action=TuningAction.REBALANCE_LAYERS,
                target_param="layer_distribution",
                adjustment_value=0.1,  # Adjustment factor
                cooldown_minutes=90,
                priority=3
            ),
            
            # Regras para Error Rate
            TuningRule(
                id="high_error_conservative_config",
                name="Configuração conservativa quando muitos erros",
                condition="error_rate > 0.1",
                action=TuningAction.DECREASE_MEMORY,
                target_param="max_memory_entries",
                adjustment_value=0.7,
                min_value=50,
                max_value=2000,
                cooldown_minutes=120,
                priority=1
            )
        ]
        
        # Salvar regras no database
        try:
            conn = sqlite3.connect(self.tuning_db_path)
            cursor = conn.cursor()
            
            for rule in default_rules:
                cursor.execute('''
                    INSERT OR REPLACE INTO tuning_rules 
                    (id, name, condition_expr, action, target_param, adjustment_value,
                     min_value, max_value, cooldown_minutes, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rule.id, rule.name, rule.condition, rule.action.value,
                    rule.target_param, rule.adjustment_value,
                    rule.min_value, rule.max_value, rule.cooldown_minutes, rule.priority
                ))
                
                self.active_rules[rule.id] = rule
            
            conn.commit()
            conn.close()
            
            logger.info(f"Carregadas {len(default_rules)} regras de tuning padrão")
            
        except Exception as e:
            logger.error(f"Erro ao carregar regras padrão: {e}")
    
    async def start_tuning(self):
        """Inicia processo de tuning automático"""
        logger.info(f"Iniciando auto-tuning a cada {self.tuning_interval_minutes} minutos")
        
        # Capturar baseline inicial
        await self._capture_baseline()
        
        while True:
            try:
                # Coletar métricas atuais
                current_performance = await self._collect_performance_metrics()
                
                if current_performance:
                    # Analisar necessidade de tuning
                    tuning_needed = await self._analyze_tuning_need(current_performance)
                    
                    if tuning_needed:
                        # Aplicar tuning
                        applied_changes = await self._apply_tuning(current_performance)
                        
                        if applied_changes:
                            logger.info(f"🔧 Tuning aplicado: {len(applied_changes)} ajustes")
                            
                            # Aguardar e medir impacto
                            await asyncio.sleep(300)  # 5 minutos
                            await self._measure_tuning_impact(applied_changes)
                    
                    # Salvar histórico
                    self.performance_history.append(current_performance)
                    await self._save_config_snapshot(current_performance)
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.tuning_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Erro no ciclo de tuning: {e}")
                await asyncio.sleep(300)  # Retry em 5 minutos
    
    async def _capture_baseline(self):
        """Captura performance baseline inicial"""
        try:
            baseline = await self._collect_performance_metrics()
            if baseline:
                self.baseline_performance = baseline
                self.best_performance = baseline
                logger.info(f"📊 Baseline capturado: hit_rate={baseline.hit_rate:.1%}, response_time={baseline.response_time:.2f}s")
        except Exception as e:
            logger.warning(f"Erro ao capturar baseline: {e}")
    
    async def _collect_performance_metrics(self) -> Optional[PerformanceProfile]:
        """Coleta métricas atuais de performance"""
        try:
            if not self.cache:
                return None
            
            # Obter estatísticas do cache
            cache_stats = self.cache.get_stats()
            
            # Obter dados do analytics se disponível
            if self.analytics:
                analytics_data = await self.analytics.get_dashboard_data(hours=1)
                current_snapshot = analytics_data.get("current_snapshot", {})
            else:
                current_snapshot = {}
            
            # Construir perfil de performance
            profile = PerformanceProfile(
                timestamp=datetime.now(),
                hit_rate=cache_stats.get('hit_rate', 0.0),
                response_time=current_snapshot.get('response_time_avg', 0.0),
                memory_usage=cache_stats.get('memory_usage_mb', 0.0) / 1024,  # GB
                throughput=current_snapshot.get('throughput_qps', 0.0),
                error_rate=current_snapshot.get('error_rate', 0.0),
                cost_per_request=self._calculate_cost_per_request(cache_stats),
                user_satisfaction_score=self._calculate_satisfaction_score(cache_stats, current_snapshot)
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Erro ao coletar métricas: {e}")
            return None
    
    def _calculate_cost_per_request(self, cache_stats: Dict) -> float:
        """Calcula custo por requisição"""
        total_requests = cache_stats.get('total_requests', 1)
        total_cost = cache_stats.get('total_cost', 0.0)
        return total_cost / max(total_requests, 1)
    
    def _calculate_satisfaction_score(self, cache_stats: Dict, analytics_data: Dict) -> float:
        """Calcula score de satisfação do usuário (0-1)"""
        # Combinar múltiplos fatores
        hit_rate = cache_stats.get('hit_rate', 0.0)
        response_time = analytics_data.get('response_time_avg', 5.0)
        error_rate = analytics_data.get('error_rate', 0.1)
        
        # Normalizar fatores (0-1)
        hit_score = hit_rate
        time_score = max(0, 1.0 - (response_time / 10.0))  # 10s = score 0
        error_score = max(0, 1.0 - (error_rate * 10))      # 10% error = score 0
        
        # Score composto
        satisfaction = (hit_score * 0.4 + time_score * 0.4 + error_score * 0.2)
        return min(1.0, max(0.0, satisfaction))
    
    async def _analyze_tuning_need(self, current_performance: PerformanceProfile) -> bool:
        """Analisa se tuning é necessário"""
        try:
            # Comparar com baseline
            if self.baseline_performance:
                hit_rate_degradation = self.baseline_performance.hit_rate - current_performance.hit_rate
                response_time_increase = current_performance.response_time - self.baseline_performance.response_time
                
                # Critérios para tuning
                significant_degradation = (
                    hit_rate_degradation > 0.1 or           # Hit rate caiu 10%+
                    response_time_increase > 1.0 or         # Response time aumentou 1s+
                    current_performance.error_rate > 0.05 or # Error rate > 5%
                    current_performance.user_satisfaction_score < 0.7  # Satisfação < 70%
                )
                
                if significant_degradation:
                    logger.info(f"📉 Degradação detectada - tuning necessário")
                    return True
            
            # Verificar se há regras aplicáveis
            applicable_rules = await self._find_applicable_rules(current_performance)
            return len(applicable_rules) > 0
            
        except Exception as e:
            logger.error(f"Erro na análise de tuning: {e}")
            return False
    
    async def _find_applicable_rules(self, performance: PerformanceProfile) -> List[TuningRule]:
        """Encontra regras aplicáveis baseadas na performance atual"""
        applicable_rules = []
        
        # Variáveis para avaliação das condições
        context = {
            'hit_rate': performance.hit_rate,
            'response_time': performance.response_time,
            'memory_usage': performance.memory_usage,
            'throughput': performance.throughput,
            'error_rate': performance.error_rate,
            'satisfaction': performance.user_satisfaction_score
        }
        
        for rule_id, rule in self.active_rules.items():
            try:
                # Verificar cooldown
                if rule_id in self.rule_cooldowns:
                    cooldown_end = self.rule_cooldowns[rule_id]
                    if datetime.now() < cooldown_end:
                        continue
                
                # Avaliar condição
                if eval(rule.condition, {"__builtins__": {}}, context):
                    applicable_rules.append(rule)
                    
            except Exception as e:
                logger.warning(f"Erro ao avaliar regra {rule_id}: {e}")
        
        # Ordenar por prioridade
        applicable_rules.sort(key=lambda r: r.priority)
        
        return applicable_rules
    
    async def _apply_tuning(self, performance: PerformanceProfile) -> List[TuningEvent]:
        """Aplica ajustes de tuning"""
        applicable_rules = await self._find_applicable_rules(performance)
        
        if not applicable_rules:
            return []
        
        applied_changes = []
        
        for rule in applicable_rules[:3]:  # Máximo 3 ajustes por ciclo
            try:
                # Obter configuração atual
                current_value = await self._get_current_config_value(rule.target_param)
                
                # Calcular novo valor
                new_value = await self._calculate_new_value(rule, current_value)
                
                # Aplicar mudança
                success = await self._apply_config_change(rule.target_param, new_value)
                
                if success:
                    # Registrar evento
                    event = TuningEvent(
                        timestamp=datetime.now(),
                        rule_id=rule.id,
                        action=rule.action,
                        target_param=rule.target_param,
                        old_value=current_value,
                        new_value=new_value,
                        reason=rule.name,
                        performance_before=asdict(performance)
                    )
                    
                    applied_changes.append(event)
                    
                    # Definir cooldown
                    cooldown_end = datetime.now() + timedelta(minutes=rule.cooldown_minutes)
                    self.rule_cooldowns[rule.id] = cooldown_end
                    
                    # Salvar no database
                    await self._save_tuning_event(event)
                    
                    logger.info(f"🔧 Aplicado: {rule.action.value} {rule.target_param}: {current_value} → {new_value}")
                    
            except Exception as e:
                logger.error(f"Erro ao aplicar regra {rule.id}: {e}")
        
        return applied_changes
    
    async def _get_current_config_value(self, param_name: str) -> float:
        """Obtém valor atual de configuração"""
        # Simulação - em produção viria do cache real
        defaults = {
            "default_ttl": 3600,      # 1 hour
            "max_memory_entries": 200,
            "layer_distribution": 0.7  # 70% L1, 30% L2
        }
        
        return self.current_config.get(param_name, defaults.get(param_name, 1.0))
    
    async def _calculate_new_value(self, rule: TuningRule, current_value: float) -> float:
        """Calcula novo valor baseado na regra"""
        if rule.action in [TuningAction.INCREASE_TTL, TuningAction.INCREASE_MEMORY]:
            new_value = current_value * rule.adjustment_value
        elif rule.action in [TuningAction.DECREASE_TTL, TuningAction.DECREASE_MEMORY]:
            new_value = current_value * rule.adjustment_value
        else:
            new_value = current_value + rule.adjustment_value
        
        # Aplicar limites
        if rule.min_value is not None:
            new_value = max(new_value, rule.min_value)
        if rule.max_value is not None:
            new_value = min(new_value, rule.max_value)
        
        # Aplicar estratégia de tuning
        if self.strategy == TuningStrategy.CONSERVATIVE:
            # Ajustes menores
            adjustment = (new_value - current_value) * 0.5
            new_value = current_value + adjustment
        elif self.strategy == TuningStrategy.AGGRESSIVE:
            # Ajustes maiores
            adjustment = (new_value - current_value) * 1.5
            new_value = current_value + adjustment
        
        return new_value
    
    async def _apply_config_change(self, param_name: str, new_value: float) -> bool:
        """Aplica mudança de configuração"""
        try:
            # Atualizar configuração local
            self.current_config[param_name] = new_value
            
            # Em produção, aplicaria no cache real
            if self.cache and hasattr(self.cache, 'update_config'):
                await self.cache.update_config({param_name: new_value})
            
            logger.debug(f"✅ Configuração aplicada: {param_name} = {new_value}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao aplicar configuração {param_name}: {e}")
            return False
    
    async def _save_tuning_event(self, event: TuningEvent):
        """Salva evento de tuning no database"""
        try:
            conn = sqlite3.connect(self.tuning_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO tuning_events 
                (timestamp, rule_id, action, target_param, old_value, new_value,
                 reason, performance_before, effectiveness_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp.isoformat(),
                event.rule_id,
                event.action.value,
                event.target_param,
                event.old_value,
                event.new_value,
                event.reason,
                json.dumps(event.performance_before),
                event.effectiveness_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Erro ao salvar evento de tuning: {e}")
    
    async def _measure_tuning_impact(self, applied_changes: List[TuningEvent]):
        """Mede impacto dos ajustes aplicados"""
        try:
            # Aguardar estabilização
            await asyncio.sleep(300)  # 5 minutos
            
            # Coletar métricas pós-tuning
            post_performance = await self._collect_performance_metrics()
            
            if not post_performance:
                return
            
            # Calcular efetividade de cada mudança
            for event in applied_changes:
                effectiveness = self._calculate_effectiveness(
                    event.performance_before,
                    asdict(post_performance)
                )
                
                event.performance_after = asdict(post_performance)
                event.effectiveness_score = effectiveness
                
                # Atualizar no database
                await self._update_tuning_event_effectiveness(event)
                
                # Registrar para aprendizado
                self.tuning_effectiveness[event.action].append(effectiveness)
                
                logger.info(f"📊 Efetividade {event.action.value}: {effectiveness:.2f}")
            
            # Atualizar melhor performance se necessário
            if self._is_better_performance(post_performance, self.best_performance):
                self.best_performance = post_performance
                logger.info("🏆 Nova melhor performance registrada!")
            
        except Exception as e:
            logger.error(f"Erro ao medir impacto do tuning: {e}")
    
    def _calculate_effectiveness(self, before: Dict, after: Dict) -> float:
        """Calcula efetividade de um ajuste (-1 a 1)"""
        try:
            # Fatores de melhoria
            hit_rate_improvement = after['hit_rate'] - before['hit_rate']
            response_time_improvement = before['response_time'] - after['response_time']  # Menor é melhor
            error_rate_improvement = before['error_rate'] - after['error_rate']  # Menor é melhor
            satisfaction_improvement = after['user_satisfaction_score'] - before['user_satisfaction_score']
            
            # Score composto (-1 a 1)
            effectiveness = (
                hit_rate_improvement * 0.3 +
                (response_time_improvement / 5.0) * 0.3 +  # Normalizar por 5s
                error_rate_improvement * 0.2 +
                satisfaction_improvement * 0.2
            )
            
            return max(-1.0, min(1.0, effectiveness))
            
        except Exception as e:
            logger.warning(f"Erro ao calcular efetividade: {e}")
            return 0.0
    
    def _is_better_performance(self, current: PerformanceProfile, best: Optional[PerformanceProfile]) -> bool:
        """Verifica se performance atual é melhor que a melhor registrada"""
        if not best:
            return True
        
        # Score composto de performance
        current_score = (
            current.hit_rate * 0.3 +
            (1.0 - min(current.response_time / 10.0, 1.0)) * 0.3 +
            (1.0 - current.error_rate) * 0.2 +
            current.user_satisfaction_score * 0.2
        )
        
        best_score = (
            best.hit_rate * 0.3 +
            (1.0 - min(best.response_time / 10.0, 1.0)) * 0.3 +
            (1.0 - best.error_rate) * 0.2 +
            best.user_satisfaction_score * 0.2
        )
        
        return current_score > best_score
    
    async def _update_tuning_event_effectiveness(self, event: TuningEvent):
        """Atualiza efetividade de evento no database"""
        try:
            conn = sqlite3.connect(self.tuning_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE tuning_events 
                SET performance_after = ?, effectiveness_score = ?
                WHERE timestamp = ? AND rule_id = ?
            ''', (
                json.dumps(event.performance_after),
                event.effectiveness_score,
                event.timestamp.isoformat(),
                event.rule_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Erro ao atualizar efetividade: {e}")
    
    async def _save_config_snapshot(self, performance: PerformanceProfile):
        """Salva snapshot da configuração atual"""
        try:
            conn = sqlite3.connect(self.tuning_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO config_history 
                (timestamp, config_snapshot, performance_snapshot, tuning_strategy)
                VALUES (?, ?, ?, ?)
            ''', (
                performance.timestamp.isoformat(),
                json.dumps(self.current_config),
                json.dumps(asdict(performance)),
                self.strategy.value
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Erro ao salvar snapshot de config: {e}")
    
    async def get_tuning_report(self, hours: int = 24) -> Dict[str, Any]:
        """Gera relatório de tuning"""
        try:
            conn = sqlite3.connect(self.tuning_db_path)
            cursor = conn.cursor()
            
            # Eventos de tuning no período
            cursor.execute('''
                SELECT action, COUNT(*) as count, AVG(effectiveness_score) as avg_effectiveness
                FROM tuning_events 
                WHERE timestamp >= datetime('now', '-{} hours')
                AND effectiveness_score IS NOT NULL
                GROUP BY action
            '''.format(hours))
            
            action_stats = cursor.fetchall()
            
            # Melhores e piores ajustes
            cursor.execute('''
                SELECT rule_id, action, target_param, effectiveness_score, timestamp
                FROM tuning_events 
                WHERE timestamp >= datetime('now', '-{} hours')
                AND effectiveness_score IS NOT NULL
                ORDER BY effectiveness_score DESC
                LIMIT 5
            '''.format(hours))
            
            best_adjustments = cursor.fetchall()
            
            cursor.execute('''
                SELECT rule_id, action, target_param, effectiveness_score, timestamp
                FROM tuning_events 
                WHERE timestamp >= datetime('now', '-{} hours')
                AND effectiveness_score IS NOT NULL
                ORDER BY effectiveness_score ASC
                LIMIT 5
            '''.format(hours))
            
            worst_adjustments = cursor.fetchall()
            
            conn.close()
            
            # Performance atual vs baseline
            current_performance = await self._collect_performance_metrics()
            
            report = {
                "period_hours": hours,
                "strategy": self.strategy.value,
                "baseline_performance": asdict(self.baseline_performance) if self.baseline_performance else {},
                "current_performance": asdict(current_performance) if current_performance else {},
                "best_performance": asdict(self.best_performance) if self.best_performance else {},
                "action_statistics": [
                    {"action": row[0], "count": row[1], "avg_effectiveness": row[2]}
                    for row in action_stats
                ],
                "best_adjustments": [
                    {
                        "rule_id": row[0],
                        "action": row[1], 
                        "target_param": row[2],
                        "effectiveness": row[3],
                        "timestamp": row[4]
                    }
                    for row in best_adjustments
                ],
                "worst_adjustments": [
                    {
                        "rule_id": row[0],
                        "action": row[1],
                        "target_param": row[2], 
                        "effectiveness": row[3],
                        "timestamp": row[4]
                    }
                    for row in worst_adjustments
                ],
                "current_config": self.current_config.copy(),
                "total_events": len(self.tuning_history),
                "learning_insights": self._generate_learning_insights()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de tuning: {e}")
            return {}
    
    def _generate_learning_insights(self) -> List[str]:
        """Gera insights de aprendizado baseado no histórico"""
        insights = []
        
        try:
            # Analisar efetividade por ação
            for action, effectiveness_scores in self.tuning_effectiveness.items():
                if len(effectiveness_scores) >= 3:
                    avg_effectiveness = statistics.mean(effectiveness_scores)
                    
                    if avg_effectiveness > 0.3:
                        insights.append(f"✅ {action.value} tem sido muito efetivo (score médio: {avg_effectiveness:.2f})")
                    elif avg_effectiveness < -0.1:
                        insights.append(f"❌ {action.value} tem sido prejudicial (score médio: {avg_effectiveness:.2f})")
            
            # Insights sobre estratégia
            if self.strategy == TuningStrategy.AGGRESSIVE:
                insights.append("🚀 Estratégia agressiva pode estar causando instabilidade")
            elif self.strategy == TuningStrategy.CONSERVATIVE:
                insights.append("🐌 Estratégia conservativa pode estar perdendo oportunidades de otimização")
            
            # Insights sobre performance
            if self.baseline_performance and self.best_performance:
                hit_rate_gain = self.best_performance.hit_rate - self.baseline_performance.hit_rate
                if hit_rate_gain > 0.1:
                    insights.append(f"📈 Hit rate melhorou {hit_rate_gain:.1%} com tuning automático")
                
                response_time_gain = self.baseline_performance.response_time - self.best_performance.response_time
                if response_time_gain > 0.5:
                    insights.append(f"⚡ Tempo de resposta melhorou {response_time_gain:.1f}s com tuning")
            
        except Exception as e:
            logger.warning(f"Erro ao gerar insights: {e}")
        
        return insights[:5]  # Máximo 5 insights
    
    async def optimize_strategy(self):
        """Otimiza estratégia de tuning baseada no aprendizado"""
        try:
            if len(self.tuning_history) < 10:
                return  # Dados insuficientes
            
            # Calcular efetividade média por estratégia
            recent_effectiveness = [
                event.effectiveness_score for event in self.tuning_history[-20:]
                if event.effectiveness_score is not None
            ]
            
            if not recent_effectiveness:
                return
            
            avg_effectiveness = statistics.mean(recent_effectiveness)
            
            # Mudar estratégia se necessário
            if avg_effectiveness < -0.2 and self.strategy == TuningStrategy.AGGRESSIVE:
                self.strategy = TuningStrategy.CONSERVATIVE
                logger.info("🔄 Mudando para estratégia conservativa devido a baixa efetividade")
            
            elif avg_effectiveness > 0.3 and self.strategy == TuningStrategy.CONSERVATIVE:
                self.strategy = TuningStrategy.BALANCED
                logger.info("🔄 Mudando para estratégia balanceada devido a alta efetividade")
            
            elif self.strategy == TuningStrategy.BALANCED:
                # Adaptativo baseado na variância
                variance = statistics.variance(recent_effectiveness)
                if variance > 0.1:  # Alta variabilidade
                    self.strategy = TuningStrategy.CONSERVATIVE
                    logger.info("🔄 Mudando para estratégia conservativa devido a alta variabilidade")
            
        except Exception as e:
            logger.warning(f"Erro ao otimizar estratégia: {e}")
    
    async def cleanup(self):
        """Limpeza de recursos"""
        try:
            # Limpar cooldowns expirados
            current_time = datetime.now()
            expired_cooldowns = [
                rule_id for rule_id, cooldown_end in self.rule_cooldowns.items()
                if current_time >= cooldown_end
            ]
            
            for rule_id in expired_cooldowns:
                del self.rule_cooldowns[rule_id]
            
            # Cleanup de dados antigos
            conn = sqlite3.connect(self.tuning_db_path)
            cursor = conn.cursor()
            
            # Remover eventos antigos (>90 dias)
            cursor.execute('''
                DELETE FROM tuning_events 
                WHERE timestamp < datetime('now', '-90 days')
            ''')
            
            events_deleted = cursor.rowcount
            
            # Remover configs antigas (>30 dias, mantendo 1 por dia)
            cursor.execute('''
                DELETE FROM config_history 
                WHERE timestamp < datetime('now', '-30 days')
                AND id NOT IN (
                    SELECT MIN(id) 
                    FROM config_history 
                    WHERE timestamp < datetime('now', '-30 days')
                    GROUP BY DATE(timestamp)
                )
            ''')
            
            configs_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if events_deleted > 0 or configs_deleted > 0:
                logger.info(f"🧹 Cleanup tuning: {events_deleted} eventos, {configs_deleted} configs removidos")
            
        except Exception as e:
            logger.warning(f"Erro no cleanup de tuning: {e}")


# Utilitários
def create_custom_tuning_rule(
    rule_id: str,
    name: str,
    condition: str,
    action: TuningAction,
    target_param: str,
    adjustment_value: float,
    **kwargs
) -> TuningRule:
    """Cria regra personalizada de tuning"""
    return TuningRule(
        id=rule_id,
        name=name,
        condition=condition,
        action=action,
        target_param=target_param,
        adjustment_value=adjustment_value,
        **kwargs
    ) 