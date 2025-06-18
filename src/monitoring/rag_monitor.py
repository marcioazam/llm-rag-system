"""
Sistema de Monitoring e Otimização Contínua para RAG
Implementa métricas específicas e adaptive routing
Baseado em: https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import json
from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Métricas detalhadas de uma query"""
    query_id: str
    query_text: str
    timestamp: float
    
    # Latências por componente
    embedding_latency: float = 0.0
    retrieval_latency: float = 0.0
    reranking_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    
    # Performance metrics
    cache_hit: bool = False
    cache_type: Optional[str] = None
    documents_retrieved: int = 0
    documents_after_rerank: int = 0
    
    # Resource usage
    memory_used_mb: float = 0.0
    cpu_percent: float = 0.0
    
    # Model metrics
    model_used: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    estimated_cost: float = 0.0
    
    # Quality metrics
    user_feedback: Optional[float] = None  # 1-5 rating
    relevance_score: float = 0.0
    
    # Routing info
    routing_strategy: str = "auto"
    routing_reason: str = ""

@dataclass
class ComponentMetrics:
    """Métricas agregadas por componente"""
    name: str
    
    # Latency tracking
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Error tracking
    error_count: int = 0
    success_count: int = 0
    
    # Resource usage
    memory_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_latency(self, latency: float):
        self.latencies.append(latency)
    
    def get_avg_latency(self) -> float:
        return np.mean(self.latencies) if self.latencies else 0.0
    
    def get_p95_latency(self) -> float:
        return np.percentile(self.latencies, 95) if self.latencies else 0.0

class PrometheusMetrics:
    """Métricas Prometheus para monitoring"""
    
    def __init__(self):
        # Counters
        self.query_counter = Counter(
            'rag_queries_total',
            'Total number of RAG queries',
            ['routing_strategy', 'cache_hit']
        )
        
        self.error_counter = Counter(
            'rag_errors_total',
            'Total number of errors',
            ['component', 'error_type']
        )
        
        # Histograms
        self.latency_histogram = Histogram(
            'rag_latency_seconds',
            'Query latency by component',
            ['component'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Gauges
        self.cache_hit_rate = Gauge(
            'rag_cache_hit_rate',
            'Cache hit rate',
            ['cache_type']
        )
        
        self.memory_usage = Gauge(
            'rag_memory_usage_mb',
            'Memory usage in MB',
            ['component']
        )
        
        # Summary
        self.model_cost = Summary(
            'rag_model_cost_dollars',
            'Estimated model cost',
            ['model_name']
        )

class AdaptiveRAGRouter:
    """
    Router adaptativo que escolhe estratégia baseado em métricas
    Implementa routing inteligente para otimizar performance
    """
    
    def __init__(self):
        # Histórico de performance por tipo de query
        self.performance_history: Dict[str, List[QueryMetrics]] = defaultdict(list)
        
        # Thresholds para decisões
        self.thresholds = {
            'simple_query_threshold': 0.5,  # segundos
            'complex_query_threshold': 2.0,  # segundos
            'cache_benefit_threshold': 0.7,  # 70% hit rate
            'graph_complexity_threshold': 100  # número de entities
        }
        
        # Estratégias disponíveis
        self.strategies = {
            'vector_only': {
                'description': 'Apenas busca vetorial para queries simples',
                'max_latency': 0.5,
                'suitable_for': ['simple', 'factual', 'direct']
            },
            'hybrid_search': {
                'description': 'Busca híbrida (dense + sparse)',
                'max_latency': 1.0,
                'suitable_for': ['technical', 'mixed', 'keyword_heavy']
            },
            'graph_traversal': {
                'description': 'Graph traversal para queries complexas',
                'max_latency': 3.0,
                'suitable_for': ['complex', 'multi_hop', 'relationship']
            },
            'cached_response': {
                'description': 'Resposta do cache quando disponível',
                'max_latency': 0.1,
                'suitable_for': ['repeated', 'common', 'factual']
            }
        }
    
    def analyze_query_complexity(self, query: str) -> Tuple[str, str]:
        """
        Analisa complexidade da query e retorna estratégia recomendada
        
        Returns:
            Tuple de (strategy, reason)
        """
        query_lower = query.lower()
        
        # Heurísticas para classificação
        complexity_score = 0
        
        # Comprimento da query
        if len(query.split()) > 10:
            complexity_score += 2
        
        # Palavras indicativas de complexidade
        complex_keywords = ['relationship', 'compare', 'analyze', 'explain how', 'why']
        if any(keyword in query_lower for keyword in complex_keywords):
            complexity_score += 3
        
        # Palavras indicativas de simplicidade
        simple_keywords = ['what is', 'define', 'list', 'show']
        if any(keyword in query_lower for keyword in simple_keywords):
            complexity_score -= 2
        
        # Keywords técnicas (beneficiam de hybrid search)
        technical_keywords = ['function', 'class', 'method', 'error', 'bug']
        has_technical = any(keyword in query_lower for keyword in technical_keywords)
        
        # Decidir estratégia
        if complexity_score <= 0:
            return 'vector_only', 'Query simples identificada'
        elif complexity_score >= 4:
            return 'graph_traversal', 'Query complexa requer graph traversal'
        elif has_technical:
            return 'hybrid_search', 'Query técnica beneficia de busca híbrida'
        else:
            return 'hybrid_search', 'Complexidade média - usando híbrida'
    
    def get_adaptive_strategy(
        self,
        query: str,
        recent_metrics: List[QueryMetrics]
    ) -> Tuple[str, str]:
        """
        Escolhe estratégia adaptativa baseada em métricas recentes
        """
        # Analisar performance recente
        if recent_metrics:
            avg_latency = np.mean([m.total_latency for m in recent_metrics[-10:]])
            cache_hits = sum(1 for m in recent_metrics[-20:] if m.cache_hit)
            cache_hit_rate = cache_hits / min(20, len(recent_metrics))
            
            # Se cache está performando bem, priorizar
            if cache_hit_rate > self.thresholds['cache_benefit_threshold']:
                return 'cached_response', f'Alto cache hit rate: {cache_hit_rate:.1%}'
            
            # Se latência está alta, simplificar estratégia
            if avg_latency > self.thresholds['complex_query_threshold']:
                return 'vector_only', f'Reduzindo complexidade - latência média: {avg_latency:.2f}s'
        
        # Fallback para análise de query
        return self.analyze_query_complexity(query)

class RAGMonitor:
    """
    Sistema principal de monitoring e otimização para RAG
    Coleta métricas, analisa performance e otimiza routing
    """
    
    def __init__(self, config_path: str = "config/monitoring_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Componentes
        self.prometheus_metrics = PrometheusMetrics()
        self.adaptive_router = AdaptiveRAGRouter()
        
        # Storage de métricas
        self.query_metrics: deque = deque(maxlen=10000)
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        
        # Inicializar componentes padrão
        for component in ['embedding', 'retrieval', 'reranking', 'generation', 'cache']:
            self.component_metrics[component] = ComponentMetrics(component)
        
        # Background tasks
        self.monitoring_task = None
        self.optimization_task = None
        
        logger.info("RAG Monitor inicializado")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração de monitoring"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Configuração padrão"""
        return {
            'monitoring': {
                'sample_interval': 60,  # segundos
                'metrics_retention': 7,  # dias
                'alert_thresholds': {
                    'latency_p95': 5.0,
                    'error_rate': 0.05,
                    'memory_usage_mb': 8192
                }
            },
            'optimization': {
                'enable_adaptive_routing': True,
                'optimization_interval': 300,  # 5 minutos
                'min_samples_for_optimization': 100
            }
        }
    
    async def start_monitoring(self):
        """Inicia tasks de monitoring em background"""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Monitoring iniciado")
    
    async def _monitoring_loop(self):
        """Loop principal de monitoring"""
        while True:
            try:
                # Coletar métricas do sistema
                await self._collect_system_metrics()
                
                # Verificar alertas
                await self._check_alerts()
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.config['monitoring']['sample_interval'])
                
            except Exception as e:
                logger.error(f"Erro no monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_loop(self):
        """Loop de otimização adaptativa"""
        while True:
            try:
                if self.config['optimization']['enable_adaptive_routing']:
                    await self._optimize_routing()
                
                await asyncio.sleep(self.config['optimization']['optimization_interval'])
                
            except Exception as e:
                logger.error(f"Erro no optimization loop: {e}")
                await asyncio.sleep(300)
    
    async def record_query(self, metrics: QueryMetrics):
        """Registra métricas de uma query"""
        # Adicionar ao histórico
        self.query_metrics.append(metrics)
        
        # Atualizar Prometheus
        self.prometheus_metrics.query_counter.labels(
            routing_strategy=metrics.routing_strategy,
            cache_hit=str(metrics.cache_hit)
        ).inc()
        
        # Atualizar latências por componente
        self.prometheus_metrics.latency_histogram.labels('embedding').observe(metrics.embedding_latency)
        self.prometheus_metrics.latency_histogram.labels('retrieval').observe(metrics.retrieval_latency)
        self.prometheus_metrics.latency_histogram.labels('reranking').observe(metrics.reranking_latency)
        self.prometheus_metrics.latency_histogram.labels('generation').observe(metrics.generation_latency)
        self.prometheus_metrics.latency_histogram.labels('total').observe(metrics.total_latency)
        
        # Atualizar custo
        if metrics.estimated_cost > 0:
            self.prometheus_metrics.model_cost.labels(
                model_name=metrics.model_used
            ).observe(metrics.estimated_cost)
        
        # Atualizar métricas de componente
        self.component_metrics['embedding'].add_latency(metrics.embedding_latency)
        self.component_metrics['retrieval'].add_latency(metrics.retrieval_latency)
        self.component_metrics['reranking'].add_latency(metrics.reranking_latency)
        self.component_metrics['generation'].add_latency(metrics.generation_latency)
        
        # Log se latência alta
        if metrics.total_latency > self.config['monitoring']['alert_thresholds']['latency_p95']:
            logger.warning(f"Alta latência detectada: {metrics.total_latency:.2f}s para query: {metrics.query_text[:50]}...")
    
    async def _collect_system_metrics(self):
        """Coleta métricas do sistema"""
        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # CPU usage
        cpu_percent = process.cpu_percent(interval=1)
        
        # Atualizar Prometheus
        self.prometheus_metrics.memory_usage.labels('total').set(memory_mb)
        
        # Atualizar component metrics
        for component in self.component_metrics.values():
            component.memory_samples.append(memory_mb)
            component.cpu_samples.append(cpu_percent)
    
    async def _check_alerts(self):
        """Verifica condições de alerta"""
        thresholds = self.config['monitoring']['alert_thresholds']
        
        # Verificar latência P95
        if self.query_metrics:
            recent_latencies = [m.total_latency for m in list(self.query_metrics)[-100:]]
            if recent_latencies:
                p95_latency = np.percentile(recent_latencies, 95)
                if p95_latency > thresholds['latency_p95']:
                    logger.warning(f"ALERTA: Latência P95 alta: {p95_latency:.2f}s")
        
        # Verificar taxa de erro
        for name, component in self.component_metrics.items():
            total = component.success_count + component.error_count
            if total > 100:
                error_rate = component.error_count / total
                if error_rate > thresholds['error_rate']:
                    logger.warning(f"ALERTA: Taxa de erro alta em {name}: {error_rate:.1%}")
        
        # Verificar memória
        if self.component_metrics['embedding'].memory_samples:
            avg_memory = np.mean(self.component_metrics['embedding'].memory_samples)
            if avg_memory > thresholds['memory_usage_mb']:
                logger.warning(f"ALERTA: Uso de memória alto: {avg_memory:.0f}MB")
    
    async def _optimize_routing(self):
        """Otimiza routing baseado em métricas"""
        if len(self.query_metrics) < self.config['optimization']['min_samples_for_optimization']:
            return
        
        # Analisar performance por estratégia
        strategy_metrics = defaultdict(list)
        for metric in self.query_metrics:
            strategy_metrics[metric.routing_strategy].append(metric)
        
        # Calcular performance média por estratégia
        strategy_performance = {}
        for strategy, metrics_list in strategy_metrics.items():
            if metrics_list:
                avg_latency = np.mean([m.total_latency for m in metrics_list])
                success_rate = sum(1 for m in metrics_list if m.relevance_score > 0.7) / len(metrics_list)
                
                strategy_performance[strategy] = {
                    'avg_latency': avg_latency,
                    'success_rate': success_rate,
                    'sample_size': len(metrics_list)
                }
        
        # Log otimizações sugeridas
        logger.info(f"Performance por estratégia: {json.dumps(strategy_performance, indent=2)}")
        
        # Ajustar thresholds do router se necessário
        if 'vector_only' in strategy_performance and 'hybrid_search' in strategy_performance:
            vector_perf = strategy_performance['vector_only']
            hybrid_perf = strategy_performance['hybrid_search']
            
            # Se vector_only está performando bem, aumentar seu uso
            if vector_perf['success_rate'] > 0.8 and vector_perf['avg_latency'] < 0.5:
                self.adaptive_router.thresholds['simple_query_threshold'] = 0.7
                logger.info("Otimização: Aumentando uso de vector_only para queries simples")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retorna resumo das métricas"""
        if not self.query_metrics:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.query_metrics)[-1000:]  # Últimas 1000 queries
        
        # Calcular estatísticas
        latencies = [m.total_latency for m in recent_metrics]
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        
        # Estatísticas por componente
        component_stats = {}
        for name, component in self.component_metrics.items():
            component_stats[name] = {
                'avg_latency': component.get_avg_latency(),
                'p95_latency': component.get_p95_latency(),
                'error_rate': (
                    component.error_count / (component.success_count + component.error_count)
                    if (component.success_count + component.error_count) > 0
                    else 0
                )
            }
        
        return {
            'total_queries': len(self.query_metrics),
            'recent_queries': len(recent_metrics),
            'avg_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'cache_hit_rate': cache_hits / len(recent_metrics),
            'component_stats': component_stats,
            'memory_usage_mb': np.mean(self.component_metrics['embedding'].memory_samples) if self.component_metrics['embedding'].memory_samples else 0
        }
    
    async def export_metrics(self, filepath: str):
        """Exporta métricas para arquivo"""
        metrics_data = {
            'summary': self.get_metrics_summary(),
            'recent_queries': [
                {
                    'query_id': m.query_id,
                    'timestamp': m.timestamp,
                    'total_latency': m.total_latency,
                    'routing_strategy': m.routing_strategy,
                    'cache_hit': m.cache_hit
                }
                for m in list(self.query_metrics)[-100:]
            ]
        }
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(metrics_data, indent=2))
        
        logger.info(f"Métricas exportadas para: {filepath}")
    
    async def stop_monitoring(self):
        """Para tasks de monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()
        
        logger.info("Monitoring parado")

# Factory function
def create_rag_monitor(config_path: Optional[str] = None) -> RAGMonitor:
    """Cria instância do RAG monitor"""
    if config_path is None:
        config_path = "config/monitoring_config.yaml"
    return RAGMonitor(config_path) 