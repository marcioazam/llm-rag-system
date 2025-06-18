"""
TESTE DE INTEGRAÇÃO - FASE 3: Demonstração de Recursos Avançados
Testa e demonstra todas as funcionalidades implementadas na Fase 3
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_fase3.log')
    ]
)

logger = logging.getLogger(__name__)

# Imports dos módulos da Fase 3
try:
    from cache.cache_warming import CacheWarmer, QueryPattern, WarmingTask
    from cache.cache_analytics import CacheAnalytics, MetricSnapshot, Alert, AlertLevel, MetricType
    from cache.cache_tuning import CacheTuner, TuningStrategy, TuningAction, PerformanceProfile
    from cache.redis_enterprise import RedisEnterpriseManager, RedisClusterNode, RedisPerformanceProfile
    from cache.optimized_rag_cache import OptimizedRAGCache
except ImportError as e:
    logger.error(f"Erro ao importar módulos da Fase 3: {e}")
    sys.exit(1)


class MockRAGPipeline:
    """Mock do pipeline RAG para testes"""
    
    def __init__(self):
        self.query_count = 0
        self.responses = {
            "cache em sistemas rag": {
                "answer": "Cache em sistemas RAG é essencial para reduzir latência e custos. Implementa-se através de múltiplas camadas: memória (L1), SQLite (L2) e Redis (L3). O cache armazena embeddings e respostas geradas, com TTL configurável baseado na confiança da resposta.",
                "confidence": 0.92,
                "sources": ["docs/cache_implementation.md", "papers/rag_optimization.pdf"],
                "processing_time": 2.3
            },
            "implementação de cache": {
                "answer": "A implementação de cache envolve escolha da estrutura de dados adequada, política de eviction (LRU, LFU), configuração de TTL dinâmico, e monitoramento de hit rate. É importante balancear memória, performance e consistência.",
                "confidence": 0.88,
                "sources": ["docs/system_architecture.md"],
                "processing_time": 1.8
            },
            "redis vs sqlite": {
                "answer": "Redis oferece performance superior para cache devido ao armazenamento em memória, mas SQLite é mais adequado para persistência e estruturas relacionais. Em sistemas híbridos, Redis serve como L1/L2 cache e SQLite como storage durável.",
                "confidence": 0.85,
                "sources": ["benchmarks/redis_vs_sqlite.json"],
                "processing_time": 2.1
            }
        }
    
    async def query_advanced(self, question: str, **kwargs):
        """Simula processamento de query avançada"""
        self.query_count += 1
        
        # Simular tempo de processamento
        await asyncio.sleep(0.1)
        
        # Retornar resposta simulada
        base_response = self.responses.get(question, {
            "answer": f"Resposta simulada para: {question}",
            "confidence": 0.75,
            "sources": ["mock_source.txt"],
            "processing_time": 1.5
        })
        
        # Adicionar metadados
        base_response.update({
            "timestamp": datetime.now().isoformat(),
            "query_count": self.query_count,
            "improvements_used": kwargs.get("force_improvements", [])
        })
        
        return base_response


class MockCache:
    """Mock do cache para testes"""
    
    def __init__(self):
        self.cache_data = {}
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "hit_rate": 0.0,
            "memory_usage_mb": 45.2,
            "cost_savings": 15.75,
            "tokens_saved": 8430
        }
    
    async def get(self, key: str):
        """Simula busca no cache"""
        self.stats["total_requests"] += 1
        
        if key in self.cache_data:
            self.stats["cache_hits"] += 1
            self.stats["l1_hits"] += 1
            self._update_hit_rate()
            return self.cache_data[key], "L1", {"confidence": 0.9}
        
        self._update_hit_rate()
        return None, None, None
    
    async def set(self, key: str, value: any, ttl: int = 3600):
        """Simula armazenamento no cache"""
        self.cache_data[key] = value
        return True
    
    def get_stats(self):
        """Retorna estatísticas do cache"""
        return self.stats.copy()
    
    def _update_hit_rate(self):
        """Atualiza hit rate"""
        if self.stats["total_requests"] > 0:
            self.stats["hit_rate"] = self.stats["cache_hits"] / self.stats["total_requests"]
    
    async def close(self):
        """Simula fechamento do cache"""
        pass


async def test_cache_warming():
    """Testa sistema de cache warming"""
    logger.info("🔥 TESTANDO CACHE WARMING")
    
    try:
        # Configurar mocks
        mock_cache = MockCache()
        mock_pipeline = MockRAGPipeline()
        
        # Inicializar cache warmer
        warmer = CacheWarmer(
            cache_instance=mock_cache,
            pipeline_instance=mock_pipeline,
            patterns_db_path="storage/test_cache_patterns.db",
            warming_batch_size=3
        )
        
        logger.info("✅ CacheWarmer inicializado")
        
        # Teste 1: Análise de padrões
        logger.info("📊 Testando análise de padrões...")
        patterns_analysis = await warmer.analyze_query_patterns()
        
        assert patterns_analysis["total_patterns"] > 0
        assert patterns_analysis["warming_candidates"] >= 0
        logger.info(f"✅ Padrões identificados: {patterns_analysis['total_patterns']}")
        
        # Teste 2: Criação de tarefas de warming
        logger.info("📋 Testando criação de tarefas de warming...")
        warming_tasks = await warmer.create_warming_tasks()
        
        assert len(warming_tasks) > 0
        logger.info(f"✅ Tarefas criadas: {len(warming_tasks)}")
        
        # Teste 3: Execução de warming
        logger.info("🚀 Testando execução de warming...")
        warming_result = await warmer.execute_warming(max_concurrent=2)
        
        assert warming_result["status"] == "completed"
        assert warming_result["total_tasks"] == len(warming_tasks)
        logger.info(f"✅ Warming executado: {warming_result['successful']}/{warming_result['total_tasks']} sucessos")
        
        # Teste 4: Estatísticas de warming
        logger.info("📈 Testando estatísticas de warming...")
        warming_stats = warmer.get_warming_stats()
        
        assert "total_warmed" in warming_stats
        assert "queue_status" in warming_stats
        logger.info(f"✅ Estatísticas coletadas: {warming_stats['total_warmed']} queries aquecidas")
        
        # Cleanup
        await warmer.cleanup()
        logger.info("🧹 Cleanup cache warming concluído")
        
        return {
            "status": "success",
            "patterns_found": patterns_analysis["total_patterns"],
            "tasks_created": len(warming_tasks),
            "successful_warming": warming_result["successful"],
            "total_warmed": warming_stats["total_warmed"]
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de cache warming: {e}")
        return {"status": "error", "error": str(e)}


async def test_cache_analytics():
    """Testa sistema de analytics"""
    logger.info("📊 TESTANDO CACHE ANALYTICS")
    
    try:
        # Configurar mock
        mock_cache = MockCache()
        
        # Inicializar analytics
        analytics = CacheAnalytics(
            cache_instance=mock_cache,
            analytics_db_path="storage/test_cache_analytics.db",
            snapshot_interval_minutes=1,
            retention_days=7
        )
        
        logger.info("✅ CacheAnalytics inicializado")
        
        # Teste 1: Captura de snapshot
        logger.info("📸 Testando captura de snapshot...")
        snapshot = await analytics.capture_snapshot()
        
        assert snapshot is not None
        assert snapshot.hit_rate >= 0
        assert snapshot.total_requests >= 0
        logger.info(f"✅ Snapshot capturado: hit_rate={snapshot.hit_rate:.1%}")
        
        # Teste 2: Verificação de alertas
        logger.info("🚨 Testando sistema de alertas...")
        
        # Simular condição de alerta (baixo hit rate)
        mock_cache.stats["hit_rate"] = 0.05  # 5% hit rate
        await analytics.check_alerts()
        
        assert len(analytics.active_alerts) > 0
        logger.info(f"✅ Alertas gerados: {len(analytics.active_alerts)}")
        
        # Teste 3: Dashboard
        logger.info("📋 Testando geração de dashboard...")
        dashboard_data = await analytics.get_dashboard_data(hours=1)
        
        assert "current_snapshot" in dashboard_data
        assert "health_score" in dashboard_data
        assert "recommendations" in dashboard_data
        
        health_score = dashboard_data["health_score"]
        recommendations = dashboard_data["recommendations"]
        
        logger.info(f"✅ Dashboard gerado: health_score={health_score:.1f}, {len(recommendations)} recomendações")
        
        # Teste 4: Callback de alerta
        logger.info("🔔 Testando callback de alerta...")
        alert_received = []
        
        def alert_callback(alert):
            alert_received.append(alert)
            logger.info(f"📧 Alerta recebido: {alert.title}")
        
        analytics.add_alert_callback(alert_callback)
        
        # Simular novo alerta
        analytics.record_request_time(10.0, error=True)  # Tempo alto + erro
        await analytics.check_alerts()
        
        # Cleanup
        await analytics.cleanup()
        logger.info("🧹 Cleanup analytics concluído")
        
        return {
            "status": "success",
            "snapshot_captured": snapshot is not None,
            "alerts_generated": len(analytics.active_alerts),
            "health_score": health_score,
            "recommendations_count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de analytics: {e}")
        return {"status": "error", "error": str(e)}


async def test_cache_tuning():
    """Testa sistema de auto-tuning"""
    logger.info("🔧 TESTANDO CACHE TUNING")
    
    try:
        # Configurar mocks
        mock_cache = MockCache()
        mock_analytics = Mock()
        
        # Mock analytics data
        mock_analytics.get_dashboard_data = AsyncMock(return_value={
            "current_snapshot": {
                "response_time_avg": 3.5,
                "throughput_qps": 0.8,
                "error_rate": 0.12
            }
        })
        
        # Inicializar tuner
        tuner = CacheTuner(
            cache_instance=mock_cache,
            analytics_instance=mock_analytics,
            tuning_db_path="storage/test_cache_tuning.db",
            strategy=TuningStrategy.BALANCED,
            tuning_interval_minutes=1
        )
        
        logger.info("✅ CacheTuner inicializado")
        
        # Teste 1: Coleta de métricas
        logger.info("📊 Testando coleta de métricas...")
        performance = await tuner._collect_performance_metrics()
        
        assert performance is not None
        assert performance.hit_rate >= 0
        assert performance.response_time >= 0
        logger.info(f"✅ Métricas coletadas: hit_rate={performance.hit_rate:.1%}, response_time={performance.response_time:.2f}s")
        
        # Teste 2: Análise de necessidade de tuning
        logger.info("🔍 Testando análise de tuning...")
        needs_tuning = await tuner._analyze_tuning_need(performance)
        
        logger.info(f"✅ Análise concluída: tuning_needed={needs_tuning}")
        
        # Teste 3: Busca de regras aplicáveis
        logger.info("📋 Testando regras aplicáveis...")
        applicable_rules = await tuner._find_applicable_rules(performance)
        
        assert isinstance(applicable_rules, list)
        logger.info(f"✅ Regras encontradas: {len(applicable_rules)}")
        
        # Teste 4: Aplicação de tuning (se necessário)
        if needs_tuning and applicable_rules:
            logger.info("🔧 Testando aplicação de tuning...")
            applied_changes = await tuner._apply_tuning(performance)
            
            assert isinstance(applied_changes, list)
            logger.info(f"✅ Mudanças aplicadas: {len(applied_changes)}")
        else:
            logger.info("ℹ️ Tuning não necessário no momento")
            applied_changes = []
        
        # Teste 5: Relatório de tuning
        logger.info("📄 Testando relatório de tuning...")
        tuning_report = await tuner.get_tuning_report(hours=1)
        
        assert "strategy" in tuning_report
        assert "current_config" in tuning_report
        assert "learning_insights" in tuning_report
        
        insights = tuning_report["learning_insights"]
        logger.info(f"✅ Relatório gerado: {len(insights)} insights")
        
        # Cleanup
        await tuner.cleanup()
        logger.info("🧹 Cleanup tuning concluído")
        
        return {
            "status": "success",
            "performance_collected": performance is not None,
            "tuning_needed": needs_tuning,
            "applicable_rules": len(applicable_rules),
            "changes_applied": len(applied_changes),
            "insights_generated": len(insights)
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de tuning: {e}")
        return {"status": "error", "error": str(e)}


async def test_redis_enterprise():
    """Testa configuração Redis Enterprise"""
    logger.info("🚀 TESTANDO REDIS ENTERPRISE")
    
    try:
        # Configurar cluster de teste
        test_nodes = [
            RedisClusterNode("localhost", 7000, "master", 1024, 500),
            RedisClusterNode("localhost", 7001, "master", 1024, 500),
            RedisClusterNode("localhost", 7002, "slave", 1024, 300)
        ]
        
        # Inicializar manager (sem conexões reais)
        with patch('redis.asyncio.Redis') as mock_redis:
            with patch('redis.asyncio.RedisCluster') as mock_cluster:
                
                # Mock das conexões Redis
                mock_redis_instance = AsyncMock()
                mock_redis_instance.ping = AsyncMock(return_value=True)
                mock_redis_instance.info = AsyncMock(return_value={
                    "uptime_in_seconds": 3600,
                    "connected_clients": 50,
                    "used_memory": 100 * 1024 * 1024,  # 100MB
                    "keyspace_hits": 900,
                    "keyspace_misses": 100,
                    "instantaneous_ops_per_sec": 150,
                    "evicted_keys": 5,
                    "expired_keys": 20,
                    "instantaneous_input_kbps": 10,
                    "instantaneous_output_kbps": 15,
                    "used_cpu_sys": 0.5,
                    "used_cpu_user": 0.3,
                    "role": "master"
                })
                mock_redis_instance.config_set = AsyncMock(return_value=True)
                mock_redis_instance.bgrewriteaof = AsyncMock(return_value=True)
                mock_redis_instance.close = AsyncMock()
                
                mock_redis.return_value = mock_redis_instance
                
                # Mock do cluster
                mock_cluster_instance = AsyncMock()
                mock_cluster_instance.ping = AsyncMock(return_value=True)
                mock_cluster_instance.cluster_info = AsyncMock(return_value={
                    "cluster_state": "ok",
                    "cluster_slots_assigned": 16384,
                    "cluster_known_nodes": 3
                })
                mock_cluster_instance.close = AsyncMock()
                
                mock_cluster.return_value = mock_cluster_instance
                
                # Inicializar Redis Enterprise Manager
                redis_manager = RedisEnterpriseManager(
                    cluster_nodes=test_nodes,
                    auto_optimization=True,
                    monitoring_interval_seconds=5
                )
                
                logger.info("✅ RedisEnterpriseManager inicializado")
                
                # Teste 1: Inicialização do cluster
                logger.info("🚀 Testando inicialização do cluster...")
                cluster_initialized = await redis_manager.initialize_cluster()
                
                assert cluster_initialized is True
                logger.info("✅ Cluster inicializado com sucesso")
                
                # Teste 2: Verificação de saúde
                logger.info("🏥 Testando verificação de saúde...")
                health_status = await redis_manager._check_cluster_health()
                
                assert health_status["status"] in ["healthy", "degraded"]
                logger.info(f"✅ Status de saúde: {health_status['status']}")
                
                # Teste 3: Coleta de métricas de performance
                logger.info("📊 Testando coleta de métricas...")
                performance_metrics = await redis_manager._collect_performance_metrics()
                
                assert performance_metrics is not None
                assert performance_metrics.hit_rate > 0
                logger.info(f"✅ Métricas coletadas: hit_rate={performance_metrics.hit_rate:.1%}, ops/sec={performance_metrics.ops_per_second}")
                
                # Teste 4: Status do cluster
                logger.info("📋 Testando status do cluster...")
                cluster_status = await redis_manager.get_cluster_status()
                
                assert "cluster_health" in cluster_status
                assert "current_performance" in cluster_status
                assert "node_count" in cluster_status
                logger.info(f"✅ Status obtido: {cluster_status['node_count']} nós")
                
                # Teste 5: Exportação de configuração
                logger.info("📄 Testando exportação de configuração...")
                config_file = await redis_manager.export_cluster_config("test_redis_config.yaml")
                
                if config_file:
                    logger.info(f"✅ Configuração exportada: {config_file}")
                else:
                    logger.warning("⚠️ Exportação de configuração falhou")
                
                # Cleanup
                await redis_manager.cleanup()
                logger.info("🧹 Cleanup Redis Enterprise concluído")
                
                return {
                    "status": "success",
                    "cluster_initialized": cluster_initialized,
                    "health_status": health_status["status"],
                    "performance_collected": performance_metrics is not None,
                    "node_count": cluster_status.get("node_count", 0),
                    "config_exported": bool(config_file)
                }
        
    except Exception as e:
        logger.error(f"❌ Erro no teste Redis Enterprise: {e}")
        return {"status": "error", "error": str(e)}


async def test_integrated_workflow():
    """Testa workflow integrado de todos os componentes"""
    logger.info("🔄 TESTANDO WORKFLOW INTEGRADO")
    
    try:
        # Configurar componentes
        mock_cache = MockCache()
        mock_pipeline = MockRAGPipeline()
        
        # Inicializar todos os componentes
        warmer = CacheWarmer(
            cache_instance=mock_cache,
            pipeline_instance=mock_pipeline,
            patterns_db_path="storage/test_integrated_patterns.db"
        )
        
        analytics = CacheAnalytics(
            cache_instance=mock_cache,
            analytics_db_path="storage/test_integrated_analytics.db"
        )
        
        tuner = CacheTuner(
            cache_instance=mock_cache,
            analytics_instance=analytics,
            tuning_db_path="storage/test_integrated_tuning.db"
        )
        
        logger.info("✅ Todos os componentes inicializados")
        
        # Simular workflow integrado
        logger.info("🔄 Executando workflow integrado...")
        
        # 1. Analytics captura métricas
        snapshot = await analytics.capture_snapshot()
        logger.info(f"📊 Snapshot capturado: hit_rate={snapshot.hit_rate:.1%}")
        
        # 2. Tuner analisa se precisa de ajustes
        performance = await tuner._collect_performance_metrics()
        needs_tuning = await tuner._analyze_tuning_need(performance)
        logger.info(f"🔧 Análise de tuning: necessário={needs_tuning}")
        
        # 3. Warmer executa pre-carregamento
        warming_result = await warmer.execute_warming()
        logger.info(f"🔥 Warming executado: {warming_result.get('successful', 0)} sucessos")
        
        # 4. Analytics verifica alertas
        await analytics.check_alerts()
        logger.info(f"🚨 Alertas ativos: {len(analytics.active_alerts)}")
        
        # 5. Gerar relatório integrado
        dashboard = await analytics.get_dashboard_data()
        tuning_report = await tuner.get_tuning_report()
        warming_stats = warmer.get_warming_stats()
        
        integrated_report = {
            "timestamp": datetime.now().isoformat(),
            "analytics": {
                "health_score": dashboard.get("health_score", 0),
                "active_alerts": len(analytics.active_alerts),
                "recommendations": len(dashboard.get("recommendations", []))
            },
            "tuning": {
                "strategy": tuning_report.get("strategy", "unknown"),
                "total_events": tuning_report.get("total_events", 0),
                "learning_insights": len(tuning_report.get("learning_insights", []))
            },
            "warming": {
                "total_warmed": warming_stats.get("total_warmed", 0),
                "success_rate": warming_result.get("success_rate", 0),
                "patterns_identified": len(warmer.patterns)
            }
        }
        
        logger.info("📋 Relatório integrado gerado")
        
        # Cleanup de todos os componentes
        await warmer.cleanup()
        await analytics.cleanup()
        await tuner.cleanup()
        
        logger.info("🧹 Cleanup integrado concluído")
        
        return {
            "status": "success",
            "integrated_report": integrated_report,
            "workflow_completed": True
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no workflow integrado: {e}")
        return {"status": "error", "error": str(e)}


async def main():
    """Função principal que executa todos os testes"""
    logger.info("🚀 INICIANDO TESTES DA FASE 3 - OTIMIZAÇÃO AVANÇADA")
    logger.info("=" * 80)
    
    # Criar diretório de storage para testes
    os.makedirs("storage", exist_ok=True)
    
    test_results = {}
    
    # Executar testes sequencialmente
    tests = [
        ("Cache Warming", test_cache_warming),
        ("Cache Analytics", test_cache_analytics),
        ("Cache Tuning", test_cache_tuning),
        ("Redis Enterprise", test_redis_enterprise),
        ("Workflow Integrado", test_integrated_workflow)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"EXECUTANDO: {test_name}")
        logger.info(f"{'=' * 50}")
        
        start_time = time.time()
        
        try:
            result = await test_func()
            test_results[test_name] = result
            
            elapsed = time.time() - start_time
            logger.info(f"✅ {test_name} concluído em {elapsed:.2f}s")
            
        except Exception as e:
            test_results[test_name] = {"status": "error", "error": str(e)}
            elapsed = time.time() - start_time
            logger.error(f"❌ {test_name} falhou em {elapsed:.2f}s: {e}")
    
    # Relatório final
    logger.info(f"\n{'=' * 80}")
    logger.info("📊 RELATÓRIO FINAL DOS TESTES - FASE 3")
    logger.info(f"{'=' * 80}")
    
    total_tests = len(tests)
    successful_tests = sum(1 for result in test_results.values() if result.get("status") == "success")
    
    logger.info(f"📈 RESULTADOS: {successful_tests}/{total_tests} testes bem-sucedidos")
    
    for test_name, result in test_results.items():
        status = result.get("status", "unknown")
        if status == "success":
            logger.info(f"✅ {test_name}: SUCESSO")
        else:
            logger.error(f"❌ {test_name}: FALHA - {result.get('error', 'Erro desconhecido')}")
    
    # Recursos demonstrados
    logger.info(f"\n🎯 RECURSOS DA FASE 3 DEMONSTRADOS:")
    logger.info(f"   🔥 Cache Warming - Pre-carregamento inteligente")
    logger.info(f"   📊 Cache Analytics - Dashboard e alertas")
    logger.info(f"   🔧 Cache Tuning - Auto-ajuste de parâmetros")
    logger.info(f"   🚀 Redis Enterprise - Configuração para produção")
    logger.info(f"   🔄 Integração Completa - Workflow unificado")
    
    # Status final
    if successful_tests == total_tests:
        logger.info(f"\n🎉 FASE 3 COMPLETADA COM SUCESSO TOTAL!")
        logger.info(f"   ✅ Todos os {total_tests} sistemas funcionando perfeitamente")
        logger.info(f"   📊 Sistema otimizado pronto para produção enterprise")
    else:
        logger.warning(f"\n⚠️ FASE 3 COMPLETADA COM ALGUMAS LIMITAÇÕES")
        logger.warning(f"   ✅ {successful_tests}/{total_tests} sistemas funcionais")
        logger.warning(f"   🔧 Revisar componentes com falhas")
    
    logger.info(f"\n{'=' * 80}")
    
    return test_results


if __name__ == "__main__":
    # Executar testes
    results = asyncio.run(main()) 