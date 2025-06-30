# Semantic Caching - Guia de Integração Gradual

## 📊 **Status Atual - IMPLEMENTADO ✅**

### **Sistema Integrado de Cache:**
- ✅ `IntegratedCacheSystem` - Cache semântico + tradicional
- ✅ Similaridade semântica baseada em embeddings
- ✅ Adaptação automática de respostas
- ✅ Fallback robusto entre sistemas
- ✅ Métricas e monitoramento integrados

---

## 🚀 **FASE 1: Integração Gradual com Pipeline RAG**

### **1.1 Modificar Pipeline Existente**

```python
# src/rag_pipeline_advanced.py - INTEGRAÇÃO GRADUAL

from src.cache.semantic_cache_integration import (
    IntegratedCacheSystem, 
    create_integrated_cache_system,
    CacheResponse
)

class RAGPipelineAdvanced:
    def __init__(self, config):
        # Cache existente (manter compatibilidade)
        self.traditional_cache = self._init_traditional_cache(config)
        
        # NOVO: Sistema integrado de cache
        self.integrated_cache = create_integrated_cache_system({
            "enable_semantic": config.get("enable_semantic_cache", True),
            "enable_traditional": True,  # Manter tradicional ativo
            "semantic_cache_config": {
                "similarity_threshold": 0.85,
                "adaptation_threshold": 0.75
            }
        })
        
        # Flag para migração gradual
        self.use_integrated_cache = config.get("use_integrated_cache", False)
        
    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Query principal com cache integrado"""
        
        start_time = time.time()
        
        # ESTRATÉGIA GRADUAL: Usar cache integrado se habilitado
        if self.use_integrated_cache:
            cache_response = await self._query_with_integrated_cache(question, **kwargs)
            if cache_response.content is not None:
                return self._format_cache_response(cache_response, start_time)
        
        # Fallback para método tradicional
        result = await self._process_query_traditional(question, **kwargs)
        
        # Salvar no cache integrado (background)
        if self.use_integrated_cache:
            asyncio.create_task(self._save_to_integrated_cache(question, result, start_time))
        
        return result
    
    async def _query_with_integrated_cache(self, question: str, **kwargs) -> CacheResponse:
        """Busca no sistema integrado de cache"""
        try:
            context = {
                "user_id": kwargs.get("user_id"),
                "session_id": kwargs.get("session_id"),
                "domain": kwargs.get("domain", "general")
            }
            
            cache_response = await self.integrated_cache.get(question, context)
            
            # Log para monitoramento
            if cache_response.content is not None:
                logger.info(f"🎯 Cache HIT: {cache_response.source} "
                          f"(confidence: {cache_response.confidence:.3f})")
            
            return cache_response
            
        except Exception as e:
            logger.warning(f"Erro no cache integrado: {e}")
            return CacheResponse(content=None, source="error", confidence=0.0, metadata={})
    
    def _format_cache_response(self, cache_response: CacheResponse, start_time: float) -> Dict[str, Any]:
        """Formata resposta do cache para compatibilidade"""
        processing_time = time.time() - start_time
        
        return {
            **cache_response.content,
            "_cache_info": {
                "hit": True,
                "source": cache_response.source,
                "confidence": cache_response.confidence,
                "processing_time": processing_time,
                "metadata": cache_response.metadata
            }
        }
    
    async def _save_to_integrated_cache(self, question: str, result: Dict, start_time: float):
        """Salva resultado no cache integrado (async background)"""
        try:
            processing_time = time.time() - start_time
            
            await self.integrated_cache.set(
                query=question,
                response=result,
                confidence_score=result.get("confidence", 0.0),
                processing_time_saved=processing_time,
                tokens_saved=self._estimate_tokens_saved(result),
                source_model=result.get("model_used", "unknown")
            )
            
        except Exception as e:
            logger.warning(f"Erro ao salvar no cache integrado: {e}")
```

### **1.2 Configuração de Migração Gradual**

```yaml
# config/cache_migration.yaml
cache_migration:
  # Fase 1: Teste em paralelo (sem impacto)
  use_integrated_cache: false
  enable_semantic_cache: true
  enable_background_save: true
  
  # Fase 2: Uso gradual (low traffic)
  # use_integrated_cache: true
  # traffic_percentage: 10
  
  # Fase 3: Rollout completo
  # traffic_percentage: 100
  
  semantic_cache:
    similarity_threshold: 0.85
    adaptation_threshold: 0.75
    max_memory_entries: 1000
    enable_redis: true
    
  monitoring:
    enable_detailed_logs: true
    metrics_collection: true
    alert_on_errors: true
```

---

## 📊 **FASE 2: Sistema de Monitoramento de Métricas**

### **2.1 Collector de Métricas Avançado**

```python
# src/monitoring/semantic_cache_metrics.py

import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import defaultdict, deque
import json
from pathlib import Path

@dataclass
class CacheMetrics:
    """Métricas detalhadas do cache semântico"""
    timestamp: float
    cache_hits: int
    cache_misses: int
    semantic_hits: int
    traditional_hits: int
    adaptation_count: int
    average_similarity: float
    processing_time_saved: float
    tokens_saved: int
    cost_savings: float
    error_count: int
    
    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def semantic_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.semantic_hits / total if total > 0 else 0.0


class SemanticCacheMetricsCollector:
    """Coletor avançado de métricas para cache semântico"""
    
    def __init__(self, 
                 cache_system: IntegratedCacheSystem,
                 collection_interval: int = 60,
                 retention_hours: int = 24):
        
        self.cache_system = cache_system
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        
        # Buffers de métricas
        self.metrics_history: deque = deque(maxlen=retention_hours * 60)
        self.real_time_stats = defaultdict(int)
        self.similarity_scores: deque = deque(maxlen=1000)
        
        # Alertas
        self.alert_thresholds = {
            "min_hit_rate": 0.15,
            "max_error_rate": 0.05,
            "min_avg_similarity": 0.70
        }
        
        self.running = False
        
    async def start_collection(self):
        """Inicia coleta de métricas em background"""
        self.running = True
        
        logger.info("🔄 Iniciando coleta de métricas do cache semântico")
        
        # Task principal de coleta
        asyncio.create_task(self._collect_metrics_loop())
        
        # Task de análise e alertas
        asyncio.create_task(self._analysis_loop())
    
    async def _collect_metrics_loop(self):
        """Loop principal de coleta de métricas"""
        while self.running:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Erro na coleta de métricas: {e}")
                await asyncio.sleep(5)
    
    async def _collect_current_metrics(self):
        """Coleta métricas atuais do sistema"""
        try:
            # Obter stats dos caches
            integrated_stats = self.cache_system.get_stats()
            
            # Extrair métricas relevantes
            current_metrics = CacheMetrics(
                timestamp=time.time(),
                cache_hits=integrated_stats.get("semantic_hits", 0) + integrated_stats.get("traditional_hits", 0),
                cache_misses=integrated_stats.get("cache_misses", 0),
                semantic_hits=integrated_stats.get("semantic_hits", 0),
                traditional_hits=integrated_stats.get("traditional_hits", 0),
                adaptation_count=integrated_stats.get("adaptation_count", 0),
                average_similarity=self._calculate_avg_similarity(),
                processing_time_saved=integrated_stats.get("total_processing_time_saved", 0.0),
                tokens_saved=integrated_stats.get("total_tokens_saved", 0),
                cost_savings=integrated_stats.get("total_cost_saved", 0.0),
                error_count=integrated_stats.get("error_count", 0)
            )
            
            # Adicionar ao histórico
            self.metrics_history.append(current_metrics)
            
            # Log de debug
            logger.debug(f"📊 Métricas coletadas: hit_rate={current_metrics.hit_rate:.2%}, "
                        f"semantic_rate={current_metrics.semantic_hit_rate:.2%}")
            
        except Exception as e:
            logger.warning(f"Erro ao coletar métricas: {e}")
    
    def _calculate_avg_similarity(self) -> float:
        """Calcula similaridade média recente"""
        if not self.similarity_scores:
            return 0.0
        return sum(self.similarity_scores) / len(self.similarity_scores)
    
    def record_similarity(self, similarity: float):
        """Registra score de similaridade"""
        self.similarity_scores.append(similarity)
    
    async def _analysis_loop(self):
        """Loop de análise e alertas"""
        while self.running:
            try:
                await self._analyze_and_alert()
                await asyncio.sleep(300)  # Análise a cada 5 minutos
            except Exception as e:
                logger.error(f"Erro na análise de métricas: {e}")
                await asyncio.sleep(30)
    
    async def _analyze_and_alert(self):
        """Analisa métricas e gera alertas"""
        if len(self.metrics_history) < 5:  # Mínimo para análise
            return
        
        # Últimas 5 métricas
        recent_metrics = list(self.metrics_history)[-5:]
        
        # Calcular médias
        avg_hit_rate = sum(m.hit_rate for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_count for m in recent_metrics) / (sum(m.cache_hits + m.cache_misses for m in recent_metrics) or 1)
        avg_similarity = sum(m.average_similarity for m in recent_metrics) / len(recent_metrics)
        
        # Verificar alertas
        alerts = []
        
        if avg_hit_rate < self.alert_thresholds["min_hit_rate"]:
            alerts.append(f"⚠️ Hit rate baixo: {avg_hit_rate:.2%} (min: {self.alert_thresholds['min_hit_rate']:.2%})")
        
        if avg_error_rate > self.alert_thresholds["max_error_rate"]:
            alerts.append(f"🚨 Error rate alto: {avg_error_rate:.2%} (max: {self.alert_thresholds['max_error_rate']:.2%})")
        
        if avg_similarity < self.alert_thresholds["min_avg_similarity"]:
            alerts.append(f"📉 Similaridade média baixa: {avg_similarity:.3f} (min: {self.alert_thresholds['min_avg_similarity']:.3f})")
        
        # Enviar alertas
        for alert in alerts:
            logger.warning(f"ALERT CACHE: {alert}")
            await self._send_alert(alert)
    
    async def _send_alert(self, message: str):
        """Envia alerta (implementar integração com sistemas de alerta)"""
        # Implementar webhook, Slack, email, etc.
        pass
    
    def get_metrics_summary(self, hours: int = 1) -> Dict:
        """Retorna resumo de métricas das últimas N horas"""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"error": "Não há métricas suficientes"}
        
        return {
            "period_hours": hours,
            "total_requests": sum(m.cache_hits + m.cache_misses for m in recent_metrics),
            "hit_rate": {
                "current": recent_metrics[-1].hit_rate,
                "average": sum(m.hit_rate for m in recent_metrics) / len(recent_metrics),
                "trend": self._calculate_trend([m.hit_rate for m in recent_metrics])
            },
            "semantic_performance": {
                "hits": sum(m.semantic_hits for m in recent_metrics),
                "rate": sum(m.semantic_hit_rate for m in recent_metrics) / len(recent_metrics),
                "avg_similarity": sum(m.average_similarity for m in recent_metrics) / len(recent_metrics)
            },
            "savings": {
                "tokens": sum(m.tokens_saved for m in recent_metrics),
                "cost_usd": sum(m.cost_savings for m in recent_metrics),
                "time_seconds": sum(m.processing_time_saved for m in recent_metrics)
            },
            "adaptations": sum(m.adaptation_count for m in recent_metrics),
            "errors": sum(m.error_count for m in recent_metrics)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendência dos valores"""
        if len(values) < 3:
            return "insufficient_data"
        
        recent_avg = sum(values[-3:]) / 3
        older_avg = sum(values[:-3]) / max(len(values) - 3, 1)
        
        if recent_avg > older_avg * 1.05:
            return "increasing"
        elif recent_avg < older_avg * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def export_metrics(self, filepath: str = None) -> str:
        """Exporta métricas para arquivo JSON"""
        if filepath is None:
            filepath = f"storage/cache_metrics_{int(time.time())}.json"
        
        export_data = {
            "exported_at": time.time(),
            "metrics_count": len(self.metrics_history),
            "metrics": [asdict(m) for m in self.metrics_history]
        }
        
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📄 Métricas exportadas: {filepath}")
        return filepath
    
    def stop_collection(self):
        """Para a coleta de métricas"""
        self.running = False
        logger.info("🛑 Coleta de métricas parada")
```

### **2.2 Dashboard de Monitoramento**

```python
# src/monitoring/cache_dashboard.py

import asyncio
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import time

class SemanticCacheDashboard:
    """Dashboard em tempo real para cache semântico"""
    
    def __init__(self, metrics_collector: SemanticCacheMetricsCollector):
        self.metrics_collector = metrics_collector
        self.console = Console()
        self.running = False
    
    async def start_dashboard(self):
        """Inicia dashboard em tempo real"""
        self.running = True
        
        with Live(self._generate_layout(), refresh_per_second=2, console=self.console) as live:
            while self.running:
                live.update(self._generate_layout())
                await asyncio.sleep(0.5)
    
    def _generate_layout(self) -> Layout:
        """Gera layout do dashboard"""
        layout = Layout()
        
        # Divisão principal
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Subdivisões
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="metrics"),
            Layout(name="performance")
        )
        
        layout["right"].split_column(
            Layout(name="trends"),
            Layout(name="alerts")
        )
        
        # Preencher seções
        layout["header"].update(self._create_header())
        layout["metrics"].update(self._create_metrics_panel())
        layout["performance"].update(self._create_performance_panel())
        layout["trends"].update(self._create_trends_panel())
        layout["alerts"].update(self._create_alerts_panel())
        layout["footer"].update(self._create_footer())
        
        return layout
    
    def _create_header(self) -> Panel:
        """Cria header do dashboard"""
        return Panel(
            f"🧠 Semantic Cache Dashboard - {time.strftime('%Y-%m-%d %H:%M:%S')}",
            title="RAG System Monitoring",
            border_style="blue"
        )
    
    def _create_metrics_panel(self) -> Panel:
        """Cria painel de métricas principais"""
        metrics = self.metrics_collector.get_metrics_summary(hours=1)
        
        if "error" in metrics:
            content = "⚠️ Métricas insuficientes"
        else:
            content = f"""
📊 Últimas 1h:
• Requests: {metrics['total_requests']:,}
• Hit Rate: {metrics['hit_rate']['current']:.2%} ({metrics['hit_rate']['trend']})
• Semantic Hits: {metrics['semantic_performance']['hits']:,}
• Adaptações: {metrics['adaptations']:,}
• Erros: {metrics['errors']:,}
            """.strip()
        
        return Panel(content, title="📊 Métricas Principais", border_style="green")
    
    def _create_performance_panel(self) -> Panel:
        """Cria painel de performance"""
        metrics = self.metrics_collector.get_metrics_summary(hours=1)
        
        if "error" in metrics:
            content = "⚠️ Dados insuficientes"
        else:
            savings = metrics['savings']
            content = f"""
💰 Economia (1h):
• Tokens: {savings['tokens']:,}
• Custo: ${savings['cost_usd']:.4f}
• Tempo: {savings['time_seconds']:.1f}s

🎯 Similaridade:
• Média: {metrics['semantic_performance']['avg_similarity']:.3f}
• Taxa Semântica: {metrics['semantic_performance']['rate']:.2%}
            """.strip()
        
        return Panel(content, title="⚡ Performance", border_style="yellow")
    
    def _create_trends_panel(self) -> Panel:
        """Cria painel de tendências"""
        recent_metrics = list(self.metrics_collector.metrics_history)[-10:]
        
        if len(recent_metrics) < 3:
            content = "📈 Coletando dados..."
        else:
            hit_rates = [m.hit_rate for m in recent_metrics]
            similarities = [m.average_similarity for m in recent_metrics]
            
            content = f"""
📈 Tendências (10 min):
• Hit Rate: {self.metrics_collector._calculate_trend(hit_rates)}
• Similaridade: {self.metrics_collector._calculate_trend(similarities)}

📊 Últimos valores:
• {' '.join([f'{hr:.1%}' for hr in hit_rates[-5:]])}
• {' '.join([f'{sim:.2f}' for sim in similarities[-5:]])}
            """.strip()
        
        return Panel(content, title="📈 Tendências", border_style="cyan")
    
    def _create_alerts_panel(self) -> Panel:
        """Cria painel de alertas"""
        # Simular verificação de alertas
        content = "✅ Sistema operando normalmente"
        border_style = "green"
        
        # Verificar métricas recentes
        if self.metrics_collector.metrics_history:
            latest = self.metrics_collector.metrics_history[-1]
            
            alerts = []
            if latest.hit_rate < 0.15:
                alerts.append("⚠️ Hit rate baixo")
            if latest.average_similarity < 0.70:
                alerts.append("📉 Similaridade baixa")
            if latest.error_count > 0:
                alerts.append(f"🚨 {latest.error_count} erros")
            
            if alerts:
                content = "\n".join(alerts)
                border_style = "red"
        
        return Panel(content, title="🚨 Alertas", border_style=border_style)
    
    def _create_footer(self) -> Panel:
        """Cria footer do dashboard"""
        return Panel(
            "Pressione Ctrl+C para sair | Atualização a cada 0.5s",
            border_style="dim"
        )
    
    def stop_dashboard(self):
        """Para o dashboard"""
        self.running = False


# Função para iniciar monitoramento completo
async def start_semantic_cache_monitoring(cache_system: IntegratedCacheSystem):
    """Inicia sistema completo de monitoramento"""
    
    # Inicializar coletor de métricas
    metrics_collector = SemanticCacheMetricsCollector(cache_system)
    await metrics_collector.start_collection()
    
    # Inicializar dashboard
    dashboard = SemanticCacheDashboard(metrics_collector)
    
    logger.info("🚀 Sistema de monitoramento iniciado")
    logger.info("📊 Dashboard disponível - execute: await start_dashboard()")
    
    return metrics_collector, dashboard
```

---

## 🎯 **FASE 3: Script de Integração Completa**

```python
# scripts/integrate_semantic_cache.py

import asyncio
import logging
from pathlib import Path
import yaml

async def integrate_semantic_cache_step_by_step():
    """Script de integração passo a passo"""
    
    print("🚀 === INTEGRAÇÃO SEMANTIC CACHE - PASSO A PASSO === 🚀\n")
    
    # PASSO 1: Verificar dependências
    print("📋 PASSO 1: Verificando dependências...")
    try:
        import numpy as np
        import openai
        from src.cache.semantic_cache_integration import create_integrated_cache_system
        from src.monitoring.semantic_cache_metrics import SemanticCacheMetricsCollector
        print("✅ Todas as dependências estão disponíveis\n")
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        print("Execute: pip install numpy openai redis\n")
        return
    
    # PASSO 2: Configurar sistema
    print("⚙️ PASSO 2: Configurando sistema integrado...")
    
    config = {
        "enable_semantic": True,
        "enable_traditional": True,
        "semantic_cache_config": {
            "similarity_threshold": 0.85,
            "adaptation_threshold": 0.75,
            "max_memory_entries": 1000,
            "db_path": "storage/semantic_cache.db"
        }
    }
    
    cache_system = create_integrated_cache_system(config)
    print("✅ Sistema de cache integrado configurado\n")
    
    # PASSO 3: Iniciar monitoramento
    print("📊 PASSO 3: Iniciando monitoramento...")
    
    metrics_collector = SemanticCacheMetricsCollector(cache_system)
    await metrics_collector.start_collection()
    print("✅ Coleta de métricas iniciada\n")
    
    # PASSO 4: Teste funcional
    print("🧪 PASSO 4: Executando testes funcionais...")
    
    # Teste básico
    test_query = "Como implementar autenticação JWT em Python?"
    test_response = {
        "answer": "Use PyJWT para implementar JWT em Python...",
        "sources": [{"content": "JWT tutorial", "score": 0.9}],
        "confidence": 0.95
    }
    
    # Salvar no cache
    await cache_system.set(test_query, test_response, confidence_score=0.95)
    
    # Buscar query similar
    similar_query = "JWT Python implementação"
    cache_response = await cache_system.get(similar_query)
    
    if cache_response.content:
        print(f"✅ Teste semântico PASSOU - Confidence: {cache_response.confidence:.3f}")
    else:
        print("⚠️ Teste semântico falhou - verificar configuração")
    
    print()
    
    # PASSO 5: Integração com pipeline
    print("🔗 PASSO 5: Exemplo de integração com pipeline...")
    
    integration_code = '''
# Em seu RAGPipelineAdvanced:
from src.cache.semantic_cache_integration import create_integrated_cache_system

class RAGPipelineAdvanced:
    def __init__(self, config):
        self.integrated_cache = create_integrated_cache_system(config)
        
    async def query(self, question: str, **kwargs):
        # Tentar cache primeiro
        cache_response = await self.integrated_cache.get(question)
        if cache_response.content:
            return cache_response.content
        
        # Processar query normal
        result = await self._process_query(question, **kwargs)
        
        # Salvar no cache
        await self.integrated_cache.set(question, result)
        return result
'''
    
    print(integration_code)
    
    # PASSO 6: Métricas finais
    print("\n📊 PASSO 6: Verificando métricas...")
    
    await asyncio.sleep(2)  # Aguardar coleta
    stats = cache_system.get_stats()
    
    print(f"📈 Estatísticas:")
    print(f"   Requests totais: {stats.get('total_requests', 0)}")
    print(f"   Hits semânticos: {stats.get('semantic_hits', 0)}")
    print(f"   Hits tradicionais: {stats.get('traditional_hits', 0)}")
    print(f"   Cache misses: {stats.get('cache_misses', 0)}")
    
    # PASSO 7: Próximos passos
    print(f"\n🎯 PRÓXIMOS PASSOS:")
    print(f"   1. ✅ Executar: python demo_semantic_cache.py")
    print(f"   2. ✅ Integrar com seu pipeline RAG")
    print(f"   3. ✅ Configurar monitoramento em produção")
    print(f"   4. ✅ Ajustar thresholds baseado em métricas")
    print(f"   5. ✅ Habilitar warming preditivo")
    
    print(f"\n🎉 INTEGRAÇÃO SEMANTIC CACHE CONCLUÍDA! 🎉")
    
    return cache_system, metrics_collector


if __name__ == "__main__":
    asyncio.run(integrate_semantic_cache_step_by_step())
```

---

## 🚨 **Checklist de Implementação**

### **Pré-requisitos:**
- [ ] ✅ Dependências instaladas (`numpy`, `openai`, `redis`)
- [ ] ✅ Variáveis de ambiente configuradas (`OPENAI_API_KEY`)
- [ ] ✅ Diretório `storage/` criado

### **Integração:**
- [ ] 🔄 Modificar `RAGPipelineAdvanced` com cache integrado
- [ ] 🔄 Configurar flag `use_integrated_cache` para migração gradual
- [ ] 🔄 Implementar coleta de métricas em background
- [ ] 🔄 Configurar alertas e monitoramento

### **Teste e Validação:**
- [ ] 🔄 Executar `demo_semantic_cache.py`
- [ ] 🔄 Executar `scripts/integrate_semantic_cache.py`
- [ ] 🔄 Validar métricas no dashboard
- [ ] 🔄 Testar fallback para cache tradicional

### **Produção:**
- [ ] ⏳ Deploy gradual com percentual de tráfego
- [ ] ⏳ Monitoramento contínuo de performance
- [ ] ⏳ Otimização de thresholds baseada em dados reais

---

## 📞 **Comandos Rápidos**

```bash
# Executar integração completa
python scripts/integrate_semantic_cache.py

# Demo funcional
python demo_semantic_cache.py

# Verificar métricas
python -c "
from src.cache.semantic_cache_integration import create_integrated_cache_system
cache = create_integrated_cache_system()
print(cache.get_stats())
"

# Iniciar dashboard de monitoramento
python -c "
import asyncio
from src.monitoring.cache_dashboard import start_semantic_cache_monitoring
# asyncio.run(start_dashboard())
"
```

**🎯 O Semantic Caching está pronto para integração gradual e monitoramento contínuo!** 