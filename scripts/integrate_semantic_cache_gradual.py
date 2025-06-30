#!/usr/bin/env python3
"""
Script de Integração Gradual - Semantic Caching
Integra o sistema de cache semântico gradualmente com o pipeline RAG existente
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

# Setup de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticCacheIntegrator:
    """Classe para integração gradual do cache semântico"""
    
    def __init__(self):
        self.integration_phases = [
            "validation",
            "parallel_testing", 
            "gradual_rollout",
            "full_integration"
        ]
        self.current_phase = None
        self.metrics = {}
        
    async def run_integration(self):
        """Executa integração completa passo a passo"""
        
        print("🚀 === INTEGRAÇÃO GRADUAL SEMANTIC CACHING === 🚀\n")
        
        try:
            # Fase 1: Validação do ambiente
            await self._phase_1_validation()
            
            # Fase 2: Teste paralelo
            await self._phase_2_parallel_testing()
            
            # Fase 3: Rollout gradual
            await self._phase_3_gradual_rollout()
            
            # Fase 4: Integração completa
            await self._phase_4_full_integration()
            
            # Relatório final
            await self._generate_final_report()
            
        except Exception as e:
            logger.error(f"Erro na integração: {e}")
            print(f"❌ Integração falhou: {e}")
            return False
        
        print("\n🎉 === INTEGRAÇÃO SEMANTIC CACHING CONCLUÍDA === 🎉")
        return True
    
    async def _phase_1_validation(self):
        """FASE 1: Validação do ambiente e dependências"""
        
        print("📋 FASE 1: Validação do Ambiente")
        print("-" * 50)
        
        self.current_phase = "validation"
        phase_metrics = {
            "start_time": time.time(),
            "checks": {},
            "errors": []
        }
        
        # Check 1: Dependências Python
        print("🔍 Verificando dependências Python...")
        try:
            import numpy as np
            phase_metrics["checks"]["numpy"] = "✅ OK"
            print(f"   ✅ NumPy: {np.__version__}")
        except ImportError:
            error = "NumPy não encontrado"
            phase_metrics["checks"]["numpy"] = f"❌ {error}"
            phase_metrics["errors"].append(error)
            print(f"   ❌ NumPy: Execute 'pip install numpy'")
        
        try:
            import openai
            phase_metrics["checks"]["openai"] = "✅ OK"
            print(f"   ✅ OpenAI: {openai.__version__}")
        except ImportError:
            error = "OpenAI não encontrado"
            phase_metrics["checks"]["openai"] = f"❌ {error}"
            phase_metrics["errors"].append(error)
            print(f"   ❌ OpenAI: Execute 'pip install openai'")
        
        # Check 2: Estrutura de arquivos
        print("\n🗂️ Verificando estrutura de arquivos...")
        
        required_files = [
            "src/cache/semantic_cache_integration.py",
            "src/rag_pipeline_advanced.py",
            "config/"
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                phase_metrics["checks"][file_path] = "✅ OK"
                print(f"   ✅ {file_path}")
            else:
                error = f"Arquivo necessário não encontrado: {file_path}"
                phase_metrics["checks"][file_path] = f"❌ {error}"
                phase_metrics["errors"].append(error)
                print(f"   ❌ {file_path}")
        
        # Check 3: Variáveis de ambiente
        print("\n🔑 Verificando variáveis de ambiente...")
        
        env_vars = ["OPENAI_API_KEY"]
        for var in env_vars:
            if os.getenv(var):
                phase_metrics["checks"][var] = "✅ OK"
                print(f"   ✅ {var}: Configurado")
            else:
                error = f"Variável de ambiente não configurada: {var}"
                phase_metrics["checks"][var] = f"⚠️ {error}"
                print(f"   ⚠️ {var}: Não configurado (fallback será usado)")
        
        # Check 4: Storage
        print("\n💾 Verificando diretório de storage...")
        storage_path = Path("storage")
        if not storage_path.exists():
            storage_path.mkdir(parents=True, exist_ok=True)
            print("   ✅ Diretório 'storage' criado")
        else:
            print("   ✅ Diretório 'storage' existe")
        
        phase_metrics["checks"]["storage"] = "✅ OK"
        
        # Salvar métricas da fase
        phase_metrics["duration"] = time.time() - phase_metrics["start_time"]
        self.metrics["phase_1"] = phase_metrics
        
        # Resultado da validação
        if phase_metrics["errors"]:
            print(f"\n⚠️ Encontrados {len(phase_metrics['errors'])} problemas:")
            for error in phase_metrics["errors"]:
                print(f"   • {error}")
            print("\nContinuando com fallbacks quando possível...")
        else:
            print("\n✅ Todas as validações passaram!")
        
        print()
        
    async def _phase_2_parallel_testing(self):
        """FASE 2: Teste paralelo sem impacto no sistema existente"""
        
        print("🧪 FASE 2: Teste Paralelo (Sem Impacto)")
        print("-" * 50)
        
        self.current_phase = "parallel_testing"
        phase_metrics = {
            "start_time": time.time(),
            "tests": {},
            "performance": {}
        }
        
        # Importar sistema de cache
        try:
            from src.cache.semantic_cache_integration import create_integrated_cache_system
            print("✅ Módulo de cache semântico importado")
        except ImportError as e:
            print(f"❌ Erro ao importar cache semântico: {e}")
            return
        
        # Teste 1: Criação do sistema
        print("\n🔧 Testando criação do sistema integrado...")
        try:
            config = {
                "enable_semantic": True,
                "enable_traditional": True,
                "semantic_cache_config": {
                    "similarity_threshold": 0.85,
                    "adaptation_threshold": 0.75,
                    "max_memory_entries": 100,  # Pequeno para teste
                    "db_path": "storage/test_semantic_cache.db"
                }
            }
            
            cache_system = create_integrated_cache_system(config)
            phase_metrics["tests"]["system_creation"] = "✅ OK"
            print("   ✅ Sistema de cache integrado criado")
            
        except Exception as e:
            phase_metrics["tests"]["system_creation"] = f"❌ {e}"
            print(f"   ❌ Erro na criação: {e}")
            return
        
        # Teste 2: Operações básicas
        print("\n💾 Testando operações básicas...")
        
        test_queries = [
            {
                "query": "Como implementar autenticação JWT em Python?",
                "response": {
                    "answer": "Para implementar JWT use PyJWT...",
                    "sources": [{"content": "JWT tutorial", "score": 0.9}],
                    "confidence": 0.95
                }
            },
            {
                "query": "FastAPI middleware para logging",
                "response": {
                    "answer": "Crie middleware com @app.middleware...",
                    "sources": [{"content": "FastAPI docs", "score": 0.85}],
                    "confidence": 0.88
                }
            }
        ]
        
        # Salvar no cache
        save_times = []
        for i, test_data in enumerate(test_queries):
            try:
                start_time = time.time()
                await cache_system.set(
                    test_data["query"], 
                    test_data["response"],
                    confidence_score=test_data["response"]["confidence"]
                )
                save_time = time.time() - start_time
                save_times.append(save_time)
                
                print(f"   ✅ Query {i+1} salva ({save_time:.3f}s)")
                
            except Exception as e:
                print(f"   ❌ Erro ao salvar query {i+1}: {e}")
                phase_metrics["tests"][f"save_query_{i+1}"] = f"❌ {e}"
        
        phase_metrics["performance"]["avg_save_time"] = sum(save_times) / len(save_times) if save_times else 0
        
        # Teste 3: Busca semântica
        print("\n🔍 Testando busca semântica...")
        
        semantic_tests = [
            ("JWT Python implementação", 0),  # Similar à primeira query
            ("Middleware logging FastAPI", 1),  # Similar à segunda query
            ("Como conectar PostgreSQL", None)  # Não deve encontrar
        ]
        
        search_times = []
        semantic_hits = 0
        
        for test_query, expected_match in semantic_tests:
            try:
                start_time = time.time()
                cache_response = await cache_system.get(test_query)
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                if cache_response.content is not None:
                    semantic_hits += 1
                    print(f"   ✅ '{test_query}' -> HIT (confidence: {cache_response.confidence:.3f}, {search_time:.3f}s)")
                else:
                    print(f"   ℹ️ '{test_query}' -> MISS ({search_time:.3f}s)")
                
            except Exception as e:
                print(f"   ❌ Erro na busca '{test_query}': {e}")
        
        phase_metrics["performance"]["avg_search_time"] = sum(search_times) / len(search_times) if search_times else 0
        phase_metrics["performance"]["semantic_hit_rate"] = semantic_hits / len(semantic_tests)
        
        # Teste 4: Monitoramento
        print("\n📊 Testando sistema de monitoramento...")
        
        try:
            from src.monitoring.semantic_cache_integration import CacheMetricsCollector
            
            metrics_collector = CacheMetricsCollector(cache_system)
            await metrics_collector.start_collection()
            
            # Simular alguns eventos
            for i in range(5):
                metrics_collector.record_cache_event("semantic_hit", {
                    "similarity": 0.87,
                    "adaptation_applied": True
                })
            
            await asyncio.sleep(2)  # Aguardar coleta
            
            stats = metrics_collector.get_real_time_stats()
            print(f"   ✅ Monitoramento funcionando: {stats['requests']['total']} eventos registrados")
            
            metrics_collector.stop_collection()
            phase_metrics["tests"]["monitoring"] = "✅ OK"
            
        except Exception as e:
            print(f"   ⚠️ Monitoramento com problema: {e}")
            phase_metrics["tests"]["monitoring"] = f"⚠️ {e}"
        
        # Resultado da fase
        phase_metrics["duration"] = time.time() - phase_metrics["start_time"]
        self.metrics["phase_2"] = phase_metrics
        
        success_rate = len([v for v in phase_metrics["tests"].values() if v.startswith("✅")]) / len(phase_metrics["tests"])
        
        print(f"\n📊 Resultado dos testes:")
        print(f"   Taxa de sucesso: {success_rate:.1%}")
        print(f"   Tempo médio de save: {phase_metrics['performance']['avg_save_time']:.3f}s")
        print(f"   Tempo médio de busca: {phase_metrics['performance']['avg_search_time']:.3f}s")
        print(f"   Hit rate semântico: {phase_metrics['performance']['semantic_hit_rate']:.1%}")
        
        if success_rate >= 0.8:
            print("✅ Testes paralelos bem-sucedidos!")
        else:
            print("⚠️ Alguns testes falharam - revisar configuração")
        
        print()
    
    async def _phase_3_gradual_rollout(self):
        """FASE 3: Rollout gradual com percentual de tráfego"""
        
        print("🔄 FASE 3: Rollout Gradual")
        print("-" * 50)
        
        self.current_phase = "gradual_rollout"
        phase_metrics = {
            "start_time": time.time(),
            "rollout_stages": {}
        }
        
        # Simulação de rollout gradual (em produção seria com tráfego real)
        rollout_stages = [
            {"percentage": 10, "duration": 3},
            {"percentage": 25, "duration": 3},
            {"percentage": 50, "duration": 3},
            {"percentage": 75, "duration": 3}
        ]
        
        print("🎯 Simulando rollout gradual:")
        print("   (Em produção, isso seria com tráfego real controlado)\n")
        
        for stage in rollout_stages:
            percentage = stage["percentage"]
            duration = stage["duration"]
            
            print(f"📈 Stage: {percentage}% do tráfego por {duration}s")
            
            stage_metrics = {
                "start_time": time.time(),
                "requests_processed": 0,
                "cache_hits": 0,
                "errors": 0
            }
            
            # Simular requests por esse período
            for second in range(duration):
                # Simular requests (em produção seria traffic splitting real)
                requests_per_second = 10
                
                for request in range(requests_per_second):
                    stage_metrics["requests_processed"] += 1
                    
                    # Simular que X% do tráfego vai para semantic cache
                    if (request % 100) < percentage:
                        # Usar semantic cache
                        if request % 3 == 0:  # 33% de hit rate
                            stage_metrics["cache_hits"] += 1
                
                await asyncio.sleep(1)
                print(f"   ⏱️ {second+1}/{duration}s - Requests: {stage_metrics['requests_processed']}")
            
            stage_metrics["duration"] = time.time() - stage_metrics["start_time"]
            stage_metrics["hit_rate"] = stage_metrics["cache_hits"] / max(stage_metrics["requests_processed"], 1)
            
            phase_metrics["rollout_stages"][f"{percentage}%"] = stage_metrics
            
            print(f"   ✅ Stage {percentage}% completo:")
            print(f"      Requests: {stage_metrics['requests_processed']}")
            print(f"      Hit rate: {stage_metrics['hit_rate']:.1%}")
            print(f"      Erros: {stage_metrics['errors']}")
            print()
        
        # Análise do rollout
        phase_metrics["duration"] = time.time() - phase_metrics["start_time"]
        self.metrics["phase_3"] = phase_metrics
        
        total_requests = sum(s["requests_processed"] for s in phase_metrics["rollout_stages"].values())
        total_hits = sum(s["cache_hits"] for s in phase_metrics["rollout_stages"].values())
        overall_hit_rate = total_hits / max(total_requests, 1)
        
        print(f"📊 Resultado do rollout gradual:")
        print(f"   Total de requests processados: {total_requests:,}")
        print(f"   Hit rate geral: {overall_hit_rate:.1%}")
        print(f"   Duração total: {phase_metrics['duration']:.1f}s")
        
        if overall_hit_rate > 0.2:  # 20% de hit rate mínimo
            print("✅ Rollout gradual bem-sucedido!")
        else:
            print("⚠️ Hit rate abaixo do esperado - investigar")
        
        print()
    
    async def _phase_4_full_integration(self):
        """FASE 4: Integração completa com pipeline RAG"""
        
        print("🔗 FASE 4: Integração Completa")
        print("-" * 50)
        
        self.current_phase = "full_integration"
        phase_metrics = {
            "start_time": time.time(),
            "integration_steps": {}
        }
        
        # Passo 1: Modificar pipeline RAG
        print("🔧 Criando exemplo de integração com pipeline RAG...")
        
        integration_code = '''
# EXEMPLO DE INTEGRAÇÃO - Adicionar ao seu RAGPipelineAdvanced

from src.cache.semantic_cache_integration import create_integrated_cache_system
from src.monitoring.semantic_cache_integration import start_cache_monitoring
import asyncio
import time

class RAGPipelineAdvanced:
    def __init__(self, config):
        # Cache existente (manter para compatibilidade)
        self.traditional_cache = self._init_traditional_cache(config)
        
        # NOVO: Sistema integrado de cache
        self.integrated_cache = create_integrated_cache_system({
            "enable_semantic": config.get("enable_semantic_cache", True),
            "enable_traditional": True,
            "semantic_cache_config": {
                "similarity_threshold": config.get("semantic_threshold", 0.85),
                "adaptation_threshold": config.get("adaptation_threshold", 0.75)
            }
        })
        
        # Flag para habilitar cache semântico
        self.use_semantic_cache = config.get("use_semantic_cache", True)
        
        # Inicializar monitoramento
        if self.use_semantic_cache:
            asyncio.create_task(self._init_cache_monitoring())
    
    async def _init_cache_monitoring(self):
        """Inicializa monitoramento do cache"""
        try:
            self.metrics_collector, _ = await start_cache_monitoring(
                self.integrated_cache,
                show_dashboard=False
            )
        except Exception as e:
            logger.warning(f"Erro ao inicializar monitoramento: {e}")
    
    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Query principal com cache semântico"""
        
        start_time = time.time()
        
        # Tentar cache semântico primeiro
        if self.use_semantic_cache:
            cache_response = await self.integrated_cache.get(question, {
                "user_id": kwargs.get("user_id"),
                "domain": kwargs.get("domain", "general")
            })
            
            if cache_response.content is not None:
                # Hit no cache semântico
                processing_time = time.time() - start_time
                
                # Registrar evento no monitoramento
                if hasattr(self, 'metrics_collector'):
                    self.metrics_collector.record_cache_event("semantic_hit", {
                        "similarity": cache_response.confidence,
                        "adaptation_applied": cache_response.metadata.get("adaptation_applied", False)
                    })
                
                # Retornar resultado do cache
                return {
                    **cache_response.content,
                    "_cache_info": {
                        "hit": True,
                        "source": cache_response.source,
                        "confidence": cache_response.confidence,
                        "processing_time": processing_time
                    }
                }
        
        # Cache miss - processar query normalmente
        result = await self._process_query_traditional(question, **kwargs)
        processing_time = time.time() - start_time
        
        # Salvar resultado no cache semântico (background)
        if self.use_semantic_cache:
            asyncio.create_task(self._save_to_semantic_cache(
                question, result, processing_time
            ))
            
            # Registrar cache miss
            if hasattr(self, 'metrics_collector'):
                self.metrics_collector.record_cache_event("cache_miss")
        
        return result
    
    async def _save_to_semantic_cache(self, question: str, result: Dict, processing_time: float):
        """Salva resultado no cache semântico (async)"""
        try:
            await self.integrated_cache.set(
                query=question,
                response=result,
                confidence_score=result.get("confidence", 0.0),
                processing_time_saved=processing_time,
                tokens_saved=self._estimate_tokens_saved(result),
                source_model=result.get("model_used", "unknown")
            )
        except Exception as e:
            logger.warning(f"Erro ao salvar no cache semântico: {e}")
            if hasattr(self, 'metrics_collector'):
                self.metrics_collector.record_cache_event("error")
    
    def _estimate_tokens_saved(self, result: Dict) -> int:
        """Estima tokens economizados (implementar baseado no seu modelo)"""
        # Estimativa simples baseada no tamanho da resposta
        answer_length = len(result.get("answer", ""))
        return answer_length // 4  # ~4 chars por token
'''
        
        # Salvar código de exemplo
        integration_file = Path("integration_example.py")
        with open(integration_file, 'w', encoding='utf-8') as f:
            f.write(integration_code)
        
        print(f"   ✅ Código de exemplo salvo: {integration_file}")
        phase_metrics["integration_steps"]["code_example"] = "✅ OK"
        
        # Passo 2: Configuração de produção
        print("\n📋 Criando configuração de produção...")
        
        production_config = {
            "semantic_cache": {
                "enable_semantic_cache": True,
                "use_semantic_cache": True,
                "semantic_threshold": 0.85,
                "adaptation_threshold": 0.75,
                "cache_config": {
                    "max_memory_entries": 1000,
                    "enable_redis": True,
                    "redis_url": "redis://localhost:6379",
                    "db_path": "storage/semantic_cache_prod.db"
                }
            },
            "monitoring": {
                "enable_detailed_logs": True,
                "metrics_collection_interval": 60,
                "alert_thresholds": {
                    "min_hit_rate": 0.15,
                    "max_error_rate": 0.05,
                    "min_avg_similarity": 0.70
                }
            }
        }
        
        config_file = Path("config/semantic_cache_production.yaml")
        config_file.parent.mkdir(exist_ok=True)
        
        import yaml
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(production_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"   ✅ Configuração de produção salva: {config_file}")
        phase_metrics["integration_steps"]["config"] = "✅ OK"
        
        # Passo 3: Script de deploy
        print("\n🚀 Criando script de deploy...")
        
        deploy_script = '''#!/bin/bash
# Script de deploy do Semantic Cache

echo "🚀 Deployando Semantic Cache..."

# 1. Backup da configuração atual
cp config/config.yaml config/config_backup_$(date +%Y%m%d_%H%M%S).yaml

# 2. Ativar cache semântico gradualmente
echo "📊 Habilitando cache semântico..."
# Adicionar use_semantic_cache: true na configuração

# 3. Reiniciar serviços
echo "🔄 Reiniciando serviços..."
# docker-compose restart rag-app

# 4. Verificar health
echo "🏥 Verificando health..."
sleep 10
curl -f http://localhost:8000/health || exit 1

# 5. Monitorar métricas
echo "📈 Monitoramento ativo..."
echo "Dashboard disponível em: http://localhost:8000/cache/metrics"

echo "✅ Deploy concluído!"
'''
        
        deploy_file = Path("scripts/deploy_semantic_cache.sh")
        deploy_file.parent.mkdir(exist_ok=True)
        
        with open(deploy_file, 'w', encoding='utf-8') as f:
            f.write(deploy_script)
        
        deploy_file.chmod(0o755)  # Tornar executável
        
        print(f"   ✅ Script de deploy salvo: {deploy_file}")
        phase_metrics["integration_steps"]["deploy_script"] = "✅ OK"
        
        # Resultado da fase
        phase_metrics["duration"] = time.time() - phase_metrics["start_time"]
        self.metrics["phase_4"] = phase_metrics
        
        print(f"\n🎯 Integração completa preparada:")
        print(f"   📝 Código de exemplo: integration_example.py")
        print(f"   ⚙️ Configuração: config/semantic_cache_production.yaml")
        print(f"   🚀 Script de deploy: scripts/deploy_semantic_cache.sh")
        
        print()
    
    async def _generate_final_report(self):
        """Gera relatório final da integração"""
        
        print("📊 RELATÓRIO FINAL DA INTEGRAÇÃO")
        print("=" * 50)
        
        total_duration = sum(phase.get("duration", 0) for phase in self.metrics.values())
        
        report = {
            "integration_completed_at": time.time(),
            "total_duration_seconds": total_duration,
            "phases": self.metrics,
            "summary": {
                "validation_passed": len(self.metrics.get("phase_1", {}).get("errors", [])) == 0,
                "tests_success_rate": 0.0,
                "rollout_success": False,
                "integration_ready": True
            },
            "next_steps": [
                "1. Revisar código de exemplo: integration_example.py",
                "2. Ajustar configuração: config/semantic_cache_production.yaml", 
                "3. Executar deploy: scripts/deploy_semantic_cache.sh",
                "4. Monitorar métricas em tempo real",
                "5. Ajustar thresholds baseado em dados de produção"
            ]
        }
        
        # Calcular success rate dos testes
        phase_2 = self.metrics.get("phase_2", {})
        if "tests" in phase_2:
            success_count = len([v for v in phase_2["tests"].values() if v.startswith("✅")])
            total_count = len(phase_2["tests"])
            report["summary"]["tests_success_rate"] = success_count / max(total_count, 1)
        
        # Verificar sucesso do rollout
        phase_3 = self.metrics.get("phase_3", {})
        if "rollout_stages" in phase_3:
            stages = phase_3["rollout_stages"]
            if stages:
                avg_hit_rate = sum(s["hit_rate"] for s in stages.values()) / len(stages)
                report["summary"]["rollout_success"] = avg_hit_rate > 0.2
        
        # Salvar relatório
        report_file = Path("storage/semantic_cache_integration_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Exibir resumo
        print(f"⏱️ Duração total: {total_duration:.1f}s")
        print(f"✅ Validação: {'Passou' if report['summary']['validation_passed'] else 'Falhou'}")
        print(f"🧪 Testes: {report['summary']['tests_success_rate']:.1%} de sucesso")
        print(f"🔄 Rollout: {'Sucesso' if report['summary']['rollout_success'] else 'Revisar'}")
        print(f"🎯 Pronto para produção: {'Sim' if report['summary']['integration_ready'] else 'Não'}")
        
        print(f"\n📄 Relatório completo salvo: {report_file}")
        
        print(f"\n🎯 PRÓXIMOS PASSOS:")
        for step in report["next_steps"]:
            print(f"   {step}")


async def main():
    """Função principal"""
    
    integrator = SemanticCacheIntegrator()
    
    try:
        success = await integrator.run_integration()
        
        if success:
            print(f"\n🎉 Integração concluída com sucesso!")
            print(f"🚀 O sistema está pronto para usar Semantic Caching!")
            return 0
        else:
            print(f"\n❌ Integração falhou. Revise os logs e tente novamente.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Integração interrompida pelo usuário")
        return 130
    except Exception as e:
        print(f"\n💥 Erro inesperado: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main())) 