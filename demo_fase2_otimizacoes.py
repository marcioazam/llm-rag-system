#!/usr/bin/env python3
"""
Demonstração - Fase 2 Otimizações
Sistema Unificado de Prompts + Enhanced Tree-sitter Integration
"""

import asyncio
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def demonstrate_unified_prompt_system():
    """Demonstra o sistema unificado de prompts em ação."""
    logger.info("🎯 === DEMONSTRAÇÃO: Sistema Unificado de Prompts ===")
    
    try:
        from src.augmentation.unified_prompt_system import UnifiedPromptSystem
        
        prompt_system = UnifiedPromptSystem()
        
        # Cenários de demonstração
        scenarios = [
            {
                "name": "🐛 Debugging de Código",
                "query": "Este código Python está com erro de IndexError quando processa uma lista vazia",
                "context": ["def process_items(items):", "    first_item = items[0]", "    return first_item.upper()"]
            },
            {
                "name": "👀 Code Review", 
                "query": "Revisar esta implementação de cache com foco em thread safety",
                "context": ["class MemoryCache:", "    def __init__(self):", "        self.cache = {}"]
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            logger.info(f"\n{scenario['name']}")
            logger.info("─" * 40)
            
            # Gerar prompt otimizado
            result = await prompt_system.generate_optimal_prompt(
                query=scenario["query"],
                context_chunks=scenario["context"],
                language="Português",
                depth="quick"
            )
            
            # Log dos resultados
            logger.info(f"🎯 Tipo detectado: {result.task_type}")
            logger.info(f"📋 Template usado: {result.template_id}")
            logger.info(f"⭐ Confiança: {result.confidence:.2f}")
            logger.info(f"📏 Tamanho: {len(result.final_prompt)} chars")
            
            results.append({
                "scenario": scenario["name"],
                "task_type": result.task_type,
                "confidence": result.confidence,
                "prompt_length": len(result.final_prompt)
            })
        
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        logger.info(f"\n📊 Confiança média: {avg_confidence:.2f}")
        
        return {
            "scenarios_tested": len(results),
            "avg_confidence": avg_confidence,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        return None

async def demonstrate_integration():
    """Demonstra a integração dos componentes no pipeline."""
    logger.info("\n🔗 === DEMONSTRAÇÃO: Integração com Pipeline ===")
    
    try:
        from src.rag_pipeline_advanced import AdvancedRAGPipeline
        
        pipeline = AdvancedRAGPipeline()
        
        if hasattr(pipeline, 'prompt_system'):
            logger.info("✅ Sistema Unificado de Prompts integrado")
            
            test_query = "Como otimizar esta query SQL lenta?"
            suggestions = pipeline.prompt_system.get_task_suggestions(test_query)
            
            logger.info(f"🎯 Query teste: {test_query}")
            logger.info(f"📋 Tipo detectado: {suggestions['task_type']}")
            logger.info(f"🌡️ Temperature: {suggestions['suggested_temperature']}")
            
            return {"integrated": True, "suggestions": suggestions}
        else:
            logger.warning("⚠️ Sistema não integrado")
            return {"integrated": False}
            
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        return None

async def main():
    """Função principal da demonstração."""
    start_time = datetime.now()
    
    logger.info("🚀 === DEMONSTRAÇÃO FASE 2 - OTIMIZAÇÕES ===")
    logger.info("=" * 60)
    
    results = {}
    
    # Demonstração 1: Sistema de Prompts
    prompt_results = await demonstrate_unified_prompt_system()
    results["prompt_system"] = prompt_results
    
    # Demonstração 2: Integração
    integration_results = await demonstrate_integration()
    results["integration"] = integration_results
    
    # Resumo final
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n🎯 === RESUMO FINAL ===")
    logger.info("=" * 40)
    
    success_count = 0
    
    if results.get("prompt_system"):
        logger.info("✅ Sistema Unificado de Prompts: OK")
        success_count += 1
    else:
        logger.error("❌ Sistema de Prompts: FALHOU")
    
    if results.get("integration", {}).get("integrated"):
        logger.info("✅ Integração com Pipeline: OK")
        success_count += 1
    else:
        logger.error("❌ Integração: FALHOU")
    
    success_rate = success_count / 2
    
    logger.info(f"📊 Taxa de sucesso: {success_rate:.1%}")
    logger.info(f"⏱️ Duração: {duration:.1f}s")
    
    # Salvar relatório
    report = {
        "timestamp": start_time.isoformat(),
        "duration": duration,
        "success_rate": success_rate,
        "results": results
    }
    
    with open("fase2_demo_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info("📄 Relatório salvo: fase2_demo_report.json")
    
    if success_rate >= 0.5:
        logger.info("🎉 FASE 2 IMPLEMENTADA COM SUCESSO!")
        return 0
    else:
        logger.error("💥 FASE 2 PRECISA DE AJUSTES")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)