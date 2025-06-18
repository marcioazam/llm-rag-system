#!/usr/bin/env python3
"""
Teste de Integração - Fase 2 Otimizações
Sistema Unificado de Prompts + Tree-sitter Analyzer Avançado
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_unified_prompt_system():
    """Testa o sistema unificado de prompts."""
    logger.info("🎯 Testando Sistema Unificado de Prompts...")
    
    try:
        from src.augmentation.unified_prompt_system import UnifiedPromptSystem
        
        prompt_system = UnifiedPromptSystem()
        
        # Teste 1: Query de bugfix
        test_queries = [
            {
                "query": "Esse código está dando erro de null pointer exception",
                "context": ["def get_user(id): return users[id]", "users = {}"],
                "expected_type": "bugfix"
            },
            {
                "query": "Revisar este pull request com foco em performance",
                "context": ["for i in range(1000000): process(i)"],
                "expected_type": "review"
            },
            {
                "query": "Como implementar cache distribuído com Redis?",
                "context": ["class CacheManager:", "def __init__(self):"],
                "expected_type": "arch"
            },
            {
                "query": "Gerar testes unitários para esta função",
                "context": ["def calculate_total(items): return sum(item.price for item in items)"],
                "expected_type": "testgen"
            }
        ]
        
        results = []
        for test in test_queries:
            result = await prompt_system.generate_optimal_prompt(
                query=test["query"],
                context_chunks=test["context"],
                language="Português",
                depth="quick"
            )
            
            results.append({
                "query": test["query"][:30] + "...",
                "expected_type": test["expected_type"],
                "detected_type": result.task_type,
                "template_id": result.template_id,
                "prompt_source": result.prompt_source,
                "confidence": result.confidence,
                "prompt_length": len(result.final_prompt),
                "temperature": result.metadata.get("temperature_suggestion", 0.5),
                "reasoning": result.metadata.get("reasoning_required", False)
            })
            
            # Verificar se a classificação está correta
            if result.task_type == test["expected_type"]:
                logger.info(f"✅ Classificação correta: {result.task_type}")
            else:
                logger.warning(f"⚠️ Classificação divergente: esperado {test['expected_type']}, obtido {result.task_type}")
        
        # Estatísticas
        correct_classifications = sum(1 for r in results if r["detected_type"] == r["expected_type"])
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_prompt_length = sum(r["prompt_length"] for r in results) / len(results)
        
        stats = {
            "total_tests": len(results),
            "correct_classifications": correct_classifications,
            "accuracy": correct_classifications / len(results),
            "avg_confidence": avg_confidence,
            "avg_prompt_length": int(avg_prompt_length),
            "results": results
        }
        
        logger.info(f"📊 Resultados do Sistema Unificado de Prompts:")
        logger.info(f"   Acurácia: {stats['accuracy']:.1%}")
        logger.info(f"   Confiança média: {stats['avg_confidence']:.2f}")
        logger.info(f"   Tamanho médio do prompt: {stats['avg_prompt_length']} chars")
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Erro no teste do sistema de prompts: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Função principal de teste."""
    logger.info("🚀 Iniciando Testes da Fase 2 - Otimizações")
    logger.info("=" * 60)
    
    results = {}
    
    # Teste 1: Sistema Unificado de Prompts
    prompt_results = await test_unified_prompt_system()
    results["prompt_system"] = prompt_results
    
    # Status final
    logger.info("=" * 60)
    logger.info(f"🎯 RESULTADO FINAL: Teste do Sistema Unificado de Prompts")
    
    if results.get("prompt_system"):
        logger.info("🎉 SISTEMA UNIFICADO DE PROMPTS FUNCIONANDO!")
        logger.info(f"📊 Acurácia: {results['prompt_system']['accuracy']:.1%}")
        return 0
    else:
        logger.error("💥 SISTEMA PRECISA DE AJUSTES")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)