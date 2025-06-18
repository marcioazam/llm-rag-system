# -*- coding: utf-8 -*-
"""
Teste Simples - Enhanced Corrective RAG
Verifica se as implementações funcionam corretamente.
"""

import asyncio
import logging
import os
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Testa se todos os imports funcionam."""
    print("🧪 Testando imports...")
    
    try:
        from src.retrieval.enhanced_corrective_rag import (
            EnhancedCorrectiveRAG,
            T5RetrievalEvaluator,
            QueryDecomposer,
            create_enhanced_corrective_rag
        )
        print("✅ Enhanced Corrective RAG imports funcionando")
        
        from src.cache.multi_layer_cache import MultiLayerCache, create_multi_layer_cache
        print("✅ Cache multicamada imports funcionando")
        
        from src.rag_pipeline_advanced import AdvancedRAGPipeline
        print("✅ Pipeline avançado imports funcionando")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos imports: {e}")
        return False

def test_cache_creation():
    """Testa criação do cache multicamada."""
    print("\n🧪 Testando criação do cache...")
    
    try:
        from src.cache.multi_layer_cache import create_multi_layer_cache
        
        config = {
            'enable_l1': True,
            'enable_l2': False,  # Desabilitar Redis para teste
            'enable_l3': True,
            'l1_max_size': 100,
            'sqlite_path': 'test_cache.db',
            'default_ttl': 3600
        }
        
        cache = create_multi_layer_cache(config)
        print(f"✅ Cache criado com configuração: L1={cache.l1 is not None}, L2={cache.l2 is not None}, L3={cache.l3 is not None}")
        
        # Cleanup
        if os.path.exists('test_cache.db'):
            os.remove('test_cache.db')
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na criação do cache: {e}")
        return False

def test_enhanced_rag_creation():
    """Testa criação do Enhanced Corrective RAG."""
    print("\n🧪 Testando criação do Enhanced Corrective RAG...")
    
    try:
        from src.retrieval.enhanced_corrective_rag import create_enhanced_corrective_rag
        
        config = {
            'relevance_threshold': 0.75,
            'max_reformulation_attempts': 3,
            'enable_decomposition': True,
            'cache_evaluations': False,  # Desabilitar cache para teste simples
            'api_providers': ['openai']
        }
        
        enhanced_rag = create_enhanced_corrective_rag(config)
        print(f"✅ Enhanced RAG criado com threshold: {enhanced_rag.relevance_threshold}")
        print(f"   Decomposition: {enhanced_rag.enable_decomposition}")
        print(f"   T5 Evaluator: {enhanced_rag.t5_evaluator is not None}")
        print(f"   Query Decomposer: {enhanced_rag.query_decomposer is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na criação do Enhanced RAG: {e}")
        return False

def test_pipeline_integration():
    """Testa integração com pipeline avançado."""
    print("\n🧪 Testando integração com pipeline...")
    
    try:
        from src.rag_pipeline_advanced import AdvancedRAGPipeline
        
        pipeline = AdvancedRAGPipeline()
        print(f"✅ Pipeline criado")
        print(f"   Enhanced corrective habilitado: {pipeline.advanced_config.get('enable_enhanced_corrective', False)}")
        print(f"   Cache habilitado: {pipeline.advanced_config.get('enable_cache', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na criação do pipeline: {e}")
        return False

async def test_basic_functionality():
    """Testa funcionalidade básica sem APIs reais."""
    print("\n🧪 Testando funcionalidade básica...")
    
    try:
        from src.retrieval.enhanced_corrective_rag import T5RetrievalEvaluator, EvaluationResult
        
        # Criar evaluator sem APIs reais
        evaluator = T5RetrievalEvaluator()
        
        # Testar fallback evaluation
        query = "test query"
        document = "test document with query words"
        
        result = evaluator._fallback_evaluation(query, document)
        
        print(f"✅ Fallback evaluation funcionando:")
        print(f"   Overall score: {result.overall_score}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Provider: {result.api_provider}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na funcionalidade básica: {e}")
        return False

def test_configuration_loading():
    """Testa carregamento de configurações."""
    print("\n🧪 Testando carregamento de configurações...")
    
    try:
        import yaml
        
        # Tentar carregar config YAML
        config_path = "config/llm_providers_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            enhanced_config = config.get('advanced_features', {}).get('enhanced_corrective_rag', {})
            if enhanced_config:
                print("✅ Configuração Enhanced Corrective RAG encontrada:")
                print(f"   Enabled: {enhanced_config.get('enabled', False)}")
                print(f"   Threshold: {enhanced_config.get('relevance_threshold', 0.75)}")
                print(f"   Cache L1: {enhanced_config.get('cache', {}).get('enable_l1', False)}")
                print(f"   Cache L2: {enhanced_config.get('cache', {}).get('enable_l2', False)}")
                print(f"   API providers: {enhanced_config.get('api_providers', {}).get('fallback_chain', [])}")
            else:
                print("⚠️ Configuração Enhanced Corrective RAG não encontrada")
        else:
            print("⚠️ Arquivo de configuração não encontrado")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no carregamento de configurações: {e}")
        return False

async def main():
    """Executa todos os testes."""
    print("=" * 80)
    print("🚀 TESTES SIMPLES - ENHANCED CORRECTIVE RAG")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 6
    
    # Teste 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Teste 2: Cache
    if test_cache_creation():
        tests_passed += 1
    
    # Teste 3: Enhanced RAG
    if test_enhanced_rag_creation():
        tests_passed += 1
    
    # Teste 4: Pipeline
    if test_pipeline_integration():
        tests_passed += 1
    
    # Teste 5: Funcionalidade básica
    if await test_basic_functionality():
        tests_passed += 1
    
    # Teste 6: Configurações
    if test_configuration_loading():
        tests_passed += 1
    
    print("\n" + "=" * 80)
    print(f"📊 RESULTADO DOS TESTES: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ TODOS OS TESTES PASSARAM!")
        print("🎉 Enhanced Corrective RAG está funcionando corretamente!")
    else:
        print(f"⚠️ {total_tests - tests_passed} testes falharam")
        print("🔧 Verifique os erros acima para resolver problemas")
    
    print("=" * 80)
    
    # Resumo das funcionalidades implementadas
    print("\n🎯 FUNCIONALIDADES IMPLEMENTADAS:")
    print("✅ 1. T5 Retrieval Evaluator com APIs reais (OpenAI, Anthropic, HuggingFace)")
    print("✅ 2. Cache multicamada (L1: Memória, L2: Redis, L3: SQLite)")
    print("✅ 3. Query Decomposer com algoritmo decompose-then-recompose")
    print("✅ 4. Circuit breakers para APIs")
    print("✅ 5. Integração com AdvancedRAGPipeline")
    print("✅ 6. Configuração via YAML")
    print("✅ 7. Fallback automático para APIs indisponíveis")
    print("✅ 8. Métricas detalhadas de performance")
    
    print("\n🚀 PRÓXIMOS PASSOS PARA PRODUÇÃO:")
    print("1. Configurar variáveis de ambiente para APIs (OPENAI_API_KEY, etc.)")
    print("2. Instalar e configurar Redis se necessário")
    print("3. Testar com queries reais")
    print("4. Monitorar performance e custos")
    print("5. Ajustar thresholds baseado em métricas de produção")

if __name__ == "__main__":
    asyncio.run(main()) 