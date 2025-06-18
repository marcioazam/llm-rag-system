#!/usr/bin/env python3
"""
Script Simplificado de Teste: HyDE + RAGAS
Testa as funcionalidades implementadas de forma básica
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path

# Configurar path do Python
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Testa se as importações funcionam"""
    print("🔍 Testando importações...")
    
    try:
        # Testar importação HyDE
        from src.retrieval.hyde_enhancer import HyDEEnhancer, HyDEResult
        print("✅ HyDE Enhancer importado com sucesso")
        
        # Testar importação RAGAS
        from src.monitoring.rag_evaluator import RAGEvaluator, RAGTestCase, RAGEvaluationResult
        print("✅ RAG Evaluator importado com sucesso")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False

def test_configurations():
    """Testa se as configurações estão corretas"""
    print("\n⚙️ Testando configurações...")
    
    try:
        import yaml
        
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Arquivo de configuração não encontrado: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Verificar seção HyDE
        hyde_config = config.get("hyde")
        if hyde_config:
            print(f"✅ Configuração HyDE encontrada")
            print(f"   Enabled: {hyde_config.get('enabled')}")
            print(f"   Docs por query: {hyde_config.get('num_hypothetical_docs')}")
        else:
            print("⚠️  Configuração HyDE não encontrada")
        
        # Verificar seção de avaliação
        eval_config = config.get("evaluation")
        if eval_config:
            print(f"✅ Configuração de avaliação encontrada")
            print(f"   Enabled: {eval_config.get('enabled')}")
        else:
            print("⚠️  Configuração de avaliação não encontrada")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar configurações: {e}")
        return False

async def test_hyde_basic():
    """Teste básico do HyDE"""
    print("\n🔍 Testando HyDE (básico)...")
    
    try:
        from src.retrieval.hyde_enhancer import HyDEEnhancer
        
        # Criar instância
        hyde = HyDEEnhancer()
        print("✅ HyDE Enhancer inicializado")
        
        # Testar configuração
        config = hyde.hyde_config
        print(f"📄 Docs por query: {config['num_hypothetical_docs']}")
        print(f"🌡️  Temperatura: {config['temperature']}")
        
        # Testar métrica sem processamento
        metrics = hyde.get_metrics()
        print(f"📊 Métricas iniciais: cache_size={metrics['cache_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste HyDE básico: {e}")
        return False

async def test_ragas_basic():
    """Teste básico do RAGAS"""
    print("\n📊 Testando RAGAS (básico)...")
    
    try:
        from src.monitoring.rag_evaluator import RAGEvaluator, RAGTestCase
        
        # Criar instância
        evaluator = RAGEvaluator()
        print("✅ RAG Evaluator inicializado")
        
        # Testar configuração
        config = evaluator.eval_config
        print(f"🌡️  Temperatura: {config['temperature']}")
        print(f"📊 Métricas habilitadas: {list(config['metrics'].keys())}")
        
        # Teste simples com fallback
        test_case = RAGTestCase(
            question="Teste simples",
            contexts=["Contexto de teste"],
            answer="Resposta de teste"
        )
        
        print("🧪 Executando teste básico...")
        result = await evaluator.evaluate_single(test_case)
        
        print(f"✅ Teste concluído:")
        print(f"   Faithfulness: {result.faithfulness:.3f}")
        print(f"   Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"   Overall Score: {result.overall_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste RAGAS básico: {e}")
        return False

def test_file_structure():
    """Verifica se os arquivos foram criados corretamente"""
    print("\n📁 Verificando estrutura de arquivos...")
    
    files_to_check = [
        "src/retrieval/hyde_enhancer.py",
        "src/monitoring/rag_evaluator.py",
        "examples/hyde_ragas_example.py",
        "scripts/test_hyde_ragas.py",
        "config/config.yaml"
    ]
    
    all_exist = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - não encontrado")
            all_exist = False
    
    return all_exist

async def main():
    """Função principal"""
    print("🧪 TESTE SIMPLIFICADO: HyDE + RAGAS")
    print("="*50)
    
    tests = [
        ("Estrutura de Arquivos", test_file_structure),
        ("Importações", test_imports),
        ("Configurações", test_configurations),
        ("HyDE Básico", test_hyde_basic),
        ("RAGAS Básico", test_ragas_basic)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-'*20} {test_name} {'-'*20}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            results.append((test_name, success))
            
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo
    print(f"\n{'='*50}")
    print("📊 RESUMO DOS TESTES")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name:.<25} {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 IMPLEMENTAÇÃO CONCLUÍDA COM SUCESSO!")
        print("✅ HyDE (Hypothetical Document Embeddings) implementado")
        print("✅ RAGAS (RAG Assessment) implementado") 
        print("✅ Configurações atualizadas")
        print("✅ Exemplos criados")
        
        print("\n📚 Próximos passos:")
        print("1. Execute: python examples/hyde_ragas_example.py")
        print("2. Configure suas API keys no .env")
        print("3. Teste com dados reais")
        
        return 0
    else:
        print(f"\n⚠️  {total-passed} teste(s) falharam")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Teste interrompido")
        sys.exit(1) 