#!/usr/bin/env python3
"""
Script de Teste: HyDE + RAGAS
Valida que as implementações HyDE e RAGAS estão funcionando corretamente
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_hyde_enhancer():
    """Testa o HyDE Enhancer"""
    print("\n🔍 Testando HyDE Enhancer...")
    
    try:
        from src.retrieval.hyde_enhancer import HyDEEnhancer
        
        # Inicializar HyDE
        hyde = HyDEEnhancer()
        
        # Teste básico
        test_query = "O que é machine learning?"
        print(f"Query de teste: {test_query}")
        
        # Processar com HyDE
        start_time = time.time()
        result = await hyde.enhance_query(test_query)
        processing_time = time.time() - start_time
        
        # Verificar resultados
        print(f"✅ HyDE processado em {processing_time:.3f}s")
        print(f"📄 Documentos hipotéticos gerados: {len(result.hypothetical_documents)}")
        print(f"🎯 Confiança: {result.confidence_score:.3f}")
        print(f"🔍 Embedding shape: {result.enhanced_embedding.shape}")
        
        if result.hypothetical_documents:
            print(f"📖 Exemplo de documento hipotético:")
            print(f"   '{result.hypothetical_documents[0][:100]}...'")
        
        # Testar métricas
        metrics = hyde.get_metrics()
        print(f"📊 Métricas HyDE: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste HyDE: {e}")
        return False

async def test_rag_evaluator():
    """Testa o RAG Evaluator"""
    print("\n📊 Testando RAG Evaluator...")
    
    try:
        from src.monitoring.rag_evaluator import RAGEvaluator, RAGTestCase
        
        # Inicializar avaliador
        evaluator = RAGEvaluator()
        
        # Caso de teste simples
        test_case = RAGTestCase(
            question="O que é inteligência artificial?",
            contexts=[
                "Inteligência artificial é uma área da ciência da computação que busca criar sistemas capazes de realizar tarefas que normalmente requerem inteligência humana.",
                "IA pode incluir aprendizado de máquina, processamento de linguagem natural e visão computacional."
            ],
            answer="Inteligência artificial é uma área da ciência da computação focada em criar sistemas que podem realizar tarefas que tradicionalmente requerem inteligência humana, como aprendizado e reconhecimento de padrões.",
            ground_truth="IA é um campo da computação que desenvolve sistemas inteligentes capazes de simular capacidades humanas."
        )
        
        print(f"Caso de teste: {test_case.question}")
        
        # Avaliar
        start_time = time.time()
        result = await evaluator.evaluate_single(test_case)
        evaluation_time = time.time() - start_time
        
        # Verificar resultados
        print(f"✅ Avaliação concluída em {evaluation_time:.3f}s")
        print(f"📈 Faithfulness: {result.faithfulness:.3f}")
        print(f"📈 Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"📈 Context Precision: {result.context_precision:.3f}")
        if result.context_recall:
            print(f"📈 Context Recall: {result.context_recall:.3f}")
        print(f"🎯 Score Geral: {result.overall_score:.3f}")
        
        # Testar resumo
        summary = evaluator.get_evaluation_summary()
        print(f"📊 Resumo: {summary.get('total_evaluations', 0)} avaliações")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste RAGAS: {e}")
        return False

async def test_integration():
    """Testa integração com sistema RAG existente"""
    print("\n🔗 Testando integração com sistema RAG...")
    
    try:
        # Importar componentes do sistema
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        # Testar HybridRetriever com HyDE
        retriever = HybridRetriever()
        
        test_query = "Como funciona deep learning?"
        print(f"Query de teste: {test_query}")
        
        # Testar busca com HyDE
        start_time = time.time()
        results = await retriever.retrieve(
            query=test_query,
            use_hyde=True,
            limit=5
        )
        retrieval_time = time.time() - start_time
        
        print(f"✅ Busca híbrida com HyDE em {retrieval_time:.3f}s")
        print(f"📄 Resultados encontrados: {len(results)}")
        
        # Verificar métricas do retriever
        metrics = retriever.get_metrics()
        hyde_time = metrics.get("hyde_time", 0)
        hyde_success = metrics.get("hyde_success_count", 0)
        
        print(f"⏱️  Tempo HyDE acumulado: {hyde_time:.3f}s")
        print(f"✅ HyDE sucessos: {hyde_success}")
        
        # Verificar metadados HyDE nos resultados
        hyde_enhanced_count = 0
        for result in results:
            if result.metadata.get("hyde_enhanced"):
                hyde_enhanced_count += 1
        
        print(f"🔍 Resultados com HyDE: {hyde_enhanced_count}/{len(results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de integração: {e}")
        return False

async def test_configuration():
    """Testa configurações do sistema"""
    print("\n⚙️ Testando configurações...")
    
    try:
        import yaml
        
        # Verificar config.yaml
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Verificar seções HyDE e avaliação
            hyde_config = config.get("hyde")
            eval_config = config.get("evaluation")
            
            if hyde_config:
                print(f"✅ Configuração HyDE encontrada:")
                print(f"   Enabled: {hyde_config.get('enabled')}")
                print(f"   Docs por query: {hyde_config.get('num_hypothetical_docs')}")
                print(f"   Modelo: {hyde_config.get('model')}")
            else:
                print(f"⚠️  Configuração HyDE não encontrada")
            
            if eval_config:
                print(f"✅ Configuração de avaliação encontrada:")
                print(f"   Enabled: {eval_config.get('enabled')}")
                print(f"   Modelo: {eval_config.get('model')}")
                print(f"   Métricas: {list(eval_config.get('metrics', {}).keys())}")
            else:
                print(f"⚠️  Configuração de avaliação não encontrada")
            
            # Verificar retrieval config
            retrieval_config = config.get("retrieval", {})
            use_hyde = retrieval_config.get("use_hyde", False)
            print(f"🔍 HyDE habilitado no retrieval: {use_hyde}")
            
        else:
            print(f"⚠️  Arquivo config.yaml não encontrado em {config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de configuração: {e}")
        return False

async def main():
    """Função principal de teste"""
    print("🧪 TESTE COMPLETO: HyDE + RAGAS")
    print("="*50)
    
    tests = [
        ("Configurações", test_configuration),
        ("HyDE Enhancer", test_hyde_enhancer),
        ("RAG Evaluator", test_rag_evaluator),
        ("Integração Sistema", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = await test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
                
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")
            results.append((test_name, False))
    
    # Resumo final
    print(f"\n{'='*50}")
    print("📊 RESUMO DOS TESTES")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name:.<30} {status}")
    
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! HyDE + RAGAS implementados com sucesso!")
        return 0
    else:
        print("⚠️  Alguns testes falharam. Verifique os logs acima.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erro fatal: {e}")
        sys.exit(1) 