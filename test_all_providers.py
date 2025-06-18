#!/usr/bin/env python3
"""
Script de demonstração do sistema RAG 100% baseado em APIs
Testa todos os 4 provedores: OpenAI, Claude, Gemini, DeepSeek
"""

import os
import sys
import yaml
import time
from typing import Dict, List
from src.models.api_model_router import APIModelRouter, TaskType

def load_config():
    """Carrega configuração dos provedores"""
    try:
        with open("config/llm_providers_config.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        sys.exit(1)

def check_api_keys() -> List[str]:
    """Verifica quais provedores estão configurados"""
    required_keys = {
        "OPENAI_API_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic",
        "GOOGLE_API_KEY": "google", 
        "DEEPSEEK_API_KEY": "deepseek"
    }
    
    available_providers = []
    for key, provider in required_keys.items():
        if os.getenv(key):
            available_providers.append(provider)
    
    return available_providers

def test_provider_specialties(router: APIModelRouter, available_providers: List[str]):
    """Testa as especialidades de cada provedor"""
    
    # Testes específicos para cada provedor
    test_cases = {
        "openai": {
            "task": TaskType.CODE_GENERATION,
            "query": "Crie uma função Python para calcular fibonacci de forma otimizada",
            "specialty": "Geração de código eficiente"
        },
        "anthropic": {
            "task": TaskType.DOCUMENT_ANALYSIS,
            "query": "Analise este documento técnico e extraia os pontos principais sobre arquitetura de software",
            "context": "Este documento descreve padrões de design para sistemas distribuídos...",
            "specialty": "Análise profunda de documentos"
        },
        "google": {
            "task": TaskType.QUICK_QUERIES,
            "query": "Qual é a diferença entre REST e GraphQL? Resposta rápida.",
            "specialty": "Respostas rápidas e contexto longo"
        },
        "deepseek": {
            "task": TaskType.CODE_REVIEW,
            "query": "Revise este código Python e sugira melhorias de performance",
            "context": "def slow_function(data): return [x*2 for x in data if x > 0]",
            "specialty": "Análise e otimização de código"
        }
    }
    
    print("\n🎯 TESTANDO ESPECIALIDADES DOS PROVEDORES")
    print("=" * 60)
    
    for provider in available_providers:
        if provider in test_cases:
            test = test_cases[provider]
            print(f"\n🤖 {provider.upper()} - {test['specialty']}")
            print("-" * 40)
            print(f"Tarefa: {test['task'].value}")
            print(f"Query: {test['query'][:50]}...")
            
            try:
                # Forçar uso do provedor específico
                available_models = router.get_available_models()
                provider_models = [m for m in available_models['models'].keys() if m.startswith(provider)]
                
                if provider_models:
                    force_model = provider_models[0]  # Usar primeiro modelo do provedor
                    
                    start_time = time.time()
                    response = router.generate_response(
                        query=test['query'],
                        context=test.get('context', ''),
                        task_type=test['task'],
                        force_model=force_model
                    )
                    duration = time.time() - start_time
                    
                    print(f"✅ Sucesso em {duration:.2f}s")
                    print(f"Modelo usado: {response.model}")
                    print(f"Custo: ${response.cost:.4f}")
                    print(f"Resposta: {response.content[:100]}...")
                else:
                    print(f"❌ Nenhum modelo disponível para {provider}")
                    
            except Exception as e:
                print(f"❌ Erro: {str(e)}")

def demonstrate_intelligent_routing(router: APIModelRouter):
    """Demonstra o roteamento inteligente de tarefas"""
    
    print("\n🧠 DEMONSTRAÇÃO DO ROTEAMENTO INTELIGENTE")
    print("=" * 60)
    
    # Diferentes tipos de queries para mostrar roteamento
    queries = [
        {
            "query": "Crie uma função para ordenar uma lista",
            "expected_task": TaskType.CODE_GENERATION,
            "description": "Geração de código"
        },
        {
            "query": "Este código tem um bug, pode corrigir?",
            "context": "def divide(a, b): return a/b",
            "expected_task": TaskType.DEBUGGING,
            "description": "Correção de bugs"
        },
        {
            "query": "Resuma este texto em 3 frases",
            "context": "Uma longa explicação sobre machine learning...",
            "expected_task": TaskType.SUMMARIZATION,
            "description": "Sumarização"
        }
    ]
    
    for i, test in enumerate(queries, 1):
        print(f"\n{i}. {test['description']}")
        print(f"Query: {test['query']}")
        
        # Detectar tipo de tarefa
        detected_task = router.detect_task_type(test['query'], test.get('context', ''))
        print(f"Tarefa detectada: {detected_task.value}")
        
        # Selecionar melhor modelo
        context_length = len(test['query']) + len(test.get('context', ''))
        selected_model = router.select_best_model(detected_task, context_length)
        
        if selected_model:
            provider = selected_model.split('.')[0]
            print(f"Modelo selecionado: {selected_model} ({provider.upper()})")
        else:
            print("❌ Nenhum modelo disponível")

def show_system_overview(router: APIModelRouter):
    """Mostra visão geral do sistema"""
    
    print("\n📊 VISÃO GERAL DO SISTEMA")
    print("=" * 60)
    
    # Modelos disponíveis
    models_info = router.get_available_models()
    print(f"🤖 Total de modelos: {models_info['total']}")
    print(f"🏢 Provedores ativos: {', '.join(models_info['providers'])}")
    
    print("\n📋 Distribuição por provedor:")
    provider_counts = {}
    for model_key in models_info['models'].keys():
        provider = model_key.split('.')[0]
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
    
    for provider, count in provider_counts.items():
        print(f"  {provider.upper()}: {count} modelos")
    
    # Estatísticas de uso
    stats = router.get_stats()
    if stats['total_requests'] > 0:
        print(f"\n📈 Estatísticas de uso:")
        print(f"  Total de requisições: {stats['total_requests']}")
        print(f"  Custo total: ${stats['total_cost']:.4f}")
        print(f"  Tempo médio: {stats['average_response_time']:.2f}s")

def main():
    """Função principal"""
    print("🚀 DEMONSTRAÇÃO - SISTEMA RAG 100% APIs")
    print("OpenAI • Anthropic • Google • DeepSeek")
    print("=" * 60)
    
    # Verificar configuração
    available_providers = check_api_keys()
    if not available_providers:
        print("❌ Nenhuma API key configurada!")
        print("📝 Configure pelo menos uma API key no arquivo .env")
        print("💡 Use 'cp config/env_example.txt .env' e edite o arquivo")
        return
    
    print(f"✅ Provedores disponíveis: {', '.join(p.upper() for p in available_providers)}")
    
    # Carregar configuração e inicializar router
    config = load_config()
    router = APIModelRouter(config)
    
    # Mostrar visão geral
    show_system_overview(router)
    
    # Demonstrar roteamento inteligente
    demonstrate_intelligent_routing(router)
    
    # Testar provedores disponíveis
    if len(available_providers) > 0:
        print(f"\n🎯 Testando {len(available_providers)} provedores configurados...")
        test_provider_specialties(router, available_providers)
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRAÇÃO CONCLUÍDA!")
    print("=" * 60)
    print("\n🎯 CARACTERÍSTICAS DO SISTEMA:")
    print("✅ Roteamento inteligente baseado no tipo de tarefa")
    print("✅ 4 provedores LLM com especialidades distintas")
    print("✅ Controle automático de custos e cache")
    print("✅ Fallback entre provedores para robustez")
    print("✅ Zero dependências de modelos locais")
    print("✅ Escalabilidade ilimitada via APIs")
    
    if len(available_providers) == 4:
        print("\n🏆 Sistema COMPLETO - Todos os 4 provedores configurados!")
    else:
        missing = 4 - len(available_providers)
        print(f"\n⚠️  Configure {missing} provedores adicionais para máxima eficiência")
    
    print("\n💡 Para uso completo, configure todas as API keys em .env")

if __name__ == "__main__":
    main() 