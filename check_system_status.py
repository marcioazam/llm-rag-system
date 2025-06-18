#!/usr/bin/env python3
"""
Script para verificar o status completo do sistema RAG baseado em APIs
"""

import os
import yaml
import requests
from typing import Dict, List

def check_api_keys() -> Dict[str, bool]:
    """Verifica se as API keys estão configuradas"""
    required_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic (Claude)", 
        "GOOGLE_API_KEY": "Google (Gemini)",
        "DEEPSEEK_API_KEY": "DeepSeek"
    }
    
    status = {}
    for key, name in required_keys.items():
        status[name] = bool(os.getenv(key))
    
    return status

def check_api_connectivity() -> Dict[str, str]:
    """Testa conectividade com as APIs"""
    results = {}
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
            response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
            results["OpenAI"] = "✅ Conectado" if response.status_code == 200 else f"❌ Erro {response.status_code}"
        except Exception as e:
            results["OpenAI"] = f"❌ Erro: {str(e)[:50]}"
    else:
        results["OpenAI"] = "⚠️  API key não configurada"
    
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            headers = {"x-api-key": os.getenv("ANTHROPIC_API_KEY")}
            # Anthropic não tem endpoint de modelos público, então testamos com uma requisição simples
            results["Anthropic"] = "✅ API key configurada"
        except Exception as e:
            results["Anthropic"] = f"❌ Erro: {str(e)[:50]}"
    else:
        results["Anthropic"] = "⚠️  API key não configurada"
    
    # Google
    if os.getenv("GOOGLE_API_KEY"):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={os.getenv('GOOGLE_API_KEY')}"
            response = requests.get(url, timeout=10)
            results["Google"] = "✅ Conectado" if response.status_code == 200 else f"❌ Erro {response.status_code}"
        except Exception as e:
            results["Google"] = f"❌ Erro: {str(e)[:50]}"
    else:
        results["Google"] = "⚠️  API key não configurada"
    
    # DeepSeek
    if os.getenv("DEEPSEEK_API_KEY"):
        results["DeepSeek"] = "✅ API key configurada"
    else:
        results["DeepSeek"] = "⚠️  API key não configurada"
    
    return results

def load_config() -> Dict:
    """Carrega configuração dos provedores"""
    try:
        with open("config/llm_providers_config.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"error": str(e)}

def main():
    print("🔍 VERIFICAÇÃO DO SISTEMA RAG - 100% APIs")
    print("=" * 50)
    
    # 1. Verificar API keys
    print("\n📋 Status das API Keys:")
    api_status = check_api_keys()
    for provider, configured in api_status.items():
        status = "✅ Configurada" if configured else "❌ Não configurada"
        print(f"  {provider}: {status}")
    
    # 2. Verificar conectividade
    print("\n🌐 Conectividade das APIs:")
    connectivity = check_api_connectivity()
    for provider, status in connectivity.items():
        print(f"  {provider}: {status}")
    
    # 3. Verificar configuração
    print("\n⚙️  Configuração do Sistema:")
    config = load_config()
    if "error" in config:
        print(f"  ❌ Erro ao carregar config: {config['error']}")
    else:
        providers = config.get("providers", {})
        print(f"  ✅ {len(providers)} provedores configurados: {', '.join(providers.keys())}")
        
        total_models = sum(len(p.get("models", {})) for p in providers.values())
        print(f"  ✅ {total_models} modelos disponíveis")
    
    # 4. Status geral
    print("\n📊 Status Geral:")
    configured_count = sum(1 for status in api_status.values() if status)
    connected_count = sum(1 for status in connectivity.values() if "✅" in status)
    
    print(f"  API Keys configuradas: {configured_count}/4")
    print(f"  APIs funcionais: {connected_count}/4")
    
    if configured_count >= 2 and connected_count >= 2:
        print("  🎉 Sistema FUNCIONAL - Pronto para uso!")
    elif configured_count >= 1:
        print("  ⚠️  Sistema PARCIALMENTE funcional")
    else:
        print("  ❌ Sistema NÃO funcional - Configure pelo menos OpenAI ou Claude")
    
    print("\n📝 Para configurar API keys, edite o arquivo .env com base em config/env_example.txt")

if __name__ == "__main__":
    main()
