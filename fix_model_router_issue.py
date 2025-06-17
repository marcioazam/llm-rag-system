#!/usr/bin/env python3
"""
Script para corrigir o erro no model_router.py
O erro "'name'" indica que o código está tentando acessar uma chave 'name' 
que não existe na resposta do Ollama
"""

import json
import requests
import sys

def test_ollama_response():
    """Testa a resposta da API do Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            print("✅ Ollama respondeu com sucesso")
            print("📋 Estrutura da resposta:")
            print(json.dumps(data, indent=2))
            
            # Verificar se tem modelos
            if 'models' in data:
                print(f"\n📊 {len(data['models'])} modelo(s) encontrado(s):")
                for i, model in enumerate(data['models']):
                    print(f"  {i+1}. Chaves disponíveis: {list(model.keys())}")
                    if 'name' in model:
                        print(f"     - name: {model['name']}")
                    elif 'model' in model:
                        print(f"     - model: {model['model']}")
            else:
                print("❌ Chave 'models' não encontrada na resposta")
            
            return data
        else:
            print(f"❌ Ollama retornou status {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Erro ao conectar com Ollama: {e}")
        return None

def generate_model_router_fix(ollama_data):
    """Gera código corrigido para o model_router baseado na resposta real do Ollama"""
    
    if not ollama_data or 'models' not in ollama_data:
        return None
    
    # Verificar qual chave usar para o nome do modelo
    sample_model = ollama_data['models'][0] if ollama_data['models'] else {}
    name_key = 'name' if 'name' in sample_model else 'model' if 'model' in sample_model else None
    
    if not name_key:
        print("❌ Não foi possível determinar a chave para o nome do modelo")
        return None
    
    fix_code = f'''
# Correção para model_router.py
# Substitua a linha que causa erro por:

# Ao invés de:
# model_name = model["name"]  # Isso causa o erro

# Use:
model_name = model.get("{name_key}", "unknown")  # Método seguro

# Ou, se quiser ser mais robusto:
model_name = model.get("{name_key}") or model.get("name") or model.get("model") or "unknown"

# Exemplo de função corrigida:
def get_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("models", []):
                # Método seguro para acessar o nome
                name = model.get("{name_key}") or model.get("name") or model.get("model") or "unknown"
                models.append({{
                    "name": name,
                    "size": model.get("size", 0),
                    "modified_at": model.get("modified_at", "")
                }})
            return models
        return []
    except Exception as e:
        logger.error(f"Erro ao obter modelos: {{e}}")
        return []
'''
    
    return fix_code

def main():
    print("🔍 Diagnosticando problema do model_router...")
    
    # Testar Ollama
    ollama_data = test_ollama_response()
    
    if ollama_data:
        # Gerar correção
        fix = generate_model_router_fix(ollama_data)
        if fix:
            print("\n🔧 Correção sugerida:")
            print("=" * 50)
            print(fix)
            print("=" * 50)
            
            # Salvar correção em arquivo
            with open("model_router_fix.py", "w") as f:
                f.write(fix)
            print("\n💾 Correção salva em 'model_router_fix.py'")
            
        else:
            print("\n❌ Não foi possível gerar correção automática")
    else:
        print("\n❌ Problema de conectividade com Ollama")
        print("Verifique se o Ollama está rodando: ollama ps")

if __name__ == "__main__":
    main()
