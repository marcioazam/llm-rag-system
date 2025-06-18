#!/usr/bin/env python3
"""
Teste da API Unificada - Demonstra uso para MCP e Cursor
"""

import json
import asyncio
import httpx

# Configuração
API_BASE = "http://localhost:8000"

async def test_api_unified():
    print("🎯 TESTANDO API UNIFICADA")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        
        # 1. Teste básico (como MCP usaria)
        print("\n📱 1. USO BÁSICO (MCP)")
        print("-" * 30)
        
        basic_request = {
            "question": "O que é Python?",
            "k": 3
        }
        
        try:
            response = await client.post(f"{API_BASE}/query", json=basic_request)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"🎯 Resposta: {result.get('answer', 'N/A')[:100]}...")
                print(f"⚡ Tempo: {result.get('processing_time', 'N/A')}s")
                print(f"📊 Modo: {result.get('mode', 'N/A')}")
            else:
                print(f"❌ Erro: {response.status_code}")
        except Exception as e:
            print(f"❌ Exceção: {e}")
        
        # 2. Teste otimizado para Cursor
        print("\n🖥️ 2. USO OTIMIZADO (CURSOR)")
        print("-" * 30)
        
        cursor_request = {
            "question": "Como melhorar esta função?",
            "context": "def calculate_sum(a, b):\n    return a + b",
            "file_type": ".py",
            "project_context": "Sistema de cálculos matemáticos",
            "quick_mode": True,
            "k": 2
        }
        
        try:
            response = await client.post(f"{API_BASE}/query", json=cursor_request)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"🎯 Resposta: {result.get('answer', 'N/A')[:100]}...")
                print(f"⚡ Tempo: {result.get('processing_time', 'N/A')}s")
                print(f"📊 Modo: {result.get('mode', 'N/A')}")
                print(f"📈 K usado: {result.get('k_used', 'N/A')}")
            else:
                print(f"❌ Erro: {response.status_code}")
        except Exception as e:
            print(f"❌ Exceção: {e}")
        
        # 3. Teste de health check
        print("\n💊 3. HEALTH CHECK")
        print("-" * 30)
        
        try:
            response = await client.get(f"{API_BASE}/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Status: {health.get('status', 'N/A')}")
                print(f"⏱️ Tempo de resposta: {health.get('response_time_ms', 'N/A')}ms")
                print(f"🔧 Componentes: {list(health.get('components', {}).keys())}")
            else:
                print(f"❌ Health check falhou: {response.status_code}")
        except Exception as e:
            print(f"❌ Exceção: {e}")

def test_sync_usage():
    """Exemplo de uso síncrono (para MCP e outras aplicações)"""
    print("\n🔄 4. EXEMPLO SÍNCRONO")
    print("-" * 30)
    
    import requests
    
    def ask_rag(question: str, **kwargs) -> dict:
        """Função helper para usar o RAG"""
        try:
            response = requests.post(
                f"{API_BASE}/query",
                json={"question": question, **kwargs},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    # Teste simples
    result = ask_rag("Explique list comprehension em Python")
    
    if "error" not in result:
        print(f"✅ Resposta recebida")
        print(f"⚡ Tempo: {result.get('processing_time', 'N/A')}s")
        print(f"🤖 Modelo: {result.get('model', 'N/A')}")
    else:
        print(f"❌ Erro: {result['error']}")

def show_mcp_example():
    """Mostra como o MCP poderia usar a API"""
    print("\n🔌 5. EXEMPLO MCP INTEGRATION")
    print("-" * 30)
    
    mcp_code = '''
# tools/rag_query.py
import httpx

async def rag_query(question: str) -> dict:
    """Tool do MCP para consultar o RAG"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/query",
            json={
                "question": question,
                "k": 5,
                "use_hybrid": True
            }
        )
        return response.json()

# Uso:
# result = await rag_query("Como implementar cache?")
'''
    print(mcp_code)

async def main():
    """Função principal"""
    await test_api_unified()
    test_sync_usage()
    show_mcp_example()
    
    print("\n" + "=" * 50)
    print("🎉 TESTE CONCLUÍDO!")
    print("\n✅ BENEFÍCIOS DA UNIFICAÇÃO:")
    print("• Um endpoint para tudo (/query)")
    print("• MCP pode usar diretamente")
    print("• Cursor ganha otimizações automáticas")
    print("• Menos código para manter")
    print("• Backward compatibility garantida")

if __name__ == "__main__":
    print("🚀 Para testar, inicie primeiro a API:")
    print("   python -m src.api.main")
    print("\nEm seguida execute este script:")
    print("   python test_unified_api.py")
    
    # Teste básico sem servidor (apenas mostra estrutura)
    show_mcp_example()
    
    # Para testar com servidor ativo, descomente:
    # asyncio.run(main()) 