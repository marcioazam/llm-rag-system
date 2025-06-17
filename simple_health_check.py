#!/usr/bin/env python3
"""
Verificação simples da API sem dependências externas
"""

import urllib.request
import urllib.error
import json
import time
import sys

def check_api(url, timeout=5):
    """Verifica se a API está respondendo"""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.getcode()
            content = response.read().decode('utf-8')
            return {
                'status': status,
                'content': content[:200] + "..." if len(content) > 200 else content,
                'success': True
            }
    except urllib.error.HTTPError as e:
        return {
            'status': e.code,
            'error': f'HTTP Error: {e.reason}',
            'success': False
        }
    except urllib.error.URLError as e:
        return {
            'status': 'CONNECTION_ERROR',
            'error': f'Connection Error: {e.reason}',
            'success': False
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e),
            'success': False
        }

def main():
    urls_to_test = [
        "http://localhost:8000",
        "http://localhost:8000/health",
        "http://localhost:8000/docs",
        "http://localhost:8001",  # Porta do teste manual
    ]
    
    print("🔍 Verificação Simples da API RAG")
    print("=" * 40)
    
    working_apis = []
    
    for url in urls_to_test:
        print(f"\n🌐 Testando: {url}")
        result = check_api(url)
        
        if result['success']:
            print(f"✅ Status: {result['status']}")
            if result['content']:
                print(f"📄 Conteúdo: {result['content']}")
            working_apis.append(url)
        else:
            print(f"❌ Status: {result['status']}")
            print(f"❌ Erro: {result['error']}")
    
    print("\n" + "=" * 40)
    print(f"📊 Resumo: {len(working_apis)}/{len(urls_to_test)} URLs funcionando")
    
    if working_apis:
        print("✅ APIs funcionando:")
        for api in working_apis:
            print(f"   - {api}")
        
        print("\n🎉 Sucesso! Você pode acessar:")
        for api in working_apis:
            if "/docs" in api:
                print(f"   📖 Documentação: {api}")
            elif "/health" not in api:
                print(f"   🌐 API Principal: {api}")
    else:
        print("\n❌ Nenhuma API está respondendo")
        print("\n🔧 Próximos passos:")
        print("   1. Execute: python fix_model_router_issue.py")
        print("   2. Corrija o model_router.py")
        print("   3. Reinicie: ./scripts/start_services.sh")

if __name__ == "__main__":
    main()
