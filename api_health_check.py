#!/usr/bin/env python3
"""
Script para verificar o status da API RAG
"""
import asyncio
import aiohttp
import time
import json
import sys
from urllib.parse import urljoin

class APIHealthChecker:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.endpoints = [
            "/",
            "/health",
            "/docs",
            "/openapi.json"
        ]
    
    async def check_endpoint(self, session, endpoint):
        """Verifica um endpoint específico"""
        url = urljoin(self.base_url, endpoint)
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with session.get(url, timeout=timeout) as response:
                status = response.status
                content_type = response.headers.get('content-type', '')
                
                # Tentar ler uma pequena parte da resposta
                try:
                    text = await response.text()
                    content_preview = text[:100] + "..." if len(text) > 100 else text
                except:
                    content_preview = "[Não foi possível ler conteúdo]"
                
                return {
                    'endpoint': endpoint,
                    'status': status,
                    'content_type': content_type,
                    'preview': content_preview,
                    'success': 200 <= status < 400
                }
        except asyncio.TimeoutError:
            return {
                'endpoint': endpoint,
                'status': 'TIMEOUT',
                'error': 'Timeout de 5 segundos excedido',
                'success': False
            }
        except Exception as e:
            return {
                'endpoint': endpoint,
                'status': 'ERROR',
                'error': str(e),
                'success': False
            }
    
    async def check_all_endpoints(self):
        """Verifica todos os endpoints"""
        print(f"🔍 Verificando API em {self.base_url}")
        print("=" * 50)
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.check_endpoint(session, endpoint) for endpoint in self.endpoints]
            results = await asyncio.gather(*tasks)
        
        success_count = 0
        for result in results:
            endpoint = result['endpoint']
            if result['success']:
                print(f"✅ {endpoint} - Status: {result['status']}")
                if 'content_type' in result:
                    print(f"   Content-Type: {result['content_type']}")
                if 'preview' in result and result['preview'].strip():
                    print(f"   Preview: {result['preview']}")
                success_count += 1
            else:
                print(f"❌ {endpoint} - Status: {result['status']}")
                if 'error' in result:
                    print(f"   Erro: {result['error']}")
        
        print("=" * 50)
        print(f"📊 Resumo: {success_count}/{len(results)} endpoints funcionando")
        
        return success_count > 0
    
    def check_process_running(self):
        """Verifica se há processos relacionados rodando"""
        import subprocess
        
        print("\n🔍 Verificando processos...")
        try:
            # Verificar processos Python/uvicorn
            result = subprocess.run(['pgrep', '-f', 'uvicorn'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("✅ Processo uvicorn encontrado")
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        ps_result = subprocess.run(['ps', '-p', pid, '-o', 'pid,cmd'], 
                                                 capture_output=True, text=True)
                        print(f"   PID {pid}: {ps_result.stdout.split('\\n')[1] if len(ps_result.stdout.split('\\n')) > 1 else 'N/A'}")
                    except:
                        pass
            else:
                print("❌ Nenhum processo uvicorn encontrado")
                
            # Verificar porta 8000
            result = subprocess.run(['netstat', '-tulpn'], capture_output=True, text=True)
            if ':8000' in result.stdout:
                print("✅ Porta 8000 está em uso")
                for line in result.stdout.split('\n'):
                    if ':8000' in line:
                        print(f"   {line.strip()}")
            else:
                print("❌ Porta 8000 não está em uso")
                
        except Exception as e:
            print(f"❌ Erro ao verificar processos: {e}")

async def main():
    print("🚀 Iniciando verificação de saúde da API RAG")
    
    checker = APIHealthChecker()
    
    # Verificar processos primeiro
    checker.check_process_running()
    
    # Aguardar um pouco
    print("\n⏳ Aguardando 3 segundos...")
    await asyncio.sleep(3)
    
    # Verificar endpoints
    api_working = await checker.check_all_endpoints()
    
    if api_working:
        print("\n🎉 API está funcionando!")
        print("\n📋 Próximos passos:")
        print("   1. Acesse http://localhost:8000/docs para ver a documentação")
        print("   2. Teste os endpoints da API")
        print("   3. Verifique os logs para mais detalhes")
    else:
        print("\n❌ API não está respondendo corretamente")
        print("\n🔧 Soluções sugeridas:")
        print("   1. Verifique os logs: tail -f logs/rag_api.log")
        print("   2. Pare e reinicie: ./scripts/stop_services.sh && ./scripts/start_services.sh")
        print("   3. Execute o diagnóstico: python diagnose_api.py")
        print("   4. Inicie manualmente: python -m uvicorn src.api.main:app --reload")

if __name__ == "__main__":
    asyncio.run(main())
