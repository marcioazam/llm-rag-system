"""
Demonstração: Cache RAG com Redis Configurado
Mostra como configurar e usar Redis via variáveis de ambiente
"""

import os
import asyncio
import tempfile
import time
from typing import Dict, Any

# Configurar variáveis de ambiente para demonstração
os.environ.update({
    "ENVIRONMENT": "production",
    "CACHE_ENABLE_REDIS": "false",  # Iniciar como false para demonstrar fallback
    "REDIS_URL": "redis://localhost:6379",
    "REDIS_DB": "1",  # Usar DB 1 para testes
    "CACHE_MAX_MEMORY_ENTRIES": "500",
    "CACHE_DB_PATH": "storage/demo_cache.db"
})

from src.cache.optimized_rag_cache import OptimizedRAGCache
from src.config.cache_config import get_cache_config, print_cache_config


async def simular_query_custosa(query: str) -> Dict[str, Any]:
    """Simula processamento RAG custoso"""
    await asyncio.sleep(0.3)  # Simular API call
    
    return {
        "answer": f"Resposta elaborada para: {query}",
        "confidence": 0.92,
        "tokens_used": 250,
        "processing_time": 0.3,
        "sources": ["enterprise_doc1.pdf", "knowledge_base.md"],
        "metadata": {
            "model": "gpt-4",
            "timestamp": time.time(),
            "query_type": "complex"
        }
    }


async def demo_cache_configurado():
    """Demonstração do cache híbrido configurado via ambiente"""
    
    print("🚀 DEMONSTRAÇÃO: CACHE RAG CONFIGURADO VIA AMBIENTE")
    print("=" * 65)
    
    # Mostrar configuração atual
    print("\n📋 CONFIGURAÇÃO ATUAL:")
    print_cache_config()
    
    print("\n🔧 INICIALIZANDO CACHE...")
    
    try:
        # Inicializar cache com configurações do ambiente
        cache = OptimizedRAGCache()
        
        print("✅ Cache inicializado com sucesso!")
        
        # Verificar status do Redis
        if cache.redis_client:
            print("🎯 Redis CONECTADO e funcionando!")
        else:
            print("💾 Rodando em modo L1+L2 (sem Redis)")
        
        print(f"📊 Configurações carregadas do ambiente:")
        print(f"  • Max memory entries: {cache.max_memory_entries}")
        print(f"  • DB path: {cache.db_path}")
        print(f"  • Redis enabled: {cache.enable_redis}")
        
        # Testes de performance
        print("\n🧪 TESTANDO PERFORMANCE...")
        
        queries = [
            "Como configurar cache RAG?",
            "Diferenças entre SQLite e Redis?",
            "Como configurar cache RAG?",     # Repetida - hit L1
            "Boas práticas de cache?",
            "Diferenças entre SQLite e Redis?", # Repetida - hit L1
        ]
        
        start_time = time.time()
        cache_hits = 0
        
        for i, query in enumerate(queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            
            query_start = time.time()
            
            # Tentar obter do cache
            cached_result, source, metadata = await cache.get(query)
            
            if cached_result:
                query_time = time.time() - query_start
                print(f"  🎯 CACHE HIT ({source}) em {query_time*1000:.1f}ms")
                print(f"     Confidence: {metadata.get('confidence', 0):.2f}")
                cache_hits += 1
            else:
                print(f"  🔄 CACHE MISS - processando...")
                result = await simular_query_custosa(query)
                
                # Salvar no cache
                await cache.set(
                    query,
                    result,
                    confidence=result["confidence"],
                    tokens_saved=result["tokens_used"],
                    processing_time_saved=result["processing_time"],
                    cost_savings=result["tokens_used"] * 0.0001
                )
                
                query_time = time.time() - query_start
                print(f"  ✅ Processada em {query_time:.2f}s e cacheada")
        
        total_time = time.time() - start_time
        
        # Estatísticas finais
        stats = cache.get_stats()
        
        print(f"\n📊 RESULTADOS:")
        print(f"  • Tempo total: {total_time:.2f}s")
        print(f"  • Cache hits: {cache_hits}/{len(queries)} ({cache_hits/len(queries)*100:.1f}%)")
        print(f"  • Tokens economizados: {stats['tokens_saved']}")
        print(f"  • Economia: ${stats['cost_savings']:.4f}")
        
        print(f"\n🧠 DISTRIBUIÇÃO DOS HITS:")
        print(f"  • L1 (Memória): {stats['l1_hits']} hits")
        print(f"  • L2 (SQLite):  {stats['l2_hits']} hits")
        print(f"  • L3 (Redis):   {stats['l3_hits']} hits")
        
        cache.close()
        print(f"\n✅ Demonstração concluída!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        return False


async def demo_redis_habilitado():
    """Tenta demonstrar com Redis habilitado"""
    
    print("\n" + "="*65)
    print("🔴 TESTE: HABILITANDO REDIS VIA VARIÁVEL DE AMBIENTE")
    print("="*65)
    
    # Habilitar Redis
    os.environ["CACHE_ENABLE_REDIS"] = "true"
    
    print("\n📋 NOVA CONFIGURAÇÃO (Redis habilitado):")
    print_cache_config()
    
    try:
        cache = OptimizedRAGCache()
        
        if cache.redis_client:
            print("🎯 Redis CONECTADO com sucesso!")
            
            # Teste rápido
            test_query = "Teste Redis habilitado"
            result = await simular_query_custosa(test_query)
            
            await cache.set(test_query, result, confidence=0.95)
            cached_result, source, metadata = await cache.get(test_query)
            
            if source == "memory":
                print(f"✅ Cache funcionando - dados em memória")
            
        else:
            print("⚠️  Redis não conectou - verifique se está rodando")
            print("   Execute: redis-server")
            print("   Ou instale: pip install redis")
        
        cache.close()
        
    except Exception as e:
        print(f"❌ Erro com Redis: {e}")
        print("💡 Isso é normal se Redis não estiver instalado/rodando")


async def main():
    """Executa demonstrações completas"""
    
    print("🎯 DEMONSTRAÇÃO COMPLETA: CONFIGURAÇÃO DE CACHE VIA .ENV")
    print("="*70)
    
    # 1. Demonstrar configuração via ambiente (sem Redis)
    success = await demo_cache_configurado()
    
    # 2. Tentar habilitar Redis
    if success:
        await demo_redis_habilitado()
    
    print(f"\n🎯 RESUMO DA CONFIGURAÇÃO:")
    print(f"="*50)
    print(f"✅ Todas as configurações carregadas via variáveis de ambiente")
    print(f"✅ Cache funciona com ou sem Redis")
    print(f"✅ Fallback automático quando Redis não disponível")
    print(f"✅ Configuração flexível por ambiente")
    
    print(f"\n📝 VARIÁVEIS DE AMBIENTE IMPORTANTES:")
    print(f"  CACHE_ENABLE_REDIS=true|false")
    print(f"  REDIS_URL=redis://localhost:6379")
    print(f"  CACHE_MAX_MEMORY_ENTRIES=2000")
    print(f"  CACHE_DB_PATH=storage/rag_cache.db")
    
    print(f"\n🔧 PARA HABILITAR REDIS:")
    print(f"  1. Instalar: pip install redis")
    print(f"  2. Iniciar servidor: redis-server")
    print(f"  3. No .env: CACHE_ENABLE_REDIS=true")
    print(f"  4. Reiniciar aplicação")


if __name__ == "__main__":
    asyncio.run(main())