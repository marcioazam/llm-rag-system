"""
Teste de Demonstração - Cache Otimizado para RAG
Compara performance entre sem cache, cache simples e cache híbrido
"""

import asyncio
import time
import json
import tempfile
import os
from typing import Dict, Any

# Importar o cache otimizado
from src.cache.optimized_rag_cache import OptimizedRAGCache


async def simular_query_rag(query: str, delay: float = 1.0) -> Dict[str, Any]:
    """Simula processamento RAG custoso"""
    await asyncio.sleep(delay)  # Simular API call
    
    return {
        "answer": f"Resposta para: {query}",
        "confidence": 0.85,
        "tokens_used": 150,
        "processing_time": delay,
        "sources": ["doc1.pdf", "doc2.pdf"]
    }


async def test_sem_cache():
    """Teste baseline sem cache"""
    print("\n🔍 TESTE 1: SEM CACHE")
    print("=" * 50)
    
    queries = [
        "Como implementar RAG?",
        "O que é embedding?", 
        "Como implementar RAG?",  # Repetida
        "Qual melhor modelo LLM?",
        "O que é embedding?",     # Repetida
    ]
    
    start_time = time.time()
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        result = await simular_query_rag(query, delay=0.5)
        print(f"  ✅ Processada em {result['processing_time']:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n📊 RESULTADOS SEM CACHE:")
    print(f"  • Tempo total: {total_time:.2f}s")
    print(f"  • Queries processadas: {len(queries)}")
    print(f"  • Cache hits: 0")
    print(f"  • Hit rate: 0%")
    
    return total_time


async def test_com_cache_hibrido():
    """Teste com cache híbrido otimizado"""
    print("\n🚀 TESTE 2: COM CACHE HÍBRIDO")
    print("=" * 50)
    
    # Criar cache temporário
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        cache_db = tmp.name
    
    try:
        cache = OptimizedRAGCache(
            db_path=cache_db,
            max_memory_entries=100,
            enable_redis=False  # Teste local
        )
        
        queries = [
            "Como implementar RAG?",
            "O que é embedding?", 
            "Como implementar RAG?",  # Repetida - hit L1
            "Qual melhor modelo LLM?",
            "O que é embedding?",     # Repetida - hit L1
        ]
        
        start_time = time.time()
        cache_hits = 0
        
        for i, query in enumerate(queries, 1):
            print(f"Query {i}: {query}")
            
            # Verificar cache primeiro
            cached_result, source, metadata = await cache.get(query)
            
            if cached_result:
                print(f"  🎯 Cache HIT ({source}) - confidence: {metadata.get('confidence', 0):.2f}")
                cache_hits += 1
            else:
                print(f"  🔄 Cache MISS - processando...")
                result = await simular_query_rag(query, delay=0.5)
                
                # Salvar no cache
                await cache.set(
                    query,
                    result,
                    confidence=result["confidence"],
                    tokens_saved=result["tokens_used"],
                    processing_time_saved=result["processing_time"]
                )
                print(f"  ✅ Processada e armazenada no cache")
        
        total_time = time.time() - start_time
        
        # Estatísticas do cache
        stats = cache.get_stats()
        
        print(f"\n📊 RESULTADOS COM CACHE HÍBRIDO:")
        print(f"  • Tempo total: {total_time:.2f}s")
        print(f"  • Queries processadas: {len(queries)}")
        print(f"  • Cache hits: {cache_hits}")
        print(f"  • Hit rate: {cache_hits/len(queries)*100:.1f}%")
        print(f"  • Tokens economizados: {stats['tokens_saved']}")
        print(f"  • Tempo economizado: {stats['processing_time_saved']:.2f}s")
        
        print(f"\n🧠 DETALHES DO CACHE:")
        print(f"  • L1 (Memória): {stats['l1_hits']} hits")
        print(f"  • L2 (SQLite): {stats['l2_hits']} hits") 
        print(f"  • L3 (Redis): {stats['l3_hits']} hits")
        print(f"  • Cache sizes: {stats['cache_sizes']}")
        
        cache.close()
        return total_time, stats
        
    finally:
        # Limpar arquivo temporário
        if os.path.exists(cache_db):
            os.unlink(cache_db)


async def test_cache_persistencia():
    """Teste de persistência entre sessões"""
    print("\n💾 TESTE 3: PERSISTÊNCIA ENTRE SESSÕES")
    print("=" * 50)
    
    # Criar cache temporário
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        cache_db = tmp.name
    
    try:
        # Sessão 1: Popular cache
        print("📝 SESSÃO 1: Populando cache...")
        cache1 = OptimizedRAGCache(db_path=cache_db, enable_redis=False)
        
        result = await simular_query_rag("Como usar Python?", delay=0.5)
        await cache1.set(
            "Como usar Python?",
            result,
            confidence=0.9,
            tokens_saved=200,
            processing_time_saved=0.5
        )
        
        stats1 = cache1.get_stats()
        print(f"  ✅ Cache populado: {stats1['cache_sizes']['sqlite']} entradas no SQLite")
        cache1.close()
        
        # Sessão 2: Usar cache persistido
        print("\n🔄 SESSÃO 2: Usando cache persistido...")
        cache2 = OptimizedRAGCache(db_path=cache_db, enable_redis=False)
        
        cached_result, source, metadata = await cache2.get("Como usar Python?")
        
        if cached_result:
            print(f"  🎯 Cache HIT persistido ({source})")
            print(f"  📊 Confiança: {metadata.get('confidence', 0):.2f}")
            print(f"  ⏱️  Age: {metadata.get('age', 0):.1f}s")
            print(f"  🔢 Access count: {metadata.get('access_count', 0)}")
        else:
            print("  ❌ Cache persistência falhou")
        
        stats2 = cache2.get_stats()
        print(f"\n📊 ESTATÍSTICAS DA SESSÃO 2:")
        print(f"  • SQLite entries: {stats2['cache_sizes']['sqlite']}")
        print(f"  • L2 hits: {stats2['l2_hits']}")
        
        cache2.close()
        
    finally:
        # Limpar arquivo temporário
        if os.path.exists(cache_db):
            os.unlink(cache_db)


async def main():
    """Executa todos os testes de comparação"""
    print("🎯 DEMONSTRAÇÃO DO CACHE OTIMIZADO PARA RAG")
    print("=" * 60)
    
    # Teste sem cache
    time_no_cache = await test_sem_cache()
    
    # Teste com cache híbrido
    time_with_cache, cache_stats = await test_com_cache_hibrido()
    
    # Teste de persistência
    await test_cache_persistencia()
    
    # Comparação final
    print("\n🏆 COMPARAÇÃO FINAL")
    print("=" * 50)
    improvement = ((time_no_cache - time_with_cache) / time_no_cache) * 100
    print(f"📈 MELHORIA DE PERFORMANCE:")
    print(f"  • Sem cache: {time_no_cache:.2f}s")
    print(f"  • Com cache: {time_with_cache:.2f}s")
    print(f"  • Melhoria: {improvement:.1f}% mais rápido")
    
    print(f"\n💰 ECONOMIA ESTIMADA:")
    print(f"  • Tokens economizados: {cache_stats['tokens_saved']}")
    print(f"  • Tempo economizado: {cache_stats['processing_time_saved']:.2f}s")
    print(f"  • Hit rate: {cache_stats['hit_rate']:.1%}")
    
    print(f"\n✅ VANTAGENS DO CACHE HÍBRIDO:")
    print(f"  • 🚀 Performance: {improvement:.1f}% mais rápido")
    print(f"  • 💾 Persistência: Dados sobrevivem a reinicializações")
    print(f"  • 🎯 Simplicidade: Zero configuração externa")
    print(f"  • 📊 Métricas: Tracking detalhado de economia")
    print(f"  • 🔄 Escalabilidade: Adicione Redis quando necessário")


if __name__ == "__main__":
    asyncio.run(main()) 