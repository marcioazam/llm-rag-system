"""
Demo simples do RAPTOR - Teste rápido
"""

import asyncio
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def demo_raptor():
    """Demo rápido do RAPTOR"""
    
    print("🚀 RAPTOR Demo - Teste Rápido")
    print("=" * 40)
    
    try:
        from src.retrieval.raptor_retriever import RaptorRetriever, get_default_raptor_config
        
        # Configuração simples
        config = get_default_raptor_config()
        config.update({
            "chunk_size": 150,
            "max_levels": 3,
            "min_cluster_size": 2
        })
        
        print("✓ Criando RAPTOR retriever...")
        raptor = RaptorRetriever(**config)
        
        # Documentos simples
        docs = [
            "Python é uma linguagem de programação. É fácil de aprender e usar.",
            "Machine Learning usa algoritmos para aprender padrões dos dados.",
            "RAG combina busca e geração de texto para melhores respostas.",
            "Cloud computing oferece recursos computacionais via internet.",
            "DevOps integra desenvolvimento e operações de software."
        ]
        
        print(f"✓ Construindo árvore com {len(docs)} documentos...")
        stats = await raptor.build_tree(docs)
        
        print(f"✅ Árvore construída!")
        print(f"   • Nós: {stats.total_nodes}")
        print(f"   • Níveis: {stats.levels}")
        print(f"   • Tempo: {stats.construction_time:.2f}s")
        
        # Teste de busca
        query = "Como usar Python para machine learning?"
        print(f"\n🔍 Buscando: {query}")
        
        results = raptor.search(query, k=3)
        
        print(f"\n📋 Resultados ({len(results)}):")
        for i, result in enumerate(results, 1):
            score = result['score']
            level = result['metadata']['level']
            content = result['content'][:100]
            print(f"   {i}. [Score: {score:.3f}, Nível: {level}]")
            print(f"      {content}...")
        
        print("\n✅ Demo RAPTOR concluído!")
        
    except ImportError as e:
        print(f"❌ Dependências não encontradas: {e}")
        print("💡 Instale: pip install scikit-learn umap-learn sentence-transformers")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_raptor())