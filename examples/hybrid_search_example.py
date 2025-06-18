"""
Exemplo Completo de Hybrid Search com Qdrant 1.8.0
Demonstra implementação de sparse + dense vectors para RAG avançado
Performance: 16x improvement em sparse vector search
"""

import asyncio
import logging
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.hybrid_indexing_pipeline import HybridIndexingPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectordb.hybrid_qdrant_store import HybridQdrantStore
from src.embeddings.sparse_vector_service import AdvancedSparseVectorService

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridSearchDemo:
    """
    Demonstração completa do sistema de busca híbrida
    """
    
    def __init__(self):
        self.indexing_pipeline = HybridIndexingPipeline()
        self.retriever = HybridRetriever()
        
    async def run_complete_demo(self):
        """
        Executa demonstração completa do sistema híbrido
        """
        logger.info("🚀 Iniciando demonstração de Hybrid Search com Qdrant 1.8.0")
        
        # 1. Preparar documentos de exemplo
        await self._prepare_sample_documents()
        
        # 2. Indexar documentos
        await self._index_documents()
        
        # 3. Demonstrar diferentes tipos de busca
        await self._demonstrate_search_strategies()
        
        # 4. Benchmark de performance
        await self._benchmark_performance()
        
        # 5. Mostrar estatísticas
        await self._show_statistics()
        
        logger.info("✅ Demonstração concluída!")
    
    async def _prepare_sample_documents(self):
        """
        Cria documentos de exemplo para demonstração
        """
        logger.info("📄 Preparando documentos de exemplo")
        
        # Criar diretório de exemplos se não existir
        examples_dir = Path("data/examples")
        examples_dir.mkdir(parents=True, exist_ok=True)
        
        # Documentos de exemplo com diferentes características
        documents = {
            "python_basics.txt": """
            Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.
            Foi criada por Guido van Rossum em 1991. Python é conhecida por sua sintaxe clara e legível.
            
            Características principais:
            - Sintaxe simples e intuitiva
            - Tipagem dinâmica
            - Interpretada (não compilada)
            - Orientada a objetos
            - Suporte a múltiplos paradigmas
            
            Exemplo de código Python:
            def hello_world():
                print("Hello, World!")
                return True
            
            Python é amplamente utilizada em:
            - Desenvolvimento web (Django, Flask)
            - Ciência de dados (Pandas, NumPy)
            - Machine Learning (TensorFlow, PyTorch)
            - Automação e scripts
            """,
            
            "rag_systems.txt": """
            RAG (Retrieval-Augmented Generation) é uma arquitetura que combina recuperação de informações
            com geração de texto usando modelos de linguagem grandes (LLMs).
            
            Componentes principais de um sistema RAG:
            1. Vector Database - armazena embeddings de documentos
            2. Retriever - busca documentos relevantes
            3. Generator - LLM que gera respostas baseadas no contexto
            
            Tipos de busca em RAG:
            - Dense retrieval: usa embeddings semânticos
            - Sparse retrieval: usa keywords e BM25
            - Hybrid retrieval: combina dense + sparse
            
            Vantagens do RAG:
            - Conhecimento atualizado
            - Reduz alucinações
            - Transparência nas fontes
            - Escalabilidade
            """,
            
            "vector_databases.txt": """
            Vector databases são sistemas especializados em armazenar e buscar vetores de alta dimensionalidade.
            
            Principais vector databases:
            - Qdrant: Rust-based, alta performance
            - Pinecone: Managed service
            - Weaviate: Open source com GraphQL
            - Chroma: Simples e local
            - Milvus: Distribuído e escalável
            
            Qdrant 1.8.0 introduziu melhorias significativas:
            - Sparse vectors com 16x performance improvement
            - CPU resource management
            - Melhor indexação para dados de texto
            
            Operações principais:
            - Upsert: inserir/atualizar vetores
            - Search: busca por similaridade
            - Filter: filtrar por metadata
            - Delete: remover vetores
            """,
            
            "machine_learning.txt": """
            Machine Learning é um subcampo da inteligência artificial que permite que sistemas
            aprendam e melhorem automaticamente através da experiência.
            
            Tipos de Machine Learning:
            1. Supervised Learning - aprendizado supervisionado
            2. Unsupervised Learning - aprendizado não supervisionado  
            3. Reinforcement Learning - aprendizado por reforço
            
            Algoritmos populares:
            - Linear Regression
            - Random Forest
            - Support Vector Machines
            - Neural Networks
            - K-Means Clustering
            
            Aplicações:
            - Reconhecimento de imagem
            - Processamento de linguagem natural
            - Sistemas de recomendação
            - Detecção de fraude
            - Carros autônomos
            """
        }
        
        # Salvar documentos
        for filename, content in documents.items():
            file_path = examples_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"✅ Criados {len(documents)} documentos de exemplo")
    
    async def _index_documents(self):
        """
        Indexa documentos usando o pipeline híbrido
        """
        logger.info("🔄 Indexando documentos no pipeline híbrido")
        
        # Encontrar documentos
        examples_dir = Path("data/examples")
        document_paths = list(examples_dir.glob("*.txt"))
        
        if not document_paths:
            logger.error("❌ Nenhum documento encontrado para indexação")
            return
        
        # Indexar
        start_time = time.time()
        stats = await self.indexing_pipeline.index_documents([str(p) for p in document_paths])
        end_time = time.time()
        
        logger.info(f"✅ Indexação concluída em {end_time - start_time:.2f}s")
        logger.info(f"📊 Estatísticas: {stats}")
    
    async def _demonstrate_search_strategies(self):
        """
        Demonstra diferentes estratégias de busca
        """
        logger.info("🔍 Demonstrando estratégias de busca")
        
        # Queries de teste
        test_queries = [
            {
                "query": "Como funciona Python?",
                "description": "Query semântica - deve usar dense search",
                "expected_strategy": "semantic"
            },
            {
                "query": "função print Python código",
                "description": "Query por keywords - deve usar sparse search",
                "expected_strategy": "keyword"
            },
            {
                "query": "Qdrant 1.8.0 performance improvement",
                "description": "Query híbrida - deve usar busca combinada",
                "expected_strategy": "hybrid"
            },
            {
                "query": "machine learning algoritmos tipos",
                "description": "Query mista - deve usar estratégia automática",
                "expected_strategy": "auto"
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"\n--- Teste {i}: {test_case['description']} ---")
            logger.info(f"Query: '{test_case['query']}'")
            
            # Busca com estratégia automática
            start_time = time.time()
            results = await self.retriever.retrieve(
                query=test_case['query'],
                limit=3,
                strategy="auto"
            )
            search_time = time.time() - start_time
            
            logger.info(f"⏱️  Tempo de busca: {search_time:.3f}s")
            logger.info(f"📊 Resultados encontrados: {len(results)}")
            
            # Mostrar top resultado
            if results:
                top_result = results[0]
                logger.info(f"🎯 Top resultado:")
                logger.info(f"   - Dense score: {top_result.dense_score:.3f}")
                logger.info(f"   - Sparse score: {top_result.sparse_score:.3f}")
                logger.info(f"   - Combined score: {top_result.combined_score:.3f}")
                logger.info(f"   - Método: {top_result.retrieval_method}")
                logger.info(f"   - Explicação: {top_result.query_match_explanation}")
                logger.info(f"   - Conteúdo: {top_result.content[:200]}...")
    
    async def _benchmark_performance(self):
        """
        Executa benchmark de performance
        """
        logger.info("\n🏆 Executando benchmark de performance")
        
        # Queries para benchmark
        benchmark_queries = [
            "Python programming language",
            "RAG system architecture",
            "vector database operations",
            "machine learning algorithms",
            "Qdrant sparse vectors",
            "dense retrieval embeddings",
            "hybrid search performance",
            "BM25 keyword matching"
        ]
        
        # Benchmark diferentes estratégias
        strategies = ["dense_only", "sparse_only", "hybrid"]
        
        for strategy in strategies:
            logger.info(f"\n--- Benchmark: {strategy} ---")
            
            times = []
            total_results = 0
            
            for query in benchmark_queries:
                start_time = time.time()
                results = await self.retriever.retrieve(
                    query=query,
                    limit=5,
                    strategy=strategy,
                    use_reranking=False  # Sem reranking para benchmark puro
                )
                end_time = time.time()
                
                times.append(end_time - start_time)
                total_results += len(results)
            
            # Estatísticas
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            avg_results = total_results / len(benchmark_queries)
            
            logger.info(f"⏱️  Tempo médio: {avg_time:.3f}s")
            logger.info(f"⏱️  Tempo mín/máx: {min_time:.3f}s / {max_time:.3f}s")
            logger.info(f"📊 Resultados médios: {avg_results:.1f}")
    
    async def _show_statistics(self):
        """
        Mostra estatísticas detalhadas do sistema
        """
        logger.info("\n📈 Estatísticas do Sistema")
        
        # Estatísticas do pipeline
        pipeline_stats = self.indexing_pipeline.get_stats()
        logger.info(f"📊 Pipeline Stats: {pipeline_stats}")
        
        # Estatísticas do retriever
        retriever_metrics = self.retriever.get_metrics()
        logger.info(f"🔍 Retriever Metrics: {retriever_metrics}")
        
        # Estatísticas do vector store
        vector_store_info = await self.retriever.vector_store.get_collection_info()
        logger.info(f"🗄️  Vector Store Info: {vector_store_info}")

async def main():
    """
    Função principal da demonstração
    """
    demo = HybridSearchDemo()
    await demo.run_complete_demo()

def run_interactive_demo():
    """
    Executa demo interativo
    """
    print("🚀 Hybrid Search Demo - Qdrant 1.8.0")
    print("=====================================")
    
    retriever = HybridRetriever()
    
    while True:
        print("\nOpções:")
        print("1. Buscar com estratégia automática")
        print("2. Buscar apenas dense vectors")
        print("3. Buscar apenas sparse vectors")
        print("4. Buscar híbrido")
        print("5. Ver métricas")
        print("6. Sair")
        
        choice = input("\nEscolha uma opção (1-6): ").strip()
        
        if choice == "6":
            break
        elif choice == "5":
            metrics = retriever.get_metrics()
            print(f"\n📊 Métricas: {metrics}")
            continue
        
        query = input("Digite sua query: ").strip()
        if not query:
            continue
        
        strategy_map = {
            "1": "auto",
            "2": "dense_only", 
            "3": "sparse_only",
            "4": "hybrid"
        }
        
        strategy = strategy_map.get(choice, "auto")
        
        async def search():
            start_time = time.time()
            results = await retriever.retrieve(query, limit=3, strategy=strategy)
            end_time = time.time()
            
            print(f"\n⏱️  Tempo: {end_time - start_time:.3f}s")
            print(f"📊 Resultados: {len(results)}")
            
            for i, result in enumerate(results, 1):
                print(f"\n--- Resultado {i} ---")
                print(f"Dense: {result.dense_score:.3f} | Sparse: {result.sparse_score:.3f}")
                print(f"Combined: {result.combined_score:.3f}")
                print(f"Método: {result.retrieval_method}")
                print(f"Conteúdo: {result.content[:300]}...")
        
        asyncio.run(search())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_demo()
    else:
        asyncio.run(main()) 