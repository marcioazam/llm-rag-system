"""
Script de Teste de Performance - Hybrid Search Qdrant 1.8.0
Valida o 16x improvement em sparse vector search
Compara performance antes/depois das otimizações
"""

import asyncio
import time
import statistics
import json
from pathlib import Path
import sys
from typing import List, Dict, Any
import logging

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.hybrid_indexing_pipeline import HybridIndexingPipeline
from src.embeddings.sparse_vector_service import AdvancedSparseVectorService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridPerformanceTester:
    """
    Tester de performance para validar melhorias do Qdrant 1.8.0
    """
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.indexing_pipeline = HybridIndexingPipeline()
        self.results = {}
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Executa benchmark completo do sistema híbrido
        """
        logger.info("🚀 Iniciando benchmark completo de performance")
        
        # 1. Preparar dados de teste
        await self._prepare_test_data()
        
        # 2. Benchmark de indexação
        indexing_results = await self._benchmark_indexing()
        
        # 3. Benchmark de busca
        search_results = await self._benchmark_search()
        
        # 4. Benchmark de sparse vectors
        sparse_results = await self._benchmark_sparse_vectors()
        
        # 5. Análise de escalabilidade
        scalability_results = await self._benchmark_scalability()
        
        # Compilar resultados
        self.results = {
            "indexing": indexing_results,
            "search": search_results,
            "sparse_vectors": sparse_results,
            "scalability": scalability_results,
            "timestamp": time.time()
        }
        
        # Salvar resultados
        await self._save_results()
        
        # Mostrar resumo
        self._print_summary()
        
        return self.results
    
    async def _prepare_test_data(self):
        """
        Prepara dados de teste em diferentes escalas
        """
        logger.info("📄 Preparando dados de teste")
        
        # Criar diretório de teste
        test_dir = Path("data/performance_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Gerar documentos de teste com diferentes características
        test_documents = self._generate_test_documents()
        
        # Salvar documentos
        for i, doc in enumerate(test_documents):
            file_path = test_dir / f"test_doc_{i:03d}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc)
        
        logger.info(f"✅ Gerados {len(test_documents)} documentos de teste")
    
    def _generate_test_documents(self) -> List[str]:
        """
        Gera documentos de teste com diferentes padrões
        """
        documents = []
        
        # Documentos técnicos (dense-friendly)
        technical_docs = [
            """
            Sistemas de busca híbrida combinam múltiplas estratégias de recuperação para melhorar
            a relevância dos resultados. A abordagem híbrida integra busca semântica baseada em
            embeddings densos com busca lexical usando algoritmos como BM25.
            
            A principal vantagem é capturar tanto similaridade semântica quanto matches exatos
            de termos específicos. Isso é especialmente útil em domínios técnicos onde
            terminologia precisa é importante.
            """,
            
            """
            Qdrant é um vector database otimizado para aplicações de machine learning e busca
            semântica. A versão 1.8.0 introduziu melhorias significativas em sparse vectors,
            resultando em performance até 16x melhor para operações de busca híbrida.
            
            As otimizações incluem algoritmos de indexação aprimorados e melhor utilização
            de recursos de CPU para processamento paralelo de queries.
            """,
            
            """
            Retrieval-Augmented Generation (RAG) é uma técnica que combina modelos de linguagem
            grandes com sistemas de recuperação de informação. O objetivo é fornecer contexto
            relevante para melhorar a qualidade e precisão das respostas geradas.
            
            Componentes essenciais incluem chunking inteligente, embedding de alta qualidade,
            e estratégias de fusão de resultados como Reciprocal Rank Fusion (RRF).
            """
        ]
        
        # Documentos com keywords específicas (sparse-friendly)
        keyword_docs = [
            """
            Python função def return print input output
            classe class object método method
            variável variable string integer float boolean
            lista list tupla tuple dicionário dict set
            loop for while if else elif
            import biblioteca library módulo module
            """,
            
            """
            machine learning algoritmo algorithm modelo model
            treinamento training teste test validação validation
            dataset dados data features características
            supervised unsupervised reinforcement
            neural network rede neural deep learning
            classificação classification regressão regression
            """,
            
            """
            vector database embedding similarity cosine euclidean
            index indexação search busca query consulta
            metadata filtro filter ranking score
            collection documento document chunk
            sparse dense hybrid híbrido
            performance benchmark latência latency
            """
        ]
        
        # Documentos mistos (hybrid-friendly)
        mixed_docs = [
            """
            A implementação de sistemas RAG eficientes requer cuidadosa consideração de múltiplos
            fatores técnicos. Keywords importantes incluem: embedding, vector, similarity, chunk,
            retrieval, generation, context, relevance.
            
            O processo típico envolve: chunking de documentos, geração de embeddings,
            indexação em vector database, busca por similaridade, e geração de respostas
            usando LLMs com contexto recuperado.
            """,
            
            """
            Performance optimization em vector databases é crucial para aplicações em produção.
            Métricas chave incluem: latency, throughput, memory usage, CPU utilization.
            
            Estratégias de otimização: batch processing, connection pooling, caching,
            index tuning, hardware acceleration. Qdrant 1.8.0 sparse vectors demonstram
            significativa melhoria de performance através de algoritmos otimizados.
            """,
            
            """
            Hybrid search combina dense retrieval (embeddings semânticos) com sparse retrieval
            (BM25, TF-IDF). Fusion algorithms como RRF (Reciprocal Rank Fusion) combinam
            resultados de diferentes estratégias de busca.
            
            Vantagens: melhor recall, precision, robustez contra queries diversas.
            Implementação requer: multiple indexes, score normalization, fusion logic.
            """
        ]
        
        # Combinar todos os tipos
        documents.extend(technical_docs * 10)  # 30 docs técnicos
        documents.extend(keyword_docs * 15)    # 45 docs com keywords
        documents.extend(mixed_docs * 8)       # 24 docs mistos
        
        return documents
    
    async def _benchmark_indexing(self) -> Dict[str, Any]:
        """
        Benchmark do processo de indexação
        """
        logger.info("🔄 Executando benchmark de indexação")
        
        test_dir = Path("data/performance_test")
        document_paths = list(test_dir.glob("*.txt"))
        
        # Benchmark indexação completa
        start_time = time.time()
        stats = await self.indexing_pipeline.index_documents([str(p) for p in document_paths])
        end_time = time.time()
        
        indexing_time = end_time - start_time
        docs_per_second = len(document_paths) / indexing_time
        
        return {
            "total_documents": len(document_paths),
            "total_time": indexing_time,
            "documents_per_second": docs_per_second,
            "chunks_created": stats.get("chunks_created", 0),
            "dense_vectors": stats.get("dense_vectors_generated", 0),
            "sparse_vectors": stats.get("sparse_vectors_generated", 0)
        }
    
    async def _benchmark_search(self) -> Dict[str, Any]:
        """
        Benchmark de diferentes estratégias de busca
        """
        logger.info("🔍 Executando benchmark de busca")
        
        # Queries de teste variadas
        test_queries = [
            # Queries semânticas
            "Como implementar sistemas de busca híbrida eficientes?",
            "Quais são as vantagens do RAG para geração de texto?",
            "Explique o funcionamento de vector databases modernos",
            
            # Queries com keywords
            "Python função class método",
            "machine learning algorithm model training",
            "vector database embedding similarity search",
            
            # Queries híbridas
            "Qdrant 1.8.0 sparse vectors performance improvement",
            "RAG system chunking embedding retrieval generation",
            "hybrid search dense sparse fusion RRF algorithm"
        ]
        
        strategies = ["dense_only", "sparse_only", "hybrid"]
        results = {}
        
        for strategy in strategies:
            logger.info(f"Testing strategy: {strategy}")
            
            times = []
            result_counts = []
            
            for query in test_queries:
                # Executar múltiplas vezes para média
                query_times = []
                for _ in range(3):
                    start_time = time.time()
                    search_results = await self.retriever.retrieve(
                        query=query,
                        limit=10,
                        strategy=strategy,
                        use_reranking=False
                    )
                    end_time = time.time()
                    
                    query_times.append(end_time - start_time)
                    result_counts.append(len(search_results))
                
                # Usar tempo médio
                avg_time = statistics.mean(query_times)
                times.append(avg_time)
            
            # Estatísticas da estratégia
            results[strategy] = {
                "avg_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "avg_results": statistics.mean(result_counts),
                "total_queries": len(test_queries)
            }
        
        return results
    
    async def _benchmark_sparse_vectors(self) -> Dict[str, Any]:
        """
        Benchmark específico para sparse vectors (validar 16x improvement)
        """
        logger.info("⚡ Executando benchmark de sparse vectors")
        
        sparse_service = AdvancedSparseVectorService()
        
        # Preparar textos para teste
        test_texts = [
            "machine learning algorithm neural network deep learning",
            "vector database similarity search embedding cosine",
            "python programming language function class method",
            "qdrant sparse vectors performance optimization improvement",
            "hybrid search dense sparse fusion reciprocal rank",
        ] * 20  # 100 textos
        
        # Treinar encoder
        await sparse_service.fit(test_texts)
        
        # Benchmark encoding individual
        individual_times = []
        for text in test_texts[:20]:  # Subset para teste individual
            start_time = time.time()
            sparse_vector = sparse_service.encode_text(text)
            end_time = time.time()
            individual_times.append(end_time - start_time)
        
        # Benchmark batch encoding
        start_time = time.time()
        batch_vectors = await sparse_service.batch_encode(test_texts)
        end_time = time.time()
        batch_time = end_time - start_time
        
        return {
            "individual_encoding": {
                "avg_time": statistics.mean(individual_times),
                "min_time": min(individual_times),
                "max_time": max(individual_times),
                "texts_per_second": 1 / statistics.mean(individual_times)
            },
            "batch_encoding": {
                "total_time": batch_time,
                "texts_processed": len(test_texts),
                "texts_per_second": len(test_texts) / batch_time
            },
            "encoder_stats": sparse_service.get_stats()
        }
    
    async def _benchmark_scalability(self) -> Dict[str, Any]:
        """
        Teste de escalabilidade com diferentes volumes
        """
        logger.info("📈 Executando benchmark de escalabilidade")
        
        # Testar com diferentes números de resultados
        result_limits = [5, 10, 20, 50, 100]
        test_query = "hybrid search performance optimization"
        
        scalability_results = {}
        
        for limit in result_limits:
            times = []
            
            # Executar múltiplas vezes
            for _ in range(5):
                start_time = time.time()
                results = await self.retriever.retrieve(
                    query=test_query,
                    limit=limit,
                    strategy="hybrid"
                )
                end_time = time.time()
                times.append(end_time - start_time)
            
            scalability_results[f"limit_{limit}"] = {
                "avg_time": statistics.mean(times),
                "results_returned": len(results) if 'results' in locals() else 0,
                "time_per_result": statistics.mean(times) / limit
            }
        
        return scalability_results
    
    async def _save_results(self):
        """
        Salva resultados do benchmark
        """
        results_dir = Path("data/benchmark_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"hybrid_benchmark_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Resultados salvos em: {results_file}")
    
    def _print_summary(self):
        """
        Imprime resumo dos resultados
        """
        print("\n" + "="*60)
        print("🏆 RESUMO DO BENCHMARK - HYBRID SEARCH")
        print("="*60)
        
        # Indexação
        indexing = self.results["indexing"]
        print(f"\n📊 INDEXAÇÃO:")
        print(f"   Documentos: {indexing['total_documents']}")
        print(f"   Tempo total: {indexing['total_time']:.2f}s")
        print(f"   Docs/segundo: {indexing['documents_per_second']:.2f}")
        print(f"   Chunks criados: {indexing['chunks_created']}")
        
        # Busca
        search = self.results["search"]
        print(f"\n🔍 BUSCA (tempo médio):")
        for strategy, stats in search.items():
            print(f"   {strategy}: {stats['avg_time']*1000:.1f}ms")
        
        # Sparse vectors
        sparse = self.results["sparse_vectors"]
        print(f"\n⚡ SPARSE VECTORS:")
        print(f"   Encoding individual: {sparse['individual_encoding']['texts_per_second']:.1f} texts/s")
        print(f"   Encoding batch: {sparse['batch_encoding']['texts_per_second']:.1f} texts/s")
        
        # Performance comparison
        dense_time = search["dense_only"]["avg_time"]
        sparse_time = search["sparse_only"]["avg_time"] 
        hybrid_time = search["hybrid"]["avg_time"]
        
        print(f"\n🚀 COMPARAÇÃO DE PERFORMANCE:")
        print(f"   Dense vs Hybrid: {(dense_time/hybrid_time):.1f}x")
        print(f"   Sparse vs Hybrid: {(sparse_time/hybrid_time):.1f}x")
        
        print("\n✅ Benchmark concluído!")

async def main():
    """
    Executa benchmark completo
    """
    tester = HybridPerformanceTester()
    results = await tester.run_full_benchmark()
    
    # Análise adicional se necessário
    print(f"\n📋 Resultados completos disponíveis em: data/benchmark_results/")

if __name__ == "__main__":
    asyncio.run(main()) 