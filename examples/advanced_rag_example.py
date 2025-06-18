"""
Exemplo Prático: Sistema RAG Avançado Completo
Demonstra todas as features implementadas trabalhando em conjunto
"""

import asyncio
import logging
from typing import Dict, Any, List
import time
from pathlib import Path

# Importar componentes avançados
from src.chunking.language_aware_chunker import LanguageAwareChunker
from src.graphrag.graph_rag_enhancer import GraphRAGEnhancer, CodeEntity
from src.retrieval.colbert_reranker import HybridReranker
from src.cache.multi_layer_cache import create_multi_layer_cache
from src.monitoring.rag_monitor import RAGMonitor, QueryMetrics
from src.optimization.performance_tuner import create_performance_tuner
from src.retrieval.hybrid_retriever import HybridRetriever
from src.embeddings.api_embedding_service import APIEmbeddingService
from src.models.api_model_router import APIModelRouter

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRAGSystem:
    """
    Sistema RAG completo com todas as features avançadas
    """
    
    def __init__(self):
        logger.info("Inicializando Sistema RAG Avançado...")
        
        # Componentes core
        self.embedding_service = APIEmbeddingService()
        self.model_router = APIModelRouter()
        
        # Componentes avançados
        self.language_chunker = LanguageAwareChunker()
        self.graph_enhancer = GraphRAGEnhancer()
        self.hybrid_reranker = HybridReranker()
        self.hybrid_retriever = HybridRetriever()
        self.monitor = RAGMonitor()
        
        # Cache e performance
        self.cache = None
        self.performance_tuner = None
        
        # Estatísticas
        self.stats = {
            'queries_processed': 0,
            'cache_hits': 0,
            'avg_latency': 0.0
        }
    
    async def initialize(self):
        """Inicializa componentes assíncronos"""
        # Inicializar cache multi-layer
        self.cache = await create_multi_layer_cache(
            semantic_threshold=0.95,
            enable_redis=True
        )
        
        # Executar performance tuning
        self.performance_tuner = await create_performance_tuner()
        
        # Iniciar monitoring
        await self.monitor.start_monitoring()
        
        logger.info("Sistema RAG Avançado inicializado com sucesso!")
    
    async def index_codebase(self, code_directory: str):
        """
        Indexa codebase com chunking inteligente e GraphRAG
        """
        logger.info(f"Indexando codebase: {code_directory}")
        
        code_entities = []
        chunks_created = 0
        
        # Processar arquivos de código
        for file_path in Path(code_directory).rglob("*.py"):
            # Ler conteúdo
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Language-aware chunking
            chunks = self.language_chunker.chunk_code(
                code=content,
                language='python',
                file_path=str(file_path)
            )
            
            # Criar entidades para GraphRAG
            for i, chunk in enumerate(chunks):
                entity = CodeEntity(
                    id=f"{file_path}_{i}",
                    name=f"{file_path.stem}_{chunk.chunk_type}_{i}",
                    type=chunk.chunk_type,
                    content=chunk.content,
                    file_path=str(file_path),
                    metadata={
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line,
                        'has_context': bool(chunk.context)
                    }
                )
                code_entities.append(entity)
                chunks_created += 1
        
        logger.info(f"Criados {chunks_created} chunks de {len(code_entities)} arquivos")
        
        # Construir knowledge graph
        graph = await self.graph_enhancer.build_knowledge_graph(code_entities)
        
        # Detectar comunidades
        communities = await self.graph_enhancer.detect_communities()
        logger.info(f"Detectadas {len(communities)} comunidades de código")
        
        # Indexar no Qdrant (hybrid search)
        await self.hybrid_retriever.index_documents(
            [{'content': e.content, 'metadata': e.metadata} for e in code_entities]
        )
        
        return {
            'files_processed': len(set(e.file_path for e in code_entities)),
            'chunks_created': chunks_created,
            'communities_detected': len(communities),
            'graph_nodes': graph.number_of_nodes(),
            'graph_edges': graph.number_of_edges()
        }
    
    async def query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Processa query com pipeline completo
        """
        start_time = time.time()
        self.stats['queries_processed'] += 1
        
        # 1. Verificar cache se habilitado
        if use_cache:
            cached_result, cache_type = await self.cache.get(query)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.info(f"Cache hit ({cache_type}) para: {query[:50]}...")
                return cached_result
        
        # 2. Adaptive routing
        strategy, reason = self.monitor.adaptive_router.analyze_query_complexity(query)
        logger.info(f"Estratégia selecionada: {strategy} - {reason}")
        
        # 3. Processar baseado na estratégia
        if strategy == 'graph_traversal':
            result = await self._process_with_graphrag(query)
        elif strategy == 'hybrid_search':
            result = await self._process_with_hybrid_search(query)
        else:
            result = await self._process_simple_search(query)
        
        # 4. Adicionar ao cache
        if use_cache:
            await self.cache.set(
                key=query,
                value=result,
                cache_types=['semantic', 'prefix', 'kv']
            )
        
        # 5. Registrar métricas
        total_latency = time.time() - start_time
        await self._record_metrics(query, result, strategy, total_latency)
        
        # Atualizar estatísticas
        self.stats['avg_latency'] = (
            (self.stats['avg_latency'] * (self.stats['queries_processed'] - 1) + total_latency) /
            self.stats['queries_processed']
        )
        
        return result
    
    async def _process_with_graphrag(self, query: str) -> Dict[str, Any]:
        """Processa query complexa com GraphRAG"""
        logger.info("Processando com GraphRAG (multi-hop reasoning)")
        
        # Embedding da query
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Encontrar entidades iniciais relevantes
        initial_results = await self.hybrid_retriever.retrieve(
            query=query,
            top_k=5
        )
        
        # Multi-hop reasoning
        graph_results = []
        for result in initial_results[:3]:
            entity_id = result.get('metadata', {}).get('entity_id')
            if entity_id:
                hop_result = await self.graph_enhancer.multi_hop_reasoning(
                    start_entity_id=entity_id,
                    query=query,
                    max_hops=3
                )
                graph_results.append(hop_result)
        
        # Combinar resultados
        combined_context = await self._combine_graph_results(graph_results)
        
        # Gerar resposta
        response = await self.model_router.route_request(
            prompt=self._create_graphrag_prompt(query, combined_context),
            task_type="complex_reasoning"
        )
        
        return {
            'answer': response,
            'strategy': 'graph_traversal',
            'graph_paths': len(graph_results),
            'context_sources': combined_context['sources']
        }
    
    async def _process_with_hybrid_search(self, query: str) -> Dict[str, Any]:
        """Processa com busca híbrida e reranking"""
        logger.info("Processando com Hybrid Search + ColBERT Reranking")
        
        # Busca híbrida
        initial_results = await self.hybrid_retriever.retrieve(
            query=query,
            top_k=20  # Mais resultados para reranking
        )
        
        # Preparar documentos para reranking
        documents = [
            {
                'id': str(i),
                'content': result['content'],
                'score': result['score'],
                'metadata': result.get('metadata', {})
            }
            for i, result in enumerate(initial_results)
        ]
        
        # ColBERT reranking
        reranked_results = await self.hybrid_reranker.rerank_with_strategy(
            query=query,
            documents=documents,
            strategy="auto"
        )
        
        # Usar top resultados rerankeados
        top_results = reranked_results[:5]
        context = "\n\n".join([r.content for r in top_results])
        
        # Gerar resposta
        response = await self.model_router.route_request(
            prompt=self._create_hybrid_prompt(query, context),
            task_type="technical_analysis"
        )
        
        return {
            'answer': response,
            'strategy': 'hybrid_search',
            'documents_retrieved': len(initial_results),
            'documents_after_rerank': len(top_results),
            'rerank_improvement': self._calculate_rerank_improvement(documents, reranked_results)
        }
    
    async def _process_simple_search(self, query: str) -> Dict[str, Any]:
        """Processa query simples com busca vetorial"""
        logger.info("Processando com busca vetorial simples")
        
        # Busca vetorial básica
        results = await self.hybrid_retriever.vector_store.search(
            query=query,
            top_k=5
        )
        
        # Contexto dos resultados
        context = "\n\n".join([r['content'] for r in results])
        
        # Gerar resposta
        response = await self.model_router.route_request(
            prompt=self._create_simple_prompt(query, context),
            task_type="simple_qa"
        )
        
        return {
            'answer': response,
            'strategy': 'vector_only',
            'documents_retrieved': len(results)
        }
    
    async def _record_metrics(self, query: str, result: Dict, strategy: str, latency: float):
        """Registra métricas detalhadas"""
        metrics = QueryMetrics(
            query_id=f"q_{int(time.time())}",
            query_text=query,
            timestamp=time.time(),
            total_latency=latency,
            routing_strategy=strategy,
            routing_reason=f"Complexity analysis: {strategy}",
            documents_retrieved=result.get('documents_retrieved', 0),
            cache_hit=False
        )
        
        await self.monitor.record_query(metrics)
    
    def _create_graphrag_prompt(self, query: str, context: Dict) -> str:
        """Cria prompt para GraphRAG"""
        return f"""Based on the following graph traversal analysis and code relationships:

{context['summary']}

Related code paths:
{context['paths']}

Question: {query}

Provide a comprehensive answer that explains the relationships and connections found."""
    
    def _create_hybrid_prompt(self, query: str, context: str) -> str:
        """Cria prompt para hybrid search"""
        return f"""Based on the following code context retrieved through hybrid search:

{context}

Question: {query}

Provide a detailed technical answer."""
    
    def _create_simple_prompt(self, query: str, context: str) -> str:
        """Cria prompt simples"""
        return f"""Context:
{context}

Question: {query}

Answer:"""
    
    async def _combine_graph_results(self, graph_results: List[Dict]) -> Dict[str, Any]:
        """Combina resultados do GraphRAG"""
        all_paths = []
        all_entities = set()
        
        for result in graph_results:
            all_paths.extend(result.get('relevant_paths', []))
            for path in result.get('relevant_paths', []):
                all_entities.update(path['path'])
        
        return {
            'summary': f"Analyzed {len(all_entities)} entities across {len(all_paths)} paths",
            'paths': '\n'.join([str(p) for p in all_paths[:5]]),
            'sources': list(all_entities)[:10]
        }
    
    def _calculate_rerank_improvement(self, original: List[Dict], reranked: List[Any]) -> float:
        """Calcula improvement do reranking"""
        if not original or not reranked:
            return 0.0
        
        # Simplificado: comparar scores médios do top-5
        original_avg = sum(d['score'] for d in original[:5]) / min(5, len(original))
        reranked_avg = sum(r.rerank_score for r in reranked[:5]) / min(5, len(reranked))
        
        improvement = ((reranked_avg - original_avg) / original_avg) * 100
        return round(improvement, 1)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema"""
        cache_stats = self.cache.get_stats()
        monitor_summary = self.monitor.get_metrics_summary()
        
        return {
            'system_stats': self.stats,
            'cache_stats': cache_stats,
            'monitor_summary': monitor_summary,
            'performance_profile': self.performance_tuner.get_optimization_summary()
        }

async def main():
    """Demonstração do sistema RAG avançado"""
    # Inicializar sistema
    rag_system = AdvancedRAGSystem()
    await rag_system.initialize()
    
    # Indexar código de exemplo
    logger.info("\n=== INDEXANDO CODEBASE ===")
    index_stats = await rag_system.index_codebase("./src")
    logger.info(f"Indexação completa: {index_stats}")
    
    # Testar diferentes tipos de queries
    test_queries = [
        # Query simples
        "O que é a classe APIEmbeddingService?",
        
        # Query técnica (beneficia de hybrid search)
        "Como implementar cache semântico com threshold de similaridade?",
        
        # Query complexa (requer GraphRAG)
        "Explique a relação entre o sistema de cache e o monitoring, e como eles interagem para otimizar performance",
        
        # Query de debugging
        "Quais são os possíveis problemas de performance no reranking e como resolver?"
    ]
    
    logger.info("\n=== TESTANDO QUERIES ===")
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        
        # Processar query
        result = await rag_system.query(query)
        
        logger.info(f"Estratégia usada: {result['strategy']}")
        logger.info(f"Resposta: {result['answer'][:200]}...")
        
        # Testar cache (segunda vez deve ser hit)
        logger.info("Testando cache...")
        cached_result = await rag_system.query(query)
        
        await asyncio.sleep(1)  # Pequena pausa entre queries
    
    # Mostrar estatísticas finais
    logger.info("\n=== ESTATÍSTICAS DO SISTEMA ===")
    stats = await rag_system.get_system_stats()
    
    logger.info(f"Queries processadas: {stats['system_stats']['queries_processed']}")
    logger.info(f"Cache hits: {stats['system_stats']['cache_hits']}")
    logger.info(f"Latência média: {stats['system_stats']['avg_latency']:.2f}s")
    logger.info(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
    logger.info(f"P95 latency: {stats['monitor_summary'].get('p95_latency', 0):.2f}s")
    
    # Exportar métricas
    await rag_system.monitor.export_metrics("metrics/rag_metrics.json")
    logger.info("\nMétricas exportadas para metrics/rag_metrics.json")

if __name__ == "__main__":
    asyncio.run(main()) 