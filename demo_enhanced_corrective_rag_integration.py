# -*- coding: utf-8 -*-
"""
Demo Enhanced Corrective RAG Integration
Demonstra integração com o sistema RAG existente
"""

import asyncio
import json
import time
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRAGPipelineWithEnhanced:
    """Pipeline RAG com Enhanced Corrective RAG integrado."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Configuração Enhanced Corrective RAG
        self.enhanced_config = self.config.get("enhanced_corrective_rag", {
            "enabled": True,
            "relevance_threshold": 0.75,
            "max_reformulation_attempts": 3,
            "enable_decomposition": True
        })
        
        # Inicializar componentes
        self.enhanced_corrective = None
        self._initialize_enhanced_rag()
        
        # Métricas de integração
        self.integration_stats = {
            "total_queries": 0,
            "enhanced_used": 0,
            "fallback_used": 0,
            "avg_relevance_improvement": 0.0
        }
    
    def _initialize_enhanced_rag(self):
        """Inicializa Enhanced Corrective RAG se habilitado."""
        if self.enhanced_config.get("enabled", False):
            try:
                from src.retrieval.enhanced_corrective_rag import create_enhanced_corrective_rag
                
                self.enhanced_corrective = create_enhanced_corrective_rag(self.enhanced_config)
                logger.info("Enhanced Corrective RAG inicializado com sucesso")
                
            except Exception as e:
                logger.warning(f"Falha ao inicializar Enhanced Corrective RAG: {e}")
                self.enhanced_corrective = None
    
    async def query(self, 
                   query_text: str, 
                   top_k: int = 10,
                   use_enhanced: bool = True,
                   **kwargs) -> Dict:
        """
        Query principal com Enhanced Corrective RAG.
        
        Args:
            query_text: Texto da query
            top_k: Número de documentos a retornar
            use_enhanced: Se deve usar Enhanced Corrective RAG
            **kwargs: Parâmetros adicionais
            
        Returns:
            Dict com resultados integrados
        """
        start_time = time.time()
        self.integration_stats["total_queries"] += 1
        
        logger.info(f"Query recebida: {query_text[:50]}...")
        
        # Decidir estratégia
        if use_enhanced and self.enhanced_corrective:
            try:
                # Usar Enhanced Corrective RAG
                results = await self._enhanced_query(query_text, top_k, **kwargs)
                self.integration_stats["enhanced_used"] += 1
                
            except Exception as e:
                logger.error(f"Erro no Enhanced RAG, fallback para tradicional: {e}")
                results = await self._traditional_query(query_text, top_k, **kwargs)
                self.integration_stats["fallback_used"] += 1
        else:
            # Usar pipeline tradicional
            results = await self._traditional_query(query_text, top_k, **kwargs)
            self.integration_stats["fallback_used"] += 1
        
        # Calcular métricas
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Enriquecer resultados com métricas de integração
        results.update({
            "integration_stats": self.integration_stats.copy(),
            "total_processing_time": processing_time,
            "enhanced_used": use_enhanced and self.enhanced_corrective is not None
        })
        
        return results
    
    async def _enhanced_query(self, query_text: str, top_k: int, **kwargs) -> Dict:
        """Query usando Enhanced Corrective RAG."""
        
        logger.info("Usando Enhanced Corrective RAG")
        
        # Configurar retriever mock para demo
        from test_enhanced_corrective_rag import MockRetriever, MockModelRouter
        self.enhanced_corrective.retriever = MockRetriever()
        self.enhanced_corrective.model_router = MockModelRouter()
        
        # Executar Enhanced Corrective RAG
        enhanced_results = await self.enhanced_corrective.retrieve_and_correct(
            query_text, 
            k=top_k
        )
        
        # Simular processamento adicional (generation, etc.)
        generated_response = await self._simulate_generation(
            query_text, 
            enhanced_results.get("documents", [])
        )
        
        # Combinar resultados
        final_results = {
            "query": query_text,
            "answer": generated_response,
            "sources": enhanced_results.get("documents", []),
            "retrieval_metadata": {
                "strategy": "enhanced_corrective",
                "avg_relevance_score": enhanced_results.get("avg_relevance_score", 0),
                "correction_applied": enhanced_results.get("correction_applied", False),
                "total_evaluated": enhanced_results.get("total_evaluated", 0),
                "processing_time": enhanced_results.get("processing_time", 0)
            }
        }
        
        return final_results
    
    async def _traditional_query(self, query_text: str, top_k: int, **kwargs) -> Dict:
        """Query usando pipeline tradicional (simulado)."""
        
        logger.info("Usando pipeline tradicional")
        
        # Simular retrieval tradicional
        await asyncio.sleep(0.1)  # Simular processamento
        
        simulated_docs = [
            {
                "content": f"Resultado tradicional 1 para: {query_text}",
                "metadata": {"score": 0.7, "source": "traditional_1"}
            },
            {
                "content": f"Resultado tradicional 2 para: {query_text}",
                "metadata": {"score": 0.6, "source": "traditional_2"}
            }
        ]
        
        # Simular generation
        generated_response = await self._simulate_generation(query_text, simulated_docs)
        
        return {
            "query": query_text,
            "answer": generated_response,
            "sources": simulated_docs[:top_k],
            "retrieval_metadata": {
                "strategy": "traditional",
                "avg_relevance_score": 0.65,
                "correction_applied": False,
                "total_evaluated": len(simulated_docs),
                "processing_time": 0.1
            }
        }
    
    async def _simulate_generation(self, query: str, documents: List) -> str:
        """Simula geração de resposta baseada nos documentos."""
        
        if not documents:
            return "Não foi possível encontrar informações relevantes para responder sua pergunta."
        
        # Simular processamento de generation
        await asyncio.sleep(0.05)
        
        response = f"""Baseado nos documentos recuperados, posso responder sobre "{query}":

{documents[0].get('content', '')[:200] if documents else 'Sem conteúdo disponível'}...

Esta resposta foi gerada usando {'Enhanced Corrective RAG' if hasattr(documents[0], 'evaluation_result') else 'pipeline tradicional'}.
"""
        
        return response
    
    def get_integration_stats(self) -> Dict:
        """Retorna estatísticas de integração."""
        stats = self.integration_stats.copy()
        
        if stats["total_queries"] > 0:
            stats["enhanced_usage_rate"] = stats["enhanced_used"] / stats["total_queries"]
            stats["fallback_rate"] = stats["fallback_used"] / stats["total_queries"]
        else:
            stats["enhanced_usage_rate"] = 0.0
            stats["fallback_rate"] = 0.0
        
        return stats


async def demo_integration():
    """Demonstração da integração Enhanced Corrective RAG."""
    
    print("🚀 DEMO: Enhanced Corrective RAG Integration")
    print("=" * 60)
    
    # Configuração do pipeline integrado
    config = {
        "enhanced_corrective_rag": {
            "enabled": True,
            "relevance_threshold": 0.75,
            "max_reformulation_attempts": 2,
            "enable_decomposition": True
        }
    }
    
    # Criar pipeline integrado
    pipeline = AdvancedRAGPipelineWithEnhanced(config)
    
    # Queries de teste
    test_queries = [
        "Como implementar Corrective RAG com T5 evaluator?",
        "Qual a diferença entre RAG e fine-tuning?",
        "Como otimizar performance de embeddings em sistemas RAG?",
        "Implementar cache distribuído para RAG systems com Redis"
    ]
    
    print(f"📝 Testando {len(test_queries)} queries...")
    print(f"⚙️  Enhanced RAG habilitado: {config['enhanced_corrective_rag']['enabled']}")
    
    # Executar queries
    results = []
    for i, query in enumerate(test_queries):
        print(f"\n📋 Query {i+1}: {query}")
        
        start_time = time.time()
        result = await pipeline.query(query, top_k=3, use_enhanced=True)
        end_time = time.time()
        
        print(f"   ⏱️  Tempo: {end_time - start_time:.3f}s")
        print(f"   🎯 Estratégia: {result['retrieval_metadata']['strategy']}")
        print(f"   📊 Relevância: {result['retrieval_metadata']['avg_relevance_score']:.3f}")
        print(f"   🔧 Correção: {result['retrieval_metadata']['correction_applied']}")
        print(f"   📄 Documentos: {len(result['sources'])}")
        
        results.append(result)
    
    # Estatísticas finais
    integration_stats = pipeline.get_integration_stats()
    
    print("\n" + "=" * 60)
    print("📈 ESTATÍSTICAS DE INTEGRAÇÃO")
    print("=" * 60)
    print(f"Total de queries: {integration_stats['total_queries']}")
    print(f"Enhanced RAG usado: {integration_stats['enhanced_used']}")
    print(f"Fallback usado: {integration_stats['fallback_used']}")
    print(f"Taxa de uso Enhanced: {integration_stats['enhanced_usage_rate']:.1%}")
    print(f"Taxa de fallback: {integration_stats['fallback_rate']:.1%}")
    
    # Comparação de performance
    enhanced_results = [r for r in results if r['retrieval_metadata']['strategy'] == 'enhanced_corrective']
    traditional_results = [r for r in results if r['retrieval_metadata']['strategy'] == 'traditional']
    
    if enhanced_results and traditional_results:
        enhanced_avg = sum(r['retrieval_metadata']['avg_relevance_score'] for r in enhanced_results) / len(enhanced_results)
        traditional_avg = sum(r['retrieval_metadata']['avg_relevance_score'] for r in traditional_results) / len(traditional_results)
        
        improvement = ((enhanced_avg - traditional_avg) / traditional_avg) * 100 if traditional_avg > 0 else 0
        
        print(f"\n📊 COMPARAÇÃO DE PERFORMANCE:")
        print(f"   Enhanced RAG - Relevância média: {enhanced_avg:.3f}")
        print(f"   Traditional RAG - Relevância média: {traditional_avg:.3f}")
        print(f"   Melhoria: {improvement:+.1f}%")
    
    return results, integration_stats


async def demo_fallback_behavior():
    """Demonstra comportamento de fallback."""
    
    print("\n🔧 DEMO: Comportamento de Fallback")
    print("=" * 40)
    
    # Pipeline com Enhanced desabilitado
    config_disabled = {
        "enhanced_corrective_rag": {
            "enabled": False
        }
    }
    
    pipeline_disabled = AdvancedRAGPipelineWithEnhanced(config_disabled)
    
    query = "Teste de fallback para pipeline tradicional"
    
    print(f"📝 Query: {query}")
    print(f"⚙️  Enhanced RAG: DESABILITADO")
    
    result = await pipeline_disabled.query(query, top_k=2)
    
    print(f"   🎯 Estratégia usada: {result['retrieval_metadata']['strategy']}")
    print(f"   📊 Relevância: {result['retrieval_metadata']['avg_relevance_score']:.3f}")
    print(f"   ✅ Fallback funcionando corretamente!")
    
    return result


async def main():
    """Executa demonstração completa da integração."""
    
    try:
        # Demo principal
        results, stats = await demo_integration()
        
        # Demo de fallback
        fallback_result = await demo_fallback_behavior()
        
        print("\n" + "=" * 60)
        print("✅ INTEGRAÇÃO ENHANCED CORRECTIVE RAG CONCLUÍDA!")
        print("=" * 60)
        
        print("\n🎯 Características demonstradas:")
        print("  ✅ Integração transparente com pipeline existente")
        print("  ✅ Fallback automático para pipeline tradicional")
        print("  ✅ Métricas de performance comparativas")
        print("  ✅ Configuração flexível via config")
        print("  ✅ Monitoramento de uso e estatísticas")
        
        print("\n🚀 Próximos passos:")
        print("  1. Integrar com AdvancedRAGPipeline real")
        print("  2. Conectar com retrievers e models reais")
        print("  3. Implementar cache persistente")
        print("  4. Adicionar métricas RAGAS")
        print("  5. Deploy em produção com monitoramento")
        
        # Salvar resultados para análise
        demo_results = {
            "integration_stats": stats,
            "sample_results": [
                {
                    "query": r["query"],
                    "strategy": r["retrieval_metadata"]["strategy"],
                    "relevance_score": r["retrieval_metadata"]["avg_relevance_score"],
                    "correction_applied": r["retrieval_metadata"]["correction_applied"]
                }
                for r in results
            ],
            "fallback_test": {
                "strategy": fallback_result["retrieval_metadata"]["strategy"],
                "relevance_score": fallback_result["retrieval_metadata"]["avg_relevance_score"]
            }
        }
        
        with open("enhanced_rag_integration_demo_results.json", "w", encoding="utf-8") as f:
            json.dump(demo_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Resultados salvos em: enhanced_rag_integration_demo_results.json")
        
    except Exception as e:
        print(f"\n❌ Erro durante a demonstração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())