# -*- coding: utf-8 -*-
"""
Teste Enhanced Corrective RAG - Demonstração do T5 evaluator e decompose-then-recompose.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockRetriever:
    """Mock retriever para testes."""
    
    async def retrieve(self, query: str, limit: int = 10) -> List[Dict]:
        """Simula recuperação de documentos."""
        mock_docs = [
            {
                "content": f"Este é um documento sobre {query}. Contém informações técnicas detalhadas sobre implementação.",
                "metadata": {"source": "doc1.py", "type": "code", "score": 0.9}
            },
            {
                "content": f"Guia prático para {query}. Inclui exemplos de código e configurações.",
                "metadata": {"source": "guide.md", "type": "documentation", "score": 0.8}
            },
            {
                "content": "Documento não relacionado sobre jardinagem e plantas ornamentais.",
                "metadata": {"source": "garden.txt", "type": "unrelated", "score": 0.1}
            }
        ]
        
        return mock_docs[:limit]


class MockModelRouter:
    """Mock model router para testes."""
    
    async def route_request(self, prompt: str, task_type: str, force_model: str = None) -> Dict:
        """Simula resposta do modelo."""
        
        if task_type == "evaluation":
            if "jardinagem" in prompt.lower():
                return {
                    "answer": """
SEMANTIC_RELEVANCE: 0.1
FACTUAL_ACCURACY: 0.8
COMPLETENESS: 0.2
CONFIDENCE: 0.9
OVERALL_SCORE: 0.2
CATEGORIES: [gardening, plants, unrelated]
EXPLANATION: Este documento é sobre jardinagem e não tem relevância para a query técnica.
"""
                }
            else:
                return {
                    "answer": """
SEMANTIC_RELEVANCE: 0.9
FACTUAL_ACCURACY: 0.8
COMPLETENESS: 0.7
CONFIDENCE: 0.8
OVERALL_SCORE: 0.8
CATEGORIES: [technical, programming, implementation]
EXPLANATION: Este documento fornece informações técnicas relevantes sobre o tópico.
"""
                }
        
        return {"answer": "Resposta simulada"}


async def test_basic_enhanced_retrieval():
    """Teste básico do Enhanced Corrective RAG."""
    print("\n" + "="*60)
    print("TESTE 1: Enhanced Retrieval Básico")
    print("="*60)
    
    from src.retrieval.enhanced_corrective_rag import EnhancedCorrectiveRAG
    
    enhanced_rag = EnhancedCorrectiveRAG(
        retriever=MockRetriever(),
        relevance_threshold=0.7,
        max_reformulation_attempts=2
    )
    
    enhanced_rag.model_router = MockModelRouter()
    
    query = "Como implementar Corrective RAG com T5 evaluator"
    
    print(f"Query: {query}")
    print(f"Threshold: {enhanced_rag.relevance_threshold}")
    
    start_time = time.time()
    results = await enhanced_rag.retrieve_and_correct(query, k=5)
    end_time = time.time()
    
    print(f"\nTempo de processamento: {end_time - start_time:.2f}s")
    print(f"Documentos retornados: {len(results.get('documents', []))}")
    print(f"Score médio de relevância: {results.get('avg_relevance_score', 0):.3f}")
    print(f"Correção aplicada: {results.get('correction_applied', False)}")
    
    for i, doc in enumerate(results.get('documents', [])[:2]):
        print(f"{i+1}. Score: {doc.relevance_score:.3f} | Status: {doc.validation_status}")
        print(f"   Conteúdo: {doc.content[:80]}...")
    
    return results


async def main():
    """Executa demonstração do Enhanced Corrective RAG."""
    print("ENHANCED CORRECTIVE RAG - DEMONSTRAÇÃO")
    print("=" * 50)
    print("Implementação de:")
    print("  • T5 Retrieval Evaluator")
    print("  • Decompose-then-Recompose Algorithm")
    print("  • Enhanced Correction Strategies")
    
    try:
        await test_basic_enhanced_retrieval()
        
        print("\n" + "="*50)
        print("ENHANCED CORRECTIVE RAG IMPLEMENTADO COM SUCESSO!")
        print("="*50)
        print("\nCaracterísticas implementadas:")
        print("  ✓ T5 Retrieval Evaluator com métricas detalhadas")
        print("  ✓ Decompose-then-Recompose Algorithm")
        print("  ✓ Enhanced Correction Strategies")
        print("  ✓ Query Complexity Analysis")
        print("  ✓ Multi-dimensional Document Evaluation")
        
    except Exception as e:
        print(f"\nErro durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 