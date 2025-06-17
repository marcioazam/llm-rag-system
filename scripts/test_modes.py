#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline

def test_query_modes():
    print("=== Testando Diferentes Modos de Query ===\n")
    
    pipeline = RAGPipeline()
    
    # Perguntas de teste
    questions = [
        "O que é RAG?",  # Deve encontrar nos docs
        "Qual é a capital do Brasil?",  # Não está nos docs, usa LLM
        "Como funciona o chunking semântico?",  # Pode estar nos docs
        "Quem foi Albert Einstein?",  # Não está nos docs, usa LLM
    ]
    
    for question in questions:
        print(f"\nPergunta: {question}")
        print("-" * 50)
        
        # Teste 1: RAG com fallback automático
        result = pipeline.query(question)
        print(f"Modo: {result['response_mode']}")
        print(f"Fontes encontradas: {len(result['sources'])}")
        print(f"Resposta: {result['answer'][:200]}...")
        
        print("-" * 30)
        
        # Teste 2: Apenas LLM
        result_llm = pipeline.query_llm_only(question)
        print(f"Modo LLM-only: {result_llm['response_mode']}")
        print(f"Resposta: {result_llm['answer'][:200]}...")
        
        print("=" * 50)

if __name__ == "__main__":
    test_query_modes()
