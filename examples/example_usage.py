#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline

def main():
    # Inicializar pipeline
    print("Inicializando RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Exemplo 1: Indexar documentos
    print("\n1. Indexando documentos de exemplo...")
    
    # Criar documento de exemplo
    example_doc = Path("data/raw/example.txt")
    example_doc.parent.mkdir(parents=True, exist_ok=True)
    
    with open(example_doc, "w") as f:
        f.write("""
        Inteligência Artificial (IA) é a simulação de processos de inteligência humana por máquinas,
        especialmente sistemas de computador. Estes processos incluem aprendizado (a aquisição de
        informações e regras para usar as informações), raciocínio (usando regras para chegar a
        conclusões aproximadas ou definitivas) e autocorreção.
        
        Machine Learning é um subconjunto da IA que permite que sistemas aprendam e melhorem
        automaticamente a partir da experiência sem serem explicitamente programados. O aprendizado
        de máquina se concentra no desenvolvimento de programas de computador que podem acessar
        dados e usá-los para aprender por si mesmos.
        
        Deep Learning é um subconjunto do Machine Learning que utiliza redes neurais com múltiplas
        camadas (redes neurais profundas) para analisar vários fatores de dados. É especialmente
        útil para reconhecimento de padrões complexos em grandes conjuntos de dados.
        """)
    
    result = pipeline.index_documents([str(example_doc)])
    print(f"Documentos indexados: {result['total_documents']}")
    print(f"Chunks criados: {result['total_chunks']}")
    
    # Exemplo 2: Fazer queries
    print("\n2. Fazendo queries...")
    
    queries = [
        "O que é Inteligência Artificial?",
        "Qual a diferença entre Machine Learning e Deep Learning?",
        "Como funciona o aprendizado de máquina?"
    ]
    
    for query in queries:
        print(f"\nPergunta: {query}")
        response = pipeline.query(query, k=3)
        print(f"Resposta: {response['answer']}")
        print(f"Fontes encontradas: {len(response['sources'])}")

if __name__ == "__main__":
    main()
