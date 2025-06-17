#!/usr/bin/env python3

import sys
from pathlib import Path
import glob

sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline

def batch_index_directory(directory: str, pattern: str = "*"):
    """Indexar todos os arquivos de um diretório"""
    
    pipeline = RAGPipeline()
    
    # Encontrar arquivos
    files = glob.glob(f"{directory}/{pattern}")
    
    if not files:
        print(f"Nenhum arquivo encontrado em {directory} com padrão {pattern}")
        return
    
    print(f"Encontrados {len(files)} arquivos para indexar")
    
    # Indexar em batches
    batch_size = 10
    total_chunks = 0
    
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        print(f"\nIndexando batch {i//batch_size + 1}/{(len(files)-1)//batch_size + 1}")
        
        result = pipeline.index_documents(batch)
        total_chunks += result['total_chunks']
        
        if result['errors']:
            print("Erros encontrados:")
            for error in result['errors']:
                print(f"  - {error}")
    
    print(f"\nIndexação completa! Total de chunks: {total_chunks}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Indexar documentos em batch')
    parser.add_argument('directory', help='Diretório com documentos')
    parser.add_argument('--pattern', default='*.pdf', help='Padrão de arquivo (default: *.pdf)')
    
    args = parser.parse_args()
    
    batch_index_directory(args.directory, args.pattern)
