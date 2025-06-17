#!/usr/bin/env python3
"""
Script para corrigir problemas de importação no projeto RAG
"""

import os
import re
import sys
from pathlib import Path

def fix_imports_in_file(file_path: Path):
    """Corrige as importações em um arquivo específico"""
    print(f"Corrigindo: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Padrões de correção de importação
    corrections = [
        # Importações relativas incorretas
        (r'from \.\.chunking\.base_chunker import', 'from src.chunking.base_chunker import'),
        (r'from \.\.embeddings\.embedding_service import', 'from src.embeddings.embedding_service import'),
        (r'from \.\.vectordb\.chroma_store import', 'from src.vectordb.chroma_store import'),
        (r'from \.\.utils\.document_loader import', 'from src.utils.document_loader import'),
        (r'from \.\.retrieval\.retriever import', 'from src.retrieval.retriever import'),
        (r'from \.\.models\.model_router import', 'from src.models.model_router import'),
        
        # Importações com pontos extras
        (r'from \.([^.]+) import', r'from src.\1 import'),
        
        # Importações absolutas incorretas dentro do src
        (r'from vectordb\.chroma_store import', 'from src.vectordb.chroma_store import'),
        (r'from chunking\.base_chunker import', 'from src.chunking.base_chunker import'),
        (r'from embeddings\.embedding_service import', 'from src.embeddings.embedding_service import'),
        (r'from utils\.document_loader import', 'from src.utils.document_loader import'),
        (r'from retrieval\.retriever import', 'from src.retrieval.retriever import'),
        (r'from models\.model_router import', 'from src.models.model_router import'),
    ]
    
    # Aplica as correções
    for pattern, replacement in corrections:
        content = re.sub(pattern, replacement, content)
    
    # Salva apenas se houve mudanças
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Arquivo corrigido")
        return True
    else:
        print(f"  - Nenhuma correção necessária")
        return False

def add_init_files():
    """Adiciona arquivos __init__.py necessários"""
    init_files = [
        'src/__init__.py',
        'src/api/__init__.py',
        'src/chunking/__init__.py',
        'src/embeddings/__init__.py',
        'src/models/__init__.py',
        'src/retrieval/__init__.py',
        'src/utils/__init__.py',
        'src/vectordb/__init__.py',
        'src/cli/__init__.py',
        'src/client/__init__.py'
    ]
    
    created_files = []
    for init_file in init_files:
        if not os.path.exists(init_file):
            os.makedirs(os.path.dirname(init_file), exist_ok=True)
            with open(init_file, 'w') as f:
                f.write('# -*- coding: utf-8 -*-\n')
            created_files.append(init_file)
    
    if created_files:
        print(f"Arquivos __init__.py criados: {len(created_files)}")
        for file in created_files:
            print(f"  + {file}")
    else:
        print("Todos os arquivos __init__.py já existem")

def fix_rag_pipeline():
    """Corrige especificamente o arquivo rag_pipeline.py"""
    rag_file = Path('src/rag_pipeline.py')
    if not rag_file.exists():
        print(f"❌ Arquivo {rag_file} não encontrado")
        return
    
    print(f"Corrigindo importações específicas em {rag_file}")
    
    with open(rag_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Correções específicas para rag_pipeline.py
    specific_corrections = [
        # Problema específico detectado no erro
        ('from vectordb.chroma_store import ChromaStore as ChromaVectorStore', 
         'from src.vectordb.chroma_store import ChromaStore as ChromaVectorStore'),
        
        # Outras possíveis importações problemáticas
        ('from .models.model_router import ModelRouter, AdvancedModelRouter',
         'from src.models.model_router import ModelRouter, AdvancedModelRouter'),
    ]
    
    for old_import, new_import in specific_corrections:
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"  ✓ Corrigido: {old_import}")
    
    with open(rag_file, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Função principal"""
    print("=== Corrigindo Importações do Projeto RAG ===\n")
    
    # Verifica se estamos no diretório correto
    if not os.path.exists('src'):
        print("❌ Diretório 'src' não encontrado. Execute este script na raiz do projeto.")
        sys.exit(1)
    
    # 1. Adiciona arquivos __init__.py
    print("1. Adicionando arquivos __init__.py...")
    add_init_files()
    print()
    
    # 2. Corrige importações específicas
    print("2. Corrigindo rag_pipeline.py...")
    fix_rag_pipeline()
    print()
    
    # 3. Procura e corrige todos os arquivos Python
    print("3. Corrigindo importações em todos os arquivos Python...")
    src_dir = Path('src')
    python_files = list(src_dir.rglob('*.py'))
    
    corrected_files = 0
    for py_file in python_files:
        if fix_imports_in_file(py_file):
            corrected_files += 1
    
    print(f"\n✅ Correção concluída!")
    print(f"   - Arquivos Python encontrados: {len(python_files)}")
    print(f"   - Arquivos corrigidos: {corrected_files}")
    print(f"\n💡 Próximos passos:")
    print(f"   1. Execute: python3 -m src.api.main")
    print(f"   2. Ou use: ./scripts/start_services.sh")

if __name__ == '__main__':
    main()
