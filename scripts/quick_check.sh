#!/bin/bash

echo "=== Verificação Rápida do Sistema RAG ==="

# Verificar arquivos principais
echo -e "\n1. Verificando arquivos principais..."

files_to_check=(
    "src/chunking/base_chunker.py"
    "config/config.yaml"
    "src/rag_pipeline.py"
    "src/api/main.py"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        # Verificar se é Python e tem sintaxe válida
        if [[ "$file" == *.py ]]; then
            if python -m py_compile "$file" 2>/dev/null; then
                echo "✓ $file - OK"
            else
                echo "✗ $file - Erro de sintaxe!"
            fi
        else
            echo "✓ $file - Existe"
        fi
    else
        echo "✗ $file - Não encontrado!"
    fi
done

# Verificar se base_chunker.py não tem conteúdo YAML
echo -e "\n2. Verificando conteúdo de base_chunker.py..."
if grep -q "llm:" src/chunking/base_chunker.py 2>/dev/null; then
    echo "✗ base_chunker.py contém conteúdo YAML incorreto!"
else
    echo "✓ base_chunker.py parece estar correto"
fi

# Verificar imports
echo -e "\n3. Testando imports principais..."
python -c "from src.chunking.base_chunker import BaseChunker, Chunk" 2>/dev/null && echo "✓ Import base_chunker OK" || echo "✗ Import base_chunker FALHOU"

echo -e "\nVerificação concluída!"
