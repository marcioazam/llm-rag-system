#!/bin/bash

# Script de setup do sistema RAG

echo "=== RAG System Setup ==="

# Criar ambiente virtual
echo "Criando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
echo "Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

# Criar diretórios necessários
echo "Criando estrutura de diretórios..."
mkdir -p data/{raw,processed,indexes/chroma}
mkdir -p logs

# Baixar modelo de embeddings (opcional - será baixado automaticamente)
echo "Pré-baixando modelo de embeddings..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Verificar Ollama
echo "Verificando Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama não encontrado. Por favor, instale o Ollama primeiro."
    echo "curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Verificar se o modelo está disponível
echo "Verificando modelo llama3.1:8b-instruct-q4_K_M-instruct-q4_K_M..."
if ! ollama list | grep -q "llama3.1:8b-instruct-q4_K_M-instruct-q4_K_M"; then
    echo "Modelo llama3.1:8b-instruct-q4_K_M-instruct-q4_K_M não encontrado. Baixando..."
    ollama pull llama3.1:8b-instruct-q4_K_M-instruct-q4_K_M
fi

echo "Setup concluído!"
