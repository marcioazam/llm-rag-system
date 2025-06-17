#!/bin/bash

echo "=== Corrigindo Problemas da API RAG ==="

# 1. Instalar dependência faltante
echo "1. Instalando aiohttp..."
pip install aiohttp

# 2. Verificar e corrigir o model_router
echo "2. Verificando model_router..."
if [ -f "src/models/model_router.py" ]; then
    echo "Verificando erro do 'name' no model_router..."
    grep -n "name" src/models/model_router.py | head -10
else
    echo "model_router.py não encontrado"
fi

# 3. Verificar resposta do Ollama
echo "3. Testando resposta do Ollama..."
curl -s http://localhost:11434/api/tags | jq '.' 2>/dev/null || curl -s http://localhost:11434/api/tags

# 4. Parar todos os processos relacionados
echo "4. Parando processos existentes..."
pkill -f uvicorn
pkill -f "python.*api"
sleep 2

# 5. Verificar se não há processos órfãos
echo "5. Verificando processos remanescentes..."
ps aux | grep -E "(uvicorn|python.*api)" | grep -v grep || echo "Nenhum processo encontrado"

# 6. Verificar portas em uso
echo "6. Verificando portas 8000 e 8001..."
ss -tulpn | grep -E ":(8000|8001)" || echo "Portas liberadas"

echo "7. Pronto para reiniciar!"
echo "Execute: ./scripts/start_services.sh"
