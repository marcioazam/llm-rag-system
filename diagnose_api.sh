#!/bin/bash

echo "=== Diagnóstico da API RAG ==="

# 1. Verificar se a API está realmente rodando
echo "1. Verificando processos da API..."
ps aux | grep -E "(uvicorn|python.*api)" | grep -v grep

# 2. Verificar portas em uso
echo -e "\n2. Verificando porta 8000..."
netstat -tulpn | grep :8000 || ss -tulpn | grep :8000

# 3. Testar conexão básica
echo -e "\n3. Testando conexão HTTP..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000 2>/dev/null || echo "Conexão falhou"

# 4. Verificar logs detalhados
echo -e "\n4. Últimas linhas do log (se existir)..."
if [ -f "logs/rag_api.log" ]; then
    echo "Log encontrado:"
    tail -20 logs/rag_api.log
else
    echo "Arquivo de log não encontrado em logs/rag_api.log"
fi

# 5. Verificar estrutura de diretórios
echo -e "\n5. Verificando estrutura de arquivos críticos..."
for file in "src/api/main.py" "src/rag_pipeline.py" "requirements.txt"; do
    if [ -f "$file" ]; then
        echo "✓ $file existe"
    else
        echo "✗ $file não encontrado"
    fi
done

# 6. Verificar se há erros no processo uvicorn
echo -e "\n6. Verificando processos uvicorn em execução..."
if pgrep -f uvicorn > /dev/null; then
    echo "Processo uvicorn encontrado:"
    pgrep -f uvicorn | xargs ps -p
else
    echo "Nenhum processo uvicorn em execução"
fi

# 7. Tentar iniciar a API manualmente para ver erros
echo -e "\n7. Testando inicialização manual..."
echo "Executando: cd $(pwd) && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 0"
echo "Aguarde 10 segundos..."

# Executar em background e capturar saída
timeout 10s python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 0 &
PID=$!
sleep 3

# Verificar se o processo ainda está rodando
if kill -0 $PID 2>/dev/null; then
    echo "✓ API iniciou com sucesso na porta 8001"
    # Testar a API
    sleep 2
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001 2>/dev/null)
    echo "Código de resposta HTTP: $HTTP_CODE"
    kill $PID 2>/dev/null
else
    echo "✗ API falhou ao iniciar"
fi

echo -e "\n8. Verificando dependências críticas..."
python -c "
import sys
print(f'Python: {sys.version}')

modules = ['fastapi', 'uvicorn', 'sentence_transformers', 'chromadb', 'ollama']
for module in modules:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError as e:
        print(f'✗ {module}: {e}')
"

echo -e "\n=== Fim do Diagnóstico ==="
