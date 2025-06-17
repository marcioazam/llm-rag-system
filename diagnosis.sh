#!/bin/bash

echo "=== DIAGNÓSTICO DA API RAG ==="
echo

# 1. Verificar se a API está rodando na porta correta
echo "1. Verificando portas em uso:"
netstat -tlnp | grep :8000
echo

# 2. Testar conectividade básica
echo "2. Testando conectividade HTTP:"
curl -v http://localhost:8000/health 2>&1 | head -10
echo

# 3. Verificar logs em tempo real
echo "3. Últimas linhas do log:"
if [ -f logs/rag_api.log ]; then
    tail -20 logs/rag_api.log
else
    echo "Arquivo logs/rag_api.log não encontrado"
fi
echo

# 4. Verificar processos relacionados
echo "4. Processos Python/Uvicorn:"
ps aux | grep -E "(python|uvicorn)" | grep -v grep
echo

# 5. Verificar status do Ollama
echo "5. Status do Ollama:"
curl -s http://localhost:11434/api/tags | python3 -m json.tool 2>/dev/null || echo "Ollama não responde"
echo

# 6. Verificar uso de recursos
echo "6. Uso de memória:"
free -h
echo

# 7. Testar endpoint específico
echo "7. Testando endpoint /docs:"
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs
echo " <- Código de resposta do /docs"
echo

# 8. Verificar se há erros no processo
echo "8. Verificando logs de erro do sistema:"
journalctl --user -u your-service --since "5 minutes ago" --no-pager 2>/dev/null || echo "Sem logs systemd específicos"
echo

echo "=== TESTES ADICIONAIS ==="

# 9. Testar com timeout maior
echo "9. Teste com timeout de 10s:"
timeout 10s curl -v http://localhost:8000/health 2>&1 || echo "Timeout ou erro"
echo

# 10. Verificar firewall/iptables
echo "10. Regras de firewall (se aplicável):"
sudo iptables -L INPUT | grep -E "(8000|REJECT|DROP)" 2>/dev/null || echo "Sem regras de firewall restritivas visíveis"
echo

echo "=== COMANDOS PARA INVESTIGAÇÃO MANUAL ==="
echo "# Ver logs em tempo real:"
echo "tail -f logs/rag_api.log"
echo
echo "# Testar API manualmente:"
echo "curl -X GET http://localhost:8000/health"
echo "curl -X GET http://localhost:8000/docs"
echo
echo "# Verificar modelos Ollama:"
echo "ollama list"
echo "ollama ps"
echo
echo "# Reiniciar apenas a API:"
echo "./scripts/stop_services.sh && ./scripts/start_services.sh"
