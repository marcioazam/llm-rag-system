#!/bin/bash

# Script para iniciar todos os serviços com suporte multi-modelo

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Iniciando Sistema RAG Multi-Modelo ===${NC}"

# Ativar ambiente virtual
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Ambiente virtual ativado${NC}"
else
    echo -e "${RED}✗ Ambiente virtual não encontrado!${NC}"
    exit 1
fi

# Configurar variáveis de ambiente para otimização
export OLLAMA_NUM_PARALLEL=2  # Máx 2 modelos simultâneos
export OLLAMA_NUM_THREAD=6    # 6 threads para i5 8gen
echo -e "${GREEN}✓ Variáveis de ambiente configuradas${NC}"

# Iniciar Ollama (se não estiver rodando)
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}Iniciando Ollama...${NC}"
    ollama serve &
    sleep 5
    
    # Verificar se Ollama iniciou corretamente
    if pgrep -x "ollama" > /dev/null; then
        echo -e "${GREEN}✓ Ollama iniciado${NC}"
    else
        echo -e "${RED}✗ Falha ao iniciar Ollama${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Ollama já está rodando${NC}"
fi

# Verificar modelos disponíveis
echo -e "\n${YELLOW}Modelos LLM disponíveis:${NC}"
ollama list

# Verificar modelos essenciais
REQUIRED_MODELS=("llama3.1:8b-instruct-q4_K_M" "codellama:7b-instruct")
MISSING_MODELS=()

for model in "${REQUIRED_MODELS[@]}"; do
    model_base=$(echo $model | cut -d':' -f1)
    if ! ollama list | grep -q "$model_base"; then
        MISSING_MODELS+=($model)
    fi
done

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo -e "${RED}✗ Modelos obrigatórios faltando:${NC}"
    for model in "${MISSING_MODELS[@]}"; do
        echo "  - $model"
    done
    echo -e "${YELLOW}Instale com: ollama pull <modelo>${NC}"
fi

# Verificar uso de memória
echo -e "\n${YELLOW}Status da Memória:${NC}"
free -h | grep -E "^Mem|^Swap"

# Perguntar sobre modo de inicialização
echo -e "\n${YELLOW}Escolha o modo de inicialização:${NC}"
echo "1) Normal - Roteamento simples (2 modelos)"
echo "2) Avançado - Multi-modelo (requer mais RAM)"
echo "3) Econômico - Apenas modelo principal"
read -p "Opção (1-3): " mode

case $mode in
    2)
        export RAG_ROUTING_MODE="advanced"
        echo -e "${GREEN}Modo avançado selecionado${NC}"
        ;;
    3)
        export RAG_ROUTING_MODE="simple"
        export RAG_USE_HYBRID="false"
        echo -e "${GREEN}Modo econômico selecionado${NC}"
        ;;
    *)
        export RAG_ROUTING_MODE="simple"
        echo -e "${GREEN}Modo normal selecionado${NC}"
        ;;
esac

# Iniciar API FastAPI
echo -e "\n${YELLOW}Iniciando API RAG...${NC}"
cd /home/$USER/llm-rag-system

# Verificar se a porta está livre
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${RED}✗ Porta 8000 já está em uso!${NC}"
    echo "Deseja parar o processo existente? (s/n)"
    read -p "> " kill_process
    if [ "$kill_process" = "s" ]; then
        kill $(lsof -Pi :8000 -sTCP:LISTEN -t)
        sleep 2
    else
        echo "Abortando..."
        exit 1
    fi
fi

# Iniciar API com logs coloridos
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload --log-config=log_config.json &
API_PID=$!

# Aguardar API iniciar
echo -e "${YELLOW}Aguardando API iniciar...${NC}"
sleep 15

# Verificar se API está respondendo
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ API está rodando${NC}"
else
    echo -e "${RED}✗ API não está respondendo${NC}"
fi

# Exibir informações finais
echo -e "\n${GREEN}=== Serviços Iniciados ===${NC}"
echo -e "API disponível em: ${GREEN}http://localhost:8000${NC}"
echo -e "Documentação: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "Modo de roteamento: ${GREEN}$RAG_ROUTING_MODE${NC}"
echo -e "\n${YELLOW}Comandos úteis:${NC}"
echo "- Ver logs: tail -f logs/rag_api.log"
echo "- Ver modelos ativos: ollama ps"
echo "- Monitorar recursos: htop"
echo "- Parar serviços: ./scripts/stop_services.sh"

# Salvar PIDs para script de parada
echo $API_PID > /tmp/rag_api.pid

# Opcional: abrir browser
if command -v xdg-open > /dev/null; then
    echo -e "\n${YELLOW}Abrir documentação no browser? (s/n)${NC}"
    read -p "> " open_browser
    if [ "$open_browser" = "s" ]; then
        xdg-open http://localhost:8000/docs
    fi
fi

echo -e "\n${GREEN}Sistema pronto! 🚀${NC}"
