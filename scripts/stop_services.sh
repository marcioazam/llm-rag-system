#!/bin/bash

# Script para parar os serviços do sistema RAG

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}=== Parando Sistema RAG ===${NC}"

# Parar API FastAPI
if [ -f /tmp/rag_api.pid ]; then
    API_PID=$(cat /tmp/rag_api.pid)
    if ps -p $API_PID > /dev/null; then
        echo -e "${YELLOW}Parando API (PID: $API_PID)...${NC}"
        kill $API_PID
        rm /tmp/rag_api.pid
        echo -e "${GREEN}✓ API parada${NC}"
    else
        echo -e "${YELLOW}API não está rodando${NC}"
        rm /tmp/rag_api.pid
    fi
else
    # Tentar encontrar processo pela porta
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}Parando processo na porta 8000...${NC}"
        kill $(lsof -Pi :8000 -sTCP:LISTEN -t)
        echo -e "${GREEN}✓ Processo parado${NC}"
    else
        echo -e "${YELLOW}Nenhum processo encontrado na porta 8000${NC}"
    fi
fi

# Perguntar sobre Ollama
echo -e "\n${YELLOW}Deseja parar o Ollama também? (s/n)${NC}"
echo -e "${RED}Nota: Isso vai descarregar todos os modelos da memória${NC}"
read -p "> " stop_ollama

if [ "$stop_ollama" = "s" ]; then
    if pgrep -x "ollama" > /dev/null; then
        echo -e "${YELLOW}Parando Ollama...${NC}"
        pkill ollama
        sleep 2
        echo -e "${GREEN}✓ Ollama parado${NC}"
    else
        echo -e "${YELLOW}Ollama não está rodando${NC}"
    fi
else
    echo -e "${GREEN}Ollama mantido rodando${NC}"
    
    # Mostrar modelos carregados
    if command -v ollama >/dev/null 2>&1; then
        echo -e "\n${YELLOW}Modelos atualmente carregados:${NC}"
        ollama ps
    fi
fi

# Limpar variáveis de ambiente
unset RAG_ROUTING_MODE
unset RAG_USE_HYBRID
unset OLLAMA_NUM_PARALLEL
unset OLLAMA_NUM_THREAD

echo -e "\n${GREEN}=== Sistema RAG parado ===${NC}"

# Mostrar uso de memória após parada
echo -e "\n${YELLOW}Memória liberada:${NC}"
free -h | grep -E "^Mem"
