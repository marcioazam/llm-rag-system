#!/bin/bash

# Script com debug detalhado
set -e  # Para em caso de erro
set -x  # Mostra comandos sendo executados

echo "=== DEBUG: CORREÇÃO COMPLETA DO SISTEMA RAG ==="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCESSO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[AVISO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERRO]${NC} $1"
}

# Verificar diretório atual
log_info "Diretório atual: $(pwd)"
log_info "Arquivos no diretório: $(ls -la)"

# 1. Parar todos os processos relacionados (com mais cuidado)
log_info "1. Parando processos existentes..."
echo "Processos Python rodando:"
ps aux | grep python | grep -v grep || echo "Nenhum processo Python encontrado"

# Tentar parar processos de forma mais segura
pids=$(pgrep -f "uvicorn\|rag_api\|fastapi" 2>/dev/null || true)
if [ ! -z "$pids" ]; then
    echo "Parando PIDs: $pids"
    kill $pids 2>/dev/null || true
    sleep 2
else
    echo "Nenhum processo relacionado encontrado"
fi

log_success "Etapa 1 concluída"

# 2. Verificar se Python está disponível
log_info "2. Verificando Python..."
python3 --version || {
    log_error "Python3 não encontrado!"
    exit 1
}
log_success "Python3 encontrado: $(python3 --version)"

# 3. Verificar pip
log_info "3. Verificando pip..."
python3 -m pip --version || {
    log_error "pip não encontrado!"
    exit 1
}
log_success "pip encontrado"

# 4. Verificar/criar ambiente virtual
log_info "4. Configurando ambiente virtual..."
if [ -d "venv" ]; then
    log_info "Ambiente virtual existe, removendo para recriar..."
    rm -rf venv
fi

log_info "Criando novo ambiente virtual..."
python3 -m venv venv
log_success "Ambiente virtual criado"

# Ativar ambiente virtual
log_info "Ativando ambiente virtual..."
source venv/bin/activate
which python
which pip
log_success "Ambiente virtual ativado"

# 5. Atualizar pip
log_info "5. Atualizando pip..."
pip install --upgrade pip
log_success "pip atualizado"

# 6. Instalar dependências essenciais uma por uma
log_info "6. Instalando dependências essenciais..."

log_info "Instalando uvicorn..."
pip install uvicorn
log_success "uvicorn instalado"

log_info "Instalando fastapi..."
pip install fastapi
log_success "fastapi instalado"

log_info "Instalando dependências adicionais..."
pip install python-multipart requests aiofiles
log_success "Dependências básicas instaladas"

# 7. Verificar Ollama
log_info "7. Verificando Ollama..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    log_success "Ollama está rodando"
else
    log_warning "Ollama não está acessível - pode não estar rodando"
fi

# 8. Criar estrutura de diretórios
log_info "8. Criando estrutura de diretórios..."
mkdir -p logs data storage uploads
log_success "Diretórios criados"

# 9. Criar arquivo de configuração simples
log_info "9. Criando configuração..."
cat > config.py << 'EOF'
import os

# Configurações da API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = False  # Desabilitado para evitar problemas

# Configurações do Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"

# Diretórios
UPLOAD_DIR = "uploads"
STORAGE_DIR = "storage"
LOG_DIR = "logs"

# Configurações do RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 2000

# Criar diretórios se não existirem
for dir_path in [UPLOAD_DIR, STORAGE_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)
EOF
log_success "Configuração criada"

# 10. Criar API básica
log_info "10. Criando API básica..."
cat > main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from datetime import datetime

# Configurações básicas
API_HOST = "0.0.0.0"
API_PORT = 8000
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"

# Criar diretórios
os.makedirs("logs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("storage", exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API System",
    description="Sistema de Retrieval-Augmented Generation",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "RAG API System is running", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ollama_url": OLLAMA_BASE_URL,
            "default_model": DEFAULT_MODEL
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch models from Ollama")
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Iniciando RAG API System...")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
EOF
log_success "API básica criada"

# 11. Testar importações
log_info "11. Testando importações..."
python3 -c "import uvicorn, fastapi; print('Importações OK')" || {
    log_error "Erro nas importações!"
    exit 1
}
log_success "Importações funcionando"

# 12. Iniciar API
log_info "12. Iniciando API RAG..."
python3 main.py &
API_PID=$!
echo $API_PID > api.pid
log_success "API iniciada com PID: $API_PID"

# 13. Aguardar inicialização
log_info "13. Aguardando inicialização (10s)..."
sleep 10

# 14. Testes
log_info "14. Executando testes..."
curl -s http://localhost:8000/health && log_success "Health check OK" || log_error "Health check FALHOU"

echo ""
echo "=== RESUMO FINAL ==="
echo "PID da API: $(cat api.pid 2>/dev/null || echo 'N/A')"
echo "Logs: logs/rag_api.log"
echo "Health Check: http://localhost:8000/health"
echo "Documentação: http://localhost:8000/docs"
echo ""
echo "Para parar: kill \$(cat api.pid)"
echo "Para logs: tail -f logs/rag_api.log"
