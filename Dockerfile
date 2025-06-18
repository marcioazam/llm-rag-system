# Multi-stage Dockerfile para sistema RAG
# Etapa 1: Build dependencies
FROM python:3.11-slim as builder

# Instalar dependências do sistema necessárias para build
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd --create-home --shell /bin/bash rag-user

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro para aproveitar cache do Docker
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Etapa 2: Production image
FROM python:3.11-slim as production

# Instalar apenas dependências de runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Criar usuário não-root
RUN useradd --create-home --shell /bin/bash rag-user

# Copiar dependências Python do builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Definir diretório de trabalho
WORKDIR /app

# Copiar código da aplicação
COPY --chown=rag-user:rag-user . .

# Criar diretórios necessários
RUN mkdir -p logs data/raw data/processed chroma_db storage && \
    chown -R rag-user:rag-user /app

# Mudar para usuário não-root
USER rag-user

# Definir variáveis de ambiente
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    ENVIRONMENT=production

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando padrão
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Etapa 3: Development image
FROM production as development

USER root

# Instalar dependências de desenvolvimento
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências de desenvolvimento Python
COPY requirements.txt .
RUN pip install --no-cache-dir pytest pytest-cov pytest-mock black flake8 ruff mypy

USER rag-user

# Comando para desenvolvimento
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Etapa 4: Testing image
FROM development as testing

USER root

# Copiar configurações de teste
COPY pytest.ini .
COPY tests/ tests/

# Instalar todas as dependências de teste
RUN pip install --no-cache-dir \
    pytest-asyncio \
    pytest-timeout \
    pytest-benchmark \
    pytest-xdist \
    bandit \
    safety

USER rag-user

# Comando para executar testes
CMD ["pytest", "tests/", "-v", "--tb=short", "--cov=src"] 