# Makefile para Sistema RAG
# Sistema RAG - Makefile Avançado
# Automatiza comandos de desenvolvimento, teste, deploy e monitoramento

.PHONY: help install install-dev test test-unit test-integration test-security test-performance
.PHONY: test-fast test-coverage test-html lint format clean setup-dev type-check
.PHONY: run-api run-tests-watch docs build docker-build docker-run monitoring
.PHONY: deploy-local deploy-staging deploy-prod backup restore benchmark profile
.PHONY: security-scan pre-commit env-check health status version validate

# Configurações
PYTHON := python
PIP := pip
PYTEST := pytest
DOCKER := docker
DOCKER_COMPOSE := docker-compose
SOURCE_DIR := src
TEST_DIR := tests
COVERAGE_MIN := 80
PROJECT_NAME := rag-system
VERSION := $(shell cat VERSION 2>/dev/null || echo "0.1.0")

# Cores para output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
RESET := \033[0m
NC := \033[0m # No Color

help: ## Mostra esta ajuda
	@echo "$(BLUE)Sistema RAG - Comandos Disponíveis$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Comandos Principais:$(RESET)"
	@echo "  make setup-dev       # Configurar ambiente completo"
	@echo "  make validate        # Validação completa (lint + test + security)"
	@echo "  make docker-run      # Executar com Docker Compose"
	@echo "  make monitoring      # Iniciar stack de monitoramento"
	@echo "  make deploy-local    # Deploy local completo"
	@echo ""
	@echo "$(YELLOW)Versão: $(VERSION)$(RESET)"

install: ## Instalar dependências do projeto
	@echo "$(BLUE)Instalando dependências...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependências instaladas com sucesso!$(NC)"

install-dev: ## Instalar dependências de desenvolvimento
	@echo "$(BLUE)Instalando dependências de desenvolvimento...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)✓ Dependências de desenvolvimento instaladas$(RESET)"

setup-dev: install-dev ## Configurar ambiente de desenvolvimento completo
	@echo "$(BLUE)Configurando ambiente de desenvolvimento...$(RESET)"
	pre-commit install
	@test -f .env || cp .env.example .env
	@echo "$(YELLOW)⚠ Configure as variáveis em .env antes de continuar$(RESET)"
	@echo "$(GREEN)✓ Ambiente de desenvolvimento configurado$(RESET)"

# ============================================================================
# TESTES
# ============================================================================

test: ## Executar todos os testes
	@echo "$(BLUE)Executando todos os testes...$(NC)"
	$(PYTEST) $(TEST_DIR) -v

test-unit: ## Executar apenas testes unitários
	@echo "$(BLUE)Executando testes unitários...$(NC)"
	$(PYTEST) -m unit -v

test-integration: ## Executar apenas testes de integração
	@echo "$(BLUE)Executando testes de integração...$(NC)"
	$(PYTEST) -m integration -v

test-security: ## Executar apenas testes de segurança
	@echo "$(BLUE)Executando testes de segurança...$(NC)"
	$(PYTEST) -m security -v

test-performance: ## Executar apenas testes de performance
	@echo "$(BLUE)Executando testes de performance...$(NC)"
	$(PYTEST) -m performance -v

test-edge-cases: ## Executar apenas testes de casos extremos
	@echo "$(BLUE)Executando testes de casos extremos...$(NC)"
	$(PYTEST) -m edge_case -v

test-fast: ## Executar testes rápidos (pular lentos)
	@echo "$(BLUE)Executando testes rápidos...$(NC)"
	$(PYTEST) --fast -v

test-smoke: ## Executar testes de smoke
	@echo "$(BLUE)Executando testes de smoke...$(NC)"
	$(PYTEST) --smoke -v

test-coverage: ## Executar testes com relatório de cobertura
	@echo "$(BLUE)Executando testes com cobertura...$(NC)"
	$(PYTEST) --cov=$(SOURCE_DIR) --cov-report=term-missing --cov-fail-under=$(COVERAGE_MIN)

test-html: ## Executar testes e gerar relatório HTML
	@echo "$(BLUE)Executando testes e gerando relatório HTML...$(NC)"
	$(PYTEST) --cov=$(SOURCE_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)Relatório HTML gerado em htmlcov/index.html$(NC)"

test-xml: ## Executar testes e gerar relatório XML (para CI)
	@echo "$(BLUE)Executando testes e gerando relatório XML...$(NC)"
	$(PYTEST) --cov=$(SOURCE_DIR) --cov-report=xml --junitxml=test-results.xml

test-watch: ## Executar testes em modo watch (reexecuta quando arquivos mudam)
	@echo "$(BLUE)Iniciando modo watch para testes...$(NC)"
	$(PYTEST) --watch $(SOURCE_DIR) $(TEST_DIR)

test-parallel: ## Executar testes em paralelo
	@echo "$(BLUE)Executando testes em paralelo...$(NC)"
	$(PYTEST) -n auto -v

test-debug: ## Executar testes com debugging habilitado
	@echo "$(BLUE)Executando testes com debugging...$(NC)"
	$(PYTEST) -xvs --pdb

test-specific: ## Executar teste específico (uso: make test-specific TEST=test_name)
	@echo "$(BLUE)Executando teste específico: $(TEST)$(NC)"
	$(PYTEST) -xvs -k "$(TEST)"

# ============================================================================
# QUALIDADE DE CÓDIGO
# ============================================================================

lint: ## Verificar qualidade do código (flake8, pylint, ruff)
	@echo "$(BLUE)Verificando qualidade do código...$(RESET)"
	@echo "$(YELLOW)→ Flake8$(RESET)"
	flake8 $(SOURCE_DIR) $(TEST_DIR) --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(YELLOW)→ Pylint$(RESET)"
	pylint $(SOURCE_DIR) --rcfile=.pylintrc || true
	@echo "$(YELLOW)→ Ruff$(RESET)"
	ruff check $(SOURCE_DIR) $(TEST_DIR) || true
	@echo "$(GREEN)✓ Verificação de qualidade concluída$(RESET)"

format: ## Formatar código (autoflake, isort, black)
	@echo "$(BLUE)Formatando código...$(RESET)"
	@echo "$(YELLOW)→ Autoflake$(RESET)"
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place $(SOURCE_DIR) $(TEST_DIR) || true
	@echo "$(YELLOW)→ isort$(RESET)"
	isort $(SOURCE_DIR) $(TEST_DIR) --profile black
	@echo "$(YELLOW)→ Black$(RESET)"
	black $(SOURCE_DIR) $(TEST_DIR) --line-length=88
	@echo "$(GREEN)✓ Código formatado com sucesso$(RESET)"

format-check: ## Verificar se código está formatado corretamente
	@echo "$(BLUE)Verificando formatação do código...$(NC)"
	black $(SOURCE_DIR) $(TEST_DIR) --check --line-length=88
	isort $(SOURCE_DIR) $(TEST_DIR) --check-only --profile black

type-check: ## Verificar tipos com mypy
	@echo "$(BLUE)Verificando tipos com mypy...$(RESET)"
	mypy $(SOURCE_DIR) --config-file=mypy.ini || true
	@echo "$(GREEN)✓ Verificação de tipos concluída$(RESET)"

security-check: ## Verificar vulnerabilidades de segurança
	@echo "$(BLUE)Verificando vulnerabilidades de segurança...$(RESET)"
	@mkdir -p reports
	bandit -r $(SOURCE_DIR) -f json -o reports/bandit-report.json || true
	safety check --json --output reports/safety-report.json || true
	@echo "$(GREEN)✓ Verificação de segurança concluída$(RESET)"

pre-commit: ## Executar hooks de pre-commit
	@echo "$(BLUE)Executando pre-commit hooks...$(RESET)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit hooks executados$(RESET)"

check-all: format lint type-check security-check ## Executar todas as verificações
	@echo "$(GREEN)✓ Todas as verificações concluídas$(RESET)"

# ============================================================================
# EXECUÇÃO E DESENVOLVIMENTO
# ============================================================================

run-api: ## Iniciar API de desenvolvimento
	@echo "$(BLUE)Iniciando API de desenvolvimento...$(NC)"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-api-prod: ## Iniciar API em modo produção
	@echo "$(BLUE)Iniciando API em modo produção...$(NC)"
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

run-jupyter: ## Iniciar Jupyter Lab para desenvolvimento
	@echo "$(BLUE)Iniciando Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

run-streamlit: ## Iniciar interface Streamlit (se disponível)
	@echo "$(BLUE)Iniciando interface Streamlit...$(NC)"
	streamlit run src/ui/app.py

# ============================================================================
# DOCUMENTAÇÃO
# ============================================================================

docs: ## Gerar documentação
	@echo "$(BLUE)Gerando documentação...$(NC)"
	sphinx-build -b html docs docs/_build/html
	@echo "$(GREEN)Documentação gerada em docs/_build/html/$(NC)"

docs-serve: ## Servir documentação localmente
	@echo "$(BLUE)Servindo documentação...$(NC)"
	cd docs/_build/html && python -m http.server 8080

docs-clean: ## Limpar documentação gerada
	@echo "$(BLUE)Limpando documentação...$(NC)"
	rm -rf docs/_build/

# ============================================================================
# LIMPEZA E MANUTENÇÃO
# ============================================================================

clean: ## Limpar arquivos temporários e cache
	@echo "$(BLUE)Limpando arquivos temporários...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf test-results.xml
	rm -rf security-report.json
	rm -rf .mypy_cache/
	@echo "$(GREEN)Limpeza concluída!$(NC)"

clean-all: clean ## Limpeza completa incluindo dependências
	@echo "$(BLUE)Limpeza completa...$(NC)"
	rm -rf venv/
	rm -rf .venv/
	rm -rf node_modules/
	@echo "$(GREEN)Limpeza completa concluída!$(NC)"

reset-db: ## Resetar bancos de dados de desenvolvimento
	@echo "$(BLUE)Resetando bancos de dados...$(NC)"
	# Adicionar comandos para resetar Qdrant e Neo4j se necessário
	@echo "$(GREEN)Bancos de dados resetados!$(NC)"

# ============================================================================
# DOCKER E DEPLOY
# ============================================================================

docker-build: ## Construir imagem Docker
	@echo "$(BLUE)Construindo imagem Docker...$(RESET)"
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)✓ Imagem Docker construída$(RESET)"

docker-run: ## Executar aplicação com Docker Compose
	@echo "$(BLUE)Iniciando serviços com Docker Compose...$(RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Serviços iniciados$(RESET)"
	@echo "$(YELLOW)API: http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Grafana: http://localhost:3000$(RESET)"
	@echo "$(YELLOW)Neo4j: http://localhost:7474$(RESET)"

docker-stop: ## Parar serviços Docker
	@echo "$(BLUE)Parando serviços Docker...$(RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Serviços parados$(RESET)"

docker-clean: ## Limpar recursos Docker
	@echo "$(BLUE)Limpando recursos Docker...$(RESET)"
	$(DOCKER_COMPOSE) down -v --rmi all
	$(DOCKER) system prune -f
	@echo "$(GREEN)✓ Limpeza Docker concluída$(RESET)"

docker-logs: ## Ver logs dos containers
	@echo "$(BLUE)Logs dos containers:$(RESET)"
	$(DOCKER_COMPOSE) logs -f

monitoring: ## Iniciar stack de monitoramento
	@echo "$(BLUE)Iniciando monitoramento...$(RESET)"
	$(DOCKER_COMPOSE) up -d prometheus grafana
	@echo "$(GREEN)✓ Monitoramento iniciado$(RESET)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(RESET)"
	@echo "$(YELLOW)Grafana: http://localhost:3000 (admin/devpassword)$(RESET)"

deploy-local: docker-run ## Deploy local completo
	@echo "$(GREEN)✓ Deploy local concluído$(RESET)"

deploy-staging: ## Deploy para staging
	@echo "$(BLUE)Deploy para staging...$(RESET)"
	@echo "$(YELLOW)⚠ Implementar lógica de deploy para staging$(RESET)"

deploy-prod: ## Deploy para produção
	@echo "$(BLUE)Deploy para produção...$(RESET)"
	@echo "$(YELLOW)⚠ Implementar lógica de deploy para produção$(RESET)"

# ============================================================================
# ANÁLISE E RELATÓRIOS
# ============================================================================

benchmark: ## Executar benchmarks de performance
	@echo "$(BLUE)Executando benchmarks...$(RESET)"
	@mkdir -p reports
	$(PYTEST) tests/performance --benchmark-only --benchmark-json=reports/benchmark.json || true
	@echo "$(GREEN)✓ Benchmarks concluídos$(RESET)"

profile: ## Executar profiling da aplicação
	@echo "$(BLUE)Executando profiling...$(RESET)"
	@mkdir -p reports
	py-spy record -o reports/profile.svg -- $(PYTHON) -m uvicorn $(SOURCE_DIR).main:app || true
	@echo "$(GREEN)✓ Profile gerado em reports/profile.svg$(RESET)"

security-scan: ## Scan de segurança completo
	@echo "$(BLUE)Executando scan de segurança...$(RESET)"
	@mkdir -p reports
	bandit -r $(SOURCE_DIR) -f json -o reports/bandit-report.json || true
	safety check --json --output reports/safety-report.json || true
	@echo "$(GREEN)✓ Scan de segurança concluído$(RESET)"

check-deps: ## Verificar dependências desatualizadas
	@echo "$(BLUE)Verificando dependências...$(RESET)"
	$(PIP) list --outdated

update-deps: ## Atualizar dependências
	@echo "$(BLUE)Atualizando dependências...$(RESET)"
	$(PIP) install --upgrade -r requirements.txt

freeze-deps: ## Congelar dependências atuais
	@echo "$(BLUE)Congelando dependências...$(RESET)"
	$(PIP) freeze > requirements-frozen.txt
	@echo "$(GREEN)✓ Dependências congeladas$(RESET)"

# ============================================================================
# MANUTENÇÃO E BACKUP
# ============================================================================

backup: ## Backup dos dados
	@echo "$(BLUE)Fazendo backup...$(RESET)"
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@echo "$(YELLOW)⚠ Implementar lógica de backup$(RESET)"

restore: ## Restaurar dados
	@echo "$(BLUE)Restaurando dados...$(RESET)"
	@echo "$(YELLOW)⚠ Implementar lógica de restore$(RESET)"

health: ## Verificar saúde dos serviços
	@echo "$(BLUE)Verificando saúde...$(RESET)"
	curl -f http://localhost:8000/health || echo "$(RED)✗ API não responde$(RESET)"
	curl -f http://localhost:6333/health || echo "$(RED)✗ Qdrant não responde$(RESET)"
	curl -f http://localhost:7474 || echo "$(RED)✗ Neo4j não responde$(RESET)"
	@echo "$(GREEN)✓ Verificação concluída$(RESET)"

status: ## Status dos serviços
	@echo "$(BLUE)Status dos serviços:$(RESET)"
	$(DOCKER_COMPOSE) ps

version: ## Versão atual
	@echo "$(BLUE)Versão: $(VERSION)$(RESET)"

env-check: ## Verificar configuração
	@echo "$(BLUE)Verificando configuração...$(RESET)"
	@test -f .env || (echo "$(RED)✗ .env não encontrado$(RESET)" && exit 1)
	@test -f requirements.txt || (echo "$(RED)✗ requirements.txt não encontrado$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Configuração OK$(RESET)"

# ============================================================================
# PIPELINES E VALIDAÇÃO
# ============================================================================

ci: clean install lint type-check test coverage security-scan ## Pipeline de CI
	@echo "$(GREEN)✓ Pipeline de CI concluído$(RESET)"

cd: build docker-build ## Pipeline de CD
	@echo "$(GREEN)✓ Pipeline de CD concluído$(RESET)"

validate: clean install-dev lint type-check test-all coverage security-scan ## Validação completa
	@echo "$(GREEN)✓ Validação completa concluída$(RESET)"

quick: lint type-check test-unit ## Verificações rápidas
	@echo "$(GREEN)✓ Verificações rápidas concluídas$(RESET)"

all: clean install-dev lint type-check test coverage ## Pipeline completo
	@echo "$(GREEN)✓ Pipeline completo concluído$(RESET)"

dev-setup: setup-dev docker-run ## Setup completo para desenvolvimento
	@echo "$(GREEN)✓ Setup de desenvolvimento concluído$(RESET)"

dev-reset: docker-clean clean install-dev docker-run ## Reset completo
	@echo "$(GREEN)✓ Reset de desenvolvimento concluído$(RESET)"

prod-check: validate benchmark ## Verificação para produção
	@echo "$(GREEN)✓ Verificação de produção concluída$(RESET)"

# ============================================================================
# INFORMAÇÕES
# ============================================================================

info: ## Mostrar informações do projeto
	@echo "$(BLUE)Informações do Projeto$(NC)"
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "Pytest: $$(pytest --version)"
	@echo "Diretório de origem: $(SOURCE_DIR)"
	@echo "Diretório de testes: $(TEST_DIR)"
	@echo "Cobertura mínima: $(COVERAGE_MIN)%"
	@echo ""
	@echo "$(YELLOW)Estrutura do projeto:$(NC)"
	@tree -I '__pycache__|*.pyc|.git|.pytest_cache|htmlcov|.coverage' -L 3

status: ## Mostrar status do projeto
	@echo "$(BLUE)Status do Projeto$(NC)"
	@echo "$(YELLOW)Git status:$(NC)"
	@git status --short
	@echo ""
	@echo "$(YELLOW)Últimos commits:$(NC)"
	@git log --oneline -5
	@echo ""
	@echo "$(YELLOW)Arquivos modificados:$(NC)"
	@git diff --name-only

# ============================================================================
# TARGETS ESPECIAIS
# ============================================================================

# Target padrão
.DEFAULT_GOAL := help

# Aliases úteis
test-all: test-unit test-integration test-performance test-security ## Todos os testes
build: ## Construir pacote Python
	@echo "$(BLUE)Construindo pacote...$(RESET)"
	$(PYTHON) -m build || true
	@echo "$(GREEN)✓ Pacote construído$(RESET)"

docs: ## Gerar documentação
	@echo "$(BLUE)Gerando documentação...$(RESET)"
	@mkdir -p docs/api
	curl -f http://localhost:8000/openapi.json > docs/api/openapi.json || true
	@echo "$(GREEN)✓ Documentação gerada$(RESET)"

docs-serve: ## Servir documentação
	@echo "$(BLUE)Servindo documentação...$(RESET)"
	$(PYTHON) -m http.server 8080 --directory docs