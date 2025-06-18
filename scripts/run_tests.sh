#!/bin/bash
set -e

# Script de execução de testes para CI/CD
# Executa diferentes tipos de testes baseado no parâmetro

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funções de log
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Função para verificar se o comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verificar dependências
check_dependencies() {
    log_info "Verificando dependências..."
    
    if ! command_exists python; then
        log_error "Python não encontrado"
        exit 1
    fi
    
    if ! command_exists pip; then
        log_error "pip não encontrado"
        exit 1
    fi
    
    # Verificar se pytest está instalado
    if ! python -c "import pytest" 2>/dev/null; then
        log_warning "pytest não encontrado, instalando..."
        pip install pytest pytest-cov pytest-mock
    fi
    
    log_success "Dependências verificadas"
}

# Instalar dependências de teste
install_test_deps() {
    log_info "Instalando dependências de teste..."
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    log_success "Dependências instaladas"
}

# Executar linting
run_linting() {
    log_info "Executando linting..."
    
    # Black - formatação
    if command_exists black; then
        log_info "Executando Black..."
        black --check --diff src tests scripts || {
            log_error "Black encontrou problemas de formatação"
            return 1
        }
    fi
    
    # isort - ordenação de imports
    if command_exists isort; then
        log_info "Executando isort..."
        isort --check-only --diff src tests scripts || {
            log_error "isort encontrou problemas"
            return 1
        }
    fi
    
    # Flake8 - linting
    if command_exists flake8; then
        log_info "Executando Flake8..."
        flake8 src tests scripts --max-line-length=100 --extend-ignore=E203,W503 || {
            log_error "Flake8 encontrou problemas"
            return 1
        }
    fi
    
    # Ruff - linting rápido
    if command_exists ruff; then
        log_info "Executando Ruff..."
        ruff check src tests scripts || {
            log_error "Ruff encontrou problemas"
            return 1
        }
    fi
    
    log_success "Linting concluído"
}

# Executar testes de segurança
run_security_tests() {
    log_info "Executando testes de segurança..."
    
    # Bandit - análise de segurança
    if command_exists bandit; then
        log_info "Executando Bandit..."
        bandit -r src -f json -o bandit-report.json || {
            log_warning "Bandit encontrou possíveis problemas de segurança"
        }
    fi
    
    # Safety - verificação de vulnerabilidades
    if command_exists safety; then
        log_info "Executando Safety..."
        safety check --json --output safety-report.json || {
            log_warning "Safety encontrou vulnerabilidades conhecidas"
        }
    fi
    
    # Testes de segurança com pytest
    log_info "Executando testes de segurança..."
    pytest tests/test_security.py -v --tb=short -m security || {
        log_error "Testes de segurança falharam"
        return 1
    }
    
    log_success "Testes de segurança concluídos"
}

# Executar testes unitários
run_unit_tests() {
    log_info "Executando testes unitários..."
    
    pytest tests/ -v --tb=short \
        --cov=src --cov-report=xml --cov-report=html --cov-report=term-missing \
        --cov-fail-under=70 \
        --junitxml=test-results.xml \
        -m "unit or not slow" \
        --timeout=300 || {
        log_error "Testes unitários falharam"
        return 1
    }
    
    log_success "Testes unitários concluídos"
}

# Executar testes de integração
run_integration_tests() {
    log_info "Executando testes de integração..."
    
    # Verificar se serviços estão disponíveis
    check_services() {
        if [[ -n "$QDRANT_HOST" ]]; then
            log_info "Verificando Qdrant..."
            curl -f "http://${QDRANT_HOST}:6333/health" || {
                log_warning "Qdrant não disponível, usando mocks"
            }
        fi
    }
    
    check_services
    
    pytest tests/test_rag_integration.py -v --tb=short \
        -m integration \
        --timeout=600 || {
        log_error "Testes de integração falharam"
        return 1
    }
    
    log_success "Testes de integração concluídos"
}

# Executar testes de performance
run_performance_tests() {
    log_info "Executando testes de performance..."
    
    pytest tests/test_performance.py -v --tb=short \
        -m performance \
        --benchmark-only \
        --benchmark-json=benchmark-results.json || {
        log_error "Testes de performance falharam"
        return 1
    }
    
    log_success "Testes de performance concluídos"
}

# Executar validação do sistema
run_system_validation() {
    log_info "Executando validação do sistema..."
    
    python scripts/validate_system.py || {
        log_error "Validação do sistema falhou"
        return 1
    }
    
    log_success "Validação do sistema concluída"
}

# Gerar relatório final
generate_report() {
    log_info "Gerando relatório final..."
    
    # Criar diretório de relatórios
    mkdir -p reports
    
    # Mover arquivos de relatório
    [ -f "test-results.xml" ] && mv test-results.xml reports/
    [ -f "coverage.xml" ] && mv coverage.xml reports/
    [ -f "benchmark-results.json" ] && mv benchmark-results.json reports/
    [ -f "bandit-report.json" ] && mv bandit-report.json reports/
    [ -f "safety-report.json" ] && mv safety-report.json reports/
    [ -f "validation_report.json" ] && mv validation_report.json reports/
    [ -d "htmlcov" ] && mv htmlcov reports/
    
    log_success "Relatório gerado em reports/"
}

# Função principal
main() {
    cd "$PROJECT_DIR"
    
    local test_type="${1:-all}"
    
    log_info "Iniciando execução de testes: $test_type"
    log_info "Diretório do projeto: $PROJECT_DIR"
    
    case "$test_type" in
        "lint")
            check_dependencies
            run_linting
            ;;
        "security")
            check_dependencies
            install_test_deps
            run_security_tests
            ;;
        "unit")
            check_dependencies
            install_test_deps
            run_unit_tests
            ;;
        "integration")
            check_dependencies
            install_test_deps
            run_integration_tests
            ;;
        "performance")
            check_dependencies
            install_test_deps
            run_performance_tests
            ;;
        "validation")
            check_dependencies
            install_test_deps
            run_system_validation
            ;;
        "all")
            check_dependencies
            install_test_deps
            run_linting
            run_security_tests
            run_unit_tests
            run_integration_tests
            run_system_validation
            generate_report
            ;;
        "ci")
            # Versão otimizada para CI - sem performance tests
            check_dependencies
            install_test_deps
            run_linting
            run_security_tests
            run_unit_tests
            run_system_validation
            generate_report
            ;;
        *)
            log_error "Tipo de teste desconhecido: $test_type"
            echo "Uso: $0 [lint|security|unit|integration|performance|validation|all|ci]"
            exit 1
            ;;
    esac
    
    log_success "Execução de testes concluída: $test_type"
}

# Capturar sinais e fazer cleanup
cleanup() {
    log_info "Fazendo cleanup..."
    # Matar processos background se houver
    jobs -p | xargs -r kill
}

trap cleanup EXIT

# Executar função principal
main "$@" 