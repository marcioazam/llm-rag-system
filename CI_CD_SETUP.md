# 🚀 CI/CD Setup Guide - LLM RAG System

## ⚡ **Quick Start**

Este projeto possui um sistema CI/CD completo implementado. Siga estes passos para ativar:

### 📋 **1. Configuração Inicial**

#### **Secrets do GitHub**
Configure estes secrets no repositório:
```
OPENAI_API_KEY          # Sua chave API do OpenAI
ANTHROPIC_API_KEY       # Sua chave API do Anthropic  
PYPI_API_TOKEN          # Token para publicar no PyPI (opcional)
```

#### **Environments**
Crie estes environments no GitHub:
- `staging` (deploy automático)
- `production` (requer aprovação)

### 🔧 **2. Branch Protection**

Configure branch protection para `main`:
- Require status checks: ✅
- Require branches to be up to date: ✅
- Require pull request reviews: ✅
- Required approving reviews: 2

### 🏷️ **3. Labels**

O sistema criará automaticamente estas labels:
- `dependencies`, `security`, `automerge`, `ci/cd`, `deployment`, `release`

---

## 🔄 **Workflows Disponíveis**

### 🧪 **CI Pipeline** (`ci.yml`)
- **Triggers**: Push, PR, daily schedule
- **Inclui**: Lint, tests, security, performance
- **Quality Gates**: Coverage 70%+, zero security issues

### 🔒 **Security Scan** (`security.yml`)  
- **Triggers**: Push, PR, daily schedule
- **Inclui**: Bandit, Safety, TruffleHog, Trivy
- **Reports**: Comentários automáticos em PRs

### 📊 **Code Quality** (`code-quality.yml`)
- **Triggers**: Push, PR, weekly schedule  
- **Inclui**: Formatting, linting, complexity, coverage
- **Output**: Quality score 0-100 com grade A-F

### 🚀 **Release Management** (`release.yml`)
- **Triggers**: Push to main, manual
- **Features**: Semantic versioning, changelog, Docker images
- **Publishes**: GitHub releases, PyPI (opcional)

### 🚀 **Deploy** (`deploy.yml`)
- **Triggers**: Após release, manual
- **Ambientes**: Staging (auto), Production (aprovação)
- **Strategies**: Rolling, Blue-Green, Rollback automático

---

## 🤖 **Dependabot**

### 📦 **Auto-Updates Configurados**
- **Python deps**: Semanalmente (segunda-feira)
- **GitHub Actions**: Semanalmente (terça-feira)  
- **Docker images**: Semanalmente (quarta-feira)

### 🔄 **Agrupamento Inteligente**
- **AI/ML libraries**: openai, anthropic, transformers
- **Web framework**: fastapi, uvicorn, pydantic
- **Database**: qdrant, neo4j, redis
- **Testing tools**: pytest, coverage, mock

---

## 🎯 **Quality Gates**

| Check | Threshold | Action |
|-------|-----------|--------|
| Test Coverage | ≥ 75% | ❌ Block merge |
| Security Issues | 0 Critical/High | ❌ Block merge |
| Linting Errors | 0 | ❌ Block merge |
| Type Coverage | ≥ 80% | ⚠️ Warning |

---

## 📝 **Uso Diário**

### 🌿 **Feature Development**
```bash
# 1. Criar feature branch
git checkout -b feature/nova-funcionalidade

# 2. Desenvolver e commitar
git commit -m "feat: adiciona nova funcionalidade"

# 3. Push e criar PR
git push origin feature/nova-funcionalidade
# Abrir PR no GitHub

# 4. CI roda automaticamente
# 5. Review e merge
```

### 🐛 **Bug Fix**
```bash
# 1. Criar branch de fix
git checkout -b fix/corrige-bug

# 2. Corrigir e commitar  
git commit -m "fix: corrige bug crítico"

# 3. Push e PR
# CI valida automaticamente
```

### 🚨 **Hotfix**
```bash
# 1. Branch de hotfix direto da main
git checkout -b hotfix/seguranca main

# 2. Correção crítica
git commit -m "fix: vulnerabilidade de segurança"

# 3. PR emergencial
# Deploy automático após merge
```

---

## 🔍 **Debugging**

### ❌ **Se CI Falhar**
1. **Check logs** no GitHub Actions
2. **Run local**: `pytest tests/ -v`
3. **Lint check**: `ruff check src/`
4. **Security check**: `bandit -r src/`

### 🐛 **Testes Locais**
```bash
# Instalar deps de dev
pip install -r requirements.txt
pip install ruff black isort pytest-cov bandit

# Rodar testes como CI
pytest tests/ --cov=src --cov-fail-under=75

# Lint check
ruff check src/ tests/
black --check src/ tests/
isort --check-only src/ tests/

# Security check
bandit -r src/
```

### 🔧 **Debug de Deploy**
1. **Check environment** está configurado
2. **Verify secrets** estão definidos
3. **Check Docker image** foi construída
4. **Review logs** do workflow

---

## 📊 **Monitoramento**

### 📈 **Métricas Disponíveis**
- **Build success rate**
- **Test coverage trends** 
- **Security issues tracking**
- **Deploy frequency**
- **Lead time**

### 🚨 **Alertas Automáticos**
- **Build failures**: Issue criado
- **Security issues**: Issue + PR comment
- **Coverage drop**: Warning
- **Deploy failures**: Rollback automático

### 📋 **Relatórios**
- **Daily**: Resumo de atividades
- **Weekly**: Métricas de qualidade
- **Monthly**: Análise de dependências
- **On-demand**: Via workflow manual

---

## 🎯 **Conventional Commits**

Use estes prefixos para versionamento automático:

| Prefix | Version Bump | Exemplo |
|--------|--------------|---------|
| `feat:` | Minor | `feat: adiciona busca semântica` |
| `fix:` | Patch | `fix: corrige bug de memória` |
| `BREAKING CHANGE:` | Major | Com breaking change |
| `docs:` | None | `docs: atualiza README` |
| `chore:` | None | `chore: atualiza dependências` |

---

## 🔄 **Workflow Manual**

### 🚀 **Release Manual**
```yaml
# GitHub Actions > Release Management > Run workflow
Inputs:
  - release_type: patch/minor/major
  - pre_release: true/false
  - custom_version: "2.1.0" (opcional)
```

### 🔒 **Security Scan Manual**
```yaml
# GitHub Actions > Security Scan > Run workflow  
Inputs:
  - scan_level: quick/standard/comprehensive
  - include_dependencies: true/false
```

### 🚀 **Deploy Manual**
```yaml
# GitHub Actions > Deploy > Run workflow
Inputs:
  - environment: staging/production
  - version: "2.1.0" (opcional)
  - force_deploy: true/false
```

---

## 📚 **Documentação Completa**

Para documentação detalhada, veja:
- **[README_CI_CD_SYSTEM.md](Docs/README_CI_CD_SYSTEM.md)** - Documentação completa
- **[.github/workflows/](/.github/workflows/)** - Código dos workflows
- **[.github/dependabot.yml](/.github/dependabot.yml)** - Configuração Dependabot

---

## ✅ **Status**

### 🎯 **Implementado e Funcional**
- ✅ CI Pipeline completo
- ✅ Security scanning  
- ✅ Code quality analysis
- ✅ Release management
- ✅ Dependabot configuration
- ✅ Documentation

### 📋 **Próximos Passos**
1. ⚙️ Configurar secrets e environments
2. 🧪 Testar workflows em PR
3. 🚀 Primeiro release
4. 📊 Monitorar métricas
5. 🔧 Customizar conforme necessário

---

<div align="center">

**🎉 Sistema CI/CD Pronto para Uso! 🎉**

**Questions? Check the [full documentation](Docs/README_CI_CD_SYSTEM.md)**

</div> 