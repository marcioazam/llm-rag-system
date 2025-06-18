# Documentação de Testes - Sistema RAG

Este diretório contém a suíte completa de testes para o sistema RAG (Retrieval-Augmented Generation). Os testes são organizados para garantir qualidade, robustez e manutenibilidade do código.

## 📁 Estrutura de Testes

```
tests/
├── conftest.py              # Configurações globais e fixtures
├── test_utils.py            # Utilitários de teste
├── pytest.ini              # Configuração do pytest
├── test_rag_pipeline.py     # Testes do pipeline principal
├── test_model_router.py     # Testes do roteador de modelos
├── test_edge_cases.py       # Testes de casos extremos
├── test_security_validation.py  # Testes de segurança
└── README.md               # Esta documentação
```

## 🧪 Tipos de Testes

### 1. Testes Unitários (`@pytest.mark.unit`)
- Testam componentes individuais isoladamente
- Usam mocks para dependências externas
- Execução rápida e determinística

### 2. Testes de Integração (`@pytest.mark.integration`)
- Testam interação entre componentes
- Podem usar serviços reais em ambiente controlado
- Mais lentos mas mais realistas

### 3. Testes de Casos Extremos (`@pytest.mark.edge_case`)
- Testam comportamento em situações limite
- Entradas inválidas, recursos esgotados, etc.
- Garantem robustez do sistema

### 4. Testes de Segurança (`@pytest.mark.security`)
- Validam proteções contra ataques
- Injection, XSS, path traversal, etc.
- Críticos para sistemas em produção

### 5. Testes de Performance (`@pytest.mark.performance`)
- Medem tempo de execução e uso de recursos
- Identificam gargalos e regressões
- Incluem testes de concorrência

## 🚀 Como Executar os Testes

### Execução Básica

```bash
# Executar todos os testes
pytest

# Executar com relatório de cobertura
pytest --cov=src --cov-report=html

# Executar testes específicos
pytest tests/test_rag_pipeline.py

# Executar teste específico
pytest tests/test_rag_pipeline.py::TestRAGPipeline::test_init_with_config_file
```

### Execução por Marcadores

```bash
# Apenas testes unitários
pytest -m unit

# Apenas testes de integração
pytest -m integration

# Apenas testes rápidos (pular lentos)
pytest -m "not slow"

# Apenas testes de segurança
pytest -m security

# Apenas testes críticos
pytest -m critical
```

### Opções Customizadas

```bash
# Execução rápida (pula testes lentos)
pytest --fast

# Apenas testes unitários
pytest --unit

# Apenas testes de integração
pytest --integration

# Apenas testes de smoke
pytest --smoke

# Modo verboso com detalhes
pytest -v

# Parar no primeiro erro
pytest -x

# Executar em paralelo (se pytest-xdist instalado)
pytest -n auto
```

### Relatórios de Cobertura

```bash
# Gerar relatório HTML
pytest --cov=src --cov-report=html

# Gerar relatório XML (para CI/CD)
pytest --cov=src --cov-report=xml

# Mostrar linhas não cobertas
pytest --cov=src --cov-report=term-missing

# Falhar se cobertura < 40%
pytest --cov=src --cov-fail-under=40
```

## 🔧 Configuração de Ambiente

### Variáveis de Ambiente

Os testes usam as seguintes variáveis de ambiente:

```bash
# Configuração automática (via conftest.py)
TESTING=true
LOG_LEVEL=DEBUG
OPENAI_API_KEY=test-key-mock
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test
```

### Dependências de Teste

Certifique-se de ter as dependências instaladas:

```bash
pip install pytest pytest-cov pytest-mock pytest-asyncio
pip install pytest-xdist  # Para execução paralela (opcional)
pip install pytest-html   # Para relatórios HTML (opcional)
```

## 📊 Fixtures Disponíveis

### Fixtures Globais (conftest.py)

- `setup_global_test_environment`: Configuração global da sessão
- `temp_test_dir`: Diretório temporário para a sessão
- `isolated_temp_dir`: Diretório temporário por teste
- `clean_environment`: Limpeza de ambiente
- `performance_monitor`: Monitor de performance
- `sample_documents`: Documentos de exemplo
- `sample_queries`: Queries de exemplo
- `mock_config`: Configuração mock
- `mock_openai_client`: Cliente OpenAI mockado
- `mock_qdrant_client`: Cliente Qdrant mockado
- `mock_neo4j_driver`: Driver Neo4j mockado

### Fixtures Específicas

- `base_config`: Configuração base para RAG
- `custom_config_file`: Factory para arquivos de configuração
- `setup_environment_and_mocks`: Mocks para pipeline RAG

## 🛠 Utilitários de Teste

### TestDataFactory

```python
from tests.test_utils import TestDataFactory

# Criar configuração de teste
config = TestDataFactory.create_test_config()

# Criar documentos de teste
docs = TestDataFactory.create_test_documents(count=5)

# Criar resultados de busca
results = TestDataFactory.create_search_results()
```

### MockFactory

```python
from tests.test_utils import MockFactory

# Criar mocks pré-configurados
embedding_mock = MockFactory.create_embedding_service_mock()
router_mock = MockFactory.create_model_router_mock()
openai_mock = MockFactory.create_openai_mock()
```

### TestValidators

```python
from tests.test_utils import TestValidators

# Validar resposta de query
TestValidators.validate_query_response(response)

# Validar processamento de documento
TestValidators.validate_document_processing(result)

# Validar resultados de busca
TestValidators.validate_search_results(results)
```

### PerformanceTestHelper

```python
from tests.test_utils import PerformanceTestHelper

# Medir tempo de execução
with PerformanceTestHelper.measure_time() as timer:
    # código a ser medido
    pass

print(f"Tempo: {timer.duration}s")

# Benchmark de queries
results = PerformanceTestHelper.benchmark_queries(pipeline, queries)
```

## 📈 Métricas e Monitoramento

### Cobertura de Código

- **Meta**: Mínimo 40% de cobertura
- **Atual**: Verificar com `pytest --cov=src --cov-report=term`
- **Relatórios**: Gerados em `htmlcov/` e `coverage.xml`

### Performance

- Testes com duração > 5s geram warnings
- Use `@pytest.mark.slow` para testes lentos
- Monitor automático via `performance_monitor` fixture

### Qualidade

- Todos os testes devem passar
- Sem warnings críticos
- Mocks adequados para dependências externas
- Isolamento entre testes

## 🐛 Debugging de Testes

### Logs Detalhados

```bash
# Habilitar logs detalhados
pytest -s --log-cli-level=DEBUG

# Capturar stdout/stderr
pytest -s --capture=no
```

### Debugging Específico

```bash
# Executar teste específico com debugging
pytest -xvs tests/test_rag_pipeline.py::test_specific_function

# Usar pdb para debugging interativo
pytest --pdb

# Parar no primeiro erro
pytest -x
```

### Problemas Comuns

1. **Testes falhando por dependências externas**
   - Verificar se mocks estão configurados
   - Usar `@pytest.mark.external` para testes que precisam de serviços

2. **Problemas de isolamento**
   - Usar `clean_environment` fixture
   - Verificar se variáveis globais estão sendo resetadas

3. **Testes lentos**
   - Marcar com `@pytest.mark.slow`
   - Otimizar ou usar mocks mais eficientes

4. **Problemas de cobertura**
   - Verificar se todos os caminhos estão testados
   - Adicionar testes para casos extremos

## 📝 Boas Práticas

### Escrita de Testes

1. **Nomes descritivos**: `test_should_return_error_when_config_file_not_found`
2. **Arrange-Act-Assert**: Estrutura clara dos testes
3. **Isolamento**: Cada teste deve ser independente
4. **Mocks apropriados**: Mockar dependências externas
5. **Assertions específicas**: Verificar comportamento exato

### Organização

1. **Um arquivo por módulo**: `test_module_name.py`
2. **Classes para agrupamento**: `TestClassName`
3. **Fixtures reutilizáveis**: Evitar duplicação
4. **Documentação**: Docstrings explicativas
5. **Marcadores**: Classificar tipos de teste

### Performance

1. **Testes rápidos**: Priorizar velocidade
2. **Mocks eficientes**: Evitar operações custosas
3. **Paralelização**: Usar pytest-xdist quando possível
4. **Cleanup**: Limpar recursos após testes
5. **Monitoramento**: Acompanhar tempo de execução

## 🔄 Integração Contínua

### GitHub Actions / CI

```yaml
# Exemplo de configuração CI
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml --cov-fail-under=40
    
- name: Upload coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest
      name: pytest
      entry: pytest
      language: system
      pass_filenames: false
      always_run: true
```

## 📚 Recursos Adicionais

- [Documentação do pytest](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Mocking com unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Boas práticas de teste](https://docs.pytest.org/en/stable/goodpractices.html)

---

**Nota**: Esta documentação é atualizada conforme novos testes são adicionados. Para dúvidas ou sugestões, consulte a equipe de desenvolvimento.