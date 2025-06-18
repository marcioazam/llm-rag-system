# 🚀 Sistema RAG com APIs Externas

Sistema RAG (Retrieval-Augmented Generation) totalmente baseado em APIs de provedores externos como OpenAI, Anthropic, Google e Groq. **Sem dependências de modelos locais.**

## ✨ Características Principais

### 🎯 **Roteamento Inteligente de Modelos**
- **GPT-4o**: Análise complexa, arquitetura de sistemas, code reviews
- **GPT-4o-mini**: Geração de código, debugging, testes unitários, refatoração
- **Claude 3.5 Sonnet**: Análise de documentos, criação de conteúdo, escrita técnica
- **Claude 3 Haiku**: Respostas rápidas, extração de dados, classificação
- **Gemini 1.5 Pro**: Análise multimodal, raciocínio de contexto longo

### 💰 **Controle Inteligente de Custos**
- Monitoramento de custos em tempo real
- Cache inteligente para reduzir chamadas de API
- Seleção automática do modelo mais econômico por tarefa
- Limites configuráveis de orçamento diário
- Fallback automático entre provedores

### ⚡ **Performance Otimizada**
- Cache de embeddings e respostas
- Batching automático de requests
- Rate limiting configurável
- Retry automático com backoff exponencial

## 🛠️ Configuração Rápida

### 1. **Instalar Dependências**
```bash
pip install -r requirements.txt
```

### 2. **Configurar API Keys**
```bash
# Copiar arquivo de exemplo
cp config/env_example.txt .env

# Editar com suas API keys
nano .env
```

**API Keys necessárias (mínimo OPENAI_API_KEY):**
```bash
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here  # Recomendado
GOOGLE_API_KEY=your-google-key-here               # Opcional
GROQ_API_KEY=gsk_your-groq-key-here               # Opcional
```

### 3. **Executar Migração**
```bash
python migrate_to_api_system.py
```

### 4. **Testar Sistema**
```python
from src.rag_pipeline_api import APIRAGPipeline

# Inicializar pipeline
pipeline = APIRAGPipeline()

# Verificar saúde do sistema
health = pipeline.health_check()
print(f"Status: {health['status']}")
print(f"API Keys configuradas: {health['api_keys_configured']}")
```

## 📚 Uso Básico

### **Adicionar Documentos**
```python
# Documentos para indexar
documents = [
    {
        "content": "Python é uma linguagem de programação de alto nível...",
        "metadata": {
            "source": "tutorial_python.md",
            "type": "documentation",
            "language": "pt"
        }
    },
    {
        "content": "FastAPI é um framework web moderno para Python...",
        "metadata": {
            "source": "fastapi_guide.md",
            "type": "tutorial"
        }
    }
]

# Adicionar ao índice
result = pipeline.add_documents(documents)
print(f"✅ {result['documents_processed']} documentos processados")
print(f"📄 {result['chunks_created']} chunks criados")
```

### **Fazer Consultas**
```python
# Query simples
response = pipeline.query("Como criar uma API com FastAPI?")
print(f"Resposta: {response['answer']}")
print(f"Modelo usado: {response['model_used']} ({response['provider_used']})")
print(f"Custo: ${response['cost']:.4f}")

# Query com parâmetros específicos
from src.models.api_model_router import TaskType

response = pipeline.query(
    question="Crie uma função Python para calcular fibonacci",
    k=3,  # Número de chunks para recuperar
    task_type=TaskType.CODE_GENERATION,  # Forçar tipo de tarefa
    force_model="openai.gpt4o_mini",     # Forçar modelo específico
    include_sources=True                  # Incluir fontes na resposta
)

print(f"Código gerado: {response['answer']}")
print(f"Fontes utilizadas: {len(response['sources'])}")
```

### **Monitorar Estatísticas**
```python
stats = pipeline.get_stats()
print(f"Total de queries: {stats['total_queries']}")
print(f"Custo total: ${stats['total_cost']:.4f}")
print(f"Documentos indexados: {stats['total_documents_indexed']}")
print(f"Uso por provedor: {stats['provider_usage']}")
```

## 🎯 Responsabilidades dos Modelos

### **OpenAI GPT-4o** 
- ✅ Raciocínio complexo e análise crítica
- ✅ Arquitetura de sistemas e design patterns  
- ✅ Code review e análise de qualidade
- ✅ Resolução de problemas complexos

### **OpenAI GPT-4o-mini**
- 🔧 Geração de código eficiente
- 🐛 Debugging e correção de erros
- 🧪 Criação de testes unitários
- ♻️ Refatoração de código

### **Claude 3.5 Sonnet**
- 📖 Análise profunda de documentos
- ✍️ Criação de conteúdo técnico
- 🔬 Síntese de pesquisas
- 📝 Documentação técnica

### **Claude 3 Haiku**
- ⚡ Respostas rápidas e diretas
- 📊 Extração de dados estruturados
- 🏷️ Classificação de conteúdo
- 🔍 Tarefas simples e objetivas

## ⚙️ Configurações Avançadas

### **Configurar Limites de Custo**
```yaml
# config/llm_providers_config.yaml
routing:
  cost_limits:
    daily_budget: 50.00          # Orçamento diário em USD
    per_request_limit: 1.00      # Limite por request
    warn_threshold: 0.80         # Alertar com 80% do orçamento
```

### **Estratégias de Roteamento**
```yaml
routing:
  strategy: "cost_performance_optimized"  # Opções:
  # - cost_optimized: Sempre o modelo mais barato
  # - performance_optimized: Sempre o melhor modelo
  # - balanced: Equilíbrio entre custo e qualidade
  # - cost_performance_optimized: Otimização inteligente
```

### **Configurar Cache**
```yaml
optimization:
  caching:
    enabled: true
    ttl_seconds: 7200            # Cache por 2 horas
    max_cache_size: 2000         # Máximo 2000 entradas
    cache_embeddings: true       # Cache de embeddings
    cache_responses: true        # Cache de respostas
```

## 🌐 API REST

### **Iniciar Servidor**
```bash
python main.py
# Servidor disponível em http://localhost:8000
```

### **Endpoints Principais**

#### **GET /** - Informações da API
```bash
curl http://localhost:8000/
```

#### **GET /health** - Verificar Saúde
```bash
curl http://localhost:8000/health
```

#### **POST /query** - Consulta RAG
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Como criar uma API com FastAPI?",
    "k": 5,
    "include_sources": true
  }'
```

#### **POST /documents** - Adicionar Documentos
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "Conteúdo do documento...",
        "metadata": {"source": "exemplo.md"}
      }
    ]
  }'
```

#### **GET /stats** - Estatísticas
```bash
curl http://localhost:8000/stats
```

#### **GET /models** - Modelos Disponíveis
```bash
curl http://localhost:8000/models
```

## 💡 Dicas de Otimização

### **Reduzir Custos**
```python
# 1. Use cache agressivamente
pipeline.config["optimization"]["caching"]["ttl_seconds"] = 86400  # 24h

# 2. Prefira modelos menores para tarefas simples
response = pipeline.query(
    "Qual a capital do Brasil?",
    force_model="openai.gpt35_turbo"  # Modelo mais barato
)

# 3. Configure orçamento diário
# Edite config/llm_providers_config.yaml
```

### **Melhorar Performance**
```python
# 1. Use batching para múltiplas queries
queries = ["Pergunta 1", "Pergunta 2", "Pergunta 3"]
responses = [pipeline.query(q) for q in queries]

# 2. Configure chunking apropriado
from src.chunking.recursive_chunker import RecursiveChunker
chunker = RecursiveChunker(
    chunk_size=750,      # Chunks maiores para menos calls
    chunk_overlap=75     # Overlap adequado
)

# 3. Use embeddings otimizados
response = pipeline.query(
    question="Sua pergunta",
    k=3  # Menos chunks = menos custo
)
```

## 🔍 Troubleshooting

### **Erro: API Key não configurada**
```bash
# Verificar se .env existe e tem as keys
cat .env | grep API_KEY

# Verificar se variáveis estão sendo carregadas
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT_FOUND'))"
```

### **Erro: Limite de rate excedido**
```python
# Configurar rate limiting
config["optimization"]["rate_limiting"]["requests_per_minute"] = 50
```

### **Custos muito altos**
```python
# Verificar estatísticas
stats = pipeline.get_stats()
print(f"Custo total: ${stats['total_cost']:.2f}")

# Configurar cache mais agressivo
config["optimization"]["caching"]["ttl_seconds"] = 86400  # 24h
```

## 📊 Monitoramento

### **Métricas Principais**
- **Total de requests**: Número total de chamadas
- **Custo total**: Gasto acumulado em USD
- **Cache hit rate**: Eficiência do cache
- **Tempo médio de resposta**: Performance
- **Distribuição de provedores**: Uso por API

### **Alertas Configuráveis**
- **Alto custo**: Alerta quando atingir 80% do orçamento
- **Alta latência**: Alerta para respostas > 60s
- **Taxa de erro**: Alerta para erro > 5%

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch para sua feature
3. Implemente suas mudanças
4. Adicione testes se necessário
5. Submeta um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

---

## 🎉 Benefícios do Sistema

✅ **Zero Infraestrutura Local**: Não precisa de GPUs, CUDA ou modelos grandes  
✅ **Escalabilidade Automática**: APIs escalam conforme demanda  
✅ **Modelos Estado-da-Arte**: Sempre os melhores modelos disponíveis  
✅ **Custo Controlado**: Pague apenas pelo que usar com limites configuráveis  
✅ **Alta Disponibilidade**: Múltiplos provedores com fallback automático  
✅ **Fácil Manutenção**: Sem updates de modelos ou gerenciamento de infra  

**Seu sistema RAG agora usa a melhor IA disponível no mercado! 🚀** 