# GUIA DE MIGRAÇÃO - SISTEMA RAG API

## ✅ MUDANÇAS REALIZADAS

### 1. **Novos Arquivos Criados**
- `config/llm_providers_config.yaml` - Configuração de provedores LLM
- `config/env_example.txt` - Exemplo de variáveis de ambiente
- `src/embeddings/api_embedding_service.py` - Serviço de embeddings via API
- `src/models/api_model_router.py` - Roteador de modelos via API
- `src/rag_pipeline_api.py` - Pipeline RAG baseado em APIs

### 2. **Dependências Atualizadas**
- Removidas: ollama, sentence-transformers, transformers, torch
- Adicionadas: httpx, tenacity, cachetools para APIs

## 🔧 CONFIGURAÇÃO NECESSÁRIA

### 1. **API Keys (OBRIGATÓRIO)**
Copie `config/env_example.txt` para `.env` e configure:

```bash
# Mínimo necessário
OPENAI_API_KEY=sk-your-openai-key-here

# Recomendado
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
```

### 2. **Instalar Dependências**
```bash
pip install -r requirements.txt
```

### 3. **Testar Sistema**
```bash
python -c "
from src.rag_pipeline_api import APIRAGPipeline
pipeline = APIRAGPipeline()
print(pipeline.health_check())
"
```

## 🚀 EXEMPLO DE USO

```python
from src.rag_pipeline_api import APIRAGPipeline

# Inicializar
pipeline = APIRAGPipeline()

# Adicionar documentos
docs = [{"content": "Texto do documento", "metadata": {"source": "teste"}}]
pipeline.add_documents(docs)

# Fazer query
response = pipeline.query("Sua pergunta aqui")
print(response["answer"])
```

## 💰 CONTROLE DE CUSTOS

O sistema inclui:
- Monitoramento de custos em tempo real
- Cache para reduzir chamadas de API
- Seleção automática do modelo mais econômico por tarefa
- Limites configuráveis de orçamento diário

## 📞 PRÓXIMOS PASSOS

1. Configure suas API keys no arquivo `.env`
2. Teste o sistema com uma query simples
3. Configure limites de custo em `llm_providers_config.yaml`
4. Migre seus dados existentes se necessário

Agora seu sistema RAG usa APIs de ponta sem infraestrutura local! 🎉
