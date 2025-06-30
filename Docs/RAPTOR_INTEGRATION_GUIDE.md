# 🚀 RAPTOR Enhanced - Guia de Integração ao Sistema RAG

## Visão Geral

O RAPTOR Enhanced foi projetado para integrar seamlessly ao seu sistema RAG existente, fornecendo fallback automático, configuração inteligente e melhorias significativas de qualidade.

### ✨ Principais Benefícios

- **Fallback Automático**: Funciona mesmo sem APIs ou dependências específicas
- **Configuração Inteligente**: Detecta automaticamente a melhor implementação
- **Interface Compatível**: Drop-in replacement para o RAPTOR original
- **Performance Otimizada**: Cache multicamada e processamento paralelo
- **Métricas Integradas**: Monitoramento completo integrado ao sistema

## 🔧 Como Integrar

### 1. Configuração no YAML

Adicione a seção `raptor_integration` no seu `config/llm_providers_config.yaml`:

```yaml
raptor_integration:
  # Estratégia de implementação
  preferred_implementation: "enhanced"  # enhanced, offline, original
  auto_fallback: true
  
  # Configurações Gerais
  chunk_size: 400
  chunk_overlap: 80
  max_levels: 4
  
  # Enhanced Implementation (Preferred)
  enhanced:
    enabled: true
    embedding:
      provider: "sentence_transformers"  # openai se tiver API key
      model: "all-MiniLM-L6-v2"
    clustering:
      method: "umap_gmm"  # melhor qualidade
    summarization:
      use_llm: false  # true se tiver OpenAI API key
      provider: "openai"
    cache:
      enabled: true
      type: "multi_layer"
```

### 2. Modificar o Pipeline RAG

#### No `src/rag_pipeline_advanced.py`:

```python
# Adicionar imports
from .retrieval.raptor_integration import (
    RaptorIntegrationAdapter,
    detect_optimal_raptor_config
)

class AdvancedRAGPipeline(APIRAGPipeline):
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        
        # RAPTOR Enhanced Integration
        self.raptor_adapter: Optional[RaptorIntegrationAdapter] = None
        self.raptor_integration_enabled = True
    
    async def _initialize_raptor_enhanced(self) -> None:
        """Inicializa RAPTOR Enhanced com fallback automático"""
        
        if self.raptor_adapter is not None:
            return
        
        try:
            # Usar configuração do YAML ou detectar automaticamente
            raptor_config = self.config.get("raptor_integration")
            if not raptor_config:
                raptor_config = detect_optimal_raptor_config()
            
            self.raptor_adapter = RaptorIntegrationAdapter(raptor_config)
            
            if await self.raptor_adapter.initialize():
                logger.info(f"✅ RAPTOR Enhanced inicializado: {self.raptor_adapter.implementation_type}")
            else:
                logger.error("❌ Falha ao inicializar qualquer implementação RAPTOR")
                self.raptor_adapter = None
                
        except Exception as e:
            logger.error(f"Erro ao inicializar RAPTOR Enhanced: {e}")
            self.raptor_adapter = None
    
    async def build_raptor_tree(self, documents: List[str]) -> Dict[str, Any]:
        """Constrói árvore RAPTOR usando implementação integrada"""
        
        await self._initialize_raptor_enhanced()
        
        if self.raptor_adapter is None:
            return {"error": "RAPTOR não disponível"}
        
        return await self.raptor_adapter.build_tree(documents)
    
    async def retrieve(self, query: str, retrieval_method: str = "hybrid", **kwargs) -> List[Dict]:
        """Retrieval unificado incluindo RAPTOR Enhanced"""
        
        if retrieval_method == "raptor":
            await self._initialize_raptor_enhanced()
            if self.raptor_adapter:
                return await self.raptor_adapter.search(query, k=kwargs.get("k", 10))
            else:
                logger.warning("RAPTOR não disponível, usando retrieval básico")
                return await self._basic_retrieval(query, kwargs.get("k", 10))
        
        elif retrieval_method == "hybrid":
            # Combinar RAPTOR com outros métodos
            await self._initialize_raptor_enhanced()
            
            if self.raptor_adapter and self.raptor_adapter.tree_built:
                raptor_docs = await self.raptor_adapter.search(query, k=kwargs.get("k", 10)//2)
                other_docs = await self._basic_retrieval(query, kwargs.get("k", 10)//2)
                
                # Combinar e reranquear
                all_docs = raptor_docs + other_docs
                all_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
                return all_docs[:kwargs.get("k", 10)]
            else:
                return await self._basic_retrieval(query, kwargs.get("k", 10))
        
        # Outros métodos...
        return await self._basic_retrieval(query, kwargs.get("k", 10))
```

### 3. Uso na API

#### No `src/api/main.py`:

```python
@app.post("/raptor/build")
async def build_raptor_tree(request: BuildTreeRequest):
    """Endpoint para construir árvore RAPTOR"""
    
    try:
        pipeline = get_pipeline()
        result = await pipeline.build_raptor_tree(request.documents)
        
        if result.get("success"):
            return {
                "success": True,
                "stats": result["stats"],
                "implementation": result.get("implementation")
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))
            
    except Exception as e:
        logger.error(f"Erro ao construir árvore RAPTOR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query com suporte a RAPTOR Enhanced"""
    
    pipeline = get_pipeline()
    
    # Usar RAPTOR se especificado
    retrieval_method = request.config.get("retrieval_method", "hybrid")
    
    result = await pipeline.query_advanced(
        question=request.question,
        config={
            "retrieval_method": retrieval_method,
            "k": request.config.get("k", 10)
        }
    )
    
    return result
```

## 🎯 Modos de Operação

### Modo Production (OpenAI APIs)

```yaml
raptor_integration:
  preferred_implementation: "enhanced"
  enhanced:
    embedding:
      provider: "openai"
      model: "text-embedding-3-small"
    summarization:
      use_llm: true
      provider: "openai"
      model: "gpt-4o-mini"
```

**Benefícios:**
- Máxima qualidade de embeddings e resumos
- LLM summarization inteligente
- Integração com APIs de produção

### Modo Offline (Sentence-Transformers)

```yaml
raptor_integration:
  preferred_implementation: "enhanced"
  enhanced:
    embedding:
      provider: "sentence_transformers"
      model: "all-MiniLM-L6-v2"
    clustering:
      method: "umap_gmm"
    summarization:
      use_llm: false
```

**Benefícios:**
- Funciona sem APIs externas
- Qualidade superior ao mock
- Clustering avançado com UMAP+GMM

### Modo Legacy (Compatibilidade)

```yaml
raptor_integration:
  preferred_implementation: "original"
  auto_fallback: true
```

**Benefícios:**
- Compatibilidade total com sistema existente
- Fallback para Enhanced se disponível

## 🔄 Fluxo de Fallback

O sistema tenta as implementações nesta ordem:

1. **Enhanced**: Melhor qualidade, requer sentence-transformers
2. **Offline**: Boa qualidade, dependências mínimas  
3. **Original**: Compatibilidade garantida, usa mocks

## 📊 Métricas e Monitoramento

```python
# Obter métricas de integração
metrics = pipeline.raptor_adapter.get_integration_metrics()

print(f"Implementação ativa: {metrics['implementation_used']}")
print(f"Tentativas de fallback: {metrics['fallback_attempts']}")
print(f"Total de queries: {metrics['total_queries']}")
print(f"Tempo de inicialização: {metrics['initialization_time']}")

# Resumo da árvore construída
tree_summary = pipeline.raptor_adapter.get_tree_summary()
print(f"Árvore construída: {tree_summary['tree_built']}")
print(f"Estatísticas: {tree_summary['stats']}")
```

## 🧪 Testando a Integração

### Teste Simples

```python
import asyncio
from src.retrieval.raptor_integration import RaptorIntegrationAdapter, detect_optimal_raptor_config

async def test_integration():
    # Configuração automática
    config = detect_optimal_raptor_config()
    adapter = RaptorIntegrationAdapter(config)
    
    # Inicialização
    if await adapter.initialize():
        print(f"✅ Inicializado: {adapter.implementation_type}")
        
        # Construir árvore
        documents = ["Python é uma linguagem...", "Machine Learning é..."]
        result = await adapter.build_tree(documents)
        
        if result.get("success"):
            # Buscar
            results = await adapter.search("O que é Python?", k=3)
            print(f"Encontrados {len(results)} resultados")

asyncio.run(test_integration())
```

### Demo Completo

Execute o demo para ver todas as funcionalidades:

```bash
python demo_raptor_integration.py
```

## 🎯 Próximos Passos

1. **Instalar dependências**:
   ```bash
   pip install -r requirements_raptor_enhanced.txt
   ```

2. **Configurar ambiente**:
   ```bash
   # Opcional: para máxima qualidade
   export OPENAI_API_KEY="your-key-here"
   
   # Opcional: para cache Redis
   export REDIS_URL="redis://localhost:6379"
   ```

3. **Atualizar configuração**:
   - Adicionar seção `raptor_integration` no YAML
   - Configurar parâmetros para seu domínio

4. **Modificar pipeline**:
   - Adicionar imports do raptor_integration
   - Substituir `_initialize_raptor_retriever` por `_initialize_raptor_enhanced`
   - Atualizar método `retrieve()` para suportar RAPTOR

5. **Testar**:
   - Executar demo de integração
   - Verificar métricas e logs
   - Validar qualidade dos resultados

## 🏆 Resultados Esperados

### Qualidade
- **20x melhoria** em similarity scores vs mock embeddings
- Clustering mais coerente com UMAP+GMM
- Resumos mais relevantes (com ou sem LLM)

### Performance
- Construção da árvore: ~1-2s para documentos pequenos
- Busca: ~50-200ms por query
- Cache automático reduz tempos subsequentes

### Confiabilidade
- Fallback automático garante funcionamento
- Configuração dinâmica adapta ao ambiente
- Métricas completas para monitoramento

## 🆘 Troubleshooting

### Problema: "RAPTOR Integration não disponível"
**Solução**: Instalar dependências:
```bash
pip install sentence-transformers umap-learn scikit-learn
```

### Problema: Qualidade baixa dos resultados
**Soluções**:
1. Usar OpenAI embeddings se disponível
2. Configurar `use_llm: true` para summarização
3. Ajustar parâmetros de chunking para seu domínio

### Problema: Performance lenta
**Soluções**:
1. Habilitar cache Redis
2. Ajustar `batch_size` e `max_workers`
3. Usar clustering mais simples (`pca_gmm` ou `kmeans`)

### Problema: Erros de fallback
**Verificar**:
1. Logs para ver qual implementação falhou
2. Dependências instaladas
3. Configuração do YAML válida

---

🎉 **Pronto!** O RAPTOR Enhanced agora está integrado ao seu sistema RAG com fallback automático e configuração inteligente.