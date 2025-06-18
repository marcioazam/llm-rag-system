# 🎯 API UNIFICADA: Um Endpoint para Todos

## ✅ **PROBLEMA RESOLVIDO**

**ANTES:** Dois sistemas separados
- `cursor_endpoint.py` (247 linhas) - Endpoints específicos para Cursor
- `main.py` (369 linhas) - API geral

**AGORA:** Um sistema unificado
- `main.py` (400+ linhas) - API única com otimizações automáticas
- ❌ `cursor_endpoint.py` - REMOVIDO

---

## 🚀 **COMO USAR**

### **1. 📱 Uso Geral (MCP, aplicações web, etc.)**
```json
POST /query
{
  "question": "Como implementar autenticação JWT?",
  "k": 5,
  "use_hybrid": true
}
```

### **2. 🖥️ Uso Otimizado para Cursor IDE**
```json
POST /query
{
  "question": "Como usar esta função?",
  "k": 3,
  "context": "def authenticate_user(username, password):\n    # código atual",
  "file_type": ".py",
  "project_context": "Sistema de autenticação Flask",
  "quick_mode": true,
  "max_response_time": 5
}
```

### **3. 🔧 Parâmetros Disponíveis**

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| **`question`** | string | - | **Obrigatório** - Pergunta principal |
| `k` | int | 5 | Número de chunks a recuperar |
| `use_hybrid` | bool | true | Usar busca híbrida |
| `llm_only` | bool | false | Usar apenas LLM (sem RAG) |
| `system_prompt` | string | null | Prompt customizado |
| | | | |
| **CURSOR ESPECÍFICOS** | | | |
| `context` | string | null | Código atual do arquivo |
| `file_type` | string | null | Tipo de arquivo (.py, .js, etc) |
| `project_context` | string | null | Contexto do projeto |
| `quick_mode` | bool | false | Modo rápido (menos chunks) |
| `allow_hybrid` | bool | true | Permitir busca híbrida |
| `max_response_time` | int | 15 | Timeout máximo |

---

## 🧠 **INTELIGÊNCIA AUTOMÁTICA**

### **Detecção Automática de Contexto**
A API detecta automaticamente se é uma **request do Cursor** baseado nos parâmetros enviados:

```python
# Se request contém context, file_type ou project_context
# Ativa otimizações para Cursor automaticamente
if request.context or request.file_type or request.project_context:
    # Constrói prompt especializado para IDE
    # Ajusta parâmetros para velocidade
    # Adiciona contexto específico da linguagem
```

### **Otimizações Ativadas**
- ✅ **System prompt otimizado** para cada linguagem
- ✅ **Redução de K** em modo rápido (3 em vez de 5)
- ✅ **Contexto de código** incluído no prompt
- ✅ **Métricas de performance** na resposta

---

## 📊 **EXEMPLOS PRÁTICOS**

### **MCP pode chamar diretamente:**
```python
# tools/rag_query.py para MCP
import httpx

async def rag_query(question: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/query",
            json={
                "question": question,
                "k": 5,
                "use_hybrid": True
            }
        )
        return response.json()

# Uso no MCP
result = await rag_query("Como implementar cache em Python?")
```

### **Cursor pode usar com contexto:**
```javascript
// Cursor integration
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "Como melhorar esta função?",
    context: getCurrentFileContent(),
    file_type: getCurrentFileExtension(),
    project_context: getProjectName(),
    quick_mode: true
  })
});
```

### **Aplicação web padrão:**
```python
# cliente.py
import requests

def ask_rag(question: str) -> dict:
    response = requests.post(
        "http://localhost:8000/query",
        json={"question": question}
    )
    return response.json()
```

---

## 🎯 **RESPOSTA UNIFICADA**

### **Estrutura Padrão**
```json
{
  "answer": "Resposta gerada...",
  "sources": [{"content": "...", "metadata": {...}}],
  "model": "openai.gpt4o_mini",
  "provider_used": "openai",
  "processing_time": 2.314,
  "mode": "cursor_optimized",
  "k_used": 3
}
```

### **Campos Adicionais para Cursor**
- `processing_time`: Tempo de processamento em segundos
- `mode`: "cursor_optimized" ou "standard"
- `k_used`: Número real de chunks usados
- `provider_used`: Provedor LLM utilizado

---

## ⚡ **VANTAGENS DA UNIFICAÇÃO**

### **Para Desenvolvedores**
- ✅ **Um endpoint para tudo** - Menos complexidade
- ✅ **Parâmetros opcionais** - Backward compatibility
- ✅ **Otimizações automáticas** - Detecta contexto

### **Para MCP**
- ✅ **Integração direta** - Apenas `/query`
- ✅ **Sem configuração extra** - Funciona out-of-the-box
- ✅ **Performance padrão** - Otimizada por padrão

### **Para Cursor**
- ✅ **Otimizações específicas** - Context-aware
- ✅ **Modo rápido** - Respostas em 2-5s
- ✅ **Prompts especializados** - Por linguagem

### **Para Manutenção**
- ✅ **Menos código** - 247 linhas removidas
- ✅ **Menos bugs** - Um sistema em vez de dois
- ✅ **Mais simples** - Uma API para manter

---

## 🔧 **MIGRAÇÃO**

### **Se você usava `/cursor/query` antes:**
```diff
- POST /cursor/query
+ POST /query

# Adicionar parâmetros específicos:
{
  "question": "sua pergunta",
+ "context": "código atual",
+ "file_type": ".py",
+ "quick_mode": true
}
```

### **Se você usava `/cursor/quick` antes:**
```diff
- POST /cursor/quick
+ POST /query

# Equivalente:
{
  "question": "sua pergunta",
+ "quick_mode": true,
+ "k": 3
}
```

---

## ✅ **RESULTADO FINAL**

### **API Simplificada:**
- 🎯 **1 endpoint principal**: `/query`
- 🧠 **Inteligência automática**: Detecta contexto
- ⚡ **Otimizações transparentes**: Sem configuração
- 🔄 **Compatibilidade total**: Funciona para todos

### **Todos podem usar `/query`:**
- **MCP** → Direto, sem parâmetros extras
- **Cursor** → Com contexto específico
- **Web apps** → Padrão simples
- **CLI tools** → Flexibilidade total

**🎉 Uma API para governar todas elas!** 🎯 