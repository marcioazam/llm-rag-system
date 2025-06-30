# 🏗️ Guia de Gerenciamento de Projetos - Sistema RAG

## Visão Geral

O sistema RAG agora implementa **isolamento completo por projeto**, garantindo que documentos e conhecimento sejam organizados e acessados de forma segura e controlada.

### 🔐 **Princípios de Segurança**
- **Validação Obrigatória**: Apenas projetos pré-existentes podem receber documentos
- **Isolamento Completo**: Cada projeto tem seu próprio namespace de conhecimento
- **CRUD Completo**: Criação, leitura, atualização e deleção de projetos
- **Auditoria**: Rastreamento completo de criação e modificações

## 📋 **API Endpoints de Projetos**

### **1. Criar Projeto**
```http
POST /projects
Content-Type: application/json

{
  "id": "projeto-web-ecommerce",
  "name": "E-commerce Web Application",
  "description": "Sistema de e-commerce desenvolvido em React/Node.js",
  "metadata": {
    "team": "frontend",
    "technology": "react",
    "priority": "high"
  }
}
```

### **2. Listar Projetos**
```http
GET /projects?status=active&limit=10&offset=0
```

### **3. Obter Projeto Específico**
```http
GET /projects/projeto-web-ecommerce
```

### **4. Atualizar Projeto**
```http
PUT /projects/projeto-web-ecommerce
Content-Type: application/json

{
  "name": "E-commerce Web App - Atualizado",
  "status": "inactive"
}
```

### **5. Deletar Projeto**
```http
DELETE /projects/projeto-web-ecommerce?force=true
```

### **6. Estatísticas do Projeto**
```http
GET /projects/projeto-web-ecommerce/stats
```

## 📚 **Indexação com Validação de Projeto**

### **Adicionar Documentos**
```http
POST /add_documents
Content-Type: application/json

{
  "project_id": "projeto-web-ecommerce",
  "documents": [
    {
      "content": "// Componente React para carrinho de compras",
      "source": "src/components/ShoppingCart.jsx"
    }
  ]
}
```

### **Query com Projeto Específico**
```http
POST /query
Content-Type: application/json

{
  "question": "Como implementar autenticação JWT?",
  "project_id": "projeto-web-ecommerce",
  "k": 5
}
```

## 🐍 **Cliente Python**

```python
import requests

class RAGProjectManager:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def create_project(self, project_id, name, description=None):
        data = {
            "id": project_id,
            "name": name,
            "description": description
        }
        response = requests.post(f"{self.base_url}/projects", json=data)
        return response.json()
    
    def add_documents(self, project_id, documents):
        data = {
            "project_id": project_id,
            "documents": documents
        }
        response = requests.post(f"{self.base_url}/add_documents", json=data)
        return response.json()
    
    def query_project(self, question, project_id=None):
        data = {"question": question}
        if project_id:
            data["project_id"] = project_id
        response = requests.post(f"{self.base_url}/query", json=data)
        return response.json()

# Exemplo de uso
manager = RAGProjectManager()

# 1. Criar projeto
project = manager.create_project(
    project_id="exemplo-python",
    name="Projeto Python de Exemplo"
)

# 2. Adicionar documentos
documents = [
    {
        "content": "def calcular_desconto(preco, percentual): return preco * (1 - percentual/100)",
        "metadata": {"file": "utils.py", "function": "calcular_desconto"}
    }
]
manager.add_documents("exemplo-python", documents)

# 3. Fazer query
result = manager.query_project(
    "Como calcular desconto em Python?",
    project_id="exemplo-python"
)
print(result["answer"])
```

## 🚨 **Validação e Erros**

### **Projeto Não Existe**
```json
{
  "detail": "Projeto 'projeto-inexistente' não existe. Crie o projeto primeiro."
}
```

### **ID Duplicado**
```json
{
  "detail": "Projeto com ID 'projeto-duplicado' já existe"
}
```

## 🔧 **Estrutura do Banco**

### **Tabela: projects**
```sql
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active',
    metadata JSON
);
```

### **Tabela: chunks (com Foreign Key)**
```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    content TEXT,
    metadata JSON,
    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
);
```

## 💡 **Melhores Práticas**

1. **Sempre criar projeto antes de indexar**
2. **Usar IDs descritivos**: `projeto-web-frontend`
3. **Incluir metadados estruturados**
4. **Monitorar estatísticas regularmente**
5. **Fazer backup antes de deletar**

## 🎯 **Casos de Uso**

### **Multi-tenant SaaS**
```python
# Cada cliente tem seu projeto
for cliente in clientes:
    manager.create_project(
        project_id=f"cliente-{cliente.id}",
        name=f"Projeto {cliente.nome}"
    )
```

### **Ambientes Separados**
```python
# Dev, Staging, Prod
for env in ["dev", "staging", "prod"]:
    manager.create_project(
        project_id=f"projeto-{env}",
        name=f"Projeto - {env.upper()}"
    )
```

---

## 🎉 **Resultado Final**

✅ **Isolamento Completo**: Cada projeto tem seu namespace  
✅ **Segurança**: Validação obrigatória de project_id  
✅ **CRUD Completo**: Gerenciamento completo de projetos  
✅ **Performance**: Índices otimizados  
✅ **Auditoria**: Rastreamento completo  

Sistema RAG multi-projeto robusto e escalável! 🚀 