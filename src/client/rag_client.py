import requests
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


class RAGClient:
    """Cliente Python para interagir com a API do sistema RAG"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    # Métodos de consulta
    def query(self,
              question: str,
              k: int = 5,
              system_prompt: Optional[str] = None,
              use_hybrid: bool = False,
              llm_only: bool = False) -> Dict[str, Any]:
        """
        Fazer uma pergunta ao sistema RAG
        
        Args:
            question: Pergunta a ser feita
            k: Número de documentos a recuperar
            system_prompt: Prompt customizado para o sistema
            use_hybrid: Se True, usa modo híbrido com múltiplos modelos
            llm_only: Se True, usa apenas o LLM sem RAG
        """
        payload = {
            "question": question,
            "query": question,  # Compatibilidade com ambas as APIs
            "k": k,
            "use_hybrid": use_hybrid
        }
        
        if system_prompt:
            payload["system_prompt"] = system_prompt
            
        if llm_only:
            payload["llm_only"] = True
            
        response = requests.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        return response.json()
    
    def query_llm_only(self,
                       question: str,
                       system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Fazer uma pergunta usando apenas o LLM"""
        return self.query(
            question=question,
            system_prompt=system_prompt,
            llm_only=True
        )
    
    def query_with_code(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Faz uma consulta que sempre inclui exemplos de código"""
        response = requests.post(
            f"{self.base_url}/query_with_code",
            json={
                "query": query,
                "k": k
            }
        )
        response.raise_for_status()
        return response.json()
    
    # Métodos de indexação e upload
    def index_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """Indexar documentos por paths"""
        response = requests.post(
            f"{self.base_url}/index",
            json={"document_paths": document_paths}
        )
        response.raise_for_status()
        return response.json()
    
    def add_documents(self, 
                     documents: List[Dict[str, str]], 
                     chunking_strategy: str = 'recursive',
                     chunk_size: int = 500,
                     chunk_overlap: int = 50) -> Dict[str, Any]:
        """
        Adiciona documentos ao sistema RAG
        
        Args:
            documents: Lista de documentos com 'content' e 'source'
            chunking_strategy: Estratégia de chunking
            chunk_size: Tamanho dos chunks
            chunk_overlap: Sobreposição entre chunks
        """
        payload = {
            "documents": documents,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        response = requests.post(f"{self.base_url}/add_documents", json=payload)
        response.raise_for_status()
        return response.json()
    
    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """Upload e indexação de arquivo"""
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{self.base_url}/upload",
                files=files
            )
        response.raise_for_status()
        return response.json()
    
    # Métodos de limpeza
    def clear_index(self) -> Dict[str, Any]:
        """Limpar índice"""
        response = requests.delete(f"{self.base_url}/index")
        response.raise_for_status()
        return response.json()
    
    def clear_database(self) -> Dict[str, Any]:
        """Limpa o banco de dados vetorial"""
        response = requests.delete(f"{self.base_url}/clear_database")
        response.raise_for_status()
        return response.json()
    
    # Métodos de informações e status
    def get_info(self) -> Dict[str, Any]:
        """Obter informações do sistema"""
        response = requests.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do sistema"""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica a saúde do sistema"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


# Exemplo de uso
if __name__ == "__main__":
    # Inicializa o cliente
    client = RAGClient()
    
    try:
        # Verifica a saúde do sistema
        print("Verificando saúde do sistema...")
        health = client.health_check()
        print(f"Status: {health}")
        
        # Exemplo 1: Adicionar documentos
        documents = [
            {
                "content": """Python é uma linguagem de programação de alto nível, 
                interpretada e de propósito geral. É conhecida por sua sintaxe clara 
                e legibilidade. Para criar um sistema web robusto em Python, você pode 
                usar frameworks como Django ou FastAPI.""",
                "source": "python_basics.txt"
            },
            {
                "content": """FastAPI é um framework web moderno e rápido para construir 
                APIs com Python 3.6+ baseado em type hints padrão do Python. É muito 
                rápido, fácil de usar e produz código pronto para produção.""",
                "source": "fastapi_guide.txt"
            }
        ]
        
        print("\nAdicionando documentos...")
        result = client.add_documents(documents)
        print(f"Resultado: {result}")
        
        # Exemplo 2: Query simples
        print("\n--- Query Simples ---")
        result = client.query("O que é Python?")
        print(f"Resposta: {result.get('answer', result.get('response', 'N/A'))}")
        
        # Exemplo 3: Query híbrida
        print("\n--- Query Híbrida ---")
        result = client.query("Como fazer um sistema web robusto com Python?", use_hybrid=True)
        print(f"Resposta: {result.get('answer', result.get('response', 'N/A'))}")
        
        # Exemplo 4: Query apenas LLM
        print("\n--- Query LLM Only ---")
        result = client.query_llm_only("Explique os benefícios do Python")
        print(f"Resposta: {result.get('answer', result.get('response', 'N/A'))}")
        
        # Exemplo 5: Query com código
        print("\n--- Query com Código ---")
        result = client.query_with_code("Mostre um exemplo de API FastAPI")
        print(f"Resposta: {result.get('answer', result.get('response', 'N/A'))}")
        
        # Exemplo 6: Upload de arquivo
        # print("\n--- Upload de Arquivo ---")
        # result = client.upload_file("exemplo.txt")
        # print(f"Upload resultado: {result}")
        
        # Exemplo 7: Estatísticas
        print("\n--- Estatísticas ---")
        try:
            stats = client.get_stats()
            print(json.dumps(stats, indent=2))
        except:
            info = client.get_info()
            print(json.dumps(info, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão: {e}")
    except Exception as e:
        print(f"Erro: {e}")
