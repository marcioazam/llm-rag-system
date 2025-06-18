"""
RAG Pipeline totalmente baseado em APIs externas.
Remove todas as dependências de modelos locais.
"""

import os
import yaml
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Imports dos novos serviços baseados em API
from .embeddings.api_embedding_service import APIEmbeddingService
from .models.api_model_router import APIModelRouter, TaskType
from .retrieval.retriever import HybridRetriever
from .utils.document_loader import DocumentLoader
from .chunking.recursive_chunker import RecursiveChunker
from .vectordb.qdrant_store import QdrantVectorStore

# Carregar variáveis de ambiente
load_dotenv()

logger = logging.getLogger(__name__)


class APIRAGPipeline:
    """
    Pipeline RAG baseado completamente em APIs externas.
    Não utiliza modelos locais como Ollama, sentence-transformers, etc.
    """

    def __init__(self, 
                 config_path: str = "config/llm_providers_config.yaml",
                 collection_name: Optional[str] = None):
        
        # Carregar configuração
        self.config = self._load_config(config_path)
        
        # Configurações básicas
        self.collection_name = collection_name or "rag_documents"
        
        # Configurar logging
        self._setup_logging()
        
        # Inicializar componentes
        self._initialize_components()
        
        # Estatísticas
        self.stats = {
            "total_queries": 0,
            "total_documents_indexed": 0,
            "total_cost": 0.0,
            "average_query_time": 0.0,
            "cache_hit_rate": 0.0,
            "provider_usage": {},
            "errors": 0
        }
        
        logger.info("APIRAGPipeline inicializado com sucesso")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # Substituir variáveis de ambiente
            config = self._replace_env_variables(config)
            
            return config
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            # Retornar configuração mínima de fallback
            return self._get_fallback_config()

    def _replace_env_variables(self, config: Any) -> Any:
        """Substitui variáveis de ambiente no formato ${VAR_NAME}"""
        if isinstance(config, dict):
            return {k: self._replace_env_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_variables(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        else:
            return config

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Configuração mínima de fallback"""
        return {
            "providers": {
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "models": {
                        "gpt4o_mini": {
                            "name": "gpt-4o-mini",
                            "max_tokens": 16384,
                            "temperature": 0.3,
                            "responsibilities": ["code_generation", "debugging"],
                            "context_window": 128000,
                            "cost_per_1k_tokens": 0.00015,
                            "priority": 2
                        }
                    }
                }
            },
            "embeddings": {
                "primary_provider": "openai",
                "providers": {
                    "openai": {
                        "models": {
                            "text_embedding_3_small": {
                                "name": "text-embedding-3-small",
                                "dimensions": 1536,
                                "max_input": 8191,
                                "cost_per_1k_tokens": 0.00002,
                                "use_for": ["quick_embeddings", "classification"]
                            }
                        }
                    }
                }
            },
            "routing": {
                "strategy": "cost_performance_optimized",
                "fallback_chain": ["openai.gpt4o_mini"]
            },
            "optimization": {
                "caching": {
                    "enabled": True,
                    "cache_embeddings": True,
                    "ttl_seconds": 3600
                }
            }
        }

    def _setup_logging(self):
        """Configura logging"""
        log_level = self.config.get("development", {}).get("verbose_logging", False)
        if log_level:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def _initialize_components(self):
        """Inicializa todos os componentes do pipeline"""
        
        # Serviço de embeddings via API
        self.embedding_service = APIEmbeddingService(self.config)
        
        # Roteador de modelos via API
        self.model_router = APIModelRouter(self.config)
        
        # Vector store (Qdrant)
        self.vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            embedding_function=self.embedding_service.encode,
            host="localhost",
            port=6333
        )
        
        # Chunker para documentos
        self.chunker = RecursiveChunker(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Document loader
        self.document_loader = DocumentLoader()
        
        # Retriever híbrido
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service
        )
        
        logger.info("Todos os componentes inicializados")

    def query(self, 
              question: str, 
              k: int = 5,
              task_type: Optional[TaskType] = None,
              force_model: Optional[str] = None,
              include_sources: bool = True) -> Dict[str, Any]:
        """
        Executa query no sistema RAG.
        """
        start_time = time.time()
        
        try:
            # 1. Recuperar documentos relevantes
            retrieved_docs = self.retriever.retrieve(question, k=k)
            
            if not retrieved_docs:
                # Fallback para resposta sem contexto
                logger.warning("Nenhum documento relevante encontrado")
                response = self.model_router.generate_response(
                    query=question,
                    context="",
                    task_type=task_type,
                    force_model=force_model
                )
                
                return {
                    "answer": response.content,
                    "sources": [],
                    "model_used": response.model,
                    "provider_used": response.provider,
                    "cost": response.cost,
                    "processing_time": time.time() - start_time,
                    "context_found": False
                }
            
            # 2. Construir contexto
            context_chunks = []
            sources = []
            
            for doc in retrieved_docs:
                context_chunks.append(doc["content"])
                if include_sources:
                    sources.append({
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "score": doc.get("score", 0.0)
                    })
            
            context = "\n\n".join(context_chunks)
            
            # 3. Gerar resposta
            response = self.model_router.generate_response(
                query=question,
                context=context,
                task_type=task_type,
                force_model=force_model
            )
            
            # 4. Atualizar estatísticas
            processing_time = time.time() - start_time
            self.stats["total_queries"] += 1
            self.stats["total_cost"] += response.cost
            
            result = {
                "answer": response.content,
                "sources": sources if include_sources else [],
                "model_used": response.model,
                "provider_used": response.provider,
                "cost": response.cost,
                "processing_time": processing_time,
                "context_found": True,
                "total_chunks_retrieved": len(retrieved_docs)
            }
            
            logger.info(f"Query processada em {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Erro ao processar query: {e}")
            return {
                "answer": f"Erro ao processar pergunta: {str(e)}",
                "sources": [],
                "error": str(e)
            }

    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adiciona documentos ao índice"""
        try:
            total_chunks = 0
            
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                if not content:
                    continue
                
                # Chunking
                chunks = self.chunker.chunk_text(content)
                
                # Adicionar metadados aos chunks
                chunk_docs = []
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source": metadata.get("source", "unknown")
                    }
                    chunk_docs.append({
                        "content": chunk.text,
                        "metadata": chunk_metadata
                    })
                
                # Adicionar ao vector store
                self.vector_store.add_documents(chunk_docs)
                total_chunks += len(chunks)
            
            self.stats["total_documents_indexed"] += len(documents)
            
            return {
                "success": True,
                "documents_processed": len(documents),
                "chunks_created": total_chunks
            }
            
        except Exception as e:
            logger.error(f"Erro ao adicionar documentos: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema"""
        return self.stats

    def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do sistema"""
        health = {
            "status": "healthy",
            "api_keys_configured": {},
        }
        
        # Verificar API keys
        api_keys = {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY")),
        }
        
        health["api_keys_configured"] = api_keys
        
        if not any(api_keys.values()):
            health["status"] = "unhealthy"
            health["error"] = "Nenhuma API key configurada"
        
        return health 