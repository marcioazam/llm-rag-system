"""
Serviço de Embeddings via API externa.
Substitui completamente modelos locais como sentence-transformers.
Suporta OpenAI, Google e outros provedores.
"""

import os
import asyncio
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    COHERE = "cohere"
    VOYAGE = "voyage"


@dataclass
class EmbeddingResponse:
    """Response do serviço de embedding"""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]
    provider: str
    cost: float = 0.0
    dimensions: int = 0
    processing_time: float = 0.0


@dataclass
class EmbeddingCache:
    """Cache para embeddings"""
    embeddings: List[List[float]]
    model: str
    provider: str
    timestamp: float
    dimensions: int


class APIEmbeddingService:
    """
    Serviço de embeddings via API externa.
    Suporta múltiplos provedores com fallback automático.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # FASE 1: Configuração padrão se não fornecida
        if config is None:
            config = self._get_default_config()
            
        self.config = config
        self.providers_config = config.get("embeddings", {}).get("providers", {})
        self.primary_provider = config.get("embeddings", {}).get("primary_provider", "openai")
        
        # Cache de embeddings
        self.cache: Dict[str, Dict] = {}
        self.cache_enabled = config.get("optimization", {}).get("caching", {}).get("cache_embeddings", True)
        self.cache_ttl = config.get("optimization", {}).get("caching", {}).get("ttl_seconds", 3600)
        
        # Rate limiting
        self.rate_limit_enabled = config.get("optimization", {}).get("rate_limiting", {}).get("enabled", True)
        self.requests_per_minute = config.get("optimization", {}).get("rate_limiting", {}).get("requests_per_minute", 100)
        
        # Configurar sessão HTTP com retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Monitoramento
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_cost": 0.0,
            "errors": 0,
            "provider_usage": {},
            "processing_time": []
        }
        
        logger.info(f"APIEmbeddingService inicializado com provedor primário: {self.primary_provider}")

    def _get_default_config(self) -> Dict[str, Any]:
        """FASE 1: Retorna configuração padrão para funcionamento básico"""
        return {
            "embeddings": {
                "primary_provider": "openai",
                "providers": {
                    "openai": {
                        "models": {
                            "text_embedding_ada_002": {
                                "cost_per_1k_tokens": 0.0001,
                                "max_tokens": 8191,
                                "dimensions": 1536
                            }
                        }
                    }
                }
            },
            "optimization": {
                "caching": {
                    "cache_embeddings": True,
                    "ttl_seconds": 3600
                },
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 100
                }
            }
        }

    def _get_cache_key(self, text: str, model: str, provider: str) -> str:
        """Gera chave do cache"""
        content = f"{text}|{model}|{provider}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: EmbeddingCache) -> bool:
        """Verifica se entrada do cache ainda é válida"""
        if not self.cache_enabled:
            return False
        return time.time() - cache_entry.timestamp < self.cache_ttl

    def _get_from_cache(self, text: str, model: str, provider: str) -> Optional[List[float]]:
        """Recupera embedding do cache"""
        if not self.cache_enabled:
            return None
            
        cache_key = self._get_cache_key(text, model, provider)
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry):
                self.stats["cache_hits"] += 1
                return cache_entry.embeddings[0] if cache_entry.embeddings else None
        return None

    def _save_to_cache(self, text: str, model: str, provider: str, embedding: List[float]):
        """Salva embedding no cache"""
        if not self.cache_enabled:
            return
            
        cache_key = self._get_cache_key(text, model, provider)
        self.cache[cache_key] = EmbeddingCache(
            embeddings=[embedding],
            model=model,
            provider=provider,
            timestamp=time.time(),
            dimensions=len(embedding)
        )

    def _call_openai_api(self, texts: List[str], model: str) -> EmbeddingResponse:
        """Chama API da OpenAI"""
        start_time = time.time()
        
        provider_config = self.providers_config.get("openai", {})
        api_key = os.getenv("OPENAI_API_KEY") or provider_config.get("api_key", "").replace("${OPENAI_API_KEY}", "")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY não configurada")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": texts,
            "model": model,
            "encoding_format": "float"
        }

        try:
            response = self.session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            
            # Calcular custo
            model_config = provider_config.get("models", {}).get(model.replace("-", "_"), {})
            cost_per_1k = model_config.get("cost_per_1k_tokens", 0.00002)
            total_tokens = data["usage"]["total_tokens"]
            cost = (total_tokens / 1000) * cost_per_1k
            
            processing_time = time.time() - start_time
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                usage=data["usage"],
                provider="openai",
                cost=cost,
                dimensions=len(embeddings[0]) if embeddings else 0,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Erro na API OpenAI: {e}")
            raise

    def _call_google_api(self, texts: List[str], model: str) -> EmbeddingResponse:
        """Chama API do Google"""
        start_time = time.time()
        
        provider_config = self.providers_config.get("google", {})
        api_key = os.getenv("GOOGLE_API_KEY") or provider_config.get("api_key", "").replace("${GOOGLE_API_KEY}", "")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY não configurada")

        headers = {"Content-Type": "application/json"}
        
        embeddings = []
        total_tokens = 0
        
        # Google API processa um texto por vez
        for text in texts:
            payload = {
                "model": f"models/{model}",
                "content": {"parts": [{"text": text}]}
            }
            
            try:
                response = self.session.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent",
                    headers=headers,
                    json=payload,
                    params={"key": api_key},
                    timeout=60
                )
                response.raise_for_status()
                
                data = response.json()
                embeddings.append(data["embedding"]["values"])
                total_tokens += len(text.split())  # Estimativa
                
            except Exception as e:
                logger.error(f"Erro na API Google para texto: {e}")
                raise

        # Calcular custo
        model_config = provider_config.get("models", {}).get(model.replace("-", "_"), {})
        cost_per_1k = model_config.get("cost_per_1k_tokens", 0.00001)
        cost = (total_tokens / 1000) * cost_per_1k
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            usage={"total_tokens": total_tokens},
            provider="google",
            cost=cost,
            dimensions=len(embeddings[0]) if embeddings else 0,
            processing_time=processing_time
        )

    def _get_provider_and_model(self, task: str = "semantic_search") -> tuple[str, str]:
        """Seleciona provedor e modelo baseado na tarefa"""
        
        # Mapear tarefas para modelos específicos
        task_mapping = {
            "semantic_search": ("openai", "text-embedding-3-large"),
            "document_similarity": ("openai", "text-embedding-3-large"),
            "clustering": ("openai", "text-embedding-3-large"),
            "quick_embeddings": ("openai", "text-embedding-3-small"),
            "classification": ("openai", "text-embedding-3-small"),
            "lightweight_search": ("openai", "text-embedding-3-small"),
            "multilingual_embeddings": ("google", "embedding-001"),
            "content_embeddings": ("google", "embedding-001")
        }
        
        if task in task_mapping:
            provider, model = task_mapping[task]
        else:
            # Usar provedor primário como fallback
            provider = self.primary_provider
            if provider == "openai":
                model = "text-embedding-3-large"
            elif provider == "google":
                model = "embedding-001"
            else:
                model = "text-embedding-3-large"
                provider = "openai"
        
        # Verificar se provedor está configurado
        if provider not in self.providers_config:
            logger.warning(f"Provedor {provider} não configurado, usando OpenAI")
            return "openai", "text-embedding-3-large"
            
        return provider, model

    def encode(self, 
               texts: Union[str, List[str]], 
               task: str = "semantic_search",
               normalize_embeddings: bool = True,
               show_progress_bar: bool = False) -> np.ndarray:
        """
        Gera embeddings para textos usando APIs externas.
        """
        # Converter para lista se necessário
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        # Selecionar provedor e modelo
        provider, model = self._get_provider_and_model(task)
        
        # Gerar embeddings
        try:
            if provider == "openai":
                response = self._call_openai_api(texts, model)
            elif provider == "google":
                response = self._call_google_api(texts, model)
            else:
                raise ValueError(f"Provedor {provider} não suportado")

            embeddings_array = np.array(response.embeddings, dtype=np.float32)
            
            # Normalizar se solicitado
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                embeddings_array = embeddings_array / np.maximum(norms, 1e-8)

            # Retornar embedding único se entrada foi texto único
            if single_text:
                return embeddings_array[0]
            
            return embeddings_array

        except Exception as e:
            logger.error(f"Erro ao gerar embeddings: {e}")
            raise

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """Calcula similaridade de cosseno entre embeddings"""
        # Garantir que são arrays 1D
        if embeddings1.ndim > 1:
            embeddings1 = embeddings1.flatten()
        if embeddings2.ndim > 1:
            embeddings2 = embeddings2.flatten()
            
        # Normalizar
        norm1 = np.linalg.norm(embeddings1)
        norm2 = np.linalg.norm(embeddings2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(embeddings1, embeddings2) / (norm1 * norm2))

    def get_sentence_embedding_dimension(self) -> int:
        """Retorna dimensão dos embeddings (compatibilidade)"""
        provider, model = self._get_provider_and_model()
        
        if provider == "openai":
            if "large" in model:
                return 3072
            else:
                return 1536
        elif provider == "google":
            return 768
        else:
            return 1536  # default

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso"""
        avg_processing_time = 0.0
        if self.stats["processing_time"]:
            avg_processing_time = sum(self.stats["processing_time"]) / len(self.stats["processing_time"])
            
        cache_hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / (self.stats["total_requests"] + self.stats["cache_hits"])

        return {
            **self.stats,
            "avg_processing_time": avg_processing_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache)
        }

    def clear_cache(self):
        """Limpa o cache de embeddings"""
        self.cache.clear()
        logger.info("Cache de embeddings limpo")

    def __del__(self):
        """Cleanup ao destruir objeto"""
        if hasattr(self, 'session'):
            self.session.close() 