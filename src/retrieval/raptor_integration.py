"""
RAPTOR Integration - Adaptador para integrar RAPTOR Enhanced ao pipeline RAG

Este módulo serve como ponte entre:
1. O RAPTOR Enhanced (nova implementação)
2. O pipeline RAG existente (interface atual)
3. Configurações e cache do sistema

Funcionalidades:
- Interface compatível com RaptorRetriever atual
- Fallback automático entre versões
- Cache e métricas integradas
- Configuração unificada
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
from pathlib import Path

# Imports do sistema atual
from .raptor_retriever import (
    RaptorRetriever as OriginalRaptorRetriever,
    RaptorNode as OriginalRaptorNode,
    RaptorTreeStats,
    ClusteringStrategy,
    RetrievalStrategy,
    get_default_raptor_config
)

# Imports do RAPTOR Enhanced
try:
    from .raptor_enhanced import (
        EnhancedRaptorRetriever,
        RaptorConfig as EnhancedRaptorConfig,
        EmbeddingProvider,
        ClusteringMethod,
        SummarizationProvider,
        create_openai_config,
        create_default_config
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

# Imports offline como fallback
try:
    from .raptor_enhanced_offline import (
        OfflineRaptorRetriever,
        OfflineRaptorConfig,
        RaptorNode as OfflineRaptorNode
    )
    OFFLINE_AVAILABLE = True
except ImportError:
    OFFLINE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class IntegratedRaptorConfig:
    """Configuração unificada para RAPTOR Integration"""
    
    # Estratégia de implementação
    preferred_implementation: str = "enhanced"  # "enhanced", "offline", "original"
    auto_fallback: bool = True
    
    # Configurações de embedding
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: Optional[str] = None
    
    # Configurações de clustering
    clustering_method: str = "umap_gmm"  # "umap_gmm", "pca_gmm", "kmeans"
    max_levels: int = 4
    min_cluster_size: int = 2
    max_cluster_size: int = 50
    
    # Configurações de chunking
    chunk_size: int = 400
    chunk_overlap: int = 80
    
    # Configurações de summarização
    use_llm_summarization: bool = False
    summarization_provider: str = "openai"
    summarization_model: str = "gpt-4o-mini"
    
    # Configurações de performance
    batch_size: int = 32
    max_workers: int = 4
    use_cache: bool = True
    cache_ttl: int = 3600
    
    # Configurações de compatibilidade
    enable_metrics: bool = True
    enable_debugging: bool = False

class RaptorIntegrationAdapter:
    """
    Adaptador que integra diferentes implementações do RAPTOR
    mantendo compatibilidade com a interface existente
    """
    
    def __init__(self, config: Union[Dict[str, Any], IntegratedRaptorConfig]):
        self.config = self._normalize_config(config)
        self.implementation = None
        self.implementation_type = None
        self.cache = None
        
        # Métricas de integração
        self.metrics = {
            "implementation_used": None,
            "fallback_attempts": 0,
            "initialization_time": 0,
            "tree_construction_time": 0,
            "total_queries": 0,
            "total_results": 0,
            "error_count": 0
        }
        
        # Estado da árvore
        self.tree_built = False
        self.tree_stats = None
        
    def _normalize_config(self, config: Union[Dict[str, Any], IntegratedRaptorConfig]) -> IntegratedRaptorConfig:
        """Normaliza configuração para formato padrão"""
        
        if isinstance(config, IntegratedRaptorConfig):
            return config
        
        # Converter config dict para IntegratedRaptorConfig
        normalized = IntegratedRaptorConfig()
        
        # Mapear configurações existentes
        if "embedding_model" in config:
            normalized.embedding_model = config["embedding_model"]
        
        if "clustering_strategy" in config:
            # Mapear estratégias antigas para novas
            strategy_map = {
                "global_local": "umap_gmm",
                "single_level": "kmeans",
                "adaptive": "pca_gmm"
            }
            normalized.clustering_method = strategy_map.get(config["clustering_strategy"], "umap_gmm")
        
        if "max_levels" in config:
            normalized.max_levels = config["max_levels"]
        
        if "chunk_size" in config:
            normalized.chunk_size = config["chunk_size"]
        
        if "chunk_overlap" in config:
            normalized.chunk_overlap = config["chunk_overlap"]
        
        # Detectar se deve usar LLM summarization
        if config.get("api_provider") == "openai" or config.get("model_name"):
            normalized.use_llm_summarization = True
            normalized.summarization_provider = config.get("api_provider", "openai")
            normalized.summarization_model = config.get("model_name", "gpt-4o-mini")
        
        return normalized
    
    async def initialize(self) -> bool:
        """Inicializa a melhor implementação disponível"""
        
        start_time = time.time()
        
        try:
            # Tentar implementação preferida
            if self.config.preferred_implementation == "enhanced" and ENHANCED_AVAILABLE:
                success = await self._initialize_enhanced()
                if success:
                    self.implementation_type = "enhanced"
                    logger.info("✅ RAPTOR Enhanced inicializado")
                    return True
            
            elif self.config.preferred_implementation == "offline" and OFFLINE_AVAILABLE:
                success = await self._initialize_offline()
                if success:
                    self.implementation_type = "offline"
                    logger.info("✅ RAPTOR Offline inicializado")
                    return True
            
            elif self.config.preferred_implementation == "original":
                success = await self._initialize_original()
                if success:
                    self.implementation_type = "original"
                    logger.info("✅ RAPTOR Original inicializado")
                    return True
            
            # Fallback automático se habilitado
            if self.config.auto_fallback:
                logger.info("Tentando fallback automático...")
                return await self._try_fallback_implementations()
            
            return False
            
        except Exception as e:
            logger.error(f"Erro na inicialização: {e}")
            if self.config.auto_fallback:
                return await self._try_fallback_implementations()
            return False
        
        finally:
            self.metrics["initialization_time"] = time.time() - start_time
            self.metrics["implementation_used"] = self.implementation_type
    
    async def _initialize_enhanced(self) -> bool:
        """Inicializa RAPTOR Enhanced"""
        
        try:
            # Configurar Enhanced
            if self.config.openai_api_key and self.config.use_llm_summarization:
                enhanced_config = EnhancedRaptorConfig(
                    embedding_provider=EmbeddingProvider.OPENAI if self.config.openai_api_key else EmbeddingProvider.SENTENCE_TRANSFORMERS,
                    embedding_model=self.config.embedding_model,
                    clustering_method=ClusteringMethod.UMAP_GMM if self.config.clustering_method == "umap_gmm" else ClusteringMethod.PCA_GMM,
                    summarization_provider=SummarizationProvider.OPENAI,
                    openai_api_key=self.config.openai_api_key,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    max_levels=self.config.max_levels,
                    batch_size=self.config.batch_size,
                    use_cache=self.config.use_cache
                )
            else:
                enhanced_config = create_default_config()
                enhanced_config.chunk_size = self.config.chunk_size
                enhanced_config.max_levels = self.config.max_levels
            
            self.implementation = EnhancedRaptorRetriever(enhanced_config)
            return True
            
        except Exception as e:
            logger.warning(f"Falha ao inicializar Enhanced: {e}")
            self.metrics["fallback_attempts"] += 1
            return False
    
    async def _initialize_offline(self) -> bool:
        """Inicializa RAPTOR Offline"""
        
        try:
            offline_config = OfflineRaptorConfig(
                embedding_model=self.config.embedding_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                max_levels=self.config.max_levels,
                batch_size=self.config.batch_size
            )
            
            self.implementation = OfflineRaptorRetriever(offline_config)
            return True
            
        except Exception as e:
            logger.warning(f"Falha ao inicializar Offline: {e}")
            self.metrics["fallback_attempts"] += 1
            return False
    
    async def _initialize_original(self) -> bool:
        """Inicializa RAPTOR Original"""
        
        try:
            self.implementation = OriginalRaptorRetriever(
                embedding_model=self.config.embedding_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                max_levels=self.config.max_levels,
                min_cluster_size=self.config.min_cluster_size,
                max_cluster_size=self.config.max_cluster_size
            )
            return True
            
        except Exception as e:
            logger.warning(f"Falha ao inicializar Original: {e}")
            self.metrics["fallback_attempts"] += 1
            return False
    
    async def _try_fallback_implementations(self) -> bool:
        """Tenta implementações em ordem de prioridade"""
        
        fallback_order = ["enhanced", "offline", "original"]
        
        for impl_type in fallback_order:
            if impl_type == self.config.preferred_implementation:
                continue  # Já tentou
            
            logger.info(f"Tentando fallback para {impl_type}...")
            
            if impl_type == "enhanced" and ENHANCED_AVAILABLE:
                if await self._initialize_enhanced():
                    self.implementation_type = "enhanced"
                    return True
            
            elif impl_type == "offline" and OFFLINE_AVAILABLE:
                if await self._initialize_offline():
                    self.implementation_type = "offline"
                    return True
            
            elif impl_type == "original":
                if await self._initialize_original():
                    self.implementation_type = "original"
                    return True
        
        logger.error("❌ Todas as implementações RAPTOR falharam")
        return False
    
    async def build_tree(self, documents: List[str]) -> Dict[str, Any]:
        """Constrói árvore RAPTOR usando implementação disponível"""
        
        if not self.implementation:
            if not await self.initialize():
                return {"error": "Nenhuma implementação RAPTOR disponível"}
        
        start_time = time.time()
        
        try:
            logger.info(f"Construindo árvore RAPTOR ({self.implementation_type}) com {len(documents)} documentos")
            
            # Chamar método apropriado baseado na implementação
            if self.implementation_type == "enhanced":
                stats = await self.implementation.build_tree(documents)
                self.tree_stats = self._normalize_stats(stats, "enhanced")
            
            elif self.implementation_type == "offline":
                stats = self.implementation.build_tree(documents)
                self.tree_stats = self._normalize_stats(stats, "offline")
            
            elif self.implementation_type == "original":
                stats = await self.implementation.build_tree(documents)
                self.tree_stats = self._normalize_stats(stats, "original")
            
            self.tree_built = True
            construction_time = time.time() - start_time
            self.metrics["tree_construction_time"] = construction_time
            
            logger.info(f"✅ Árvore construída em {construction_time:.2f}s")
            
            return {
                "success": True,
                "implementation": self.implementation_type,
                "stats": self.tree_stats,
                "construction_time": construction_time
            }
            
        except Exception as e:
            logger.error(f"Erro na construção da árvore: {e}")
            self.metrics["error_count"] += 1
            return {"error": str(e), "implementation": self.implementation_type}
    
    def _normalize_stats(self, stats: Any, impl_type: str) -> Dict[str, Any]:
        """Normaliza estatísticas para formato padrão"""
        
        if impl_type == "enhanced":
            return stats
        
        elif impl_type == "offline":
            return {
                "total_nodes": stats.get("total_nodes", 0),
                "max_level": stats.get("max_level", 0),
                "nodes_per_level": stats.get("nodes_per_level", {}),
                "construction_time": stats.get("construction_time", 0),
                "embedding_provider": stats.get("embedding_provider", "unknown"),
                "clustering_method": stats.get("clustering_method", "unknown")
            }
        
        elif impl_type == "original":
            if hasattr(stats, 'to_dict'):
                return stats.to_dict()
            return vars(stats) if stats else {}
        
        return stats
    
    async def search(self, 
                    query: str, 
                    k: int = 10,
                    max_tokens: int = 2000,
                    **kwargs) -> List[Dict[str, Any]]:
        """Busca na árvore RAPTOR"""
        
        if not self.implementation or not self.tree_built:
            logger.warning("RAPTOR não inicializado ou árvore não construída")
            return []
        
        try:
            start_time = time.time()
            
            # Chamar busca baseado na implementação
            if self.implementation_type in ["enhanced", "offline"]:
                results = await self.implementation.search(query, k)
            else:  # original
                results = self.implementation.search(query, k, max_tokens)
            
            search_time = time.time() - start_time
            
            # Normalizar resultados
            normalized_results = []
            for result in results:
                normalized_result = {
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "source": "raptor",
                    "implementation": self.implementation_type,
                    "metadata": {
                        **result.get("metadata", {}),
                        "search_time": search_time,
                        "raptor_implementation": self.implementation_type
                    }
                }
                normalized_results.append(normalized_result)
            
            # Atualizar métricas
            self.metrics["total_queries"] += 1
            self.metrics["total_results"] += len(normalized_results)
            
            logger.debug(f"RAPTOR search: {len(normalized_results)} resultados em {search_time:.3f}s")
            
            return normalized_results
            
        except Exception as e:
            logger.error(f"Erro na busca RAPTOR: {e}")
            self.metrics["error_count"] += 1
            return []
    
    def get_tree_summary(self) -> Dict[str, Any]:
        """Retorna resumo da árvore construída"""
        
        if not self.tree_built or not self.tree_stats:
            return {"status": "not_built"}
        
        summary = {
            "implementation": self.implementation_type,
            "tree_built": self.tree_built,
            "stats": self.tree_stats,
            "metrics": self.metrics
        }
        
        # Informações específicas da implementação
        if hasattr(self.implementation, 'get_tree_summary'):
            impl_summary = self.implementation.get_tree_summary()
            summary["implementation_details"] = impl_summary
        
        return summary
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de integração"""
        
        return {
            **self.metrics,
            "config": {
                "preferred_implementation": self.config.preferred_implementation,
                "auto_fallback": self.config.auto_fallback,
                "embedding_provider": self.config.embedding_provider,
                "clustering_method": self.config.clustering_method,
                "use_llm_summarization": self.config.use_llm_summarization
            },
            "availability": {
                "enhanced": ENHANCED_AVAILABLE,
                "offline": OFFLINE_AVAILABLE,
                "original": True
            }
        }

# Factory functions para compatibilidade

async def create_integrated_raptor_retriever(config: Dict[str, Any]) -> RaptorIntegrationAdapter:
    """Cria retriever RAPTOR integrado com fallback automático"""
    
    adapter = RaptorIntegrationAdapter(config)
    
    if await adapter.initialize():
        return adapter
    else:
        raise RuntimeError("Falha ao inicializar qualquer implementação RAPTOR")

def get_integrated_raptor_config() -> Dict[str, Any]:
    """Retorna configuração padrão para RAPTOR integrado"""
    
    return {
        "preferred_implementation": "enhanced",
        "auto_fallback": True,
        "embedding_provider": "sentence_transformers",
        "embedding_model": "all-MiniLM-L6-v2",
        "clustering_method": "umap_gmm",
        "chunk_size": 400,
        "chunk_overlap": 80,
        "max_levels": 4,
        "use_llm_summarization": False,
        "batch_size": 32,
        "use_cache": True,
        "enable_metrics": True
    }

def detect_optimal_raptor_config() -> Dict[str, Any]:
    """Detecta melhor configuração baseada no ambiente"""
    
    config = get_integrated_raptor_config()
    
    # Detectar API keys disponíveis
    import os
    
    if os.getenv("OPENAI_API_KEY"):
        config.update({
            "preferred_implementation": "enhanced",
            "embedding_provider": "openai",
            "use_llm_summarization": True,
            "summarization_provider": "openai"
        })
        logger.info("🔑 OpenAI API key detectada - usando configuração Enhanced")
    
    elif ENHANCED_AVAILABLE:
        config.update({
            "preferred_implementation": "enhanced",
            "embedding_provider": "sentence_transformers"
        })
        logger.info("🤖 Sentence-Transformers disponível - usando Enhanced offline")
    
    elif OFFLINE_AVAILABLE:
        config.update({
            "preferred_implementation": "offline"
        })
        logger.info("💻 Usando implementação offline")
    
    else:
        config.update({
            "preferred_implementation": "original"
        })
        logger.info("📦 Usando implementação original")
    
    return config