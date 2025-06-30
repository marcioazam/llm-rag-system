"""
🔗 Integração do Semantic Cache com Pipeline RAG
Conecta o cache semântico ao pipeline principal do sistema RAG
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Imports do sistema
try:
    from .semantic_cache import SemanticSimilarityCache, create_semantic_cache
    from .multi_layer_cache import MultiLayerCache
except ImportError:
    # Fallback para desenvolvimento
    SemanticSimilarityCache = None
    MultiLayerCache = None

logger = logging.getLogger(__name__)

@dataclass
class CacheResponse:
    """Resposta do sistema de cache integrado"""
    content: Optional[Dict[str, Any]]
    source: str  # "semantic", "traditional", "none"
    confidence: float
    metadata: Dict[str, Any]
    tokens_saved: int = 0
    cost_saved: float = 0.0


class IntegratedCacheSystem:
    """
    Sistema integrado de cache que combina:
    - Cache semântico (similaridade)
    - Cache tradicional (exato)
    - Predictive warming
    - Adaptive response generation
    """
    
    def __init__(self,
                 semantic_cache_config: Optional[Dict] = None,
                 traditional_cache_config: Optional[Dict] = None,
                 enable_semantic: bool = True,
                 enable_traditional: bool = True):
        
        self.enable_semantic = enable_semantic
        self.enable_traditional = enable_traditional
        
        # Inicializar caches
        self.semantic_cache = None
        self.traditional_cache = None
        
        if enable_semantic and SemanticSimilarityCache:
            try:
                self.semantic_cache = create_semantic_cache(semantic_cache_config)
                logger.info("✅ Cache semântico inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao inicializar cache semântico: {e}")
                self.enable_semantic = False
        
        if enable_traditional and MultiLayerCache:
            try:
                self.traditional_cache = MultiLayerCache(traditional_cache_config)
                logger.info("✅ Cache tradicional inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao inicializar cache tradicional: {e}")
                self.enable_traditional = False
        
        # Estatísticas integradas
        self.stats = {
            "total_requests": 0,
            "semantic_hits": 0,
            "traditional_hits": 0,
            "cache_misses": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0
        }
    
    async def get(self, query: str, context: Optional[Dict] = None) -> CacheResponse:
        """Busca integrada nos caches"""
        self.stats["total_requests"] += 1
        
        # Tentar cache semântico primeiro
        if self.enable_semantic and self.semantic_cache:
            try:
                result, similarity, metadata = await self.semantic_cache.get(query, context)
                if result is not None:
                    self.stats["semantic_hits"] += 1
                    return CacheResponse(
                        content=result,
                        source="semantic",
                        confidence=similarity,
                        metadata=metadata
                    )
            except Exception as e:
                logger.warning(f"Erro no cache semântico: {e}")
        
        # Fallback para cache tradicional
        if self.enable_traditional and self.traditional_cache:
            try:
                result, source, metadata = await self.traditional_cache.get(query)
                if result is not None:
                    self.stats["traditional_hits"] += 1
                    return CacheResponse(
                        content=result,
                        source=f"traditional_{source}",
                        confidence=1.0,
                        metadata=metadata
                    )
            except Exception as e:
                logger.warning(f"Erro no cache tradicional: {e}")
        
        # Cache miss
        self.stats["cache_misses"] += 1
        return CacheResponse(
            content=None,
            source="none",
            confidence=0.0,
            metadata={}
        )
    
    async def set(self, query: str, response: Dict[str, Any], **kwargs) -> bool:
        """Adiciona resposta aos caches"""
        success = False
        
        # Salvar no cache semântico
        if self.enable_semantic and self.semantic_cache:
            try:
                await self.semantic_cache.set(query, response, **kwargs)
                success = True
            except Exception as e:
                logger.warning(f"Erro ao salvar no cache semântico: {e}")
        
        # Salvar no cache tradicional
        if self.enable_traditional and self.traditional_cache:
            try:
                await self.traditional_cache.set(query, response)
                success = True
            except Exception as e:
                logger.warning(f"Erro ao salvar no cache tradicional: {e}")
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas integradas"""
        combined_stats = self.stats.copy()
        
        if self.semantic_cache:
            combined_stats["semantic_cache"] = self.semantic_cache.get_stats()
        
        if self.traditional_cache:
            combined_stats["traditional_cache"] = self.traditional_cache.get_stats()
        
        return combined_stats


def create_integrated_cache_system(config: Optional[Dict] = None) -> IntegratedCacheSystem:
    """Cria sistema integrado de cache"""
    default_config = {
        "enable_semantic": True,
        "enable_traditional": True
    }
    
    if config:
        default_config.update(config)
    
    return IntegratedCacheSystem(**default_config)


# Configurações padrão para diferentes cenários
CACHE_CONFIGS = {
    "development": {
        "enable_semantic": True,
        "enable_traditional": True,
        "semantic_threshold": 0.80,  # Mais permissivo para desenvolvimento
        "cache_priority": "semantic_first"
    },
    
    "production": {
        "enable_semantic": True,
        "enable_traditional": True,
        "semantic_threshold": 0.85,  # Mais restritivo para produção
        "cache_priority": "parallel"  # Máxima performance
    },
    
    "cost_optimized": {
        "enable_semantic": True,
        "enable_traditional": True,
        "semantic_threshold": 0.75,  # Favorece cache hits
        "cache_priority": "semantic_first"
    },
    
    "accuracy_optimized": {
        "enable_semantic": True,
        "enable_traditional": True,
        "semantic_threshold": 0.90,  # Apenas matches muito precisos
        "cache_priority": "traditional_first"
    }
}


# Exemplo de uso
async def demo_integrated_cache():
    """Demonstração do sistema integrado de cache"""
    cache_system = create_integrated_cache_system()
    
    # Simular algumas operações
    query1 = "Como funciona machine learning?"
    response1 = {
        "answer": "Machine learning é uma área da inteligência artificial...",
        "confidence": 0.9,
        "sources": ["fonte1", "fonte2"]
    }
    
    # Salvar resposta
    await cache_system.set(
        query=query1,
        response=response1,
        confidence=0.9,
        tokens_used=150,
        cost=0.01,
        quality_score=0.95,
        query_type="conceptual"
    )
    
    # Buscar query similar
    result = await cache_system.get("Como funciona ML?", query_type="conceptual")
    print(f"Resultado: {result.content}")
    print(f"Fonte: {result.source}")
    print(f"Confiança: {result.confidence}")
    
    # Estatísticas
    stats = cache_system.get_stats()
    print(f"Stats: {stats}")
    
    await cache_system.close()


if __name__ == "__main__":
    asyncio.run(demo_integrated_cache())
🔗 Integração do Semantic Cache com Pipeline RAG
Conecta o cache semântico ao pipeline principal do sistema RAG
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Imports do sistema
try:
    from .semantic_cache import SemanticSimilarityCache, create_semantic_cache
    from .multi_layer_cache import MultiLayerCache
except ImportError:
    # Fallback para desenvolvimento
    SemanticSimilarityCache = None
    MultiLayerCache = None

logger = logging.getLogger(__name__)

@dataclass
class CacheResponse:
    """Resposta do sistema de cache integrado"""
    content: Optional[Dict[str, Any]]
    source: str  # "semantic", "traditional", "none"
    confidence: float
    metadata: Dict[str, Any]
    tokens_saved: int = 0
    cost_saved: float = 0.0


class IntegratedCacheSystem:
    """
    Sistema integrado de cache que combina:
    - Cache semântico (similaridade)
    - Cache tradicional (exato)
    - Predictive warming
    - Adaptive response generation
    """
    
    def __init__(self,
                 semantic_cache_config: Optional[Dict] = None,
                 traditional_cache_config: Optional[Dict] = None,
                 enable_semantic: bool = True,
                 enable_traditional: bool = True):
        
        self.enable_semantic = enable_semantic
        self.enable_traditional = enable_traditional
        
        # Inicializar caches
        self.semantic_cache = None
        self.traditional_cache = None
        
        if enable_semantic and SemanticSimilarityCache:
            try:
                self.semantic_cache = create_semantic_cache(semantic_cache_config)
                logger.info("✅ Cache semântico inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao inicializar cache semântico: {e}")
                self.enable_semantic = False
        
        if enable_traditional and MultiLayerCache:
            try:
                self.traditional_cache = MultiLayerCache(traditional_cache_config)
                logger.info("✅ Cache tradicional inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao inicializar cache tradicional: {e}")
                self.enable_traditional = False
        
        # Estatísticas integradas
        self.stats = {
            "total_requests": 0,
            "semantic_hits": 0,
            "traditional_hits": 0,
            "cache_misses": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0
        }
    
    async def get(self, query: str, context: Optional[Dict] = None) -> CacheResponse:
        """Busca integrada nos caches"""
        self.stats["total_requests"] += 1
        
        # Tentar cache semântico primeiro
        if self.enable_semantic and self.semantic_cache:
            try:
                result, similarity, metadata = await self.semantic_cache.get(query, context)
                if result is not None:
                    self.stats["semantic_hits"] += 1
                    return CacheResponse(
                        content=result,
                        source="semantic",
                        confidence=similarity,
                        metadata=metadata
                    )
            except Exception as e:
                logger.warning(f"Erro no cache semântico: {e}")
        
        # Fallback para cache tradicional
        if self.enable_traditional and self.traditional_cache:
            try:
                result, source, metadata = await self.traditional_cache.get(query)
                if result is not None:
                    self.stats["traditional_hits"] += 1
                    return CacheResponse(
                        content=result,
                        source=f"traditional_{source}",
                        confidence=1.0,
                        metadata=metadata
                    )
            except Exception as e:
                logger.warning(f"Erro no cache tradicional: {e}")
        
        # Cache miss
        self.stats["cache_misses"] += 1
        return CacheResponse(
            content=None,
            source="none",
            confidence=0.0,
            metadata={}
        )
    
    async def set(self, query: str, response: Dict[str, Any], **kwargs) -> bool:
        """Adiciona resposta aos caches"""
        success = False
        
        # Salvar no cache semântico
        if self.enable_semantic and self.semantic_cache:
            try:
                await self.semantic_cache.set(query, response, **kwargs)
                success = True
            except Exception as e:
                logger.warning(f"Erro ao salvar no cache semântico: {e}")
        
        # Salvar no cache tradicional
        if self.enable_traditional and self.traditional_cache:
            try:
                await self.traditional_cache.set(query, response)
                success = True
            except Exception as e:
                logger.warning(f"Erro ao salvar no cache tradicional: {e}")
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas integradas"""
        combined_stats = self.stats.copy()
        
        if self.semantic_cache:
            combined_stats["semantic_cache"] = self.semantic_cache.get_stats()
        
        if self.traditional_cache:
            combined_stats["traditional_cache"] = self.traditional_cache.get_stats()
        
        return combined_stats


def create_integrated_cache_system(config: Optional[Dict] = None) -> IntegratedCacheSystem:
    """Cria sistema integrado de cache"""
    default_config = {
        "enable_semantic": True,
        "enable_traditional": True
    }
    
    if config:
        default_config.update(config)
    
    return IntegratedCacheSystem(**default_config)


# Configurações padrão para diferentes cenários
CACHE_CONFIGS = {
    "development": {
        "enable_semantic": True,
        "enable_traditional": True,
        "semantic_threshold": 0.80,  # Mais permissivo para desenvolvimento
        "cache_priority": "semantic_first"
    },
    
    "production": {
        "enable_semantic": True,
        "enable_traditional": True,
        "semantic_threshold": 0.85,  # Mais restritivo para produção
        "cache_priority": "parallel"  # Máxima performance
    },
    
    "cost_optimized": {
        "enable_semantic": True,
        "enable_traditional": True,
        "semantic_threshold": 0.75,  # Favorece cache hits
        "cache_priority": "semantic_first"
    },
    
    "accuracy_optimized": {
        "enable_semantic": True,
        "enable_traditional": True,
        "semantic_threshold": 0.90,  # Apenas matches muito precisos
        "cache_priority": "traditional_first"
    }
}


# Exemplo de uso
async def demo_integrated_cache():
    """Demonstração do sistema integrado de cache"""
    cache_system = create_integrated_cache_system()
    
    # Simular algumas operações
    query1 = "Como funciona machine learning?"
    response1 = {
        "answer": "Machine learning é uma área da inteligência artificial...",
        "confidence": 0.9,
        "sources": ["fonte1", "fonte2"]
    }
    
    # Salvar resposta
    await cache_system.set(
        query=query1,
        response=response1,
        confidence=0.9,
        tokens_used=150,
        cost=0.01,
        quality_score=0.95,
        query_type="conceptual"
    )
    
    # Buscar query similar
    result = await cache_system.get("Como funciona ML?", query_type="conceptual")
    print(f"Resultado: {result.content}")
    print(f"Fonte: {result.source}")
    print(f"Confiança: {result.confidence}")
    
    # Estatísticas
    stats = cache_system.get_stats()
    print(f"Stats: {stats}")
    
    await cache_system.close()


if __name__ == "__main__":
    asyncio.run(demo_integrated_cache())