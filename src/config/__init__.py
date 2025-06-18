"""
Módulo de configuração do sistema RAG
"""

from .cache_config import get_cache_config, get_redis_settings, print_cache_config

__all__ = [
    "get_cache_config",
    "get_redis_settings", 
    "print_cache_config"
] 