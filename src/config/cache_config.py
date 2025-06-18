"""
Configuração do Sistema de Cache RAG
Lê variáveis de ambiente e fornece configurações por ambiente
"""

import os
from typing import Dict, Any
from pathlib import Path

def get_cache_config() -> Dict[str, Any]:
    """
    Retorna configuração do cache baseada em variáveis de ambiente
    """
    
    # Determinar ambiente
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    # Configuração base
    base_config = {
        "db_path": os.getenv("CACHE_DB_PATH", "storage/rag_cache.db"),
        "enable_redis": _get_bool_env("CACHE_ENABLE_REDIS", False),
        "redis_url": _get_redis_url(),
        "max_memory_entries": _get_memory_entries_by_env(environment),
    }
    
    # Configurações específicas por ambiente
    env_configs = {
        "development": {
            **base_config,
            "db_path": "storage/dev_rag_cache.db",
            "max_memory_entries": 200,
            "enable_redis": False,  # Forçar desabilitado em dev
        },
        
        "testing": {
            **base_config,
            "db_path": ":memory:",  # SQLite em memória para testes
            "max_memory_entries": 50,
            "enable_redis": False,
        },
        
        "production": {
            **base_config,
            "db_path": os.getenv("CACHE_DB_PATH", "storage/prod_rag_cache.db"),
            "max_memory_entries": _get_memory_entries_by_env("production"),
            "enable_redis": _get_bool_env("CACHE_ENABLE_REDIS", False),
        }
    }
    
    config = env_configs.get(environment, env_configs["development"])
    
    # Validar configuração do Redis se habilitado
    if config["enable_redis"]:
        _validate_redis_config(config)
    
    return config


def _get_redis_url() -> str:
    """
    Constrói URL do Redis a partir das variáveis de ambiente
    """
    redis_url = os.getenv("REDIS_URL")
    
    if redis_url:
        # Se REDIS_URL está definida, usar diretamente
        return redis_url
    
    # Construir URL a partir de componentes
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    password = os.getenv("REDIS_PASSWORD", "")
    db = os.getenv("REDIS_DB", "0")
    
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    else:
        return f"redis://{host}:{port}/{db}"


def _get_memory_entries_by_env(environment: str) -> int:
    """
    Retorna número de entradas em memória baseado no ambiente
    """
    # Valor padrão baseado no ambiente
    defaults = {
        "development": 200,
        "testing": 50,
        "production": 2000,
    }
    
    # Permitir override via env var
    env_value = os.getenv("CACHE_MAX_MEMORY_ENTRIES")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    
    return defaults.get(environment, defaults["development"])


def _get_bool_env(key: str, default: bool = False) -> bool:
    """
    Converte variável de ambiente para boolean
    """
    value = os.getenv(key, "").lower()
    
    if value in ("true", "1", "yes", "on", "enabled"):
        return True
    elif value in ("false", "0", "no", "off", "disabled"):
        return False
    else:
        return default


def _validate_redis_config(config: Dict[str, Any]) -> None:
    """
    Valida configuração do Redis
    """
    redis_url = config.get("redis_url")
    
    if not redis_url:
        raise ValueError(
            "Redis habilitado mas REDIS_URL não configurada. "
            "Configure REDIS_URL no arquivo .env"
        )
    
    # Verificar se URL tem formato válido
    if not redis_url.startswith(("redis://", "rediss://")):
        raise ValueError(
            f"REDIS_URL inválida: {redis_url}. "
            "Deve começar com redis:// ou rediss://"
        )


def get_redis_settings() -> Dict[str, Any]:
    """
    Retorna configurações específicas do Redis
    """
    return {
        "url": _get_redis_url(),
        "password": os.getenv("REDIS_PASSWORD", ""),
        "db": int(os.getenv("REDIS_DB", "0")),
        "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
        "socket_timeout": float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0")),
        "socket_connect_timeout": float(os.getenv("REDIS_CONNECT_TIMEOUT", "5.0")),
        "retry_on_timeout": _get_bool_env("REDIS_RETRY_ON_TIMEOUT", True),
        "health_check_interval": int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
    }


def print_cache_config() -> None:
    """
    Imprime configuração atual do cache para debugging
    """
    config = get_cache_config()
    
    print("🎯 CONFIGURAÇÃO DO CACHE RAG")
    print("=" * 40)
    print(f"Ambiente: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"DB Path: {config['db_path']}")
    print(f"Max Memory Entries: {config['max_memory_entries']}")
    print(f"Redis Enabled: {config['enable_redis']}")
    
    if config['enable_redis']:
        redis_settings = get_redis_settings()
        print(f"Redis URL: {redis_settings['url']}")
        print(f"Redis DB: {redis_settings['db']}")
        print(f"Max Connections: {redis_settings['max_connections']}")
    
    print("=" * 40)


if __name__ == "__main__":
    # Teste da configuração
    try:
        print_cache_config()
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        import traceback
        traceback.print_exc() 