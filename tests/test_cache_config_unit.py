import os
import importlib

import pytest

from src.config import cache_config as cc


def _clear_env(monkeypatch):
    # Remove env vars used in module
    for var in [
        "ENVIRONMENT",
        "CACHE_DB_PATH",
        "CACHE_ENABLE_REDIS",
        "CACHE_MAX_MEMORY_ENTRIES",
        "REDIS_URL",
        "REDIS_HOST",
        "REDIS_PORT",
        "REDIS_PASSWORD",
        "REDIS_DB",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_get_cache_config_default_dev(monkeypatch):
    """Configuração padrão deve ser development."""
    _clear_env(monkeypatch)
    # Recarregar módulo para garantir envs limpos
    importlib.reload(cc)

    config = cc.get_cache_config()
    assert config["db_path"] == "storage/dev_rag_cache.db"
    assert config["enable_redis"] is False
    assert config["max_memory_entries"] == 200


def test_get_cache_config_testing(monkeypatch):
    """Ambiente 'testing' deve usar :memory: e menos entradas."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "testing")
    importlib.reload(cc)

    config = cc.get_cache_config()
    assert config["db_path"] == ":memory:"
    assert config["max_memory_entries"] == 50
    assert config["enable_redis"] is False


def test_get_cache_config_prod_invalid_redis(monkeypatch):
    """Se Redis habilitado sem URL deve lançar ValueError."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CACHE_ENABLE_REDIS", "true")
    # Garantir REDIS_URL não setado
    monkeypatch.delenv("REDIS_URL", raising=False)
    importlib.reload(cc)

    with pytest.raises(ValueError):
        cc.get_cache_config() 