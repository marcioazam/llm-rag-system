import os
import importlib
import pytest

from src.config.cache_config import get_cache_config


def _clear_env(keys):
    for k in keys:
        os.environ.pop(k, None)


class TestCacheConfig:
    def test_development_defaults(self, monkeypatch):
        _clear_env([
            "ENVIRONMENT",
            "CACHE_ENABLE_REDIS",
            "CACHE_DB_PATH",
            "CACHE_MAX_MEMORY_ENTRIES",
            "REDIS_URL",
        ])
        config = get_cache_config()
        assert config["db_path"] == "storage/dev_rag_cache.db"
        assert config["enable_redis"] is False
        assert config["max_memory_entries"] == 200

    def test_testing_environment(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "testing")
        config = get_cache_config()
        assert config["db_path"] == ":memory:"
        assert config["enable_redis"] is False
        assert config["max_memory_entries"] == 50

    def test_production_with_redis(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("CACHE_ENABLE_REDIS", "true")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        config = get_cache_config()
        assert config["enable_redis"] is True
        assert config["redis_url"].startswith("redis://")

    def test_redis_validation_error(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("CACHE_ENABLE_REDIS", "1")
        # Definir URL inválida para disparar erro de validação
        monkeypatch.setenv("REDIS_URL", "http://invalid-url")
        with pytest.raises(ValueError):
            get_cache_config()

    def teardown_method(self):
        # Limpar variáveis para não afetar outros testes
        for key in list(os.environ.keys()):
            if key.startswith("CACHE_") or key.startswith("REDIS_") or key == "ENVIRONMENT":
                os.environ.pop(key, None) 