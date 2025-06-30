import os
import re
from src.config import cache_config as cc

# ---------------------------------------------------------------------------
# Helper para limpar variáveis de ambiente relevantes -----------------------
# ---------------------------------------------------------------------------

def _clear_env():
    for key in list(os.environ.keys()):
        if key.startswith("CACHE_") or key.startswith("REDIS_") or key == "ENVIRONMENT":
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# _get_bool_env -------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_get_bool_env_variations():
    _clear_env()
    # Default False quando não definido
    assert cc._get_bool_env("CACHE_TEST_BOOL") is False
    # Truthy values
    for val in ["true", "1", "yes", "on", "enabled", "TRUE"]:
        os.environ["CACHE_TEST_BOOL"] = val
        assert cc._get_bool_env("CACHE_TEST_BOOL") is True
    # Falsy values
    for val in ["false", "0", "no", "off", "disabled", "FALSE"]:
        os.environ["CACHE_TEST_BOOL"] = val
        assert cc._get_bool_env("CACHE_TEST_BOOL", True) is False
    _clear_env()


# ---------------------------------------------------------------------------
# _get_redis_url ------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_get_redis_url_composition():
    _clear_env()
    # Default composition without password
    url = cc._get_redis_url()
    assert url.startswith("redis://localhost:6379/")
    # Custom host/port and password
    os.environ.update({
        "REDIS_HOST": "redis.host",
        "REDIS_PORT": "6380",
        "REDIS_PASSWORD": "secret",
        "REDIS_DB": "2",
    })
    url2 = cc._get_redis_url()
    assert url2 == "redis://:secret@redis.host:6380/2"
    # Direct REDIS_URL overrides
    os.environ["REDIS_URL"] = "redis://custom:9999/0"
    assert cc._get_redis_url() == "redis://custom:9999/0"
    _clear_env()


# ---------------------------------------------------------------------------
# _get_memory_entries_by_env -------------------------------------------------
# ---------------------------------------------------------------------------

def test_get_memory_entries_override_and_defaults():
    _clear_env()
    # Default dev
    assert cc._get_memory_entries_by_env("development") == 200
    # Override via env
    os.environ["CACHE_MAX_MEMORY_ENTRIES"] = "777"
    assert cc._get_memory_entries_by_env("development") == 777
    # Invalid override falls back
    os.environ["CACHE_MAX_MEMORY_ENTRIES"] = "not-int"
    assert cc._get_memory_entries_by_env("production") == 2000
    _clear_env()


# ---------------------------------------------------------------------------
# get_redis_settings --------------------------------------------------------
# ---------------------------------------------------------------------------

def test_get_redis_settings_types():
    _clear_env()
    settings = cc.get_redis_settings()
    assert isinstance(settings["url"], str)
    assert isinstance(settings["db"], int)
    assert isinstance(settings["max_connections"], int)
    # retry_on_timeout default True
    assert isinstance(settings["retry_on_timeout"], bool)


# ---------------------------------------------------------------------------
# print_cache_config --------------------------------------------------------
# ---------------------------------------------------------------------------

def test_print_cache_config_output(capsys, monkeypatch):
    _clear_env()
    monkeypatch.setenv("ENVIRONMENT", "testing")
    # Ensure function prints and not raise exceptions
    cc.print_cache_config()
    captured = capsys.readouterr().out
    assert "CONFIGURAÇÃO DO CACHE" in captured or "CONFIGURAÇÃO DO CACHE RAG" in captured
    assert "testing" in captured.lower()

    _clear_env() 