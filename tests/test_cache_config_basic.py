import os
from contextlib import contextmanager
from src.config.cache_config import get_cache_config, _get_bool_env

@contextmanager
def temp_env(**env):
    old = {k: os.getenv(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def test_testing_environment_defaults():
    """ENVIRONMENT=testing deve usar DB em memória e Redis desabilitado."""
    with temp_env(ENVIRONMENT="testing"):
        cfg = get_cache_config()
        assert cfg["db_path"] == ":memory:"
        assert cfg["enable_redis"] is False
        assert cfg["max_memory_entries"] == 50

def test_development_overrides():
    """Override via variáveis de ambiente específicas."""
    with temp_env(ENVIRONMENT="development", CACHE_MAX_MEMORY_ENTRIES="123"):
        cfg = get_cache_config()
        assert cfg["max_memory_entries"] == 123


def test_bool_env_helper():
    """_get_bool_env interpreta strings corretamente."""
    os.environ["TEMP_BOOL"] = "True"
    assert _get_bool_env("TEMP_BOOL", False) is True
    os.environ["TEMP_BOOL"] = "off"
    assert _get_bool_env("TEMP_BOOL", True) is False
    os.environ.pop("TEMP_BOOL")
    # default fallback
    assert _get_bool_env("TEMP_BOOL", True) is True 