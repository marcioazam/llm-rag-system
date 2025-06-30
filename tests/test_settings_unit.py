import os
import importlib
from pathlib import Path

import pytest

from src import settings as rag_settings_module


def _clear_env(monkeypatch):
    keys = [
        "ENVIRONMENT",
        "CHUNK_SIZE",
        "DEFAULT_K",
        "OPENAI_API_KEY",
    ]
    for k in keys:
        monkeypatch.delenv(k, raising=False)


def test_ragsettings_defaults(monkeypatch):
    _clear_env(monkeypatch)
    importlib.reload(rag_settings_module)
    s = rag_settings_module.RAGSettings()
    assert s.environment == "development"
    assert s.chunk_size == 1000
    assert s.llm.openai_api_key == ""


def test_ragsettings_env_overrides(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("CHUNK_SIZE", "123")
    monkeypatch.setenv("DEFAULT_K", "9")
    importlib.reload(rag_settings_module)

    s = rag_settings_module.RAGSettings()
    assert s.environment == "testing"
    assert s.chunk_size == 123
    assert s.default_k == 9


def test_ragsettings_from_yaml(tmp_path: Path):
    yaml_content = """
    environment: production
    chunk_size: 777
    default_k: 11
    """
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)

    s = rag_settings_module.RAGSettings.from_yaml(yaml_file)
    assert s.environment == "production"
    assert s.chunk_size == 777
    assert s.default_k == 11 