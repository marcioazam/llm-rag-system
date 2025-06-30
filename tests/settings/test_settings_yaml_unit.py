import os, yaml, tempfile
from src.settings import RAGSettings


def test_ragsettings_from_yaml():
    data = {
        "environment": "testing",
        "chunk_size": 777,
        "neo4j": {
            "uri": "bolt://host:9999",
            "password": "secret"
        }
    }
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as f:
        yaml.dump(data, f)
        path = f.name
    cfg = RAGSettings.from_yaml(path)
    assert cfg.environment == "testing"
    assert cfg.chunk_size == 777
    assert cfg.neo4j.uri.endswith(":9999")


def test_ragsettings_env_override(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "123")
    settings = RAGSettings()
    assert settings.chunk_size == 123 