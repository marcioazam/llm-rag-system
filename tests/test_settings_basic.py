import os
from pathlib import Path
import textwrap

import pytest

from src.settings import RAGSettings


class TestRAGSettings:
    def test_default_values(self):
        settings = RAGSettings()
        assert settings.environment == "development"
        assert settings.chunk_size == 1000
        # Verificar sub-config
        assert settings.neo4j.uri.startswith("bolt://")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "512")
        monkeypatch.setenv("ENVIRONMENT", "testing")
        settings = RAGSettings()
        assert settings.chunk_size == 512
        assert settings.environment == "testing"

    def test_yaml_loading(self, tmp_path: Path):
        yaml_content = textwrap.dedent(
            """
            chunk_size: 256
            llm:
              default_llm_model: "gpt-4o-mini"
            """
        )
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        settings = RAGSettings.from_yaml(yaml_file)
        assert settings.chunk_size == 256
        assert settings.llm.default_llm_model == "gpt-4o-mini" 