from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Neo4jSettings(BaseSettings):
    use_graph_store: bool = Field(True, env="USE_GRAPH_STORE")
    uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    user: str = Field("neo4j", env="NEO4J_USER")
    password: str = Field("arrozefeijao13", env="NEO4J_PASSWORD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class LLMSettings(BaseSettings):
    model: str = "llama3.1:8b-instruct-q4_K_M"
    code_model: str = "codellama:7b-instruct"
    base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    temperature: float = 0.7
    max_tokens: int = 2048


class RAGSettings(BaseSettings):
    """Configuração central do sistema RAG.

    Carrega valores automaticamente de variáveis de ambiente e, opcionalmente,
    de um arquivo YAML.
    """

    # caminhos
    data_dir: Path = Path("data")

    # chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # sub-configs
    neo4j: Neo4jSettings = Neo4jSettings()
    llm: LLMSettings = LLMSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # ------------------------------------------------------------------
    # YAML util
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RAGSettings":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Flatten nested keys if following same structure
        return cls(**data)

    # Convenience dump
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump() 