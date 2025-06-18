from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()


class Neo4jSettings(BaseSettings):
    use_graph_store: bool = Field(True, env="USE_GRAPH_STORE")
    uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    user: str = Field("neo4j", env="NEO4J_USER")
    password: str = Field("", env="NEO4J_PASSWORD")  # Default empty for tests

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Permitir campos extras


class LLMSettings(BaseSettings):
    # API Keys
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    google_ai_api_key: str = Field("", env="GOOGLE_AI_API_KEY")
    
    # Model settings
    model: str = "llama3.1:8b-instruct-q4_K_M"
    code_model: str = "codellama:7b-instruct"
    base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    temperature: float = Field(0.7, env="MODEL_TEMPERATURE")
    max_tokens: int = Field(2048, env="MAX_TOKENS")
    
    # Default models
    default_llm_model: str = Field("gpt-3.5-turbo", env="DEFAULT_LLM_MODEL")
    default_embedding_model: str = Field("text-embedding-ada-002", env="DEFAULT_EMBEDDING_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Permitir campos extras


class RAGSettings(BaseSettings):
    """Configuração central do sistema RAG.

    Carrega valores automaticamente de variáveis de ambiente e, opcionalmente,
    de um arquivo YAML.
    """

    # Configurações gerais da aplicação
    environment: str = Field("development", env="ENVIRONMENT")
    secret_key: str = Field("", env="SECRET_KEY")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # caminhos
    data_dir: Path = Path("data")

    # chunking
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # retrieval
    default_k: int = Field(5, env="DEFAULT_K")
    max_k: int = Field(20, env="MAX_K")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")

    # sub-configs
    neo4j: Neo4jSettings = Neo4jSettings()
    llm: LLMSettings = LLMSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Permitir campos extras

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