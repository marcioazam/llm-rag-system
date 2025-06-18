from __future__ import annotations

"""Dependência FastAPI que devolve instância singleton do AdvancedRAGPipeline (100% API)."""

from functools import lru_cache

from src.rag_pipeline_advanced import AdvancedRAGPipeline
from src.settings import RAGSettings


@lru_cache(maxsize=1)
def _create_pipeline() -> AdvancedRAGPipeline:
    """
    Cria AdvancedRAGPipeline com sistema 100% baseado em APIs externas.
    - OpenAI, Anthropic, Google, DeepSeek
    - Zero modelos locais (Ollama, sentence-transformers)
    - 5 melhorias avançadas de RAG
    """
    settings = RAGSettings()
    return AdvancedRAGPipeline(config_path="config/llm_providers_config.yaml")


def get_pipeline() -> AdvancedRAGPipeline:
    """Retorna instância singleton do AdvancedRAGPipeline"""
    return _create_pipeline() 