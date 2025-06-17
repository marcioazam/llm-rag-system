from __future__ import annotations

"""Dependência FastAPI que devolve instância singleton do RAGPipeline."""

from functools import lru_cache

from src.rag_pipeline import RAGPipeline
from src.settings import RAGSettings


@lru_cache(maxsize=1)
def _create_pipeline() -> RAGPipeline:
    settings = RAGSettings()
    return RAGPipeline(config_path=None, settings=settings.model_dump())


def get_pipeline() -> RAGPipeline:  # FastAPI Depends não precisa ser assíncrono
    return _create_pipeline() 