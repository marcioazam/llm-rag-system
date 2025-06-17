from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class RAGComponentType(str, Enum):
    """Tipos de componentes de um sistema RAG."""

    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    INDEXING = "indexing"
    RETRIEVAL = "retrieval"
    AUGMENTATION = "augmentation"
    GENERATION = "generation"
    POST_PROCESSING = "post_processing"
    FEEDBACK = "feedback"


@dataclass
class RAGArchitecture:
    """Descrição de alto nível dos componentes de uma arquitetura RAG."""

    # Pipeline de Ingestão
    document_loaders: List[str]
    preprocessors: List[str]
    chunkers: List[str]

    # Sistema de Embeddings
    embedding_models: List[str]
    vector_databases: List[str]

    # Retrieval Avançado
    retrieval_strategies: List[str]
    rerankers: List[str]

    # Geração e Otimização
    llm_models: List[str]
    prompt_optimizers: List[str]

    # Monitoramento
    metrics_collectors: List[str]
    feedback_loops: List[str]


# ------------------------------------------------------------------
# Registry simples em memória para armazenar a arquitetura ativa
# ------------------------------------------------------------------

_GLOBAL_ARCH_INFO: RAGArchitecture | None = None


def register_architecture(info: RAGArchitecture) -> None:
    """Registra/atualiza a descrição da arquitetura global em execução."""
    global _GLOBAL_ARCH_INFO
    _GLOBAL_ARCH_INFO = info


def get_architecture() -> RAGArchitecture | None:
    """Retorna a arquitetura atualmente registrada (ou None)."""
    return _GLOBAL_ARCH_INFO 