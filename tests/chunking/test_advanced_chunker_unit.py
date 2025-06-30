import math
import types
import numpy as np

import pytest

import src.chunking.advanced_chunker as ac


# ------------------------------------------------------------------
# Preparar Dummy EnhancedSemanticChunker para evitar carregar modelos pesados
# ------------------------------------------------------------------
class _DummyEnhancedSemanticChunker:  # noqa: D401
    """Stub leve que ignora processamento semântico pesado."""

    def __init__(self, *args, **kwargs):
        pass

    def chunk(self, text: str, metadata=None):  # type: ignore[override]
        return []


ac.EnhancedSemanticChunker = _DummyEnhancedSemanticChunker  # type: ignore[attr-defined]


# ------------------------------------------------------------------
# Dummy EmbeddingService
# ------------------------------------------------------------------
class _DummyEmbedService:
    def embed_texts(self, texts, show_progress=False):  # noqa: D401
        # Retorna vetores unitários para evitar divisão por zero
        return [np.array([1.0, 0.0, 0.0]) for _ in texts]


@pytest.fixture(scope="module")
def chunker():
    embedding_service = _DummyEmbedService()
    return ac.AdvancedChunker(embedding_service=embedding_service, max_chunk_size=50, chunk_overlap=10)


def test_structural_chunk(chunker):
    doc = {
        "content": "Parágrafo 1.\n\nParágrafo 2. Mais texto.",
        "metadata": {},
    }
    chunks = chunker.structural_chunk(doc)
    assert len(chunks) >= 1
    assert all(c["metadata"]["chunk_method"] == "structural" for c in chunks)


def test_sliding_window_chunk(chunker):
    text = "a" * 900  # 900 caracteres
    doc = {"content": text, "metadata": {}}
    chunks = chunker.sliding_window_chunk(doc, window=400, stride=200)
    # Espera-se ceil(900/200) = 5 janelas
    assert len(chunks) == math.ceil(900 / 200)
    assert chunks[0]["metadata"]["chunk_method"] == "sliding_window"


def test_semantic_chunk_simple(chunker):
    doc = {"content": "Primeira frase. Segunda frase. Terceira frase.", "metadata": {}}
    chunks = chunker.semantic_chunk(doc)
    # Todas frases devem se agrupar em único chunk dado threshold e max_chunk_size baixo
    assert len(chunks) >= 1
    assert chunks[0]["metadata"]["chunk_method"] in {"semantic", "enhanced_semantic"}

ac.cosine_similarity = lambda a, b: np.array([[1.0]]) 