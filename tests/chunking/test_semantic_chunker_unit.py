import types

import pytest

# Importar módulo e em seguida aplicar stubs antes de instanciar
import src.chunking.semantic_chunker as sc


# ------------------------------------------------------------------
# Stubs para dependências pesadas
# ------------------------------------------------------------------
class _DummySentenceTransformer:  # noqa: D401
    """Stub que devolve vetores unitários simples."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, sentences):  # noqa: D401
        # Retorna vetor [1, 0, 0] para cada sentença
        return [[1.0, 0.0, 0.0] for _ in sentences]


# Patchar dependências antes de uso
sc.SentenceTransformer = _DummySentenceTransformer  # type: ignore[attr-defined]
sc.cosine_similarity = lambda a, b: [[1.0]]  # type: ignore
# Garantir que np.stack existe e funciona — usar simples conversão para lista
sc.np.stack = lambda seq: list(seq)  # type: ignore


@pytest.fixture(scope="module")
def chunker():
    # min_chunk_size=1 para não filtrar
    return sc.SemanticChunker(similarity_threshold=0.1, min_chunk_size=1, max_chunk_size=100)


def test_split_sentences():
    text = "Primeira frase. Segunda frase! Terceira frase?"
    sentences = sc.SemanticChunker()._split_sentences(text)  # type: ignore
    assert len(sentences) == 3


def test_chunk_basic(chunker):
    text = "Sentença A. Sentença B."
    metadata = {"document_id": "doc1"}
    chunks = chunker.chunk(text, metadata)
    assert chunks  # deve retornar pelo menos um chunk
    assert chunks[0].metadata["document_id"] == "doc1" 