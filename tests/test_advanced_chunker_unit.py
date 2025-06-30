import numpy as np

from types import SimpleNamespace

from src.chunking.advanced_chunker import AdvancedChunker


class DummyEmbeddingService:
    def embed_texts(self, texts, show_progress=False):
        # Cria embeddings simples determinísticos
        return [np.array([i, 0, 0], dtype=float) for i, _ in enumerate(texts)]


def test_semantic_chunk_simple():
    chunker = AdvancedChunker(embedding_service=DummyEmbeddingService(), max_chunk_size=100)
    doc = {"content": "Uma frase. Outra frase. Mais uma frase.", "metadata": {"doc": "x"}}

    chunks = chunker.semantic_chunk(doc)
    assert chunks, "Deve retornar pelo menos um chunk"
    for c in chunks:
        assert c["metadata"]["chunk_method"] == "semantic"
        assert len(c["content"]) <= 100


def test_structural_chunk_split():
    chunker = AdvancedChunker(embedding_service=DummyEmbeddingService(), max_chunk_size=50)
    text = "Parágrafo um muito longo que excede cinquenta caracteres para forçar split.\n\nSegundo parágrafo também longo."
    doc = {"content": text, "metadata": {}}
    chunks = chunker.structural_chunk(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["metadata"]["chunk_method"] == "structural" 