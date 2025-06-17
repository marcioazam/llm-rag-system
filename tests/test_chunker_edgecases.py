from src.chunking.recursive_chunker import RecursiveChunker


def test_chunker_empty_document():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk("", metadata={})
    assert chunks == []


def test_chunker_long_document():
    text = "a" * 1000  # 1000 caracteres
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk(text, metadata={})
    # Espera ao menos 4 chunks considerando sobreposição
    assert len(chunks) >= 4 