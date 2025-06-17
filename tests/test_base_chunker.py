import pytest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any
from src.chunking.base_chunker import BaseChunker, Chunk


class ConcreteChunker(BaseChunker):
    """Implementação concreta do BaseChunker para testes."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Implementação simples de chunking para testes."""
        if text is None:
            raise TypeError("Text cannot be None")
        
        if not text:
            return []
        
        chunks = []
        chunk_id = 0
        
        # Calcular step, garantindo que seja pelo menos 1
        step = max(1, self.chunk_size - self.overlap)
        
        # Dividir texto em chunks simples
        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text.strip():
                chunk = Chunk(
                    content=chunk_text,
                    metadata=metadata,
                    chunk_id=f"chunk_{chunk_id}",
                    document_id=metadata.get('document_id', 'test_doc'),
                    position=chunk_id
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks


class TestBaseChunker:
    """Testes para a classe BaseChunker."""

    def test_init(self):
        """Testa a inicialização da classe concreta."""
        chunker = ConcreteChunker()
        assert chunker is not None
        assert hasattr(chunker, 'chunk_size')
        assert hasattr(chunker, 'overlap')

    def test_chunk_text_basic(self):
        """Testa chunking básico de texto."""
        chunker = ConcreteChunker(chunk_size=50)
        text = "This is a test text for chunking."
        metadata = {'document_id': 'test_doc'}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, Chunk) for chunk in result)

    def test_chunk_text_empty(self):
        """Testa chunking de texto vazio."""
        chunker = ConcreteChunker()
        metadata = {'document_id': 'test_doc'}
        result = chunker.chunk("", metadata)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_chunk_text_none(self):
        """Testa chunking de texto None."""
        chunker = ConcreteChunker()
        metadata = {'document_id': 'test_doc'}
        with pytest.raises(TypeError):
            chunker.chunk(None, metadata)

    def test_chunk_text_long(self):
        """Testa chunking de texto longo."""
        chunker = ConcreteChunker(chunk_size=100, overlap=20)
        long_text = "This is a very long text. " * 100
        metadata = {'document_id': 'test_doc'}
        result = chunker.chunk(long_text, metadata)
        assert isinstance(result, list)
        assert len(result) > 1  # Deve gerar múltiplos chunks
        assert all(isinstance(chunk, Chunk) for chunk in result)

    def test_chunk_size_property(self):
        """Testa propriedade chunk_size."""
        chunker = ConcreteChunker(chunk_size=500)
        assert chunker.chunk_size == 500

    def test_overlap_property(self):
        """Testa propriedade overlap."""
        chunker = ConcreteChunker(overlap=50)
        assert chunker.overlap == 50

    def test_chunk_with_metadata(self):
        """Testa chunking com metadata."""
        chunker = ConcreteChunker()
        text = "Test text with metadata"
        metadata = {"source": "test", "type": "document", "document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(chunk.metadata == metadata for chunk in result)

    def test_chunk_with_custom_size(self):
        """Testa chunking com tamanho customizado."""
        chunker = ConcreteChunker(chunk_size=100)
        text = "This is a test text for custom size chunking. " * 10
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0
        assert chunker.chunk_size == 100

    def test_chunk_with_overlap(self):
        """Testa chunking com overlap."""
        chunker = ConcreteChunker(chunk_size=100, overlap=20)
        text = "This is a test text for overlap chunking. " * 10
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0
        assert chunker.overlap == 20

    def test_chunk_single_word(self):
        """Testa chunking de uma única palavra."""
        chunker = ConcreteChunker()
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk("word", metadata)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunk_multiple_sentences(self):
        """Testa chunking de múltiplas sentenças."""
        chunker = ConcreteChunker()
        text = "First sentence. Second sentence. Third sentence."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_with_newlines(self):
        """Testa chunking de texto com quebras de linha."""
        chunker = ConcreteChunker()
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_with_special_chars(self):
        """Testa chunking de texto com caracteres especiais."""
        chunker = ConcreteChunker()
        text = "Text with @#$%^&*() special characters!"
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_unicode_text(self):
        """Testa chunking de texto unicode."""
        chunker = ConcreteChunker()
        text = "Texto em português com acentos: ção, ã, é, ü"
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_very_small_size(self):
        """Testa chunking com tamanho muito pequeno."""
        chunker = ConcreteChunker(chunk_size=10)
        text = "This is a test text for very small chunks."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_very_large_size(self):
        """Testa chunking com tamanho muito grande."""
        chunker = ConcreteChunker(chunk_size=10000)
        text = "Short text"
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_str_representation(self):
        """Testa representação string do chunker."""
        chunker = ConcreteChunker()
        str_repr = str(chunker)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    def test_repr_representation(self):
        """Testa representação repr do chunker."""
        chunker = ConcreteChunker()
        repr_str = repr(chunker)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0