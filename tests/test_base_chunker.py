import pytest
from unittest.mock import patch, MagicMock
from src.chunking.base_chunker import BaseChunker


class TestBaseChunker:
    """Testes para a classe BaseChunker."""

    def test_init(self):
        """Testa a inicialização da classe."""
        chunker = BaseChunker()
        assert chunker is not None

    def test_chunk_text_basic(self):
        """Testa chunking básico de texto."""
        chunker = BaseChunker()
        text = "This is a test text for chunking."
        result = chunker.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_text_empty(self):
        """Testa chunking de texto vazio."""
        chunker = BaseChunker()
        result = chunker.chunk_text("")
        assert isinstance(result, list)

    def test_chunk_text_none(self):
        """Testa chunking de texto None."""
        chunker = BaseChunker()
        result = chunker.chunk_text(None)
        assert result is None or isinstance(result, list)

    def test_chunk_text_long(self):
        """Testa chunking de texto longo."""
        chunker = BaseChunker()
        long_text = "This is a very long text. " * 100
        result = chunker.chunk_text(long_text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_size_property(self):
        """Testa propriedade chunk_size."""
        chunker = BaseChunker(chunk_size=500)
        assert hasattr(chunker, 'chunk_size') or hasattr(chunker, '_chunk_size')

    def test_overlap_property(self):
        """Testa propriedade overlap."""
        chunker = BaseChunker(overlap=50)
        assert hasattr(chunker, 'overlap') or hasattr(chunker, '_overlap')

    def test_chunk_with_metadata(self):
        """Testa chunking com metadata."""
        chunker = BaseChunker()
        text = "Test text with metadata"
        metadata = {"source": "test", "type": "document"}
        try:
            result = chunker.chunk_text(text, metadata=metadata)
            assert isinstance(result, list)
        except TypeError:
            # Se não aceita metadata, testa sem
            result = chunker.chunk_text(text)
            assert isinstance(result, list)

    def test_chunk_with_custom_size(self):
        """Testa chunking com tamanho customizado."""
        chunker = BaseChunker(chunk_size=100)
        text = "This is a test text for custom size chunking. " * 10
        result = chunker.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_with_overlap(self):
        """Testa chunking com overlap."""
        chunker = BaseChunker(chunk_size=100, overlap=20)
        text = "This is a test text for overlap chunking. " * 10
        result = chunker.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_single_word(self):
        """Testa chunking de uma única palavra."""
        chunker = BaseChunker()
        result = chunker.chunk_text("word")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunk_multiple_sentences(self):
        """Testa chunking de múltiplas sentenças."""
        chunker = BaseChunker()
        text = "First sentence. Second sentence. Third sentence."
        result = chunker.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_with_newlines(self):
        """Testa chunking de texto com quebras de linha."""
        chunker = BaseChunker()
        text = "Line 1\nLine 2\nLine 3\nLine 4"
        result = chunker.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_with_special_chars(self):
        """Testa chunking de texto com caracteres especiais."""
        chunker = BaseChunker()
        text = "Text with @#$%^&*() special characters!"
        result = chunker.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_unicode_text(self):
        """Testa chunking de texto unicode."""
        chunker = BaseChunker()
        text = "Texto em português com acentos: ção, ã, é, ü"
        result = chunker.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_very_small_size(self):
        """Testa chunking com tamanho muito pequeno."""
        chunker = BaseChunker(chunk_size=10)
        text = "This is a test text for very small chunks."
        result = chunker.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_very_large_size(self):
        """Testa chunking com tamanho muito grande."""
        chunker = BaseChunker(chunk_size=10000)
        text = "Short text"
        result = chunker.chunk_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_str_representation(self):
        """Testa representação string do chunker."""
        chunker = BaseChunker()
        str_repr = str(chunker)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    def test_repr_representation(self):
        """Testa representação repr do chunker."""
        chunker = BaseChunker()
        repr_str = repr(chunker)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0