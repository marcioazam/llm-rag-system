import pytest
from unittest.mock import patch, MagicMock
from src.chunking.base_chunker import BaseChunker, Chunk


class ConcreteChunker(BaseChunker):
    """Implementação concreta para testes da classe abstrata BaseChunker."""
    
    def chunk(self, text: str, metadata: dict) -> list:
        """Implementação simples para testes."""
        if not text:
            return []
        
        # Divide o texto em chunks de 100 caracteres
        chunk_size = 100
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            chunks.append(Chunk(
                content=chunk_text,
                metadata=metadata,
                chunk_id=f"chunk_{i}",
                document_id=metadata.get("document_id", "test_doc"),
                position=i // chunk_size
            ))
        
        return chunks


class TestBaseChunker:
    """Testes para a classe BaseChunker."""

    def test_init(self):
        """Testa a inicialização da classe concreta."""
        chunker = ConcreteChunker()
        assert chunker is not None
        assert isinstance(chunker, BaseChunker)

    def test_chunk_basic(self):
        """Testa chunking básico de texto."""
        chunker = ConcreteChunker()
        text = "This is a test text for chunking."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, Chunk) for chunk in result)
        assert result[0].content == text  # Texto pequeno fica em um chunk
        assert result[0].metadata == metadata
        assert result[0].document_id == "test_doc"

    def test_chunk_empty(self):
        """Testa chunking de texto vazio."""
        chunker = ConcreteChunker()
        result = chunker.chunk("", {})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_chunk_none(self):
        """Testa chunking de texto None."""
        chunker = ConcreteChunker()
        # A implementação pode tratar None de diferentes formas
        try:
            result = chunker.chunk(None, {})
            # Se não lançar exceção, deve retornar lista vazia ou tratar adequadamente
            assert isinstance(result, list)
        except (TypeError, AttributeError):
            # É aceitável lançar exceção para None
            pass

    def test_chunk_long_text(self):
        """Testa chunking de texto longo."""
        chunker = ConcreteChunker()
        long_text = "This is a very long text that should be split into multiple chunks. " * 10
        metadata = {"document_id": "long_doc"}
        result = chunker.chunk(long_text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 1  # Deve gerar múltiplos chunks
        
        # Verifica se todos os chunks são válidos
        for i, chunk in enumerate(result):
            assert isinstance(chunk, Chunk)
            assert chunk.metadata == metadata
            assert chunk.document_id == "long_doc"
            assert chunk.position == i
            assert len(chunk.content) <= 100  # Respeitando o tamanho máximo

    def test_chunk_with_metadata(self):
        """Testa chunking com metadados customizados."""
        chunker = ConcreteChunker()
        text = "Test text with custom metadata."
        metadata = {
            "document_id": "custom_doc",
            "author": "Test Author",
            "category": "Test Category"
        }
        result = chunker.chunk(text, metadata)
        
        assert len(result) == 1
        chunk = result[0]
        assert chunk.metadata == metadata
        assert chunk.metadata["author"] == "Test Author"
        assert chunk.metadata["category"] == "Test Category"

    def test_chunk_unicode(self):
        """Testa chunking com caracteres unicode."""
        chunker = ConcreteChunker()
        text = "Texto com acentos: ção, ã, é, ü, 中文, 🚀"
        metadata = {"document_id": "unicode_doc"}
        result = chunker.chunk(text, metadata)
        
        assert len(result) == 1
        assert result[0].content == text
        assert "🚀" in result[0].content

    def test_chunk_special_characters(self):
        """Testa chunking com caracteres especiais."""
        chunker = ConcreteChunker()
        text = "Text with special chars: @#$%^&*()[]{}|\\;':,.<>?/~`"
        metadata = {"document_id": "special_doc"}
        result = chunker.chunk(text, metadata)
        
        assert len(result) == 1
        assert result[0].content == text

    def test_chunk_newlines(self):
        """Testa chunking com quebras de linha."""
        chunker = ConcreteChunker()
        text = "Line 1\nLine 2\nLine 3\n\nLine 5"
        metadata = {"document_id": "newline_doc"}
        result = chunker.chunk(text, metadata)
        
        assert len(result) == 1
        assert "\n" in result[0].content

    def test_chunk_properties(self):
        """Testa as propriedades dos chunks gerados."""
        chunker = ConcreteChunker()
        text = "Test chunk properties."
        metadata = {"document_id": "props_doc"}
        result = chunker.chunk(text, metadata)
        
        chunk = result[0]
        assert hasattr(chunk, 'content')
        assert hasattr(chunk, 'metadata')
        assert hasattr(chunk, 'chunk_id')
        assert hasattr(chunk, 'document_id')
        assert hasattr(chunk, 'position')
        
        assert isinstance(chunk.content, str)
        assert isinstance(chunk.metadata, dict)
        assert isinstance(chunk.chunk_id, str)
        assert isinstance(chunk.document_id, str)
        assert isinstance(chunk.position, int)

    def test_abstract_base_class(self):
        """Testa que BaseChunker é uma classe abstrata."""
        with pytest.raises(TypeError):
            BaseChunker()  # Não deve ser possível instanciar diretamente

    def test_chunk_method_signature(self):
        """Testa a assinatura do método chunk."""
        chunker = ConcreteChunker()
        
        # Deve aceitar text e metadata
        result = chunker.chunk("test", {"doc_id": "test"})
        assert isinstance(result, list)
        
        # Deve funcionar com metadata vazio
        result = chunker.chunk("test", {})
        assert isinstance(result, list)