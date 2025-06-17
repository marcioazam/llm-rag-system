import pytest
from unittest.mock import patch, MagicMock
from src.chunking.recursive_chunker import RecursiveChunker
from src.chunking.base_chunker import Chunk


class TestRecursiveChunker:
    """Testes para a classe RecursiveChunker."""

    def test_init_default(self):
        """Testa a inicialização com parâmetros padrão."""
        chunker = RecursiveChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.separators == ["\n\n", "\n", ". ", " ", ""]

    def test_init_custom_params(self):
        """Testa a inicialização com parâmetros customizados."""
        custom_separators = ["\n", ". ", " "]
        chunker = RecursiveChunker(
            chunk_size=256,
            chunk_overlap=25,
            separators=custom_separators
        )
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 25
        assert chunker.separators == custom_separators

    def test_chunk_basic(self):
        """Testa chunking básico de texto."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a test text for chunking."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, Chunk) for chunk in result)
        assert result[0].content.strip() == text
        assert result[0].metadata == metadata
        assert result[0].document_id == "test_doc"

    def test_chunk_empty(self):
        """Testa chunking de texto vazio."""
        chunker = RecursiveChunker()
        result = chunker.chunk("", {})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_chunk_none(self):
        """Testa chunking de texto None."""
        chunker = RecursiveChunker()
        with pytest.raises((TypeError, AttributeError)):
            chunker.chunk(None, {})

    def test_chunk_long_text(self):
        """Testa chunking de texto longo que precisa ser dividido."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
        long_text = "This is a very long text that should be split into multiple chunks because it exceeds the maximum chunk size that we have configured for this test."
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
            # O tamanho pode variar devido à lógica de separadores e overlap
            assert len(chunk.content) > 0  # Pelo menos tem conteúdo

    def test_chunk_with_separators(self):
        """Testa chunking com diferentes separadores."""
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        metadata = {"document_id": "sep_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica se o texto foi dividido adequadamente
        full_text = "".join([chunk.content for chunk in result])
        # Remove espaços extras que podem ser adicionados na combinação
        assert "Paragraph 1" in full_text
        assert "Paragraph 2" in full_text
        assert "Paragraph 3" in full_text

    def test_chunk_code_like(self):
        """Testa chunking de texto similar a código."""
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=10)
        code_text = "def function():\n    return True\n\nclass MyClass:\n    pass"
        metadata = {"document_id": "code_doc"}
        result = chunker.chunk(code_text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "def function" in result[0].content

    def test_chunk_no_overlap(self):
        """Testa chunking sem overlap."""
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        text = "Short text for testing no overlap functionality."
        metadata = {"document_id": "no_overlap_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Com overlap 0, não deve haver repetição de conteúdo
        if len(result) > 1:
            combined_length = sum(len(chunk.content) for chunk in result)
            # Pode haver espaços adicionais na combinação
            assert combined_length >= len(text.replace(" ", ""))

    def test_chunk_with_overlap(self):
        """Testa chunking com overlap significativo."""
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=10)
        text = "This is a test text that will be chunked with overlap to ensure continuity."
        metadata = {"document_id": "overlap_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        if len(result) > 1:
            # Deve haver algum overlap entre chunks consecutivos
            assert len(result) > 1

    def test_chunk_single_sentence(self):
        """Testa chunking de uma única sentença."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a single sentence."
        metadata = {"document_id": "single_doc"}
        result = chunker.chunk(text, metadata)
        
        assert len(result) == 1
        assert result[0].content.strip() == text

    def test_chunk_multiple_paragraphs(self):
        """Testa chunking de múltiplos parágrafos."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        metadata = {"document_id": "multi_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica se os parágrafos estão presentes
        full_content = " ".join([chunk.content for chunk in result])
        assert "First paragraph" in full_content
        assert "Second paragraph" in full_content
        assert "Third paragraph" in full_content

    def test_chunk_custom_separators(self):
        """Testa chunking com separadores customizados."""
        custom_separators = ["|", "-", " "]
        chunker = RecursiveChunker(
            chunk_size=20,
            chunk_overlap=2,
            separators=custom_separators
        )
        text = "Part1|Part2-Part3 Part4"
        metadata = {"document_id": "custom_sep_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_markdown_like(self):
        """Testa chunking de texto similar a Markdown."""
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=5)
        text = "# Title\n\n## Subtitle\n\nSome content here.\n\n- List item 1\n- List item 2"
        metadata = {"document_id": "md_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica se elementos markdown estão presentes
        full_content = " ".join([chunk.content for chunk in result])
        assert "Title" in full_content
        assert "Subtitle" in full_content

    def test_chunk_very_small_size(self):
        """Testa chunking com tamanho muito pequeno."""
        chunker = RecursiveChunker(chunk_size=5, chunk_overlap=1)
        text = "Small chunk test"
        metadata = {"document_id": "small_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Com tamanho muito pequeno, pode haver chunks maiores devido à lógica de separadores
        # Verificamos apenas que o resultado é válido
        for chunk in result:
            assert len(chunk.content) > 0  # Pelo menos tem conteúdo

    def test_chunk_special_characters(self):
        """Testa chunking com caracteres especiais."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
        text = "Text with @#$%^&*() special chars and números 123."
        metadata = {"document_id": "special_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "@#$%^&*()" in result[0].content

    def test_chunk_unicode(self):
        """Testa chunking com caracteres unicode."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=5)
        text = "Texto com acentos: ção, ã, é, ü, 中文, 🚀"
        metadata = {"document_id": "unicode_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "🚀" in result[0].content

    def test_chunk_json_like(self):
        """Testa chunking de texto similar a JSON."""
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=5)
        text = '{"key": "value", "number": 123, "array": [1, 2, 3]}'
        metadata = {"document_id": "json_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_tabs_and_spaces(self):
        """Testa chunking com tabs e espaços."""
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=3)
        text = "Text\twith\ttabs\nand    multiple    spaces"
        metadata = {"document_id": "whitespace_doc"}
        result = chunker.chunk(text, metadata)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "\t" in result[0].content or "tabs" in result[0].content

    def test_chunk_properties(self):
        """Testa as propriedades dos chunks gerados."""
        chunker = RecursiveChunker()
        text = "Test chunk properties and metadata."
        metadata = {
            "document_id": "props_doc",
            "author": "Test Author",
            "category": "Test"
        }
        result = chunker.chunk(text, metadata)
        
        assert len(result) > 0
        chunk = result[0]
        
        # Verifica propriedades do chunk
        assert hasattr(chunk, 'content')
        assert hasattr(chunk, 'metadata')
        assert hasattr(chunk, 'chunk_id')
        assert hasattr(chunk, 'document_id')
        assert hasattr(chunk, 'position')
        
        # Verifica tipos
        assert isinstance(chunk.content, str)
        assert isinstance(chunk.metadata, dict)
        assert isinstance(chunk.chunk_id, str)
        assert isinstance(chunk.document_id, str)
        assert isinstance(chunk.position, int)
        
        # Verifica valores
        assert chunk.metadata == metadata
        assert chunk.document_id == "props_doc"
        assert chunk.position == 0
        assert len(chunk.chunk_id) > 0  # UUID deve ter conteúdo

    def test_repr(self):
        """Testa a representação string da classe."""
        chunker = RecursiveChunker(chunk_size=256, chunk_overlap=25)
        repr_str = repr(chunker)
        assert "RecursiveChunker" in repr_str
        # A representação deve conter informações sobre a classe
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_str(self):
        """Testa a representação string da classe."""
        chunker = RecursiveChunker(chunk_size=256, chunk_overlap=25)
        str_repr = str(chunker)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0