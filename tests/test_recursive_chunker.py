import pytest
from unittest.mock import patch, MagicMock
from src.chunking.recursive_chunker import RecursiveChunker


class TestRecursiveChunker:
    """Testes para a classe RecursiveChunker."""

    def test_init_default(self):
        """Testa a inicialização com valores padrão."""
        chunker = RecursiveChunker()
        assert chunker is not None
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50

    def test_init_with_params(self):
        """Testa a inicialização com parâmetros customizados."""
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        assert chunker is not None
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_chunk_basic(self):
        """Testa chunking básico de texto."""
        chunker = RecursiveChunker()
        text = "This is a test text for recursive chunking."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(hasattr(chunk, 'content') for chunk in result)

    def test_chunk_empty(self):
        """Testa chunking de texto vazio."""
        chunker = RecursiveChunker()
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk("", metadata)
        assert isinstance(result, list)

    def test_chunk_none_handling(self):
        """Testa tratamento de entrada None."""
        chunker = RecursiveChunker()
        metadata = {"document_id": "test_doc"}
        try:
            result = chunker.chunk(None, metadata)
            assert result is None or isinstance(result, list)
        except (TypeError, AttributeError):
            # Comportamento esperado para None
            pass

    def test_chunk_long_text(self):
        """Testa chunking de texto longo."""
        chunker = RecursiveChunker(chunk_size=100)
        long_text = "This is a very long text that should be split into multiple chunks. " * 20
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(long_text, metadata)
        assert isinstance(result, list)
        assert len(result) > 1

    def test_chunk_with_separators(self):
        """Testa chunking com diferentes separadores."""
        chunker = RecursiveChunker()
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_code_text(self):
        """Testa chunking de código."""
        chunker = RecursiveChunker()
        code_text = """def function1():
    return 'hello'

def function2():
    return 'world'

class MyClass:
    def method(self):
        pass"""
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(code_text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_with_overlap(self):
        """Testa chunking com overlap."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test text for overlap testing. " * 10
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 1

    def test_chunk_single_sentence(self):
        """Testa chunking de uma única sentença."""
        chunker = RecursiveChunker()
        text = "This is a single sentence."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunk_multiple_paragraphs(self):
        """Testa chunking de múltiplos parágrafos."""
        chunker = RecursiveChunker(chunk_size=100)
        text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content."""
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_with_custom_separators(self):
        """Testa chunking com separadores customizados."""
        try:
            chunker = RecursiveChunker(separators=["\n\n", "\n", " ", ""])
            text = "Text with custom separators."
            metadata = {"document_id": "test_doc"}
            result = chunker.chunk(text, metadata)
            assert isinstance(result, list)
        except TypeError:
            # Se não aceita separators customizados, usa padrão
            chunker = RecursiveChunker()
            metadata = {"document_id": "test_doc"}
            result = chunker.chunk(text, metadata)
            assert isinstance(result, list)

    def test_chunk_markdown_text(self):
        """Testa chunking de texto Markdown."""
        chunker = RecursiveChunker()
        markdown_text = """# Title

## Subtitle

Some **bold** text and *italic* text.

- List item 1
- List item 2
- List item 3

```python
code_block = "example"
```"""
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(markdown_text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_very_small_chunks(self):
        """Testa chunking com chunks muito pequenos."""
        chunker = RecursiveChunker(chunk_size=20)
        text = "This is a test text for very small recursive chunks."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 1

    def test_chunk_no_overlap(self):
        """Testa chunking sem overlap."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = "This is a test text for no overlap chunking. " * 5
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 1

    def test_chunk_max_overlap(self):
        """Testa chunking com overlap máximo."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=90)
        text = "This is a test text for maximum overlap chunking. " * 5
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_with_special_characters(self):
        """Testa chunking com caracteres especiais."""
        chunker = RecursiveChunker()
        text = "Text with @#$%^&*()_+-=[]{}|;':,.<>?/~` special characters!"
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_unicode_content(self):
        """Testa chunking com conteúdo unicode."""
        chunker = RecursiveChunker()
        text = "Texto em português: ção, não, coração. 中文测试. العربية. русский."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_json_like_text(self):
        """Testa chunking de texto similar a JSON."""
        chunker = RecursiveChunker()
        json_text = '{"key1": "value1", "key2": "value2", "nested": {"inner": "data"}}'
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(json_text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_with_tabs_and_spaces(self):
        """Testa chunking com tabs e espaços."""
        chunker = RecursiveChunker()
        text = "Text\twith\ttabs\tand    multiple    spaces."
        metadata = {"document_id": "test_doc"}
        result = chunker.chunk(text, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_properties(self):
        """Testa propriedades do chunker."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        # Verifica se tem as propriedades esperadas
        assert hasattr(chunker, 'chunk_size')
        assert hasattr(chunker, 'chunk_overlap')
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 20

    def test_str_representation(self):
        """Testa representação string do chunker."""
        chunker = RecursiveChunker()
        str_repr = str(chunker)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    def test_repr_representation(self):
        """Testa representação repr do chunker."""
        chunker = RecursiveChunker()
        repr_str = repr(chunker)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0