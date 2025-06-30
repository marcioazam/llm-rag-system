"""
Testes abrangentes para Language Aware Chunker - FASE 2
Cobertura atual: 0% -> Meta: 70%+
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Importar após configurar path
from chunking.language_aware_chunker import LanguageAwareChunker, CodeChunk, create_language_aware_chunker


class TestLanguageAwareChunkerBasic:
    """Testes básicos de funcionalidade"""
    
    def test_module_import(self):
        """Verifica se o módulo pode ser importado"""
        from chunking.language_aware_chunker import LanguageAwareChunker
        assert LanguageAwareChunker is not None
    
    @patch('tree_sitter_languages.get_language')
    def test_initialization(self, mock_get_language):
        """Testa inicialização básica do chunker"""
        # Mock tree-sitter language
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = LanguageAwareChunker()
        
        assert chunker.target_chunk_size == LanguageAwareChunker.DEFAULT_CHUNK_SIZE
        assert hasattr(chunker, 'parsers')
        assert hasattr(chunker, 'language_configs')
    
    @patch('tree_sitter_languages.get_language')
    def test_initialization_custom_size(self, mock_get_language):
        """Testa inicialização com tamanho customizado"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        custom_size = 1000
        chunker = LanguageAwareChunker(target_chunk_size=custom_size)
        
        assert chunker.target_chunk_size == custom_size

    @patch('tree_sitter_languages.get_language')
    def test_supported_languages(self, mock_get_language):
        """Verifica se linguagens suportadas estão configuradas"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = LanguageAwareChunker()
        
        supported_languages = ['python', 'javascript', 'typescript', 'csharp', 'java']
        for lang in supported_languages:
            assert lang in chunker.language_configs
            assert 'context_nodes' in chunker.language_configs[lang]
            assert 'chunk_boundaries' in chunker.language_configs[lang]


class TestCodeChunkDataclass:
    """Testes para o dataclass CodeChunk"""
    
    def test_code_chunk_creation(self):
        """Testa criação básica de CodeChunk"""
        chunk = CodeChunk(
            content="def hello():\n    print('hello')",
            start_line=1,
            end_line=2,
            chunk_type="function",
            language="python",
            metadata={"test": True}
        )
        
        assert chunk.content == "def hello():\n    print('hello')"
        assert chunk.start_line == 1
        assert chunk.end_line == 2
        assert chunk.chunk_type == "function"
        assert chunk.language == "python"
        assert chunk.metadata == {"test": True}
        assert chunk.context is None
    
    def test_code_chunk_post_init(self):
        """Testa cálculos automáticos no post_init"""
        content = "def test():\n    return True"
        chunk = CodeChunk(
            content=content,
            start_line=1,
            end_line=2,
            chunk_type="function",
            language="python",
            metadata={}
        )
        
        assert chunk.size == len(content)
        assert chunk.token_count == len(content.split())
    
    def test_code_chunk_with_context(self):
        """Testa CodeChunk com contexto"""
        chunk = CodeChunk(
            content="def method(self):\n    pass",
            start_line=5,
            end_line=6,
            chunk_type="method",
            language="python",
            metadata={},
            context="class MyClass:\n    pass"
        )
        
        assert chunk.context == "class MyClass:\n    pass"


class TestLanguageAwareChunkerFunctional:
    """Testes funcionais com mocks"""
    
    @patch('tree_sitter_languages.get_language')
    @patch('tree_sitter.Parser')
    def test_chunk_code_python_basic(self, mock_parser_class, mock_get_language):
        """Testa chunking básico de código Python"""
        # Setup mocks
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        
        # Mock tree structure
        mock_tree = MagicMock()
        mock_root = MagicMock()
        mock_root.children = []
        mock_tree.root_node = mock_root
        mock_parser.parse.return_value = mock_tree
        
        chunker = LanguageAwareChunker()
        
        python_code = """
def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
"""
        
        chunks = chunker.chunk_code(python_code, "python")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, CodeChunk)
            assert chunk.language == "python"
    
    @patch('tree_sitter_languages.get_language')
    def test_chunk_code_unsupported_language(self, mock_get_language):
        """Testa chunking para linguagem não suportada"""
        mock_get_language.side_effect = Exception("Language not found")
        
        chunker = LanguageAwareChunker()
        
        code = "some code in unknown language"
        chunks = chunker.chunk_code(code, "unknown")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        # Deve usar _basic_chunking como fallback
    
    @patch('tree_sitter_languages.get_language')
    def test_basic_chunking_fallback(self, mock_get_language):
        """Testa fallback para chunking básico"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = LanguageAwareChunker()
        
        code = "Line 1\nLine 2\nLine 3\nLine 4\n" * 20  # Código longo
        chunks = chunker._basic_chunking(code, "python")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert isinstance(chunk, CodeChunk)
            assert chunk.language == "python"
            assert chunk.chunk_type == "text"
            assert chunk.size <= chunker.MAX_CHUNK_SIZE


class TestLanguageConfigs:
    """Testes para configurações de linguagem"""
    
    @patch('tree_sitter_languages.get_language')
    def test_python_config(self, mock_get_language):
        """Testa configuração específica do Python"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = LanguageAwareChunker()
        python_config = chunker.language_configs['python']
        
        assert python_config['preserve_imports'] is True
        assert python_config['preserve_class_def'] is True
        assert 'import_statement' in python_config['context_nodes']
        assert 'function_definition' in python_config['chunk_boundaries']
    
    @patch('tree_sitter_languages.get_language')
    def test_javascript_config(self, mock_get_language):
        """Testa configuração específica do JavaScript"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = LanguageAwareChunker()
        js_config = chunker.language_configs['javascript']
        
        assert js_config['preserve_imports'] is True
        assert js_config['preserve_closure'] is True
        assert 'import_statement' in js_config['context_nodes']
        assert 'function_declaration' in js_config['chunk_boundaries']


class TestOptimalChunkSize:
    """Testes para otimização de tamanho de chunk"""
    
    @patch('tree_sitter_languages.get_language')
    def test_get_optimal_chunk_size_defaults(self, mock_get_language):
        """Testa tamanhos ótimos padrão"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = LanguageAwareChunker()
        
        # Testa diferentes linguagens
        size_python = chunker.get_optimal_chunk_size('python')
        size_js = chunker.get_optimal_chunk_size('javascript')
        
        assert isinstance(size_python, int)
        assert isinstance(size_js, int)
        assert size_python > 0
        assert size_js > 0
    
    @patch('tree_sitter_languages.get_language')
    def test_get_optimal_chunk_size_complexity(self, mock_get_language):
        """Testa tamanhos com diferentes complexidades"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = LanguageAwareChunker()
        
        size_simple = chunker.get_optimal_chunk_size('python', 'simple')
        size_complex = chunker.get_optimal_chunk_size('python', 'complex')
        
        assert isinstance(size_simple, int)
        assert isinstance(size_complex, int)
        # Código complexo geralmente precisa de chunks menores
        assert size_simple >= size_complex


class TestErrorHandling:
    """Testes para tratamento de erros"""
    
    @patch('tree_sitter_languages.get_language')
    def test_parser_initialization_error(self, mock_get_language):
        """Testa erro na inicialização do parser"""
        mock_get_language.side_effect = Exception("Parser error")
        
        # Não deve lançar exceção, deve continuar com parsers disponíveis
        chunker = LanguageAwareChunker()
        assert hasattr(chunker, 'parsers')
        assert isinstance(chunker.parsers, dict)
    
    @patch('tree_sitter_languages.get_language')
    @patch('tree_sitter.Parser')
    def test_parsing_error_fallback(self, mock_parser_class, mock_get_language):
        """Testa fallback quando parsing falha"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = Exception("Parsing error")
        mock_parser_class.return_value = mock_parser
        
        chunker = LanguageAwareChunker()
        
        code = "def test():\n    pass"
        chunks = chunker.chunk_code(code, "python")
        
        # Deve usar basic_chunking como fallback
        assert isinstance(chunks, list)
        assert len(chunks) > 0


class TestFactoryFunction:
    """Testes para função factory"""
    
    @patch('tree_sitter_languages.get_language')
    def test_create_language_aware_chunker_default(self, mock_get_language):
        """Testa função factory com padrões"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = create_language_aware_chunker()
        
        assert isinstance(chunker, LanguageAwareChunker)
        assert chunker.target_chunk_size == LanguageAwareChunker.DEFAULT_CHUNK_SIZE
    
    @patch('tree_sitter_languages.get_language')
    def test_create_language_aware_chunker_custom(self, mock_get_language):
        """Testa função factory com tamanho customizado"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        custom_size = 800
        chunker = create_language_aware_chunker(target_chunk_size=custom_size)
        
        assert isinstance(chunker, LanguageAwareChunker)
        assert chunker.target_chunk_size == custom_size


class TestIntegrationScenarios:
    """Testes de cenários integrados"""
    
    @patch('tree_sitter_languages.get_language')
    @patch('tree_sitter.Parser')
    def test_large_code_file(self, mock_parser_class, mock_get_language):
        """Testa chunking de arquivo grande"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        
        # Mock tree simples
        mock_tree = MagicMock()
        mock_root = MagicMock()
        mock_root.children = []
        mock_tree.root_node = mock_root
        mock_parser.parse.return_value = mock_tree
        
        chunker = LanguageAwareChunker(target_chunk_size=300)
        
        # Código grande (>1500 chars)
        large_code = """
def function_1():
    '''Large function with lots of code'''
    for i in range(100):
        print(f"Processing item {i}")
        if i % 10 == 0:
            print("Checkpoint reached")
    return True

def function_2():
    '''Another large function'''
    data = []
    for j in range(50):
        data.append(j * 2)
    return data
""" * 5
        
        chunks = chunker.chunk_code(large_code, "python")
        
        assert len(chunks) > 1  # Deve ser dividido em múltiplos chunks
        
        # Verificar tamanhos
        for chunk in chunks:
            assert chunk.size <= chunker.MAX_CHUNK_SIZE
    
    @patch('tree_sitter_languages.get_language')
    def test_empty_code(self, mock_get_language):
        """Testa chunking de código vazio"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = LanguageAwareChunker()
        
        chunks = chunker.chunk_code("", "python")
        
        # Deve retornar lista vazia ou chunk único vazio
        assert isinstance(chunks, list)
    
    @patch('tree_sitter_languages.get_language')
    def test_single_line_code(self, mock_get_language):
        """Testa chunking de código de uma linha"""
        mock_lang = MagicMock()
        mock_get_language.return_value = mock_lang
        
        chunker = LanguageAwareChunker()
        
        single_line = "print('Hello, World!')"
        chunks = chunker.chunk_code(single_line, "python")
        
        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert chunks[0].content.strip() == single_line.strip()


# Fixtures úteis
@pytest.fixture
def sample_python_code():
    """Código Python de exemplo para testes"""
    return """
import os
import sys

class TestClass:
    def __init__(self):
        self.value = 42
    
    def method1(self):
        return self.value * 2
    
    def method2(self):
        return self.value + 10

def standalone_function():
    return "Hello, World!"

if __name__ == "__main__":
    test = TestClass()
    print(test.method1())
"""

@pytest.fixture
def sample_javascript_code():
    """Código JavaScript de exemplo para testes"""
    return """
import { Component } from 'react';

class MyComponent extends Component {
    constructor(props) {
        super(props);
        this.state = { count: 0 };
    }
    
    increment() {
        this.setState({ count: this.state.count + 1 });
    }
    
    render() {
        return <div>{this.state.count}</div>;
    }
}

export default MyComponent;
""" 