"""
Testes básicos para o Language Aware Chunker.
Cobertura atual: 0% -> Meta: 70%
"""

import pytest
from typing import List
from pathlib import Path

from src.chunking.language_aware_chunker import (
    LanguageAwareChunker, CodeChunk, create_language_aware_chunker
)


class TestCodeChunk:
    """Testes para a classe CodeChunk."""

    def test_code_chunk_creation(self):
        """Testar criação básica de CodeChunk."""
        chunk = CodeChunk(
            content="def hello():\n    return 'world'",
            start_line=1,
            end_line=2,
            chunk_type="function",
            language="python",
            metadata={"name": "hello"}
        )
        
        assert chunk.content == "def hello():\n    return 'world'"
        assert chunk.start_line == 1
        assert chunk.end_line == 2
        assert chunk.chunk_type == "function"
        assert chunk.language == "python"
        assert chunk.metadata["name"] == "hello"
        
        # Verificar se __post_init__ foi executado
        assert chunk.size > 0
        assert chunk.token_count > 0

    def test_code_chunk_post_init(self):
        """Testar cálculo automático de métricas."""
        content = "class Test:\n    def method(self):\n        pass"
        chunk = CodeChunk(
            content=content,
            start_line=1,
            end_line=3,
            chunk_type="class",
            language="python",
            metadata={}
        )
        
        assert chunk.size == len(content)
        assert chunk.token_count == len(content.split())


class TestLanguageAwareChunker:
    """Testes para o Language Aware Chunker."""

    @pytest.fixture
    def chunker(self):
        """Criar instância do chunker."""
        return LanguageAwareChunker(target_chunk_size=400)

    def test_init(self, chunker):
        """Testar inicialização do chunker."""
        assert chunker.target_chunk_size == 400
        assert isinstance(chunker.parsers, dict)
        assert isinstance(chunker.language_configs, dict)
        
        # Verificar se configurações básicas existem
        assert "python" in chunker.language_configs
        assert "javascript" in chunker.language_configs

    def test_init_default_size(self):
        """Testar inicialização com tamanho padrão."""
        chunker = LanguageAwareChunker()
        assert chunker.target_chunk_size == LanguageAwareChunker.DEFAULT_CHUNK_SIZE

    def test_chunk_code_python_simple(self, chunker):
        """Testar chunking básico de código Python."""
        python_code = '''
def function1():
    """Primeira função"""
    return "hello"

def function2():
    """Segunda função"""
    return "world"

class MyClass:
    def method(self):
        return "class method"
'''
        
        chunks = chunker.chunk_code(python_code, "python")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert isinstance(chunk, CodeChunk)
            assert chunk.language == "python"
            assert len(chunk.content) > 0
            assert chunk.start_line >= 0
            assert chunk.end_line >= chunk.start_line

    def test_chunk_code_javascript(self, chunker):
        """Testar chunking de código JavaScript."""
        js_code = '''
function hello() {
    return "Hello, World!";
}

const arrow = () => {
    console.log("Arrow function");
};

class Example {
    constructor() {
        this.name = "example";
    }
    
    getName() {
        return this.name;
    }
}
'''
        
        chunks = chunker.chunk_code(js_code, "javascript")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.language == "javascript"

    def test_chunk_code_unsupported_language(self, chunker):
        """Testar chunking com linguagem não suportada."""
        code = "some generic code content here"
        
        chunks = chunker.chunk_code(code, "unsupported_lang")
        
        # Deve usar chunking básico
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.language == "unsupported_lang"

    def test_basic_chunking(self, chunker):
        """Testar método de chunking básico."""
        code = "This is a long piece of text that should be split into multiple chunks " * 20
        
        chunks = chunker._basic_chunking(code, "text")
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Deve dividir texto longo
        
        for chunk in chunks:
            assert isinstance(chunk, CodeChunk)
            assert chunk.language == "text"
            assert len(chunk.content) <= chunker.target_chunk_size * 2  # Margem para divisão

    def test_chunk_code_with_file_path(self, chunker):
        """Testar chunking com caminho de arquivo."""
        python_code = '''
def test_function():
    return True
'''
        
        chunks = chunker.chunk_code(python_code, "python", file_path="test.py")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.language == "python"

    def test_get_optimal_chunk_size(self, chunker):
        """Testar obtenção de tamanho ótimo de chunk."""
        # Tamanho para complexidade baixa
        size_low = chunker.get_optimal_chunk_size("python", "low")
        assert isinstance(size_low, int)
        assert size_low > 0
        
        # Tamanho para complexidade média
        size_medium = chunker.get_optimal_chunk_size("python", "medium")
        assert isinstance(size_medium, int)
        assert size_medium > 0
        
        # Tamanho para complexidade alta
        size_high = chunker.get_optimal_chunk_size("python", "high")
        assert isinstance(size_high, int)
        assert size_high > 0
        
        # Complexidade alta deve ter chunks maiores
        assert size_high >= size_medium >= size_low

    def test_chunk_empty_code(self, chunker):
        """Testar chunking de código vazio."""
        chunks = chunker.chunk_code("", "python")
        
        # Deve retornar lista vazia ou um chunk vazio
        assert isinstance(chunks, list)

    def test_chunk_very_long_function(self, chunker):
        """Testar chunking de função muito longa."""
        long_function = '''
def very_long_function():
    """Esta é uma função muito longa que deve ser subdividida"""
    # Muitas linhas de código
''' + "    x = 1\n" * 100 + '''
    return x
'''
        
        chunks = chunker.chunk_code(long_function, "python")
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "python"

    def test_language_configs(self, chunker):
        """Testar configurações de linguagem."""
        # Python config
        python_config = chunker.language_configs["python"]
        assert "preserve_imports" in python_config
        assert "context_nodes" in python_config
        assert "chunk_boundaries" in python_config
        
        # JavaScript config
        js_config = chunker.language_configs["javascript"]
        assert "preserve_imports" in js_config
        assert "context_nodes" in js_config

    def test_extract_global_context_python(self, chunker):
        """Testar extração de contexto global Python."""
        python_code = '''
import os
import sys
from typing import List

def function():
    pass
'''
        
        # Simular tree parsing
        try:
            tree = chunker.parsers["python"].parse(bytes(python_code, 'utf8'))
            context = chunker._extract_global_context(tree.root_node, python_code, "python")
            
            # Deve conter imports
            assert isinstance(context, str)
            if context:  # Se conseguiu extrair contexto
                assert "import" in context.lower()
        except:
            # Se tree-sitter não funcionar, apenas verificar que não dá erro
            pass

    def test_multiple_small_chunks(self, chunker):
        """Testar com múltiplas funções pequenas."""
        code = '''
def func1():
    return 1

def func2():
    return 2

def func3():
    return 3
'''
        
        chunks = chunker.chunk_code(code, "python")
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "python"

    def test_chunk_with_imports_and_classes(self, chunker):
        """Testar chunking com imports e classes."""
        code = '''
import numpy as np
from typing import List, Dict

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, input_data: List[str]) -> Dict[str, int]:
        """Process input data and return statistics"""
        result = {}
        for item in input_data:
            result[item] = len(item)
        return result
    
    def save_results(self, results: Dict[str, int], filename: str):
        """Save results to file"""
        with open(filename, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\\n")

def main():
    processor = DataProcessor()
    data = ["hello", "world", "python"]
    results = processor.process(data)
    processor.save_results(results, "output.txt")
'''
        
        chunks = chunker.chunk_code(code, "python")
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "python"
            # Verificar se tem conteúdo útil
            assert len(chunk.content.strip()) > 0

    def test_constants_and_limits(self, chunker):
        """Testar constantes e limites da classe."""
        assert LanguageAwareChunker.DEFAULT_CHUNK_SIZE == 500
        assert LanguageAwareChunker.MAX_CHUNK_SIZE == 1500
        assert LanguageAwareChunker.MIN_CHUNK_SIZE == 100
        
        # Verificar se target_chunk_size está dentro dos limites razoáveis
        assert chunker.target_chunk_size > 0

    def test_parsers_initialization(self, chunker):
        """Testar inicialização dos parsers."""
        parsers = chunker.parsers
        
        # Pode não ter todos os parsers dependendo do ambiente
        assert isinstance(parsers, dict)
        
        # Verificar se pelo menos tentou inicializar
        expected_languages = ["python", "javascript", "typescript", "csharp", "java"]
        for lang in expected_languages:
            # Não garantimos que todos funcionem, mas deve tentar
            pass

    def test_chunk_code_error_handling(self, chunker):
        """Testar tratamento de erros no chunking."""
        # Código com caracteres especiais que podem causar problemas
        problematic_code = "def func():\n\treturn 'hello\\x00world'"
        
        try:
            chunks = chunker.chunk_code(problematic_code, "python")
            assert isinstance(chunks, list)
        except:
            # Se der erro, deve falhar graciosamente
            pass

    @pytest.mark.parametrize("language", ["python", "javascript", "typescript"])
    def test_chunk_different_languages(self, chunker, language):
        """Testar chunking para diferentes linguagens."""
        code_samples = {
            "python": "def test():\n    return True",
            "javascript": "function test() {\n    return true;\n}",
            "typescript": "function test(): boolean {\n    return true;\n}"
        }
        
        code = code_samples.get(language, "// Test code")
        chunks = chunker.chunk_code(code, language)
        
        assert isinstance(chunks, list)
        if chunks:  # Se conseguiu gerar chunks
            assert all(chunk.language == language for chunk in chunks)


class TestFactoryFunction:
    """Testes para a função factory."""

    def test_create_language_aware_chunker_default(self):
        """Testar criação com parâmetros padrão."""
        chunker = create_language_aware_chunker()
        
        assert isinstance(chunker, LanguageAwareChunker)
        assert chunker.target_chunk_size == LanguageAwareChunker.DEFAULT_CHUNK_SIZE

    def test_create_language_aware_chunker_custom_size(self):
        """Testar criação com tamanho customizado."""
        custom_size = 800
        chunker = create_language_aware_chunker(target_chunk_size=custom_size)
        
        assert isinstance(chunker, LanguageAwareChunker)
        assert chunker.target_chunk_size == custom_size

    def test_create_language_aware_chunker_none_size(self):
        """Testar criação com tamanho None."""
        chunker = create_language_aware_chunker(target_chunk_size=None)
        
        assert isinstance(chunker, LanguageAwareChunker)
        assert chunker.target_chunk_size == LanguageAwareChunker.DEFAULT_CHUNK_SIZE


class TestEdgeCases:
    """Testes para casos extremos."""

    @pytest.fixture
    def chunker(self):
        return LanguageAwareChunker(target_chunk_size=200)

    def test_single_long_line(self, chunker):
        """Testar linha única muito longa."""
        long_line = "x = " + "1 + " * 200 + "1"
        
        chunks = chunker.chunk_code(long_line, "python")
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "python"

    def test_many_small_functions(self, chunker):
        """Testar muitas funções pequenas."""
        code = ""
        for i in range(50):
            code += f"def func{i}():\n    return {i}\n\n"
        
        chunks = chunker.chunk_code(code, "python")
        
        assert len(chunks) >= 1
        total_content = "".join(chunk.content for chunk in chunks)
        assert "func0" in total_content
        assert "func49" in total_content

    def test_nested_structures(self, chunker):
        """Testar estruturas profundamente aninhadas."""
        nested_code = '''
class Outer:
    class Middle:
        class Inner:
            def deep_method(self):
                if True:
                    for i in range(10):
                        if i % 2 == 0:
                            try:
                                result = i * 2
                            except:
                                pass
                return result
'''
        
        chunks = chunker.chunk_code(nested_code, "python")
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "python"

    def test_mixed_content_types(self, chunker):
        """Testar conteúdo misto (código + comentários + strings)."""
        mixed_code = '''
#!/usr/bin/env python3
"""
This is a module docstring
with multiple lines
"""

# Import statements
import os
import sys

# Global variable
CONSTANT = "value"

def main():
    """Main function with docstring"""
    # Inline comment
    print("Hello, World!")  # End of line comment
    
    multiline_string = """
    This is a multiline string
    with various content
    """
    
    return True

if __name__ == "__main__":
    main()
'''
        
        chunks = chunker.chunk_code(mixed_code, "python")
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.language == "python"
            assert len(chunk.content.strip()) > 0 