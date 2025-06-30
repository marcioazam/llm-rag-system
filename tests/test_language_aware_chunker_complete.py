"""
Testes completos para LanguageAwareChunker
Cobrindo todos os cenários não testados para aumentar a cobertura
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.chunking.language_aware_chunker import LanguageAwareChunker, ChunkType


class TestLanguageAwareChunkerBasic:
    """Testes básicos do LanguageAwareChunker"""
    
    def test_init_default(self):
        """Testa inicialização com parâmetros padrão"""
        chunker = LanguageAwareChunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50
        assert chunker.language == "auto"
        assert chunker.chunk_type == ChunkType.SEMANTIC
        
    def test_init_custom_parameters(self):
        """Testa inicialização com parâmetros customizados"""
        chunker = LanguageAwareChunker(
            chunk_size=1024,
            overlap=100,
            language="python",
            chunk_type=ChunkType.STRUCTURAL
        )
        assert chunker.chunk_size == 1024
        assert chunker.overlap == 100
        assert chunker.language == "python"
        assert chunker.chunk_type == ChunkType.STRUCTURAL
        
    def test_chunk_type_enum(self):
        """Testa os valores do enum ChunkType"""
        assert ChunkType.SEMANTIC == "semantic"
        assert ChunkType.STRUCTURAL == "structural"
        assert ChunkType.HYBRID == "hybrid"


class TestLanguageAwareChunkerDetection:
    """Testes de detecção de linguagem"""
    
    @pytest.fixture
    def chunker(self):
        return LanguageAwareChunker()
    
    def test_detect_language_python(self, chunker):
        """Testa detecção de linguagem Python"""
        code = """
def hello_world():
    print("Hello, World!")
    return True
"""
        result = chunker.detect_language(code)
        assert result == "python"
        
    def test_detect_language_javascript(self, chunker):
        """Testa detecção de linguagem JavaScript"""
        code = """
function helloWorld() {
    console.log("Hello, World!");
    return true;
}
"""
        result = chunker.detect_language(code)
        assert result == "javascript"
        
    def test_detect_language_typescript(self, chunker):
        """Testa detecção de linguagem TypeScript"""
        code = """
interface User {
    name: string;
    age: number;
}

function greetUser(user: User): string {
    return `Hello, ${user.name}!`;
}
"""
        result = chunker.detect_language(code)
        assert result == "typescript"
        
    def test_detect_language_java(self, chunker):
        """Testa detecção de linguagem Java"""
        code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
        result = chunker.detect_language(code)
        assert result == "java"
        
    def test_detect_language_csharp(self, chunker):
        """Testa detecção de linguagem C#"""
        code = """
using System;

namespace HelloWorld
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("Hello, World!");
        }
    }
}
"""
        result = chunker.detect_language(code)
        assert result == "csharp"
        
    def test_detect_language_cpp(self, chunker):
        """Testa detecção de linguagem C++"""
        code = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
        result = chunker.detect_language(code)
        assert result == "cpp"
        
    def test_detect_language_go(self, chunker):
        """Testa detecção de linguagem Go"""
        code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
"""
        result = chunker.detect_language(code)
        assert result == "go"
        
    def test_detect_language_rust(self, chunker):
        """Testa detecção de linguagem Rust"""
        code = """
fn main() {
    println!("Hello, World!");
}
"""
        result = chunker.detect_language(code)
        assert result == "rust"
        
    def test_detect_language_unknown(self, chunker):
        """Testa detecção de linguagem desconhecida"""
        code = "This is just plain text without any programming constructs."
        result = chunker.detect_language(code)
        assert result == "text"
        
    def test_detect_language_empty(self, chunker):
        """Testa detecção com texto vazio"""
        result = chunker.detect_language("")
        assert result == "text"


class TestLanguageAwareChunkerChunking:
    """Testes de chunking por linguagem"""
    
    @pytest.fixture
    def chunker(self):
        return LanguageAwareChunker(chunk_size=200, overlap=20)
    
    def test_chunk_python_semantic(self, chunker):
        """Testa chunking semântico de código Python"""
        chunker.chunk_type = ChunkType.SEMANTIC
        code = """
def function1():
    '''First function'''
    return 1

def function2():
    '''Second function'''
    return 2
"""
        chunks = chunker.chunk_text(code)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        
    def test_chunk_python_structural(self, chunker):
        """Testa chunking estrutural de código Python"""
        chunker.chunk_type = ChunkType.STRUCTURAL
        code = """
import os
import sys

def main():
    print("Hello")
    
if __name__ == "__main__":
    main()
"""
        chunks = chunker.chunk_text(code)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        
    def test_chunk_javascript_semantic(self, chunker):
        """Testa chunking semântico de código JavaScript"""
        chunker.chunk_type = ChunkType.SEMANTIC
        code = """
function calculateSum(a, b) {
    return a + b;
}

function calculateProduct(a, b) {
    return a * b;
}

const numbers = [1, 2, 3, 4, 5];
const sum = numbers.reduce((acc, num) => acc + num, 0);
"""
        chunks = chunker.chunk_text(code)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        
    def test_chunk_hybrid_mode(self, chunker):
        """Testa chunking híbrido"""
        chunker.chunk_type = ChunkType.HYBRID
        code = """
def complex_function():
    # This is a complex function
    data = []
    for i in range(100):
        data.append(i * 2)
    return data
"""
        chunks = chunker.chunk_text(code)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_with_metadata(self, chunker):
        """Testa chunking com metadados"""
        document = {
            "content": "def hello(): return 'world'",
            "metadata": {"file_path": "test.py", "author": "test"}
        }
        chunks = chunker.chunk(document)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "metadata" in chunk
            assert chunk["metadata"]["file_path"] == "test.py"
            
    def test_empty_text(self, chunker):
        """Testa texto vazio"""
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0
        
    def test_none_text(self, chunker):
        """Testa texto None"""
        chunks = chunker.chunk_text(None)
        assert len(chunks) == 0


class TestLanguageAwareChunkerAdvanced:
    """Testes avançados do LanguageAwareChunker"""
    
    @pytest.fixture
    def chunker(self):
        return LanguageAwareChunker()
    
    def test_extract_functions_python(self, chunker):
        """Testa extração de funções Python"""
        code = """
def func1():
    pass

async def async_func():
    await something()

def func2(param1, param2=None):
    return param1 + param2
"""
        functions = chunker.extract_functions(code, "python")
        assert len(functions) == 3
        assert any("func1" in func for func in functions)
        assert any("async_func" in func for func in functions)
        assert any("func2" in func for func in functions)
        
    def test_extract_functions_javascript(self, chunker):
        """Testa extração de funções JavaScript"""
        code = """
function regularFunction() {
    return true;
}

const arrowFunction = () => {
    return false;
}

async function asyncFunction() {
    await something();
}
"""
        functions = chunker.extract_functions(code, "javascript")
        assert len(functions) >= 1
        
    def test_extract_classes_python(self, chunker):
        """Testa extração de classes Python"""
        code = """
class BaseClass:
    def base_method(self):
        pass

class DerivedClass(BaseClass):
    def __init__(self):
        super().__init__()
        
    def derived_method(self):
        return "derived"
"""
        classes = chunker.extract_classes(code, "python")
        assert len(classes) == 2
        assert any("BaseClass" in cls for cls in classes)
        assert any("DerivedClass" in cls for cls in classes)
        
    def test_split_by_structure_python(self, chunker):
        """Testa divisão por estrutura Python"""
        code = """
import sys

def main():
    print("Hello")

class Test:
    pass

if __name__ == "__main__":
    main()
"""
        chunks = chunker.split_by_structure(code, "python")
        assert len(chunks) >= 1
        
    def test_split_by_structure_unknown_language(self, chunker):
        """Testa divisão por estrutura com linguagem desconhecida"""
        code = "Some random text that doesn't belong to any programming language."
        chunks = chunker.split_by_structure(code, "unknown")
        assert len(chunks) == 1
        assert chunks[0] == code
        
    def test_semantic_chunking_with_overlap(self, chunker):
        """Testa chunking semântico com overlap"""
        chunker.overlap = 50
        code = """
def function_one():
    '''This is function one'''
    result = 1 + 1
    return result

def function_two():
    '''This is function two'''
    result = 2 + 2
    return result
"""
        chunks = chunker.semantic_chunking(code, "python")
        assert len(chunks) >= 1


class TestLanguageAwareChunkerEdgeCases:
    """Testes de casos extremos"""
    
    @pytest.fixture
    def chunker(self):
        return LanguageAwareChunker()
    
    def test_very_small_chunk_size(self, chunker):
        """Testa chunk size muito pequeno"""
        chunker.chunk_size = 10
        code = "def test(): pass"
        chunks = chunker.chunk_text(code)
        assert len(chunks) >= 1
        
    def test_very_large_text(self, chunker):
        """Testa texto muito grande"""
        large_code = "def func():\n    pass\n" * 1000
        chunks = chunker.chunk_text(large_code)
        assert len(chunks) > 1
        
    def test_malformed_code(self, chunker):
        """Testa código malformado"""
        malformed_code = """
def incomplete_function(
    # Missing closing parenthesis and body
"""
        chunks = chunker.chunk_text(malformed_code)
        assert len(chunks) >= 1
        
    def test_mixed_languages(self, chunker):
        """Testa código com múltiplas linguagens"""
        mixed_code = """
# Python code
def python_func():
    return "python"

// JavaScript code
function jsFunc() {
    return "javascript";
}
"""
        chunks = chunker.chunk_text(mixed_code)
        assert len(chunks) >= 1


class TestLanguageAwareChunkerIntegration:
    """Testes de integração"""
    
    def test_complete_workflow(self):
        """Testa fluxo completo de chunking"""
        chunker = LanguageAwareChunker(
            chunk_size=300,
            overlap=30,
            language="auto",
            chunk_type=ChunkType.HYBRID
        )
        
        document = {
            "content": """
def calculate_fibonacci(n):
    '''Calculate fibonacci number'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathUtils:
    '''Utility class for math operations'''
    
    @staticmethod
    def factorial(n):
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n-1)
        
    def power(self, base, exponent):
        return base ** exponent
""",
            "metadata": {
                "file_path": "math_utils.py",
                "language": "python"
            }
        }
        
        chunks = chunker.chunk(document)
        
        # Verificações
        assert len(chunks) >= 1
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert all(chunk["metadata"]["file_path"] == "math_utils.py" for chunk in chunks)
        
    def test_different_chunk_types_same_content(self):
        """Testa diferentes tipos de chunk no mesmo conteúdo"""
        code = """
def func1():
    return 1

def func2():
    return 2
"""
        
        # Semantic chunking
        semantic_chunker = LanguageAwareChunker(chunk_type=ChunkType.SEMANTIC)
        semantic_chunks = semantic_chunker.chunk_text(code)
        
        # Structural chunking
        structural_chunker = LanguageAwareChunker(chunk_type=ChunkType.STRUCTURAL)
        structural_chunks = structural_chunker.chunk_text(code)
        
        # Hybrid chunking
        hybrid_chunker = LanguageAwareChunker(chunk_type=ChunkType.HYBRID)
        hybrid_chunks = hybrid_chunker.chunk_text(code)
        
        # Todos devem produzir chunks
        assert len(semantic_chunks) >= 1
        assert len(structural_chunks) >= 1
        assert len(hybrid_chunks) >= 1


class TestLanguageAwareChunkerPerformance:
    """Testes de performance"""
    
    def test_performance_large_file(self):
        """Testa performance com arquivo grande"""
        chunker = LanguageAwareChunker()
        
        # Simula um arquivo Python grande
        large_code = """
def function_{i}():
    '''Function number {i}'''
    result = {i} * 2
    return result

class Class_{i}:
    '''Class number {i}'''
    def method_{i}(self):
        return {i}
""" * 100
        
        import time
        start_time = time.time()
        chunks = chunker.chunk_text(large_code)
        end_time = time.time()
        
        # Deve processar em tempo razoável (menos de 5 segundos)
        assert (end_time - start_time) < 5.0
        assert len(chunks) > 1
        
    def test_memory_efficiency(self):
        """Testa eficiência de memória"""
        chunker = LanguageAwareChunker(chunk_size=100)
        
        # Gera código que seria maior que chunk_size
        code = "def test():\n    " + "x = 1\n    " * 50 + "return x"
        
        chunks = chunker.chunk_text(code)
        
        # Verifica que chunks foram criados sem exceder muito o tamanho
        assert len(chunks) >= 1
        for chunk in chunks:
            # Permite alguma flexibilidade no tamanho devido ao overlap
            assert len(chunk) <= chunker.chunk_size * 1.5 