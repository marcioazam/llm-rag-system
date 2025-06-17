import pytest
import ast
from unittest.mock import patch

from src.code_analysis.python_analyzer import PythonAnalyzer


class TestPythonAnalyzer:
    """Testes para a classe PythonAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Cria uma instância do analisador para testes."""
        return PythonAnalyzer()

    def test_language_property(self, analyzer):
        """Testa a propriedade language."""
        assert analyzer.language == "python"

    def test_extract_symbols_functions(self, analyzer):
        """Testa extração de símbolos de funções."""
        code = """
def function1():
    pass

def function2(param1, param2):
    return param1 + param2

def function_with_docstring():
    \"\"\"Esta é uma função com docstring.\"\"\"
    pass
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Deve encontrar 3 funções
        functions = [s for s in symbols if s["type"] == "function"]
        assert len(functions) == 3
        
        # Verificar nomes das funções
        function_names = [f["name"] for f in functions]
        assert "function1" in function_names
        assert "function2" in function_names
        assert "function_with_docstring" in function_names
        
        # Verificar números de linha
        for func in functions:
            assert "line" in func
            assert isinstance(func["line"], int)
            assert func["line"] > 0

    def test_extract_symbols_async_functions(self, analyzer):
        """Testa extração de símbolos de funções assíncronas."""
        code = """
import asyncio

async def async_function1():
    await asyncio.sleep(1)

async def async_function2(param):
    return await some_async_operation(param)
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Deve encontrar 2 funções assíncronas
        async_functions = [s for s in symbols if s["type"] == "async_function"]
        assert len(async_functions) == 2
        
        # Verificar nomes
        async_names = [f["name"] for f in async_functions]
        assert "async_function1" in async_names
        assert "async_function2" in async_names

    def test_extract_symbols_classes(self, analyzer):
        """Testa extração de símbolos de classes."""
        code = """
class SimpleClass:
    pass

class ClassWithMethods:
    def __init__(self):
        self.value = 0
    
    def method1(self):
        return self.value
    
    @staticmethod
    def static_method():
        return "static"

class InheritedClass(ClassWithMethods):
    def method2(self):
        return super().method1() + 1
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Deve encontrar 3 classes
        classes = [s for s in symbols if s["type"] == "class"]
        assert len(classes) == 3
        
        # Verificar nomes das classes
        class_names = [c["name"] for c in classes]
        assert "SimpleClass" in class_names
        assert "ClassWithMethods" in class_names
        assert "InheritedClass" in class_names
        
        # Deve também encontrar métodos como funções
        functions = [s for s in symbols if s["type"] == "function"]
        method_names = [f["name"] for f in functions]
        assert "__init__" in method_names
        assert "method1" in method_names
        assert "static_method" in method_names
        assert "method2" in method_names

    def test_extract_symbols_mixed_code(self, analyzer):
        """Testa extração de símbolos em código misto."""
        code = """
# Importações
import os
from typing import List, Dict

# Constante global
GLOBAL_CONSTANT = "test"

# Função standalone
def utility_function(data: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in data}

# Classe principal
class MainClass:
    def __init__(self, name: str):
        self.name = name
    
    async def async_method(self):
        await asyncio.sleep(0.1)
        return self.name
    
    @property
    def display_name(self):
        return f"Name: {self.name}"

# Função aninhada
def outer_function():
    def inner_function():
        return "inner"
    return inner_function()

# Função lambda (não deve ser capturada como símbolo nomeado)
lambda_func = lambda x: x * 2
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Verificar tipos de símbolos encontrados
        symbol_types = [s["type"] for s in symbols]
        assert "function" in symbol_types
        assert "async_function" in symbol_types
        assert "class" in symbol_types
        
        # Verificar símbolos específicos
        symbol_names = [s["name"] for s in symbols]
        assert "utility_function" in symbol_names
        assert "MainClass" in symbol_names
        assert "__init__" in symbol_names
        assert "async_method" in symbol_names
        assert "display_name" in symbol_names
        assert "outer_function" in symbol_names
        assert "inner_function" in symbol_names

    def test_extract_symbols_empty_code(self, analyzer):
        """Testa extração de símbolos em código vazio."""
        symbols = analyzer.extract_symbols("")
        assert symbols == []
        
        symbols = analyzer.extract_symbols("   \n\n   ")
        assert symbols == []

    def test_extract_symbols_comments_only(self, analyzer):
        """Testa extração de símbolos em código apenas com comentários."""
        code = """
# Este é um comentário
# Outro comentário
\"\"\"Docstring no nível do módulo\"\"\"

# Mais comentários
"""
        
        symbols = analyzer.extract_symbols(code)
        assert symbols == []

    def test_extract_symbols_syntax_error(self, analyzer):
        """Testa tratamento de erro de sintaxe."""
        invalid_code = """
def function_with_syntax_error(
    # Parênteses não fechados
    pass
"""
        
        symbols = analyzer.extract_symbols(invalid_code)
        # Deve retornar lista vazia em caso de erro de sintaxe
        assert symbols == []

    def test_extract_symbols_complex_syntax(self, analyzer):
        """Testa extração em código com sintaxe complexa."""
        code = """
from typing import TypeVar, Generic, Protocol

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T):
        self._value = value
    
    def get(self) -> T:
        return self._value

class Drawable(Protocol):
    def draw(self) -> None: ...

def process_items(*args, **kwargs):
    for arg in args:
        print(arg)
    for key, value in kwargs.items():
        print(f"{key}: {value}")

@decorator
def decorated_function():
    pass

class MetaClass(type):
    def __new__(cls, name, bases, attrs):
        return super().__new__(cls, name, bases, attrs)
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Verificar que símbolos complexos são capturados
        symbol_names = [s["name"] for s in symbols]
        assert "Container" in symbol_names
        assert "Drawable" in symbol_names
        assert "process_items" in symbol_names
        assert "decorated_function" in symbol_names
        assert "MetaClass" in symbol_names
        assert "__new__" in symbol_names

    def test_extract_symbols_line_numbers(self, analyzer):
        """Testa precisão dos números de linha."""
        code = """
# Linha 1: comentário

def first_function():  # Linha 3
    pass

# Linha 6: comentário

class TestClass:  # Linha 8
    def method(self):  # Linha 9
        pass

async def async_func():  # Linha 12
    pass
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Verificar números de linha específicos
        for symbol in symbols:
            if symbol["name"] == "first_function":
                assert symbol["line"] == 4
            elif symbol["name"] == "TestClass":
                assert symbol["line"] == 9
            elif symbol["name"] == "method":
                assert symbol["line"] == 10
            elif symbol["name"] == "async_func":
                assert symbol["line"] == 13

    def test_extract_symbols_nested_classes(self, analyzer):
        """Testa extração de classes aninhadas."""
        code = """
class OuterClass:
    def outer_method(self):
        pass
    
    class InnerClass:
        def inner_method(self):
            pass
        
        class DeeplyNestedClass:
            def deeply_nested_method(self):
                pass
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Deve encontrar todas as classes
        classes = [s for s in symbols if s["type"] == "class"]
        class_names = [c["name"] for c in classes]
        
        assert "OuterClass" in class_names
        assert "InnerClass" in class_names
        assert "DeeplyNestedClass" in class_names
        
        # Deve encontrar todos os métodos
        functions = [s for s in symbols if s["type"] == "function"]
        method_names = [f["name"] for f in functions]
        
        assert "outer_method" in method_names
        assert "inner_method" in method_names
        assert "deeply_nested_method" in method_names

    def test_extract_symbols_special_methods(self, analyzer):
        """Testa extração de métodos especiais."""
        code = """
class SpecialMethodsClass:
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"SpecialMethodsClass({self.value})"
    
    def __len__(self):
        return len(str(self.value))
    
    def __getitem__(self, key):
        return str(self.value)[key]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Verificar métodos especiais
        functions = [s for s in symbols if s["type"] == "function"]
        method_names = [f["name"] for f in functions]
        
        special_methods = [
            "__init__", "__str__", "__repr__", "__len__",
            "__getitem__", "__enter__", "__exit__"
        ]
        
        for method in special_methods:
            assert method in method_names

    def test_extract_symbols_generators_and_comprehensions(self, analyzer):
        """Testa que geradores e comprehensions não são capturados como símbolos."""
        code = """
def generator_function():
    for i in range(10):
        yield i

def function_with_comprehensions():
    list_comp = [x for x in range(10)]
    dict_comp = {x: x**2 for x in range(5)}
    set_comp = {x for x in range(3)}
    gen_exp = (x for x in range(7))
    return list_comp, dict_comp, set_comp, gen_exp
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Deve encontrar apenas as funções nomeadas
        function_names = [s["name"] for s in symbols if s["type"] == "function"]
        assert "generator_function" in function_names
        assert "function_with_comprehensions" in function_names
        
        # Não deve encontrar comprehensions como símbolos
        assert len([s for s in symbols if s["type"] == "function"]) == 2

    def test_inheritance_from_base_analyzer(self, analyzer):
        """Testa que PythonAnalyzer herda corretamente de BaseStaticAnalyzer."""
        # Verificar que é uma subclasse
        from src.code_analysis.base_analyzer import BaseStaticAnalyzer
        assert isinstance(analyzer, BaseStaticAnalyzer)
        
        # Verificar que implementa os métodos obrigatórios
        assert hasattr(analyzer, 'extract_symbols')
        assert callable(analyzer.extract_symbols)
        
        # Verificar que o método é implementado (não abstrato)
        assert analyzer.extract_symbols.__func__ is not BaseStaticAnalyzer.extract_symbols

    def test_ast_node_types_coverage(self, analyzer):
        """Testa cobertura de diferentes tipos de nós AST."""
        code = """
# Diferentes tipos de definições que devem ser capturadas
def regular_function():
    pass

async def async_function():
    pass

class RegularClass:
    pass

class ClassWithBases(object):
    pass

class ClassWithMultipleBases(dict, object):
    pass
"""
        
        symbols = analyzer.extract_symbols(code)
        
        # Verificar tipos específicos
        types_found = set(s["type"] for s in symbols)
        assert "function" in types_found
        assert "async_function" in types_found
        assert "class" in types_found
        
        # Verificar que todas as definições foram encontradas
        names_found = set(s["name"] for s in symbols)
        expected_names = {
            "regular_function", "async_function", "RegularClass",
            "ClassWithBases", "ClassWithMultipleBases"
        }
        assert expected_names.issubset(names_found)

    def test_performance_large_file(self, analyzer):
        """Testa performance com arquivo grande."""
        # Gerar código grande programaticamente
        large_code_parts = []
        
        for i in range(100):
            large_code_parts.append(f"""
def function_{i}(param1, param2):
    \"\"\"Função número {i}.\"\"\"
    result = param1 + param2
    return result * {i}

class Class_{i}:
    def __init__(self):
        self.value = {i}
    
    def method_{i}(self):
        return self.value * 2
""")
        
        large_code = "\n".join(large_code_parts)
        
        # Deve processar sem erros
        symbols = analyzer.extract_symbols(large_code)
        
        # Verificar que encontrou o número esperado de símbolos
        functions = [s for s in symbols if s["type"] == "function"]
        classes = [s for s in symbols if s["type"] == "class"]
        
        # 100 funções + 100 __init__ + 100 métodos = 300 funções
        assert len(functions) == 300
        # 100 classes
        assert len(classes) == 100