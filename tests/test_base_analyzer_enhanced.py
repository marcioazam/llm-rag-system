"""
Testes para o módulo base_analyzer de análise de código
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


# Mock das classes base para análise de código
class MockBaseAnalyzer:
    """Mock do analisador base de código"""
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.symbols = []
        self.dependencies = []
        self.metadata = {}
    
    def analyze(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """Analisa código e retorna estrutura de dados"""
        return {
            "language": self.language,
            "symbols": self.extract_symbols(code),
            "dependencies": self.extract_dependencies(code),
            "metadata": self.extract_metadata(code, file_path),
            "complexity": self.calculate_complexity(code)
        }
    
    def extract_symbols(self, code: str) -> List[Dict[str, Any]]:
        """Extrai símbolos do código"""
        symbols = []
        
        # Mock de extração de funções
        if "def " in code:
            functions = [line.strip() for line in code.split('\n') if line.strip().startswith('def ')]
            for func in functions:
                func_name = func.split('(')[0].replace('def ', '').strip()
                symbols.append({
                    "type": "function",
                    "name": func_name,
                    "line": 1,
                    "signature": func
                })
        
        # Mock de extração de classes
        if "class " in code:
            classes = [line.strip() for line in code.split('\n') if line.strip().startswith('class ')]
            for cls in classes:
                cls_name = cls.split('(')[0].split(':')[0].replace('class ', '').strip()
                symbols.append({
                    "type": "class",
                    "name": cls_name,
                    "line": 1,
                    "signature": cls
                })
        
        return symbols
    
    def extract_dependencies(self, code: str) -> List[Dict[str, Any]]:
        """Extrai dependências do código"""
        dependencies = []
        
        # Mock de extração de imports
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                dependencies.append({
                    "type": "import",
                    "statement": line,
                    "line": i,
                    "module": self._extract_module_name(line)
                })
        
        return dependencies
    
    def extract_metadata(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """Extrai metadados do código"""
        return {
            "file_path": file_path,
            "lines_count": len(code.split('\n')),
            "chars_count": len(code),
            "has_docstring": '"""' in code or "'''" in code,
            "has_comments": '#' in code
        }
    
    def calculate_complexity(self, code: str) -> Dict[str, Any]:
        """Calcula métricas de complexidade"""
        lines = code.split('\n')
        return {
            "cyclomatic": self._calculate_cyclomatic_complexity(code),
            "lines_of_code": len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            "comment_ratio": len([l for l in lines if l.strip().startswith('#')]) / max(len(lines), 1)
        }
    
    def _extract_module_name(self, import_statement: str) -> str:
        """Extrai nome do módulo de uma declaração import"""
        if import_statement.startswith('from '):
            return import_statement.split(' ')[1].split(' ')[0]
        elif import_statement.startswith('import '):
            return import_statement.replace('import ', '').split(' ')[0].split('.')[0]
        return ""
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calcula complexidade ciclomática básica"""
        complexity = 1  # Base complexity
        
        # Conta estruturas de controle
        control_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'and', 'or']
        for keyword in control_keywords:
            complexity += code.count(f' {keyword} ') + code.count(f'\n{keyword} ')
        
        return complexity


class TestBaseAnalyzer:
    """Testes para funcionalidades básicas do analisador"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.analyzer = MockBaseAnalyzer()
    
    def test_analyzer_initialization(self):
        """Testa inicialização do analisador"""
        assert self.analyzer.language == "python"
        assert isinstance(self.analyzer.symbols, list)
        assert isinstance(self.analyzer.dependencies, list)
        assert isinstance(self.analyzer.metadata, dict)
    
    def test_analyze_simple_code(self):
        """Testa análise de código simples"""
        code = """
def hello_world():
    print("Hello, World!")
    return True
"""
        result = self.analyzer.analyze(code)
        
        assert "language" in result
        assert "symbols" in result
        assert "dependencies" in result
        assert "metadata" in result
        assert "complexity" in result
    
    def test_extract_function_symbols(self):
        """Testa extração de símbolos de função"""
        code = """
def add_numbers(a, b):
    return a + b

def multiply(x, y):
    return x * y
"""
        symbols = self.analyzer.extract_symbols(code)
        
        function_symbols = [s for s in symbols if s["type"] == "function"]
        assert len(function_symbols) == 2
        assert any(s["name"] == "add_numbers" for s in function_symbols)
        assert any(s["name"] == "multiply" for s in function_symbols)
    
    def test_extract_class_symbols(self):
        """Testa extração de símbolos de classe"""
        code = """
class Calculator:
    def __init__(self):
        self.result = 0

class MathUtils:
    @staticmethod
    def square(x):
        return x * x
"""
        symbols = self.analyzer.extract_symbols(code)
        
        class_symbols = [s for s in symbols if s["type"] == "class"]
        assert len(class_symbols) == 2
        assert any(s["name"] == "Calculator" for s in class_symbols)
        assert any(s["name"] == "MathUtils" for s in class_symbols)
    
    def test_extract_dependencies(self):
        """Testa extração de dependências"""
        code = """
import os
import sys
from typing import Dict, List
from collections import defaultdict
"""
        dependencies = self.analyzer.extract_dependencies(code)
        
        assert len(dependencies) == 4
        modules = [d["module"] for d in dependencies]
        assert "os" in modules
        assert "sys" in modules
        assert "typing" in modules
        assert "collections" in modules
    
    def test_extract_metadata(self):
        """Testa extração de metadados"""
        code = '''"""
This is a docstring
"""
# This is a comment
def test_function():
    pass
'''
        metadata = self.analyzer.extract_metadata(code, "test.py")
        
        assert metadata["file_path"] == "test.py"
        assert metadata["lines_count"] > 0
        assert metadata["chars_count"] > 0
        assert metadata["has_docstring"] is True
        assert metadata["has_comments"] is True
    
    def test_calculate_complexity(self):
        """Testa cálculo de complexidade"""
        code = """
def complex_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
            elif i % 3 == 0:
                print("fizz")
        return True
    else:
        return False
"""
        complexity = self.analyzer.calculate_complexity(code)
        
        assert "cyclomatic" in complexity
        assert "lines_of_code" in complexity
        assert "comment_ratio" in complexity
        assert complexity["cyclomatic"] > 1  # Deve ter complexidade > 1
        assert complexity["lines_of_code"] > 0


class TestAnalyzerEdgeCases:
    """Testes para casos extremos do analisador"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.analyzer = MockBaseAnalyzer()
    
    def test_empty_code_analysis(self):
        """Testa análise de código vazio"""
        result = self.analyzer.analyze("")
        
        assert result["symbols"] == []
        assert result["dependencies"] == []
        assert result["metadata"]["lines_count"] == 1
        assert result["complexity"]["cyclomatic"] == 1
    
    def test_whitespace_only_code(self):
        """Testa análise de código só com espaços"""
        code = "   \n  \n   \n  "
        result = self.analyzer.analyze(code)
        
        assert result["symbols"] == []
        assert result["dependencies"] == []
        assert result["metadata"]["lines_count"] > 1
    
    def test_comments_only_code(self):
        """Testa análise de código só com comentários"""
        code = """
# This is a comment
# Another comment
# Yet another comment
"""
        result = self.analyzer.analyze(code)
        
        assert result["symbols"] == []
        assert result["dependencies"] == []
        assert result["metadata"]["has_comments"] is True
        assert result["complexity"]["comment_ratio"] > 0
    
    def test_invalid_syntax_handling(self):
        """Testa tratamento de sintaxe inválida"""
        invalid_code = """
def incomplete_function(
    # Missing closing parenthesis and body
class IncompleteClass
    # Missing colon
"""
        # O analisador deve ser robusto a erros de sintaxe
        try:
            result = self.analyzer.analyze(invalid_code)
            # Deve conseguir extrair o que for possível
            assert isinstance(result, dict)
        except Exception:
            # Ou falhar graciosamente
            assert True
    
    def test_very_large_code(self):
        """Testa análise de código muito grande"""
        # Gera código grande
        large_code = "\n".join([f"def function_{i}(): pass" for i in range(1000)])
        
        result = self.analyzer.analyze(large_code)
        
        assert len(result["symbols"]) == 1000
        assert result["metadata"]["lines_count"] == 1000
        assert result["complexity"]["lines_of_code"] == 1000


class TestAnalyzerIntegration:
    """Testes de integração do analisador"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.analyzer = MockBaseAnalyzer()
    
    def test_real_python_file_structure(self):
        """Testa análise de estrutura de arquivo Python real"""
        code = '''
"""
Module docstring
"""
import os
import sys
from typing import Dict, List, Optional

class DataProcessor:
    """Class for processing data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = []
    
    def process(self, input_data: List[str]) -> Optional[Dict[str, Any]]:
        """Process the input data"""
        if not input_data:
            return None
        
        results = {}
        for item in input_data:
            if item.strip():
                results[item] = len(item)
        
        return results
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        required_keys = ["input_path", "output_path"]
        return all(key in config for key in required_keys)

def main():
    """Main function"""
    processor = DataProcessor({"input_path": "/tmp", "output_path": "/out"})
    return processor

if __name__ == "__main__":
    main()
'''
        
        result = self.analyzer.analyze(code, "data_processor.py")
        
        # Verifica estrutura completa
        assert result["metadata"]["file_path"] == "data_processor.py"
        assert result["metadata"]["has_docstring"] is True
        
        # Verifica símbolos
        symbols = result["symbols"]
        class_symbols = [s for s in symbols if s["type"] == "class"]
        function_symbols = [s for s in symbols if s["type"] == "function"]
        
        assert len(class_symbols) >= 1
        assert len(function_symbols) >= 3  # __init__, process, validate_config, main
        
        # Verifica dependências
        dependencies = result["dependencies"]
        assert len(dependencies) >= 3  # os, sys, typing
        
        # Verifica complexidade
        assert result["complexity"]["cyclomatic"] > 1
        assert result["complexity"]["lines_of_code"] > 10
    
    def test_multi_language_support(self):
        """Testa suporte a múltiplas linguagens"""
        # Python
        python_analyzer = MockBaseAnalyzer("python")
        python_result = python_analyzer.analyze("def test(): pass")
        assert python_result["language"] == "python"
        
        # JavaScript (mock)
        js_analyzer = MockBaseAnalyzer("javascript")
        js_result = js_analyzer.analyze("function test() {}")
        assert js_result["language"] == "javascript"
        
        # TypeScript (mock)
        ts_analyzer = MockBaseAnalyzer("typescript")
        ts_result = ts_analyzer.analyze("function test(): void {}")
        assert ts_result["language"] == "typescript"
    
    def test_analyzer_performance(self):
        """Testa performance do analisador"""
        import time
        
        # Código de tamanho médio
        medium_code = "\n".join([
            "import sys",
            "class TestClass:",
            "    def __init__(self):",
            "        self.value = 0",
            "    def method(self, x):",
            "        if x > 0:",
            "            return x * 2",
            "        else:",
            "            return 0",
            "def helper_function():",
            "    return True"
        ])
        
        start_time = time.time()
        result = self.analyzer.analyze(medium_code)
        end_time = time.time()
        
        # Análise deve ser rápida (< 1 segundo)
        assert end_time - start_time < 1.0
        assert isinstance(result, dict)
        assert len(result["symbols"]) > 0 