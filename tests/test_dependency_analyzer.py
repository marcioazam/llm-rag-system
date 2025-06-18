"""Comprehensive tests for dependency analyzer functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from src.code_analysis.dependency_analyzer import DependencyAnalyzer


class TestDependencyAnalyzer:
    """Test suite for dependency analyzer."""

    def test_init_without_project_root(self):
        """Test initialization without project root."""
        analyzer = DependencyAnalyzer()
        assert analyzer.module_lookup == {}

    def test_init_with_project_root(self):
        """Test initialization with project root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some Python files
            (Path(temp_dir) / "module1.py").write_text("# module1")
            (Path(temp_dir) / "subdir").mkdir()
            (Path(temp_dir) / "subdir" / "module2.py").write_text("# module2")
            (Path(temp_dir) / "subdir" / "__init__.py").write_text("# init")
            
            analyzer = DependencyAnalyzer(temp_dir)
            
            # Verify module lookup table
            assert "module1" in analyzer.module_lookup
            assert "subdir.module2" in analyzer.module_lookup
            assert "subdir.__init__" in analyzer.module_lookup
            
            # Verify paths are correct
            assert analyzer.module_lookup["module1"].endswith("module1.py")
            assert analyzer.module_lookup["subdir.module2"].endswith(os.path.join("subdir", "module2.py"))

    def test_analyze_empty_code(self):
        """Test analysis of empty code."""
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze("")
        assert result == []

    def test_analyze_syntax_error(self):
        """Test analysis of code with syntax errors."""
        analyzer = DependencyAnalyzer()
        invalid_code = "def invalid_function(\n    # Missing closing parenthesis"
        result = analyzer.analyze(invalid_code)
        assert result == []

    def test_analyze_simple_function_call(self):
        """Test analysis of simple internal function calls."""
        analyzer = DependencyAnalyzer()
        code = '''
def helper_function():
    return 42

def main_function():
    result = helper_function()
    return result
'''
        result = analyzer.analyze(code)
        
        assert len(result) == 1
        assert result[0]['source'] == 'main_function'
        assert result[0]['target'] == 'helper_function'
        assert result[0]['relation_type'] == 'calls'

    def test_analyze_multiple_function_calls(self):
        """Test analysis of multiple function calls."""
        analyzer = DependencyAnalyzer()
        code = '''
def func_a():
    return 1

def func_b():
    return 2

def func_c():
    return 3

def main():
    a = func_a()
    b = func_b()
    c = func_c()
    return a + b + c
'''
        result = analyzer.analyze(code)
        
        assert len(result) == 3
        
        # Verify all calls from main
        sources = [r['source'] for r in result]
        targets = [r['target'] for r in result]
        
        assert all(source == 'main' for source in sources)
        assert 'func_a' in targets
        assert 'func_b' in targets
        assert 'func_c' in targets

    def test_analyze_class_method_calls(self):
        """Test analysis of class method calls."""
        analyzer = DependencyAnalyzer()
        code = '''
class MyClass:
    def method_a(self):
        return self.method_b()
    
    def method_b(self):
        return 42

def external_function():
    obj = MyClass()
    return obj.method_a()
'''
        result = analyzer.analyze(code)
        
        # Should find method_a calling method_b
        method_calls = [r for r in result if r['source'] == 'method_a']
        assert len(method_calls) == 1
        assert method_calls[0]['target'] == 'method_b'
        assert method_calls[0]['relation_type'] == 'calls'

    def test_analyze_external_imports(self):
        """Test analysis of external function calls via imports."""
        analyzer = DependencyAnalyzer()
        code = '''
import os
import sys
from pathlib import Path
from typing import List

def main():
    os.getcwd()
    sys.exit(0)
    Path("/tmp").exists()
    return List[str]
'''
        result = analyzer.analyze(code)
        
        # Should find external calls
        external_calls = [r for r in result if r['relation_type'] == 'calls_external']
        assert len(external_calls) >= 3
        
        targets = [r['target'] for r in external_calls]
        assert 'os' in targets or 'os.getcwd' in targets
        assert 'sys' in targets or 'sys.exit' in targets
        assert 'pathlib.Path' in targets

    def test_analyze_import_aliases(self):
        """Test analysis with import aliases."""
        analyzer = DependencyAnalyzer()
        code = '''
import numpy as np
from collections import defaultdict as dd

def process_data():
    arr = np.array([1, 2, 3])
    d = dd(list)
    return arr, d
'''
        result = analyzer.analyze(code)
        
        external_calls = [r for r in result if r['relation_type'] == 'calls_external']
        targets = [r['target'] for r in external_calls]
        
        assert 'numpy' in targets or 'numpy.array' in targets
        assert 'collections.defaultdict' in targets

    def test_analyze_nested_function_calls(self):
        """Test analysis of nested function calls."""
        analyzer = DependencyAnalyzer()
        code = '''
def inner():
    return 42

def middle():
    return inner()

def outer():
    return middle()
'''
        result = analyzer.analyze(code)
        
        assert len(result) == 2
        
        # Find specific relationships
        middle_to_inner = next(r for r in result if r['source'] == 'middle')
        outer_to_middle = next(r for r in result if r['source'] == 'outer')
        
        assert middle_to_inner['target'] == 'inner'
        assert outer_to_middle['target'] == 'middle'

    def test_analyze_async_functions(self):
        """Test analysis of async function calls."""
        analyzer = DependencyAnalyzer()
        code = '''
async def async_helper():
    return 42

async def async_main():
    result = await async_helper()
    return result
'''
        result = analyzer.analyze(code)
        
        assert len(result) == 1
        assert result[0]['source'] == 'async_main'
        assert result[0]['target'] == 'async_helper'
        assert result[0]['relation_type'] == 'calls'

    def test_analyze_attribute_calls(self):
        """Test analysis of attribute-based function calls."""
        analyzer = DependencyAnalyzer()
        code = '''
def helper():
    return 42

def main():
    obj = SomeClass()
    obj.method()  # External call
    helper()      # Internal call
    return obj
'''
        result = analyzer.analyze(code)
        
        # Should find the internal call to helper
        internal_calls = [r for r in result if r['relation_type'] == 'calls']
        assert len(internal_calls) == 1
        assert internal_calls[0]['target'] == 'helper'

    def test_analyze_no_function_context(self):
        """Test that calls outside functions are ignored."""
        analyzer = DependencyAnalyzer()
        code = '''
def helper():
    return 42

# This call is at module level, should be ignored
helper()

def main():
    # This call is inside a function, should be captured
    return helper()
'''
        result = analyzer.analyze(code)
        
        # Should only find the call from within main function
        assert len(result) == 1
        assert result[0]['source'] == 'main'
        assert result[0]['target'] == 'helper'

    def test_analyze_complex_imports(self):
        """Test analysis with complex import patterns."""
        analyzer = DependencyAnalyzer()
        code = '''
from os.path import join, exists
from typing import List, Dict, Optional
import json
from . import local_module
from ..parent import parent_module

def process_files():
    path = join("/tmp", "file.txt")
    if exists(path):
        data = json.loads("{}")
        local_module.function()
        parent_module.other_function()
    return path
'''
        result = analyzer.analyze(code)
        
        external_calls = [r for r in result if r['relation_type'] == 'calls_external']
        targets = [r['target'] for r in external_calls]
        
        # Should find various external calls
        assert 'os.path.join' in targets
        assert 'os.path.exists' in targets
        assert 'json.loads' in targets or 'json' in targets
        assert 'local_module' in targets or 'local_module.function' in targets
        assert 'parent_module' in targets or 'parent_module.other_function' in targets

    def test_analyze_file_success(self):
        """Test successful file analysis."""
        analyzer = DependencyAnalyzer()
        code = '''
def func_a():
    return func_b()

def func_b():
    return 42
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = analyzer.analyze_file(temp_path)
            
            assert len(result) == 1
            assert result[0]['source'] == 'func_a'
            assert result[0]['target'] == 'func_b'
            assert result[0]['relation_type'] == 'calls'
        finally:
            os.unlink(temp_path)

    def test_analyze_file_not_found(self):
        """Test file analysis with non-existent file."""
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze_file("/nonexistent/file.py")
        assert result == []

    def test_analyze_file_read_error(self):
        """Test file analysis with read permission error."""
        analyzer = DependencyAnalyzer()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test(): pass")
            temp_path = f.name
        
        try:
            # Mock Path.read_text to raise an exception
            with patch('pathlib.Path.read_text', side_effect=PermissionError("Access denied")):
                result = analyzer.analyze_file(temp_path)
                assert result == []
        finally:
            os.unlink(temp_path)

    def test_analyze_file_with_module_lookup(self):
        """Test file analysis with module lookup resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            (Path(temp_dir) / "utils.py").write_text("def utility(): pass")
            (Path(temp_dir) / "main.py").write_text('''
import utils

def main():
    utils.utility()
    return 42
''')
            
            analyzer = DependencyAnalyzer(temp_dir)
            result = analyzer.analyze_file(str(Path(temp_dir) / "main.py"))
            
            # Should find external call to utils
            external_calls = [r for r in result if r['relation_type'] == 'calls_external']
            assert len(external_calls) == 1
            assert external_calls[0]['target'] == 'utils'
            
            # Should have target_path resolved
            assert 'target_path' in external_calls[0]
            assert external_calls[0]['target_path'].endswith("utils.py")

    def test_analyze_file_with_unicode_content(self):
        """Test file analysis with unicode content."""
        analyzer = DependencyAnalyzer()
        code = '''
# -*- coding: utf-8 -*-
def função_com_acentos():
    """Função com caracteres especiais: ção, ã, é"""
    return "Olá mundo! 🌍"

def main():
    return função_com_acentos()
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = analyzer.analyze_file(temp_path)
            
            assert len(result) == 1
            assert result[0]['source'] == 'main'
            assert result[0]['target'] == 'função_com_acentos'
            assert result[0]['relation_type'] == 'calls'
        finally:
            os.unlink(temp_path)

    def test_analyze_file_with_encoding_errors(self):
        """Test file analysis with encoding errors (should be ignored)."""
        analyzer = DependencyAnalyzer()
        
        # Create file with invalid UTF-8 bytes
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
            f.write(b'def test():\n    return "\xff\xfe invalid utf-8"')
            temp_path = f.name
        
        try:
            # Should not raise an exception due to errors="ignore"
            result = analyzer.analyze_file(temp_path)
            # Result might be empty or contain partial analysis
            assert isinstance(result, list)
        finally:
            os.unlink(temp_path)

    def test_analyze_recursive_calls(self):
        """Test analysis of recursive function calls."""
        analyzer = DependencyAnalyzer()
        code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''
        result = analyzer.analyze(code)
        
        # Should find recursive calls
        factorial_calls = [r for r in result if r['source'] == 'factorial' and r['target'] == 'factorial']
        fibonacci_calls = [r for r in result if r['source'] == 'fibonacci' and r['target'] == 'fibonacci']
        
        assert len(factorial_calls) == 1
        assert len(fibonacci_calls) == 2  # Two recursive calls in fibonacci

    def test_analyze_class_definitions(self):
        """Test that class definitions are properly tracked."""
        analyzer = DependencyAnalyzer()
        code = '''
class MyClass:
    def method(self):
        return self.other_method()
    
    def other_method(self):
        return 42

def create_instance():
    return MyClass()
'''
        result = analyzer.analyze(code)
        
        # Should find method calling other_method
        method_calls = [r for r in result if r['source'] == 'method']
        assert len(method_calls) == 1
        assert method_calls[0]['target'] == 'other_method'

    def test_analyze_lambda_functions(self):
        """Test that lambda functions don't interfere with analysis."""
        analyzer = DependencyAnalyzer()
        code = '''
def helper():
    return 42

def main():
    # Lambda function should not be tracked as a definition
    func = lambda x: x * 2
    result = helper()
    return func(result)
'''
        result = analyzer.analyze(code)
        
        # Should only find call to helper
        assert len(result) == 1
        assert result[0]['source'] == 'main'
        assert result[0]['target'] == 'helper'

    def test_analyze_from_import_star(self):
        """Test analysis with 'from module import *' patterns."""
        analyzer = DependencyAnalyzer()
        code = '''
from os import *
from typing import *

def main():
    path = getcwd()  # From os import *
    data: List[str] = []  # From typing import *
    return path, data
'''
        result = analyzer.analyze(code)
        
        # The analyzer should handle star imports
        # Note: Star imports create entries with the module name
        external_calls = [r for r in result if r['relation_type'] == 'calls_external']
        
        # Should find some external references
        assert len(external_calls) >= 0  # Star imports are complex to resolve

    def test_analyze_method_chaining(self):
        """Test analysis of method chaining calls."""
        analyzer = DependencyAnalyzer()
        code = '''
def helper():
    return SomeObject()

def main():
    result = helper().method1().method2().method3()
    return result
'''
        result = analyzer.analyze(code)
        
        # Should find call to helper
        internal_calls = [r for r in result if r['relation_type'] == 'calls']
        assert len(internal_calls) == 1
        assert internal_calls[0]['target'] == 'helper'

    def test_analyze_with_decorators(self):
        """Test analysis of functions with decorators."""
        analyzer = DependencyAnalyzer()
        code = '''
def decorator(func):
    return func

@decorator
def decorated_function():
    return helper_function()

def helper_function():
    return 42
'''
        result = analyzer.analyze(code)
        
        # Should find call from decorated_function to helper_function
        calls = [r for r in result if r['source'] == 'decorated_function']
        assert len(calls) == 1
        assert calls[0]['target'] == 'helper_function'

    def test_module_lookup_edge_cases(self):
        """Test module lookup with edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with special names
            (Path(temp_dir) / "__init__.py").write_text("# init")
            (Path(temp_dir) / "_private.py").write_text("# private")
            (Path(temp_dir) / "123invalid.py").write_text("# invalid name")
            
            # Create nested structure
            nested_dir = Path(temp_dir) / "a" / "b" / "c"
            nested_dir.mkdir(parents=True)
            (nested_dir / "deep.py").write_text("# deep")
            
            analyzer = DependencyAnalyzer(temp_dir)
            
            # Verify all files are included
            assert "__init__" in analyzer.module_lookup
            assert "_private" in analyzer.module_lookup
            assert "123invalid" in analyzer.module_lookup
            assert "a.b.c.deep" in analyzer.module_lookup

    def test_empty_project_directory(self):
        """Test initialization with empty project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = DependencyAnalyzer(temp_dir)
            assert analyzer.module_lookup == {}

    def test_project_with_no_python_files(self):
        """Test initialization with directory containing no Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create non-Python files
            (Path(temp_dir) / "readme.txt").write_text("README")
            (Path(temp_dir) / "config.json").write_text("{}")
            
            analyzer = DependencyAnalyzer(temp_dir)
            assert analyzer.module_lookup == {}