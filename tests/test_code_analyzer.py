"""Comprehensive tests for code analyzer functionality."""

import pytest
import ast
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

from src.graphdb.code_analyzer import CodeAnalyzer, _attach_parents, _patched_parse
from src.graphdb.graph_models import NodeType, RelationType, GraphRelation


class TestCodeAnalyzer:
    """Test suite for code analyzer."""

    @pytest.fixture
    def mock_graph_store(self):
        """Mock Neo4jStore for testing."""
        mock_store = Mock()
        mock_store.add_code_element = Mock()
        mock_store.add_relationship = Mock()
        return mock_store

    @pytest.fixture
    def analyzer(self, mock_graph_store):
        """Create CodeAnalyzer instance with mocked store."""
        return CodeAnalyzer(mock_graph_store)

    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing."""
        return '''
import os
import sys
from pathlib import Path
from typing import List, Dict

class BaseClass:
    """Base class for testing."""
    pass

class DerivedClass(BaseClass):
    """Derived class for testing."""
    
    def __init__(self):
        super().__init__()
    
    def method_one(self, param: str) -> str:
        return param.upper()

def standalone_function(x: int, y: int) -> int:
    """Standalone function for testing."""
    return x + y

def another_function():
    """Another function."""
    pass
'''

    @pytest.fixture
    def temp_python_file(self, sample_python_code):
        """Create temporary Python file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_code)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_init(self, mock_graph_store):
        """Test CodeAnalyzer initialization."""
        analyzer = CodeAnalyzer(mock_graph_store)
        assert analyzer.graph_store == mock_graph_store

    def test_analyze_python_file_basic(self, analyzer, temp_python_file, mock_graph_store):
        """Test basic Python file analysis."""
        analyzer.analyze_python_file(temp_python_file)
        
        # Verify file node was created
        file_calls = [call for call in mock_graph_store.add_code_element.call_args_list 
                     if 'file::' in str(call)]
        assert len(file_calls) >= 1
        
        # Verify file node structure
        file_call = file_calls[0]
        file_data = file_call[0][0]
        assert file_data['type'] == NodeType.CODE_FILE.value
        assert file_data['name'] == os.path.basename(temp_python_file)
        assert file_data['file_path'] == temp_python_file
        assert len(file_data['content']) <= 1000  # Content should be truncated

    def test_analyze_python_file_imports(self, analyzer, temp_python_file, mock_graph_store):
        """Test import extraction from Python file."""
        analyzer.analyze_python_file(temp_python_file)
        
        # Check for import nodes
        import_calls = [call for call in mock_graph_store.add_code_element.call_args_list 
                       if 'import::' in str(call)]
        
        # Should have imports for: os, sys, pathlib.Path, typing.List, typing.Dict
        assert len(import_calls) >= 4
        
        # Verify specific imports
        import_names = [call[0][0]['name'] for call in import_calls]
        assert 'os' in import_names
        assert 'sys' in import_names
        assert 'pathlib.Path' in import_names
        assert 'typing.List' in import_names
        assert 'typing.Dict' in import_names

    def test_analyze_python_file_classes(self, analyzer, temp_python_file, mock_graph_store):
        """Test class extraction from Python file."""
        analyzer.analyze_python_file(temp_python_file)
        
        # Check for class nodes
        class_calls = [call for call in mock_graph_store.add_code_element.call_args_list 
                      if 'class::' in str(call) and '@file::' in str(call)]
        
        # Should have BaseClass and DerivedClass
        assert len(class_calls) >= 2
        
        class_names = [call[0][0]['name'] for call in class_calls]
        assert 'BaseClass' in class_names
        assert 'DerivedClass' in class_names
        
        # Verify class node structure
        base_class_call = next(call for call in class_calls 
                              if call[0][0]['name'] == 'BaseClass')
        base_class_data = base_class_call[0][0]
        assert base_class_data['type'] == NodeType.CLASS.value
        assert 'class::BaseClass@file::' in base_class_data['id']

    def test_analyze_python_file_functions(self, analyzer, temp_python_file, mock_graph_store):
        """Test function extraction from Python file."""
        analyzer.analyze_python_file(temp_python_file)
        
        # Check for function nodes (only top-level functions)
        func_calls = [call for call in mock_graph_store.add_code_element.call_args_list 
                     if 'func::' in str(call)]
        
        # Should have standalone_function and another_function
        assert len(func_calls) >= 2
        
        func_names = [call[0][0]['name'] for call in func_calls]
        assert 'standalone_function' in func_names
        assert 'another_function' in func_names
        
        # Should NOT have method_one (it's inside a class)
        assert 'method_one' not in func_names
        assert '__init__' not in func_names

    def test_analyze_python_file_relationships(self, analyzer, temp_python_file, mock_graph_store):
        """Test relationship creation during file analysis."""
        analyzer.analyze_python_file(temp_python_file)
        
        # Check relationship calls
        relationship_calls = mock_graph_store.add_relationship.call_args_list
        assert len(relationship_calls) > 0
        
        # Verify relationship types
        relationship_types = [call[0][0].type for call in relationship_calls]
        assert RelationType.IMPORTS.value in relationship_types
        assert RelationType.CONTAINS.value in relationship_types
        assert RelationType.EXTENDS.value in relationship_types

    def test_analyze_python_file_inheritance(self, analyzer, temp_python_file, mock_graph_store):
        """Test inheritance relationship extraction."""
        analyzer.analyze_python_file(temp_python_file)
        
        # Find EXTENDS relationships
        extends_calls = [call for call in mock_graph_store.add_relationship.call_args_list 
                        if call[0][0].type == RelationType.EXTENDS.value]
        
        assert len(extends_calls) >= 1
        
        # Verify DerivedClass extends BaseClass
        extends_rel = extends_calls[0][0][0]
        assert 'DerivedClass' in extends_rel.source_id
        assert 'BaseClass' in extends_rel.target_id

    def test_analyze_python_file_unicode_error(self, analyzer, mock_graph_store):
        """Test handling of files with unicode decode errors."""
        # Create a file with invalid encoding
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
            f.write(b'\xff\xfe# Invalid UTF-8')
            temp_path = f.name
        
        try:
            # Should not raise an exception
            analyzer.analyze_python_file(temp_path)
            
            # Should not have created any nodes
            assert mock_graph_store.add_code_element.call_count == 0
        finally:
            os.unlink(temp_path)

    def test_analyze_python_file_syntax_error(self, analyzer, mock_graph_store):
        """Test handling of files with syntax errors."""
        invalid_code = '''
def invalid_function(
    # Missing closing parenthesis and colon
    pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(invalid_code)
            temp_path = f.name
        
        try:
            # Should not raise an exception
            analyzer.analyze_python_file(temp_path)
            
            # Should not have created any nodes
            assert mock_graph_store.add_code_element.call_count == 0
        finally:
            os.unlink(temp_path)

    def test_analyze_python_file_nonexistent(self, analyzer, mock_graph_store):
        """Test handling of non-existent files."""
        # Should not raise an exception
        analyzer.analyze_python_file("/nonexistent/file.py")
        
        # Should not have created any nodes
        assert mock_graph_store.add_code_element.call_count == 0

    def test_analyze_project(self, analyzer, mock_graph_store):
        """Test project-wide analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple Python files
            files_content = {
                'module1.py': 'def func1(): pass',
                'module2.py': 'class Class1: pass',
                'subdir/module3.py': 'import os\ndef func2(): pass',
                '.hidden.py': 'def hidden(): pass',  # Should be ignored
                'not_python.txt': 'not python code'  # Should be ignored
            }
            
            for file_path, content in files_content.items():
                full_path = Path(temp_dir) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
            
            # Mock ThreadPoolExecutor to run synchronously
            with patch('src.graphdb.code_analyzer.ThreadPoolExecutor') as mock_executor:
                mock_executor.return_value.__enter__.return_value.submit = lambda func, *args: self._create_completed_future(func(*args))
                mock_executor.return_value.__enter__.return_value.__exit__ = Mock()
                
                with patch('src.graphdb.code_analyzer.as_completed') as mock_as_completed:
                    mock_as_completed.return_value = []
                    
                    analyzer.analyze_project(temp_dir)
            
            # Should have processed 3 Python files (excluding hidden and non-Python)
            # Note: The exact count depends on the mocking, but we can verify some calls were made
            assert mock_graph_store.add_code_element.call_count > 0

    def _create_completed_future(self, result):
        """Helper to create a completed future for testing."""
        future = Future()
        future.set_result(result)
        return future

    def test_add_import(self, analyzer, mock_graph_store):
        """Test import node and relationship creation."""
        file_id = "file::/test/file.py"
        import_name = "numpy"
        
        analyzer._add_import(file_id, import_name)
        
        # Verify import node creation
        import_calls = [call for call in mock_graph_store.add_code_element.call_args_list 
                       if 'import::' in str(call)]
        assert len(import_calls) == 1
        
        import_data = import_calls[0][0][0]
        assert import_data['id'] == f"import::{import_name}"
        assert import_data['name'] == import_name
        assert import_data['type'] == "Import"
        
        # Verify relationship creation
        rel_calls = mock_graph_store.add_relationship.call_args_list
        assert len(rel_calls) == 1
        
        rel = rel_calls[0][0][0]
        assert rel.source_id == file_id
        assert rel.target_id == f"import::{import_name}"
        assert rel.type == RelationType.IMPORTS.value

    def test_add_class(self, analyzer, mock_graph_store):
        """Test class node and relationship creation."""
        file_id = "file::/test/file.py"
        
        # Create a mock AST ClassDef node
        class_node = ast.ClassDef(
            name="TestClass",
            bases=[ast.Name(id="BaseClass", ctx=ast.Load())],
            keywords=[],
            decorator_list=[],
            body=[ast.Pass()]
        )
        
        analyzer._add_class(file_id, class_node)
        
        # Verify class node creation
        class_calls = [call for call in mock_graph_store.add_code_element.call_args_list 
                      if 'class::TestClass@' in str(call)]
        assert len(class_calls) == 1
        
        class_data = class_calls[0][0][0]
        assert class_data['name'] == "TestClass"
        assert class_data['type'] == NodeType.CLASS.value
        
        # Verify base class node creation
        base_calls = [call for call in mock_graph_store.add_code_element.call_args_list 
                     if 'class::BaseClass' in str(call) and '@' not in str(call)]
        assert len(base_calls) == 1
        
        # Verify relationships (CONTAINS and EXTENDS)
        rel_calls = mock_graph_store.add_relationship.call_args_list
        assert len(rel_calls) == 2
        
        rel_types = [call[0][0].type for call in rel_calls]
        assert RelationType.CONTAINS.value in rel_types
        assert RelationType.EXTENDS.value in rel_types

    def test_add_function(self, analyzer, mock_graph_store):
        """Test function node and relationship creation."""
        file_id = "file::/test/file.py"
        
        # Create a mock AST FunctionDef node
        func_node = ast.FunctionDef(
            name="test_function",
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
                kw_defaults=[], kwarg=None, defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        
        analyzer._add_function(file_id, func_node)
        
        # Verify function node creation
        func_calls = [call for call in mock_graph_store.add_code_element.call_args_list 
                     if 'func::test_function@' in str(call)]
        assert len(func_calls) == 1
        
        func_data = func_calls[0][0][0]
        assert func_data['name'] == "test_function"
        assert func_data['type'] == NodeType.FUNCTION.value
        
        # Verify CONTAINS relationship
        rel_calls = mock_graph_store.add_relationship.call_args_list
        assert len(rel_calls) == 1
        
        rel = rel_calls[0][0][0]
        assert rel.source_id == file_id
        assert rel.type == RelationType.CONTAINS.value

    def test_resolve_name_simple(self, analyzer):
        """Test resolving simple names from AST nodes."""
        # Test ast.Name
        name_node = ast.Name(id="simple_name", ctx=ast.Load())
        result = analyzer._resolve_name(name_node)
        assert result == "simple_name"
        
        # Test None case
        result = analyzer._resolve_name(ast.Pass())
        assert result is None

    def test_resolve_name_attribute(self, analyzer):
        """Test resolving attribute names from AST nodes."""
        # Create ast.Attribute node for "module.submodule.Class"
        attr_node = ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="module", ctx=ast.Load()),
                attr="submodule",
                ctx=ast.Load()
            ),
            attr="Class",
            ctx=ast.Load()
        )
        
        result = analyzer._resolve_name(attr_node)
        assert result == "module.submodule.Class"

    def test_resolve_name_complex_attribute(self, analyzer):
        """Test resolving complex attribute chains."""
        # Create a more complex attribute chain
        attr_node = ast.Attribute(
            value=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="a", ctx=ast.Load()),
                    attr="b",
                    ctx=ast.Load()
                ),
                attr="c",
                ctx=ast.Load()
            ),
            attr="d",
            ctx=ast.Load()
        )
        
        result = analyzer._resolve_name(attr_node)
        assert result == "a.b.c.d"

    def test_resolve_name_invalid_attribute(self, analyzer):
        """Test resolving attribute with invalid base."""
        # Attribute with non-Name base
        attr_node = ast.Attribute(
            value=ast.Constant(value=42),
            attr="invalid",
            ctx=ast.Load()
        )
        
        result = analyzer._resolve_name(attr_node)
        assert result is None

    def test_attach_parents(self):
        """Test AST parent attachment functionality."""
        code = '''
def func():
    class InnerClass:
        pass
    return 42
'''
        
        tree = ast.parse(code)
        _attach_parents(tree)
        
        # Find the function and class nodes
        func_node = None
        class_node = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_node = node
            elif isinstance(node, ast.ClassDef):
                class_node = node
        
        assert func_node is not None
        assert class_node is not None
        
        # Verify parent relationships
        assert hasattr(class_node, 'parent')
        assert class_node.parent == func_node
        assert hasattr(func_node, 'parent')
        assert isinstance(func_node.parent, ast.Module)

    def test_patched_parse(self):
        """Test that the patched parse function attaches parents."""
        code = 'def func(): pass'
        
        tree = _patched_parse(code)
        
        # Find the function node
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_node = node
                break
        
        assert func_node is not None
        assert hasattr(func_node, 'parent')
        assert isinstance(func_node.parent, ast.Module)

    def test_analyze_file_with_complex_imports(self, analyzer, mock_graph_store):
        """Test analysis of file with complex import patterns."""
        complex_code = '''
import os, sys
from pathlib import Path, PurePath
from typing import List, Dict, Optional
from . import local_module
from ..parent import parent_module
import numpy as np
from collections import defaultdict as dd
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_code)
            temp_path = f.name
        
        try:
            analyzer.analyze_python_file(temp_path)
            
            # Check for various import patterns
            import_calls = [call for call in mock_graph_store.add_code_element.call_args_list 
                           if 'import::' in str(call)]
            
            import_names = [call[0][0]['name'] for call in import_calls]
            
            # Verify different import types
            assert 'os' in import_names
            assert 'sys' in import_names
            assert 'pathlib.Path' in import_names
            assert 'pathlib.PurePath' in import_names
            assert 'typing.List' in import_names
            assert 'typing.Dict' in import_names
            assert 'typing.Optional' in import_names
            assert 'local_module' in import_names
            assert 'parent_module' in import_names
            assert 'numpy' in import_names
            assert 'collections.defaultdict' in import_names
            
        finally:
            os.unlink(temp_path)

    def test_analyze_file_with_multiple_inheritance(self, analyzer, mock_graph_store):
        """Test analysis of class with multiple inheritance."""
        inheritance_code = '''
class Base1:
    pass

class Base2:
    pass

class MultipleInheritance(Base1, Base2):
    pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(inheritance_code)
            temp_path = f.name
        
        try:
            analyzer.analyze_python_file(temp_path)
            
            # Check for EXTENDS relationships
            extends_calls = [call for call in mock_graph_store.add_relationship.call_args_list 
                            if call[0][0].type == RelationType.EXTENDS.value]
            
            # Should have 2 EXTENDS relationships (to Base1 and Base2)
            assert len(extends_calls) >= 2
            
            # Verify the relationships
            target_ids = [call[0][0].target_id for call in extends_calls]
            assert any('Base1' in target_id for target_id in target_ids)
            assert any('Base2' in target_id for target_id in target_ids)
            
        finally:
            os.unlink(temp_path)

    def test_analyze_project_with_cpu_count_none(self, analyzer, mock_graph_store):
        """Test project analysis when os.cpu_count() returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple Python file
            test_file = Path(temp_dir) / 'test.py'
            test_file.write_text('def test(): pass')
            
            with patch('os.cpu_count', return_value=None):
                with patch('src.graphdb.code_analyzer.ThreadPoolExecutor') as mock_executor:
                    mock_executor.return_value.__enter__.return_value.submit = lambda func, *args: self._create_completed_future(func(*args))
                    mock_executor.return_value.__enter__.return_value.__exit__ = Mock()
                    
                    with patch('src.graphdb.code_analyzer.as_completed') as mock_as_completed:
                        mock_as_completed.return_value = []
                        
                        analyzer.analyze_project(temp_dir)
                    
                    # Verify ThreadPoolExecutor was called with max_workers=4 (fallback)
                    mock_executor.assert_called_with(max_workers=4)

    def test_empty_project_analysis(self, analyzer, mock_graph_store):
        """Test analysis of empty project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory
            analyzer.analyze_project(temp_dir)
            
            # Should not have created any nodes
            assert mock_graph_store.add_code_element.call_count == 0
            assert mock_graph_store.add_relationship.call_count == 0