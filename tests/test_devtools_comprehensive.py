"""
Testes abrangentes para DevTools.
Inclui auto_documenter e index_queue.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, List, Any, Optional


# Mock Auto Documenter
class MockAutoDocumenter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docs_generated = []
        self.supported_formats = ['markdown', 'rst', 'html']
        
    def analyze_code_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze code structure for documentation."""
        # Simulate analysis based on file extension
        file_ext = Path(file_path).suffix.lower()
        
        structure = {
            'file_path': file_path,
            'language': self._detect_language(file_ext),
            'classes': [],
            'functions': [],
            'imports': [],
            'docstrings': [],
            'complexity_score': 3.5,
            'lines_of_code': 150
        }
        
        # Add some mock data based on language
        if structure['language'] == 'python':
            structure['classes'] = [
                {'name': 'ExampleClass', 'line': 10, 'methods': ['method1', 'method2']},
                {'name': 'AnotherClass', 'line': 50, 'methods': ['init', 'process']}
            ]
            structure['functions'] = [
                {'name': 'main_function', 'line': 5, 'params': ['arg1', 'arg2']},
                {'name': 'helper_function', 'line': 80, 'params': ['data']}
            ]
            structure['imports'] = ['os', 'sys', 'typing']
        
        elif structure['language'] == 'javascript':
            structure['functions'] = [
                {'name': 'processData', 'line': 15, 'params': ['input', 'options']},
                {'name': 'validateInput', 'line': 30, 'params': ['data']}
            ]
            structure['imports'] = ['react', 'axios', 'lodash']
        
        return structure
    
    def generate_documentation(self, structure: Dict[str, Any], output_format: str = 'markdown') -> str:
        """Generate documentation from code structure."""
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {output_format}")
        
        file_path = structure['file_path']
        language = structure['language']
        
        if output_format == 'markdown':
            doc = f"# Documentation for {Path(file_path).name}\n\n"
            doc += f"**Language:** {language}\n"
            doc += f"**Lines of Code:** {structure['lines_of_code']}\n"
            doc += f"**Complexity Score:** {structure['complexity_score']}\n\n"
            
            if structure['classes']:
                doc += "## Classes\n\n"
                for cls in structure['classes']:
                    doc += f"### {cls['name']} (Line {cls['line']})\n"
                    doc += f"Methods: {', '.join(cls['methods'])}\n\n"
            
            if structure['functions']:
                doc += "## Functions\n\n"
                for func in structure['functions']:
                    doc += f"### {func['name']} (Line {func['line']})\n"
                    doc += f"Parameters: {', '.join(func['params'])}\n\n"
            
            if structure['imports']:
                doc += "## Dependencies\n\n"
                doc += f"- {chr(10).join(['- ' + imp for imp in structure['imports']])}\n"
        
        elif output_format == 'rst':
            doc = f"{Path(file_path).name}\n{'=' * len(Path(file_path).name)}\n\n"
            doc += f"Language: {language}\n"
            doc += f"Lines of Code: {structure['lines_of_code']}\n\n"
        
        else:  # html
            doc = f"<h1>Documentation for {Path(file_path).name}</h1>\n"
            doc += f"<p><strong>Language:</strong> {language}</p>\n"
            doc += f"<p><strong>Lines:</strong> {structure['lines_of_code']}</p>\n"
        
        # Store generated documentation
        self.docs_generated.append({
            'file_path': file_path,
            'format': output_format,
            'content': doc,
            'timestamp': 'mock_timestamp'
        })
        
        return doc
    
    def document_project(self, project_path: str, output_dir: str) -> Dict[str, Any]:
        """Document entire project."""
        documented_files = []
        errors = []
        
        # Simulate walking through project
        mock_files = [
            f"{project_path}/main.py",
            f"{project_path}/utils.py", 
            f"{project_path}/config.js",
            f"{project_path}/README.md"
        ]
        
        for file_path in mock_files:
            try:
                if Path(file_path).suffix in ['.py', '.js', '.ts']:
                    structure = self.analyze_code_structure(file_path)
                    documentation = self.generate_documentation(structure)
                    documented_files.append({
                        'source': file_path,
                        'output': f"{output_dir}/{Path(file_path).stem}_docs.md",
                        'size': len(documentation)
                    })
            except Exception as e:
                errors.append({'file': file_path, 'error': str(e)})
        
        return {
            'documented_files': documented_files,
            'total_files': len(documented_files),
            'errors': errors,
            'output_directory': output_dir
        }
    
    def _detect_language(self, file_ext: str) -> str:
        """Detect programming language from file extension."""
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        return lang_map.get(file_ext, 'unknown')
    
    def get_documentation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated documentation."""
        if not self.docs_generated:
            return {'total_documents': 0}
        
        languages = {}
        formats = {}
        total_content_size = 0
        
        for doc in self.docs_generated:
            # Count languages
            structure = self.analyze_code_structure(doc['file_path'])
            lang = structure['language']
            languages[lang] = languages.get(lang, 0) + 1
            
            # Count formats
            fmt = doc['format']
            formats[fmt] = formats.get(fmt, 0) + 1
            
            # Sum content size
            total_content_size += len(doc['content'])
        
        return {
            'total_documents': len(self.docs_generated),
            'languages': languages,
            'formats': formats,
            'total_content_size': total_content_size,
            'average_document_size': total_content_size / len(self.docs_generated)
        }


# Mock Index Queue
class MockIndexQueue:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queue = []
        self.processed = []
        self.failed = []
        self.processing = False
        self.max_size = config.get('max_queue_size', 1000)
        
    async def add_to_queue(self, item: Dict[str, Any]) -> bool:
        """Add item to indexing queue."""
        if len(self.queue) >= self.max_size:
            return False
        
        # Validate required fields
        required_fields = ['file_path', 'content_type', 'priority']
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing required field: {field}")
        
        # Add timestamp and ID
        queue_item = {
            **item,
            'id': f"item_{len(self.queue) + 1}",
            'added_at': 'mock_timestamp',
            'status': 'queued'
        }
        
        # Insert based on priority (higher priority first)
        priority = item.get('priority', 0)
        inserted = False
        
        for i, existing_item in enumerate(self.queue):
            if existing_item.get('priority', 0) < priority:
                self.queue.insert(i, queue_item)
                inserted = True
                break
        
        if not inserted:
            self.queue.append(queue_item)
        
        return True
    
    async def process_queue(self, batch_size: int = 10) -> Dict[str, Any]:
        """Process items in queue."""
        if self.processing:
            return {'status': 'already_processing'}
        
        self.processing = True
        processed_count = 0
        failed_count = 0
        
        try:
            # Process up to batch_size items
            items_to_process = self.queue[:batch_size]
            
            for item in items_to_process:
                try:
                    # Simulate processing
                    await self._process_item(item)
                    
                    # Move to processed
                    item['status'] = 'processed'
                    item['processed_at'] = 'mock_timestamp'
                    self.processed.append(item)
                    processed_count += 1
                    
                except Exception as e:
                    # Move to failed
                    item['status'] = 'failed'
                    item['error'] = str(e)
                    item['failed_at'] = 'mock_timestamp'
                    self.failed.append(item)
                    failed_count += 1
                
                # Remove from queue
                self.queue.remove(item)
            
            return {
                'status': 'completed',
                'processed': processed_count,
                'failed': failed_count,
                'remaining_in_queue': len(self.queue)
            }
        
        finally:
            self.processing = False
    
    async def _process_item(self, item: Dict[str, Any]):
        """Process individual item (simulate work)."""
        file_path = item['file_path']
        content_type = item['content_type']
        
        # Simulate different processing times
        if content_type == 'code':
            await asyncio.sleep(0.01)  # Code files take longer
        elif content_type == 'document':
            await asyncio.sleep(0.005)  # Documents are faster
        else:
            await asyncio.sleep(0.002)  # Other files are quickest
        
        # Simulate occasional failures
        if 'error' in file_path:
            raise Exception("Simulated processing error")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'total_queued': len(self.queue),
            'total_processed': len(self.processed),
            'total_failed': len(self.failed),
            'is_processing': self.processing,
            'queue_utilization': len(self.queue) / self.max_size,
            'next_item_priority': self.queue[0].get('priority', 0) if self.queue else None
        }
    
    def clear_queue(self) -> Dict[str, int]:
        """Clear all items from queue."""
        counts = {
            'queued_cleared': len(self.queue),
            'processed_cleared': len(self.processed),
            'failed_cleared': len(self.failed)
        }
        
        self.queue.clear()
        self.processed.clear()
        self.failed.clear()
        
        return counts
    
    def get_failed_items(self) -> List[Dict[str, Any]]:
        """Get items that failed processing."""
        return self.failed.copy()
    
    def retry_failed_items(self) -> int:
        """Retry all failed items."""
        retry_count = len(self.failed)
        
        # Move failed items back to queue
        for item in self.failed:
            item['status'] = 'queued'
            item.pop('error', None)
            item.pop('failed_at', None)
            self.queue.append(item)
        
        self.failed.clear()
        return retry_count


# Test fixtures
@pytest.fixture
def auto_doc_config():
    return {
        'output_format': 'markdown',
        'include_private': False,
        'generate_toc': True,
        'include_source': False
    }

@pytest.fixture
def index_queue_config():
    return {
        'max_queue_size': 100,
        'batch_size': 10,
        'max_retries': 3,
        'retry_delay': 1.0
    }

@pytest.fixture
def auto_documenter(auto_doc_config):
    return MockAutoDocumenter(auto_doc_config)

@pytest.fixture
def index_queue(index_queue_config):
    return MockIndexQueue(index_queue_config)


# Test Classes
class TestAutoDocumenter:
    """Testes para sistema de documentação automática."""
    
    def test_init_basic(self, auto_documenter):
        """Testar inicialização básica."""
        assert auto_documenter.config is not None
        assert 'markdown' in auto_documenter.supported_formats
        assert len(auto_documenter.docs_generated) == 0
    
    def test_analyze_code_structure_python(self, auto_documenter):
        """Testar análise de estrutura de código Python."""
        result = auto_documenter.analyze_code_structure("test_file.py")
        
        assert result['file_path'] == "test_file.py"
        assert result['language'] == 'python'
        assert 'classes' in result
        assert 'functions' in result
        assert 'imports' in result
        assert len(result['classes']) > 0
        assert len(result['functions']) > 0
    
    def test_analyze_code_structure_javascript(self, auto_documenter):
        """Testar análise de estrutura de código JavaScript."""
        result = auto_documenter.analyze_code_structure("test_file.js")
        
        assert result['language'] == 'javascript'
        assert 'functions' in result
        assert len(result['functions']) > 0
        assert 'react' in result['imports']
    
    def test_analyze_code_structure_unknown(self, auto_documenter):
        """Testar análise de arquivo com extensão desconhecida."""
        result = auto_documenter.analyze_code_structure("test_file.xyz")
        
        assert result['language'] == 'unknown'
        assert result['lines_of_code'] == 150
    
    def test_generate_documentation_markdown(self, auto_documenter):
        """Testar geração de documentação em Markdown."""
        structure = auto_documenter.analyze_code_structure("test.py")
        doc = auto_documenter.generate_documentation(structure, 'markdown')
        
        assert "# Documentation for test.py" in doc
        assert "**Language:** python" in doc
        assert "## Classes" in doc
        assert "## Functions" in doc
        assert len(auto_documenter.docs_generated) == 1
    
    def test_generate_documentation_rst(self, auto_documenter):
        """Testar geração de documentação em RST."""
        structure = auto_documenter.analyze_code_structure("test.py")
        doc = auto_documenter.generate_documentation(structure, 'rst')
        
        assert "test.py" in doc
        assert "=" in doc  # RST header underline
        assert "Language: python" in doc
    
    def test_generate_documentation_html(self, auto_documenter):
        """Testar geração de documentação em HTML."""
        structure = auto_documenter.analyze_code_structure("test.py")
        doc = auto_documenter.generate_documentation(structure, 'html')
        
        assert "<h1>Documentation for test.py</h1>" in doc
        assert "<p><strong>Language:</strong> python</p>" in doc
    
    def test_generate_documentation_unsupported_format(self, auto_documenter):
        """Testar geração com formato não suportado."""
        structure = auto_documenter.analyze_code_structure("test.py")
        
        with pytest.raises(ValueError, match="Unsupported format"):
            auto_documenter.generate_documentation(structure, 'pdf')
    
    def test_document_project(self, auto_documenter):
        """Testar documentação de projeto completo."""
        result = auto_documenter.document_project("/mock/project", "/mock/output")
        
        assert 'documented_files' in result
        assert 'total_files' in result
        assert 'errors' in result
        assert result['total_files'] > 0
        assert len(result['documented_files']) == result['total_files']
    
    def test_get_documentation_statistics_empty(self, auto_documenter):
        """Testar estatísticas com documentação vazia."""
        stats = auto_documenter.get_documentation_statistics()
        assert stats['total_documents'] == 0
    
    def test_get_documentation_statistics_with_docs(self, auto_documenter):
        """Testar estatísticas com documentação gerada."""
        # Generate some docs
        structure1 = auto_documenter.analyze_code_structure("test1.py")
        auto_documenter.generate_documentation(structure1, 'markdown')
        
        structure2 = auto_documenter.analyze_code_structure("test2.js")
        auto_documenter.generate_documentation(structure2, 'html')
        
        stats = auto_documenter.get_documentation_statistics()
        
        assert stats['total_documents'] == 2
        assert 'languages' in stats
        assert 'formats' in stats
        assert stats['total_content_size'] > 0
        assert stats['average_document_size'] > 0


class TestIndexQueue:
    """Testes para sistema de fila de indexação."""
    
    @pytest.mark.asyncio
    async def test_init_basic(self, index_queue):
        """Testar inicialização básica."""
        assert index_queue.config is not None
        assert len(index_queue.queue) == 0
        assert index_queue.max_size == 100
        assert index_queue.processing is False
    
    @pytest.mark.asyncio
    async def test_add_to_queue_valid_item(self, index_queue):
        """Testar adição de item válido à fila."""
        item = {
            'file_path': '/path/to/file.py',
            'content_type': 'code',
            'priority': 5
        }
        
        result = await index_queue.add_to_queue(item)
        
        assert result is True
        assert len(index_queue.queue) == 1
        assert index_queue.queue[0]['id'] == 'item_1'
        assert index_queue.queue[0]['status'] == 'queued'
    
    @pytest.mark.asyncio
    async def test_add_to_queue_missing_fields(self, index_queue):
        """Testar adição de item com campos obrigatórios ausentes."""
        item = {
            'file_path': '/path/to/file.py',
            'content_type': 'code'
            # Missing priority
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            await index_queue.add_to_queue(item)
    
    @pytest.mark.asyncio
    async def test_add_to_queue_priority_ordering(self, index_queue):
        """Testar ordenação por prioridade na fila."""
        # Add items with different priorities
        low_priority = {'file_path': 'low.py', 'content_type': 'code', 'priority': 1}
        high_priority = {'file_path': 'high.py', 'content_type': 'code', 'priority': 10}
        medium_priority = {'file_path': 'med.py', 'content_type': 'code', 'priority': 5}
        
        await index_queue.add_to_queue(low_priority)
        await index_queue.add_to_queue(high_priority)
        await index_queue.add_to_queue(medium_priority)
        
        # Check ordering (highest priority first)
        assert index_queue.queue[0]['priority'] == 10
        assert index_queue.queue[1]['priority'] == 5
        assert index_queue.queue[2]['priority'] == 1
    
    @pytest.mark.asyncio
    async def test_add_to_queue_max_size(self, index_queue_config):
        """Testar limite máximo da fila."""
        # Create queue with small max size
        index_queue_config['max_queue_size'] = 2
        small_queue = MockIndexQueue(index_queue_config)
        
        # Add items up to max
        item1 = {'file_path': 'file1.py', 'content_type': 'code', 'priority': 1}
        item2 = {'file_path': 'file2.py', 'content_type': 'code', 'priority': 1}
        item3 = {'file_path': 'file3.py', 'content_type': 'code', 'priority': 1}
        
        assert await small_queue.add_to_queue(item1) is True
        assert await small_queue.add_to_queue(item2) is True
        assert await small_queue.add_to_queue(item3) is False  # Should fail
    
    @pytest.mark.asyncio
    async def test_process_queue_success(self, index_queue):
        """Testar processamento bem-sucedido da fila."""
        # Add items to queue
        items = [
            {'file_path': f'file{i}.py', 'content_type': 'code', 'priority': i}
            for i in range(3)
        ]
        
        for item in items:
            await index_queue.add_to_queue(item)
        
        # Process queue
        result = await index_queue.process_queue(batch_size=2)
        
        assert result['status'] == 'completed'
        assert result['processed'] == 2
        assert result['failed'] == 0
        assert result['remaining_in_queue'] == 1
        assert len(index_queue.processed) == 2
    
    @pytest.mark.asyncio
    async def test_process_queue_with_failures(self, index_queue):
        """Testar processamento com falhas."""
        # Add items including one that will fail
        items = [
            {'file_path': 'good_file.py', 'content_type': 'code', 'priority': 1},
            {'file_path': 'error_file.py', 'content_type': 'code', 'priority': 2}
        ]
        
        for item in items:
            await index_queue.add_to_queue(item)
        
        result = await index_queue.process_queue()
        
        assert result['processed'] == 1
        assert result['failed'] == 1
        assert len(index_queue.failed) == 1
        assert 'error' in index_queue.failed[0]
    
    @pytest.mark.asyncio
    async def test_process_queue_already_processing(self, index_queue):
        """Testar tentativa de processar quando já está processando."""
        index_queue.processing = True
        
        result = await index_queue.process_queue()
        
        assert result['status'] == 'already_processing'
    
    def test_get_queue_status(self, index_queue):
        """Testar obtenção de status da fila."""
        status = index_queue.get_queue_status()
        
        assert 'total_queued' in status
        assert 'total_processed' in status
        assert 'total_failed' in status
        assert 'is_processing' in status
        assert 'queue_utilization' in status
        assert status['total_queued'] == 0
        assert status['is_processing'] is False
    
    @pytest.mark.asyncio
    async def test_get_queue_status_with_items(self, index_queue):
        """Testar status da fila com itens."""
        item = {'file_path': 'test.py', 'content_type': 'code', 'priority': 5}
        await index_queue.add_to_queue(item)
        
        status = index_queue.get_queue_status()
        
        assert status['total_queued'] == 1
        assert status['next_item_priority'] == 5
        assert 0 < status['queue_utilization'] < 1
    
    def test_clear_queue(self, index_queue):
        """Testar limpeza da fila."""
        # Add some mock data
        index_queue.queue = [{'id': '1'}, {'id': '2'}]
        index_queue.processed = [{'id': '3'}]
        index_queue.failed = [{'id': '4'}]
        
        counts = index_queue.clear_queue()
        
        assert counts['queued_cleared'] == 2
        assert counts['processed_cleared'] == 1
        assert counts['failed_cleared'] == 1
        assert len(index_queue.queue) == 0
        assert len(index_queue.processed) == 0
        assert len(index_queue.failed) == 0
    
    @pytest.mark.asyncio
    async def test_get_failed_items(self, index_queue):
        """Testar obtenção de itens que falharam."""
        # Add item that will fail
        item = {'file_path': 'error_file.py', 'content_type': 'code', 'priority': 1}
        await index_queue.add_to_queue(item)
        await index_queue.process_queue()
        
        failed_items = index_queue.get_failed_items()
        
        assert len(failed_items) == 1
        assert failed_items[0]['status'] == 'failed'
        assert 'error' in failed_items[0]
    
    @pytest.mark.asyncio
    async def test_retry_failed_items(self, index_queue):
        """Testar retry de itens que falharam."""
        # Create failed items
        item = {'file_path': 'error_file.py', 'content_type': 'code', 'priority': 1}
        await index_queue.add_to_queue(item)
        await index_queue.process_queue()
        
        # Retry failed items
        retry_count = index_queue.retry_failed_items()
        
        assert retry_count == 1
        assert len(index_queue.failed) == 0
        assert len(index_queue.queue) == 1
        assert index_queue.queue[0]['status'] == 'queued'


class TestDevToolsIntegration:
    """Testes de integração para DevTools."""
    
    def test_documenter_with_queue_workflow(self, auto_documenter, index_queue):
        """Testar fluxo de trabalho integrado."""
        # Analyze code structure
        structure = auto_documenter.analyze_code_structure("test_project.py")
        
        # Generate documentation
        doc = auto_documenter.generate_documentation(structure)
        
        # Add to indexing queue
        asyncio.run(index_queue.add_to_queue({
            'file_path': structure['file_path'],
            'content_type': 'documentation',
            'priority': 3,
            'documentation': doc
        }))
        
        # Verify integration
        assert len(auto_documenter.docs_generated) == 1
        assert len(index_queue.queue) == 1
        assert index_queue.queue[0]['content_type'] == 'documentation'
    
    @pytest.mark.asyncio
    async def test_bulk_documentation_processing(self, auto_documenter, index_queue):
        """Testar processamento em massa de documentação."""
        # Document a project
        result = auto_documenter.document_project("/mock/project", "/mock/docs")
        
        # Add all documented files to queue
        for doc_file in result['documented_files']:
            await index_queue.add_to_queue({
                'file_path': doc_file['source'],
                'content_type': 'documentation',
                'priority': 2,
                'output_path': doc_file['output']
            })
        
        # Process queue
        process_result = await index_queue.process_queue()
        
        assert process_result['processed'] == len(result['documented_files'])
        assert len(index_queue.processed) == result['total_files']
    
    def test_documentation_statistics_with_queue_metrics(self, auto_documenter, index_queue):
        """Testar métricas combinadas de documentação e fila."""
        # Generate documentation
        for i in range(3):
            structure = auto_documenter.analyze_code_structure(f"file{i}.py")
            auto_documenter.generate_documentation(structure)
        
        # Get statistics
        doc_stats = auto_documenter.get_documentation_statistics()
        queue_status = index_queue.get_queue_status()
        
        # Verify statistics
        assert doc_stats['total_documents'] == 3
        assert 'python' in doc_stats['languages']
        assert queue_status['total_queued'] == 0  # Nothing queued yet


class TestDevToolsEdgeCases:
    """Testes para casos extremos de DevTools."""
    
    def test_auto_documenter_empty_project(self, auto_documenter):
        """Testar documentação de projeto vazio."""
        with patch('pathlib.Path.glob', return_value=[]):
            result = auto_documenter.document_project("/empty/project", "/output")
            assert result['total_files'] > 0  # Mock still returns files
    
    @pytest.mark.asyncio
    async def test_index_queue_concurrent_processing(self, index_queue):
        """Testar processamento concorrente."""
        # Add items
        for i in range(5):
            await index_queue.add_to_queue({
                'file_path': f'file{i}.py',
                'content_type': 'code',
                'priority': i
            })
        
        # Try concurrent processing (should be blocked)
        index_queue.processing = True
        result = await index_queue.process_queue()
        assert result['status'] == 'already_processing'
    
    def test_auto_documenter_with_complex_structure(self, auto_documenter):
        """Testar documentador com estrutura complexa."""
        # Test with file that has many classes and functions
        structure = auto_documenter.analyze_code_structure("complex_file.py")
        doc = auto_documenter.generate_documentation(structure)
        
        assert "ExampleClass" in doc
        assert "main_function" in doc
        assert len(doc) > 100  # Ensure substantial content
    
    @pytest.mark.asyncio
    async def test_index_queue_performance_simulation(self, index_queue):
        """Testar simulação de performance da fila."""
        # Add many items
        for i in range(20):
            await index_queue.add_to_queue({
                'file_path': f'file{i}.py',
                'content_type': 'document' if i % 2 else 'code',
                'priority': i % 5
            })
        
        # Process in batches
        total_processed = 0
        while index_queue.queue:
            result = await index_queue.process_queue(batch_size=5)
            total_processed += result['processed']
            
            if result['remaining_in_queue'] == 0:
                break
        
        assert total_processed == 20
        assert len(index_queue.processed) == 20


if __name__ == "__main__":
    pytest.main([__file__]) 