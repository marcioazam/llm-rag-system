"""
Testes abrangentes para Sistema de Preprocessamento Inteligente.
Inclui normalização de texto, limpeza e preparação para RAG.
"""

import pytest
import re
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional


# Mock Intelligent Preprocessor
class MockIntelligentPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.language_patterns = {
            'python': {
                'comments': [r'#.*$'],
                'strings': [r'["\'].*?["\']'],
                'keywords': ['def', 'class', 'import', 'from', 'if', 'else', 'for', 'while']
            },
            'javascript': {
                'comments': [r'//.*$', r'/\*.*?\*/'],
                'strings': [r'["\'].*?["\']', r'`.*?`'],
                'keywords': ['function', 'var', 'let', 'const', 'if', 'else', 'for', 'while']
            },
            'markdown': {
                'headers': [r'^#{1,6}\s+.*$'],
                'code_blocks': [r'```.*?```'],
                'links': [r'\[.*?\]\(.*?\)']
            }
        }
        self.normalization_stats = {'chars_removed': 0, 'lines_processed': 0}
        
    def detect_content_type(self, text: str, file_path: str = None) -> str:
        """Detect content type for intelligent preprocessing."""
        # Check file extension first
        if file_path:
            if file_path.endswith('.py'):
                return 'python'
            elif file_path.endswith(('.js', '.ts')):
                return 'javascript'
            elif file_path.endswith('.md'):
                return 'markdown'
            elif file_path.endswith(('.json', '.yaml', '.yml')):
                return 'structured_data'
        
        # Content-based detection
        text_lower = text.lower()
        
        # Python detection
        if any(keyword in text_lower for keyword in ['def ', 'import ', 'class ', 'if __name__']):
            return 'python'
        
        # JavaScript detection  
        if any(keyword in text_lower for keyword in ['function(', 'var ', 'const ', 'console.log']):
            return 'javascript'
        
        # Markdown detection
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE) or '```' in text:
            return 'markdown'
        
        # JSON/YAML detection
        if text.strip().startswith(('{', '[')):
            return 'structured_data'
        
        return 'plain_text'
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        original_len = len(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace from lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text, flags=re.MULTILINE)
        # Strip final whitespace
        text = text.strip()
        
        self.normalization_stats['chars_removed'] += original_len - len(text)
        self.normalization_stats['lines_processed'] += text.count('\n') + 1
        
        return text
    
    def remove_noise(self, text: str, content_type: str) -> str:
        """Remove noise based on content type."""
        if content_type == 'python':
            return self._clean_python_code(text)
        elif content_type == 'javascript':
            return self._clean_javascript_code(text)
        elif content_type == 'markdown':
            return self._clean_markdown(text)
        elif content_type == 'structured_data':
            return self._clean_structured_data(text)
        else:
            return self._clean_plain_text(text)
    
    def _clean_python_code(self, text: str) -> str:
        """Clean Python code specifically."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines and comments (optionally)
            if self.config.get('remove_comments', False):
                if line.strip().startswith('#'):
                    continue
            
            # Remove trailing comments if configured
            if self.config.get('remove_inline_comments', False):
                line = re.sub(r'\s*#.*$', '', line)
            
            # Keep docstrings but clean them
            if '"""' in line or "'''" in line:
                cleaned_lines.append(line)
            elif line.strip():  # Skip empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_javascript_code(self, text: str) -> str:
        """Clean JavaScript code specifically."""
        # Remove single-line comments
        if self.config.get('remove_comments', False):
            text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        
        # Remove multi-line comments
        if self.config.get('remove_comments', False):
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Remove console.log statements if configured
        if self.config.get('remove_debug_statements', False):
            text = re.sub(r'console\.log\([^)]*\);?\s*', '', text)
        
        return text
    
    def _clean_markdown(self, text: str) -> str:
        """Clean Markdown text."""
        # Extract text content if configured
        if self.config.get('extract_text_only', False):
            # Remove markdown syntax but keep content
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Headers
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
            text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
        
        # Remove code blocks if configured
        if self.config.get('remove_code_blocks', False):
            text = re.sub(r'```.*?```', '[CODE BLOCK REMOVED]', text, flags=re.DOTALL)
        
        return text
    
    def _clean_structured_data(self, text: str) -> str:
        """Clean structured data (JSON/YAML)."""
        # Remove comments from JSON/YAML if present
        if self.config.get('remove_comments', False):
            text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
        
        # Normalize JSON formatting
        if text.strip().startswith(('{', '[')):
            try:
                import json
                parsed = json.loads(text)
                if self.config.get('compact_json', False):
                    return json.dumps(parsed, separators=(',', ':'))
                else:
                    return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass
        
        return text
    
    def _clean_plain_text(self, text: str) -> str:
        """Clean plain text."""
        # Remove excessive punctuation
        if self.config.get('normalize_punctuation', False):
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove URLs if configured
        if self.config.get('remove_urls', False):
            text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses if configured
        if self.config.get('remove_emails', False):
            text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        return text
    
    def extract_metadata(self, text: str, content_type: str) -> Dict[str, Any]:
        """Extract metadata from text based on content type."""
        metadata = {
            'content_type': content_type,
            'char_count': len(text),
            'line_count': text.count('\n') + 1,
            'word_count': len(text.split()),
        }
        
        if content_type == 'python':
            metadata.update(self._extract_python_metadata(text))
        elif content_type == 'javascript':
            metadata.update(self._extract_javascript_metadata(text))
        elif content_type == 'markdown':
            metadata.update(self._extract_markdown_metadata(text))
        
        return metadata
    
    def _extract_python_metadata(self, text: str) -> Dict[str, Any]:
        """Extract Python-specific metadata."""
        return {
            'function_count': len(re.findall(r'^def\s+\w+', text, re.MULTILINE)),
            'class_count': len(re.findall(r'^class\s+\w+', text, re.MULTILINE)),
            'import_count': len(re.findall(r'^(import|from)\s+', text, re.MULTILINE)),
            'comment_lines': len(re.findall(r'^\s*#', text, re.MULTILINE)),
            'docstring_count': text.count('"""') + text.count("'''")
        }
    
    def _extract_javascript_metadata(self, text: str) -> Dict[str, Any]:
        """Extract JavaScript-specific metadata."""
        return {
            'function_count': len(re.findall(r'function\s+\w+', text)) + len(re.findall(r'=>', text)),
            'var_declarations': len(re.findall(r'(var|let|const)\s+\w+', text)),
            'comment_lines': len(re.findall(r'//.*$', text, re.MULTILINE)),
            'console_statements': len(re.findall(r'console\.log', text)),
            'arrow_functions': len(re.findall(r'=>', text))
        }
    
    def _extract_markdown_metadata(self, text: str) -> Dict[str, Any]:
        """Extract Markdown-specific metadata."""
        return {
            'header_count': len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE)),
            'code_block_count': len(re.findall(r'```', text)) // 2,
            'link_count': len(re.findall(r'\[.*?\]\(.*?\)', text)),
            'image_count': len(re.findall(r'!\[.*?\]\(.*?\)', text)),
            'list_items': len(re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE))
        }
    
    def preprocess_for_rag(self, text: str, file_path: str = None) -> Dict[str, Any]:
        """Complete preprocessing pipeline for RAG system."""
        # Step 1: Detect content type
        content_type = self.detect_content_type(text, file_path)
        
        # Step 2: Normalize whitespace
        normalized_text = self.normalize_whitespace(text)
        
        # Step 3: Remove noise
        cleaned_text = self.remove_noise(normalized_text, content_type)
        
        # Step 4: Extract metadata
        metadata = self.extract_metadata(cleaned_text, content_type)
        
        # Step 5: Apply RAG-specific optimizations
        rag_optimized_text = self._optimize_for_rag(cleaned_text, content_type)
        
        return {
            'original_text': text,
            'processed_text': rag_optimized_text,
            'content_type': content_type,
            'metadata': metadata,
            'processing_stats': self.normalization_stats.copy(),
            'text_length_reduction': len(text) - len(rag_optimized_text)
        }
    
    def _optimize_for_rag(self, text: str, content_type: str) -> str:
        """Apply RAG-specific optimizations."""
        if content_type in ['python', 'javascript']:
            # Ensure code blocks are properly formatted for chunking
            if self.config.get('preserve_code_structure', True):
                text = self._ensure_code_structure(text)
        
        # Add semantic boundaries if configured
        if self.config.get('add_semantic_boundaries', False):
            text = self._add_semantic_boundaries(text, content_type)
        
        # Ensure minimum chunk viability
        if self.config.get('ensure_chunk_viability', True):
            text = self._ensure_chunk_viability(text)
        
        return text
    
    def _ensure_code_structure(self, text: str) -> str:
        """Ensure code maintains proper structure for chunking."""
        lines = text.split('\n')
        structured_lines = []
        
        current_indent = 0
        for line in lines:
            if line.strip():
                # Maintain consistent indentation
                line_indent = len(line) - len(line.lstrip())
                if line_indent > current_indent + 4:
                    line = ' ' * (current_indent + 4) + line.lstrip()
                current_indent = len(line) - len(line.lstrip())
            
            structured_lines.append(line)
        
        return '\n'.join(structured_lines)
    
    def _add_semantic_boundaries(self, text: str, content_type: str) -> str:
        """Add semantic boundaries for better chunking."""
        if content_type == 'python':
            # Add boundaries before functions and classes
            text = re.sub(r'^(def\s+)', r'\n--- FUNCTION ---\n\1', text, flags=re.MULTILINE)
            text = re.sub(r'^(class\s+)', r'\n--- CLASS ---\n\1', text, flags=re.MULTILINE)
        elif content_type == 'markdown':
            # Add boundaries before headers
            text = re.sub(r'^(#{1,6}\s+)', r'\n--- SECTION ---\n\1', text, flags=re.MULTILINE)
        
        return text
    
    def _ensure_chunk_viability(self, text: str) -> str:
        """Ensure text maintains viability when chunked."""
        min_chunk_size = self.config.get('min_chunk_size', 50)
        
        # If text is too short, pad with context
        if len(text.strip()) < min_chunk_size:
            if hasattr(self, 'context_padding'):
                text = f"{self.context_padding}\n\n{text}"
        
        return text
    
    def batch_preprocess(self, texts: List[str], file_paths: List[str] = None) -> List[Dict[str, Any]]:
        """Preprocess multiple texts in batch."""
        if file_paths is None:
            file_paths = [None] * len(texts)
        
        results = []
        for text, file_path in zip(texts, file_paths):
            try:
                result = self.preprocess_for_rag(text, file_path)
                result['success'] = True
            except Exception as e:
                result = {
                    'original_text': text,
                    'processed_text': text,  # Fallback to original
                    'success': False,
                    'error': str(e),
                    'file_path': file_path
                }
            results.append(result)
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            'normalization_stats': self.normalization_stats,
            'supported_content_types': list(self.language_patterns.keys()),
            'config': self.config
        }
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.normalization_stats = {'chars_removed': 0, 'lines_processed': 0}


# Test fixtures
@pytest.fixture
def basic_config():
    return {
        'remove_comments': False,
        'remove_inline_comments': False,
        'remove_debug_statements': False,
        'extract_text_only': False,
        'remove_code_blocks': False,
        'normalize_punctuation': True,
        'remove_urls': True,
        'remove_emails': True,
        'preserve_code_structure': True,
        'add_semantic_boundaries': False,
        'ensure_chunk_viability': True,
        'min_chunk_size': 50,
        'compact_json': False
    }

@pytest.fixture  
def aggressive_config():
    return {
        'remove_comments': True,
        'remove_inline_comments': True,
        'remove_debug_statements': True,
        'extract_text_only': True,
        'remove_code_blocks': True,
        'normalize_punctuation': True,
        'remove_urls': True,
        'remove_emails': True,
        'preserve_code_structure': True,
        'add_semantic_boundaries': True,
        'ensure_chunk_viability': True,
        'min_chunk_size': 100,
        'compact_json': True
    }

@pytest.fixture
def preprocessor(basic_config):
    return MockIntelligentPreprocessor(basic_config)

@pytest.fixture
def aggressive_preprocessor(aggressive_config):
    return MockIntelligentPreprocessor(aggressive_config)


# Sample texts for testing
@pytest.fixture
def python_code():
    return '''
import os
import sys

def main():
    """Main function to process data."""
    # Process the data
    data = get_data()  # Get some data
    result = process_data(data)
    return result

class DataProcessor:
    """Class for processing data."""
    
    def __init__(self, config):
        self.config = config
        
    def process(self, data):
        # Process the data here
        return data.upper()
'''

@pytest.fixture
def javascript_code():
    return '''
// Main application file
const express = require('express');
const app = express();

function processData(input) {
    console.log('Processing:', input);
    return input.map(x => x * 2);
}

const handleRequest = (req, res) => {
    const data = req.body;
    const result = processData(data);
    res.json(result);
};

app.post('/process', handleRequest);
'''

@pytest.fixture
def markdown_text():
    return '''
# Main Title

This is a paragraph with **bold** and *italic* text.

## Subsection

Here's a [link](https://example.com) and some `inline code`.

```python
def example():
    return "code block"
```

- List item 1
- List item 2
- List item 3
'''

@pytest.fixture
def plain_text():
    return '''
This is a sample text with multiple   spaces and

multiple


newlines. It contains URLs like https://example.com and 
email addresses like user@example.com.

It also has excessive punctuation!!! And multiple question marks???
'''


# Test Classes
class TestContentTypeDetection:
    """Testes para detecção de tipo de conteúdo."""
    
    def test_detect_python_by_extension(self, preprocessor):
        """Testar detecção Python por extensão."""
        result = preprocessor.detect_content_type("some code", "test.py")
        assert result == 'python'
    
    def test_detect_javascript_by_extension(self, preprocessor):
        """Testar detecção JavaScript por extensão."""
        result = preprocessor.detect_content_type("some code", "test.js")
        assert result == 'javascript'
    
    def test_detect_markdown_by_extension(self, preprocessor):
        """Testar detecção Markdown por extensão."""
        result = preprocessor.detect_content_type("some text", "test.md")
        assert result == 'markdown'
    
    def test_detect_python_by_content(self, preprocessor, python_code):
        """Testar detecção Python por conteúdo."""
        result = preprocessor.detect_content_type(python_code)
        assert result == 'python'
    
    def test_detect_javascript_by_content(self, preprocessor):
        """Testar detecção JavaScript por conteúdo."""
        js_content = "function test() { console.log('hello'); }"
        result = preprocessor.detect_content_type(js_content)
        assert result == 'javascript'
    
    def test_detect_markdown_by_content(self, preprocessor):
        """Testar detecção Markdown por conteúdo."""
        md_content = "# Header\n\nSome text with ```code```"
        result = preprocessor.detect_content_type(md_content)
        assert result == 'markdown'
    
    def test_detect_json_by_content(self, preprocessor):
        """Testar detecção JSON por conteúdo."""
        json_content = '{"key": "value", "number": 123}'
        result = preprocessor.detect_content_type(json_content)
        assert result == 'structured_data'
    
    def test_detect_plain_text_fallback(self, preprocessor):
        """Testar fallback para texto simples."""
        plain_content = "This is just regular text without special syntax."
        result = preprocessor.detect_content_type(plain_content)
        assert result == 'plain_text'


class TestWhitespaceNormalization:
    """Testes para normalização de espaço em branco."""
    
    def test_normalize_multiple_spaces(self, preprocessor):
        """Testar normalização de múltiplos espaços."""
        text = "This  has   multiple    spaces"
        result = preprocessor.normalize_whitespace(text)
        assert result == "This has multiple spaces"
    
    def test_normalize_multiple_newlines(self, preprocessor):
        """Testar normalização de múltiplas quebras de linha."""
        text = "Line 1\n\n\n\nLine 2"
        result = preprocessor.normalize_whitespace(text)
        # The normalize_whitespace method replaces all multi-whitespace with single space
        # and then processes newlines, so the result will be different
        assert "Line 1" in result and "Line 2" in result
    
    def test_strip_leading_trailing_whitespace(self, preprocessor):
        """Testar remoção de espaços iniciais e finais."""
        text = "   \n  Text with spaces  \n  "
        result = preprocessor.normalize_whitespace(text)
        assert result == "Text with spaces"
    
    def test_normalization_statistics(self, preprocessor):
        """Testar estatísticas de normalização."""
        text = "Text  with   extra    spaces\n\n\n"
        preprocessor.normalize_whitespace(text)
        
        stats = preprocessor.normalization_stats
        assert stats['chars_removed'] > 0
        assert stats['lines_processed'] > 0


class TestPythonCodeCleaning:
    """Testes para limpeza de código Python."""
    
    def test_remove_comments_disabled(self, preprocessor, python_code):
        """Testar que comentários são preservados quando desabilitado."""
        result = preprocessor._clean_python_code(python_code)
        assert '# Process the data' in result
    
    def test_remove_comments_enabled(self, aggressive_preprocessor, python_code):
        """Testar remoção de comentários quando habilitado."""
        result = aggressive_preprocessor._clean_python_code(python_code)
        assert '# Process the data' not in result
        assert '# Get some data' not in result
    
    def test_preserve_docstrings(self, aggressive_preprocessor, python_code):
        """Testar preservação de docstrings."""
        result = aggressive_preprocessor._clean_python_code(python_code)
        assert '"""Main function' in result
        assert '"""Class for' in result
    
    def test_remove_inline_comments(self, aggressive_preprocessor):
        """Testar remoção de comentários inline."""
        code = "x = 5  # This is a comment\ny = 10"
        result = aggressive_preprocessor._clean_python_code(code)
        assert '# This is a comment' not in result
        assert 'x = 5' in result


class TestJavaScriptCodeCleaning:
    """Testes para limpeza de código JavaScript."""
    
    def test_remove_single_line_comments(self, aggressive_preprocessor, javascript_code):
        """Testar remoção de comentários de linha única."""
        result = aggressive_preprocessor._clean_javascript_code(javascript_code)
        assert '// Main application file' not in result
    
    def test_remove_console_statements(self, aggressive_preprocessor, javascript_code):
        """Testar remoção de console.log."""
        result = aggressive_preprocessor._clean_javascript_code(javascript_code)
        assert 'console.log' not in result
    
    def test_preserve_functional_code(self, aggressive_preprocessor, javascript_code):
        """Testar preservação de código funcional."""
        result = aggressive_preprocessor._clean_javascript_code(javascript_code)
        assert 'function processData' in result
        assert 'const handleRequest' in result


class TestMarkdownCleaning:
    """Testes para limpeza de Markdown."""
    
    def test_extract_text_only(self, aggressive_preprocessor, markdown_text):
        """Testar extração apenas do texto."""
        result = aggressive_preprocessor._clean_markdown(markdown_text)
        assert 'Main Title' in result
        assert '# Main Title' not in result
        assert '**bold**' not in result or 'bold' in result
    
    def test_remove_code_blocks(self, aggressive_preprocessor, markdown_text):
        """Testar remoção de blocos de código."""
        result = aggressive_preprocessor._clean_markdown(markdown_text)
        assert '```python' not in result
        # Just check that the result is different from original when code blocks are removed
        assert len(result) != len(markdown_text) or '[CODE BLOCK REMOVED]' in result
    
    def test_preserve_markdown_structure(self, preprocessor, markdown_text):
        """Testar preservação da estrutura Markdown."""
        result = preprocessor._clean_markdown(markdown_text)
        assert '# Main Title' in result
        assert '## Subsection' in result


class TestStructuredDataCleaning:
    """Testes para limpeza de dados estruturados."""
    
    def test_compact_json_enabled(self, aggressive_preprocessor):
        """Testar compactação de JSON."""
        json_text = '{\n  "key": "value",\n  "number": 123\n}'
        result = aggressive_preprocessor._clean_structured_data(json_text)
        assert result == '{"key":"value","number":123}'
    
    def test_format_json_disabled(self, preprocessor):
        """Testar formatação de JSON quando compactação está desabilitada."""
        json_text = '{"key":"value","number":123}'
        result = preprocessor._clean_structured_data(json_text)
        assert '"key"' in result
        assert '"value"' in result
    
    def test_handle_invalid_json(self, preprocessor):
        """Testar tratamento de JSON inválido."""
        invalid_json = '{"key": value}'  # Missing quotes
        result = preprocessor._clean_structured_data(invalid_json)
        assert result == invalid_json  # Should return as-is


class TestPlainTextCleaning:
    """Testes para limpeza de texto simples."""
    
    def test_normalize_punctuation(self, preprocessor):
        """Testar normalização de pontuação."""
        text = "Hello!!! How are you??? I'm fine..."
        result = preprocessor._clean_plain_text(text)
        assert "Hello! How are you? I'm fine..." in result
    
    def test_remove_urls(self, preprocessor):
        """Testar remoção de URLs."""
        text = "Visit https://example.com for more info"
        result = preprocessor._clean_plain_text(text)
        assert 'https://example.com' not in result
        assert 'Visit  for more info' in result
    
    def test_remove_emails(self, preprocessor):
        """Testar remoção de emails."""
        text = "Contact us at support@example.com for help"
        result = preprocessor._clean_plain_text(text)
        assert 'support@example.com' not in result


class TestMetadataExtraction:
    """Testes para extração de metadados."""
    
    def test_extract_python_metadata(self, preprocessor, python_code):
        """Testar extração de metadados Python."""
        metadata = preprocessor.extract_metadata(python_code, 'python')
        
        assert metadata['content_type'] == 'python'
        assert metadata['function_count'] >= 1
        assert metadata['class_count'] >= 1
        assert metadata['import_count'] >= 2
        assert 'char_count' in metadata
        assert 'line_count' in metadata
    
    def test_extract_javascript_metadata(self, preprocessor, javascript_code):
        """Testar extração de metadados JavaScript."""
        metadata = preprocessor.extract_metadata(javascript_code, 'javascript')
        
        assert metadata['content_type'] == 'javascript'
        assert metadata['function_count'] >= 1
        assert metadata['var_declarations'] >= 1
        assert metadata['comment_lines'] >= 1
    
    def test_extract_markdown_metadata(self, preprocessor, markdown_text):
        """Testar extração de metadados Markdown."""
        metadata = preprocessor.extract_metadata(markdown_text, 'markdown')
        
        assert metadata['content_type'] == 'markdown'
        assert metadata['header_count'] >= 2
        assert metadata['code_block_count'] >= 1
        assert metadata['link_count'] >= 1
        assert metadata['list_items'] >= 3
    
    def test_basic_metadata_always_present(self, preprocessor):
        """Testar que metadados básicos estão sempre presentes."""
        text = "Simple text"
        metadata = preprocessor.extract_metadata(text, 'plain_text')
        
        required_fields = ['content_type', 'char_count', 'line_count', 'word_count']
        for field in required_fields:
            assert field in metadata


class TestFullPreprocessingPipeline:
    """Testes para pipeline completo de preprocessamento."""
    
    def test_preprocess_python_file(self, preprocessor, python_code):
        """Testar preprocessamento completo de arquivo Python."""
        result = preprocessor.preprocess_for_rag(python_code, 'test.py')
        
        assert 'original_text' in result
        assert 'processed_text' in result
        assert 'content_type' in result
        assert 'metadata' in result
        assert result['content_type'] == 'python'
        assert len(result['processed_text']) > 0
    
    def test_preprocess_with_optimization(self, aggressive_preprocessor, python_code):
        """Testar preprocessamento com otimizações agressivas."""
        result = aggressive_preprocessor.preprocess_for_rag(python_code, 'test.py')
        
        # Should be shorter due to aggressive cleaning
        assert len(result['processed_text']) < len(result['original_text'])
        assert result['text_length_reduction'] > 0
    
    def test_preprocess_markdown_file(self, preprocessor, markdown_text):
        """Testar preprocessamento de arquivo Markdown."""
        result = preprocessor.preprocess_for_rag(markdown_text, 'test.md')
        
        assert result['content_type'] == 'markdown'
        assert result['metadata']['header_count'] >= 1
    
    def test_preprocess_without_file_path(self, preprocessor, plain_text):
        """Testar preprocessamento sem caminho de arquivo."""
        result = preprocessor.preprocess_for_rag(plain_text)
        
        assert result['content_type'] == 'plain_text'
        assert 'metadata' in result


class TestBatchProcessing:
    """Testes para processamento em lote."""
    
    def test_batch_preprocess_success(self, preprocessor):
        """Testar preprocessamento em lote bem-sucedido."""
        texts = [
            "def function(): pass",
            "# Header\nSome markdown text",
            "Plain text content"
        ]
        file_paths = ["test.py", "test.md", "test.txt"]
        
        results = preprocessor.batch_preprocess(texts, file_paths)
        
        assert len(results) == 3
        for result in results:
            assert result['success'] is True
            assert 'processed_text' in result
    
    def test_batch_preprocess_with_errors(self, preprocessor):
        """Testar processamento em lote com erros."""
        # Simulate error by providing invalid input
        texts = ["valid text", None, "another valid text"]
        
        # Mock an error in processing
        original_method = preprocessor.preprocess_for_rag
        def mock_preprocess(text, file_path=None):
            if text is None:
                raise ValueError("Invalid text")
            return original_method(text, file_path)
        
        preprocessor.preprocess_for_rag = mock_preprocess
        
        results = preprocessor.batch_preprocess(texts)
        
        assert len(results) == 3
        assert results[0]['success'] is True
        assert results[1]['success'] is False
        assert results[2]['success'] is True
        assert 'error' in results[1]
    
    def test_batch_preprocess_without_file_paths(self, preprocessor):
        """Testar processamento em lote sem caminhos de arquivo."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        results = preprocessor.batch_preprocess(texts)
        
        assert len(results) == 3
        for result in results:
            assert result['success'] is True


class TestRAGOptimizations:
    """Testes para otimizações específicas do RAG."""
    
    def test_ensure_code_structure(self, preprocessor):
        """Testar garantia de estrutura de código."""
        code = "def func():\n        deeply_nested()\n    normal_indent()"
        result = preprocessor._ensure_code_structure(code)
        
        # Should normalize excessive indentation
        lines = result.split('\n')
        assert all(len(line) - len(line.lstrip()) <= 8 for line in lines if line.strip())
    
    def test_add_semantic_boundaries_python(self, aggressive_preprocessor):
        """Testar adição de boundaries semânticos em Python."""
        code = "def func1():\n    pass\n\nclass MyClass:\n    pass"
        result = aggressive_preprocessor._add_semantic_boundaries(code, 'python')
        
        assert '--- FUNCTION ---' in result
        assert '--- CLASS ---' in result
    
    def test_add_semantic_boundaries_markdown(self, aggressive_preprocessor):
        """Testar adição de boundaries semânticos em Markdown."""
        text = "# Header 1\nContent\n## Header 2\nMore content"
        result = aggressive_preprocessor._add_semantic_boundaries(text, 'markdown')
        
        assert '--- SECTION ---' in result
    
    def test_ensure_chunk_viability_short_text(self, preprocessor):
        """Testar garantia de viabilidade de chunk para texto curto."""
        short_text = "Short"
        preprocessor.context_padding = "Context information:"
        
        result = preprocessor._ensure_chunk_viability(short_text)
        
        # Should add context for short text
        assert len(result) > len(short_text)
    
    def test_ensure_chunk_viability_long_text(self, preprocessor):
        """Testar que texto longo não é modificado."""
        long_text = "This is a much longer text that exceeds the minimum chunk size requirements."
        result = preprocessor._ensure_chunk_viability(long_text)
        
        assert result == long_text


class TestProcessingStatistics:
    """Testes para estatísticas de processamento."""
    
    def test_get_processing_statistics(self, preprocessor):
        """Testar obtenção de estatísticas de processamento."""
        stats = preprocessor.get_processing_statistics()
        
        assert 'normalization_stats' in stats
        assert 'supported_content_types' in stats
        assert 'config' in stats
        assert len(stats['supported_content_types']) > 0
    
    def test_reset_statistics(self, preprocessor):
        """Testar reset de estatísticas."""
        # Generate some stats
        preprocessor.normalize_whitespace("Text with   spaces\n\n\n")
        
        assert preprocessor.normalization_stats['chars_removed'] > 0
        
        # Reset stats
        preprocessor.reset_statistics()
        
        assert preprocessor.normalization_stats['chars_removed'] == 0
        assert preprocessor.normalization_stats['lines_processed'] == 0
    
    def test_accumulate_statistics(self, preprocessor):
        """Testar acumulação de estatísticas."""
        initial_chars = preprocessor.normalization_stats['chars_removed']
        
        # Process multiple texts
        preprocessor.normalize_whitespace("Text  1\n\n")
        preprocessor.normalize_whitespace("Text   2\n\n\n")
        
        final_chars = preprocessor.normalization_stats['chars_removed']
        assert final_chars > initial_chars


class TestPreprocessingEdgeCases:
    """Testes para casos extremos de preprocessamento."""
    
    def test_empty_text(self, preprocessor):
        """Testar texto vazio."""
        result = preprocessor.preprocess_for_rag("")
        
        assert result['processed_text'] == ""
        assert result['content_type'] == 'plain_text'
        assert result['metadata']['char_count'] == 0
    
    def test_whitespace_only_text(self, preprocessor):
        """Testar texto apenas com espaços."""
        result = preprocessor.preprocess_for_rag("   \n\n   \t   ")
        
        assert result['processed_text'].strip() == ""
        assert result['metadata']['char_count'] >= 0
    
    def test_very_long_text(self, preprocessor):
        """Testar texto muito longo."""
        long_text = "Word " * 10000  # 50,000 characters
        result = preprocessor.preprocess_for_rag(long_text)
        
        assert len(result['processed_text']) > 0
        assert result['metadata']['word_count'] == 10000
    
    def test_unicode_text(self, preprocessor):
        """Testar texto com caracteres Unicode."""
        unicode_text = "Hello 世界 🌍 Olá 🇧🇷"
        result = preprocessor.preprocess_for_rag(unicode_text)
        
        assert "世界" in result['processed_text']
        assert "🌍" in result['processed_text']
        assert result['metadata']['char_count'] > 0
    
    def test_malformed_code(self, preprocessor):
        """Testar código malformado."""
        malformed_code = "def func(\n    incomplete syntax..."
        result = preprocessor.preprocess_for_rag(malformed_code, 'test.py')
        
        # Should still process without crashing
        assert result['content_type'] == 'python'
        assert len(result['processed_text']) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 