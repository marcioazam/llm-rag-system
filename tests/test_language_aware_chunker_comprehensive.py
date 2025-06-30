"""
Testes abrangentes para Language Aware Chunker.
Integra detecção de linguagem com estratégias de chunking específicas.
"""

import pytest
from unittest.mock import Mock, patch


# Mock implementation of LanguageAwareChunker
class LanguageDetector:
    def __init__(self):
        self.supported_languages = {
            'python', 'javascript', 'java', 'cpp', 'markdown', 'json', 'yaml', 'sql', 'text'
        }
        
    def detect(self, file_path=None, content=None):
        """Detect language from file path or content."""
        if file_path:
            extension = file_path.split('.')[-1].lower()
            ext_map = {
                'py': 'python', 'js': 'javascript', 'java': 'java',
                'cpp': 'cpp', 'md': 'markdown', 'json': 'json',
                'yaml': 'yaml', 'yml': 'yaml', 'sql': 'sql'
            }
            return ext_map.get(extension, 'text')
            
        if content:
            if 'def ' in content or 'import ' in content:
                return 'python'
            elif 'function' in content and '(' in content:
                return 'javascript'
            elif 'public class' in content:
                return 'java'
            elif content.strip().startswith('#'):
                return 'markdown'
            elif content.strip().startswith('{'):
                return 'json'
                
        return 'text'


class CodeChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_code(self, text, language):
        """Chunk code with language-specific awareness."""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            line_size = len(line) + 1
            
            # Language-specific chunking rules
            is_boundary = self._is_chunk_boundary(line, language)
            
            if (current_size + line_size > self.chunk_size and 
                current_chunk and is_boundary):
                
                chunk = {
                    'text': '\n'.join(current_chunk),
                    'start_line': i - len(current_chunk),
                    'end_line': i,
                    'language': language,
                    'chunk_type': 'code',
                    'metadata': self._extract_code_metadata(current_chunk, language)
                }
                chunks.append(chunk)
                
                # Smart overlap for code
                overlap_lines = self._get_code_overlap(current_chunk, language)
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
            else:
                current_chunk.append(line)
                current_size += line_size
                
        # Add final chunk
        if current_chunk:
            chunk = {
                'text': '\n'.join(current_chunk),
                'start_line': len(lines) - len(current_chunk),
                'end_line': len(lines),
                'language': language,
                'chunk_type': 'code',
                'metadata': self._extract_code_metadata(current_chunk, language)
            }
            chunks.append(chunk)
            
        return chunks
        
    def _is_chunk_boundary(self, line, language):
        """Determine if line is a good boundary for chunking."""
        stripped = line.strip()
        
        if language == 'python':
            return (stripped.startswith(('def ', 'class ', 'if __name__')) or
                    stripped.startswith(('import ', 'from ')))
        elif language == 'javascript':
            return (stripped.startswith(('function ', 'class ', 'const ', 'let ')) or
                    stripped.startswith(('import ', 'export ')))
        elif language == 'java':
            return (stripped.startswith(('public ', 'private ', 'protected ')) or
                    stripped.startswith(('import ', 'package ')))
        
        return stripped == '' or stripped.startswith(('#', '//', '/*'))
        
    def _get_code_overlap(self, lines, language):
        """Get smart overlap for code based on language."""
        # For code, try to include complete function/class definitions
        overlap_lines = []
        
        for line in reversed(lines[-10:]):  # Look at last 10 lines
            if self._is_chunk_boundary(line, language):
                overlap_lines.insert(0, line)
                if len(overlap_lines) >= 5:  # Max 5 lines overlap
                    break
            elif overlap_lines:
                overlap_lines.insert(0, line)
                
        return overlap_lines
        
    def _extract_code_metadata(self, lines, language):
        """Extract metadata from code chunk."""
        metadata = {
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity': 0
        }
        
        for line in lines:
            stripped = line.strip()
            
            if language == 'python':
                if stripped.startswith('def '):
                    func_name = stripped.split('(')[0].replace('def ', '')
                    metadata['functions'].append(func_name)
                elif stripped.startswith('class '):
                    class_name = stripped.split('(')[0].split(':')[0].replace('class ', '')
                    metadata['classes'].append(class_name)
                elif stripped.startswith(('import ', 'from ')):
                    metadata['imports'].append(stripped)
                elif stripped.startswith(('if ', 'for ', 'while ', 'try ')):
                    metadata['complexity'] += 1
                    
        return metadata


class DocumentChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_document(self, text, language):
        """Chunk documents with format-specific awareness."""
        if language == 'markdown':
            return self._chunk_markdown(text)
        elif language == 'json':
            return self._chunk_json(text)
        elif language == 'yaml':
            return self._chunk_yaml(text)
        else:
            return self._chunk_text(text, language)
            
    def _chunk_markdown(self, text):
        """Chunk markdown by headers."""
        import re
        chunks = []
        sections = re.split(r'\n(#{1,6}\s+.*)\n', text)
        
        current_chunk = ""
        current_header = None
        
        for section in sections:
            if section.strip().startswith('#'):
                # New header
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'language': 'markdown',
                        'chunk_type': 'document',
                        'header': current_header,
                        'level': len(current_header.split('#')) - 1 if current_header else 0
                    })
                    
                current_header = section.strip()
                current_chunk = section
            else:
                current_chunk += section
                
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'language': 'markdown',
                'chunk_type': 'document',
                'header': current_header,
                'level': len(current_header.split('#')) - 1 if current_header else 0
            })
            
        return chunks
        
    def _chunk_json(self, text):
        """Chunk JSON by top-level keys."""
        import json
        
        try:
            data = json.loads(text)
            
            if isinstance(data, dict) and len(str(data)) > self.chunk_size:
                chunks = []
                for key, value in data.items():
                    chunk_data = {key: value}
                    chunk_text = json.dumps(chunk_data, indent=2)
                    
                    chunks.append({
                        'text': chunk_text,
                        'language': 'json',
                        'chunk_type': 'document',
                        'json_key': key,
                        'data_type': type(value).__name__
                    })
                    
                return chunks
                
        except json.JSONDecodeError:
            pass
            
        return [{
            'text': text,
            'language': 'json',
            'chunk_type': 'document'
        }]
        
    def _chunk_yaml(self, text):
        """Chunk YAML by sections."""
        chunks = []
        sections = text.split('\n---\n')
        
        for i, section in enumerate(sections):
            if section.strip():
                chunks.append({
                    'text': section.strip(),
                    'language': 'yaml',
                    'chunk_type': 'document',
                    'section_number': i
                })
                
        return chunks
        
    def _chunk_text(self, text, language):
        """Chunk plain text."""
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'language': language,
                'chunk_type': 'document'
            }]
            
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'language': language,
                    'chunk_type': 'document'
                })
                
                # Overlap with last sentence
                overlap = '. '.join(current_chunk.split('. ')[-2:])
                current_chunk = overlap + '. ' + sentence
            else:
                current_chunk += ('. ' + sentence if current_chunk else sentence)
                
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'language': language,
                'chunk_type': 'document'
            })
            
        return chunks


class LanguageAwareChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200, strategy='adaptive'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.language_detector = LanguageDetector()
        self.code_chunker = CodeChunker(chunk_size, chunk_overlap)
        self.document_chunker = DocumentChunker(chunk_size, chunk_overlap)
        
    def chunk_text(self, text, file_path=None, metadata=None):
        """Chunk text with language awareness."""
        if not text.strip():
            return []
            
        # Detect language
        language = self.language_detector.detect(file_path=file_path, content=text)
        
        # Override with metadata if provided
        if metadata and 'language' in metadata:
            language = metadata['language']
            
        # Choose chunking strategy based on language
        if language in ['python', 'javascript', 'java', 'cpp']:
            chunks = self.code_chunker.chunk_code(text, language)
        else:
            chunks = self.document_chunker.chunk_document(text, language)
            
        # Add common metadata
        for i, chunk in enumerate(chunks):
            chunk.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'detected_language': language,
                'file_path': file_path,
                'chunking_strategy': self.strategy,
                'chunk_size_config': self.chunk_size,
                'overlap_config': self.chunk_overlap
            })
            
            if metadata:
                chunk['original_metadata'] = metadata
                
        return chunks
        
    def get_chunking_strategy_for_language(self, language):
        """Get optimal chunking strategy for a language."""
        strategies = {
            'python': 'function_aware',
            'javascript': 'function_aware',
            'java': 'class_aware',
            'cpp': 'function_aware',
            'markdown': 'header_aware',
            'json': 'key_aware',
            'yaml': 'section_aware',
            'sql': 'statement_aware',
            'text': 'sentence_aware'
        }
        
        return strategies.get(language, 'generic')
        
    def analyze_chunking_efficiency(self, chunks):
        """Analyze efficiency of chunking results."""
        if not chunks:
            return {
                'total_chunks': 0,
                'efficiency_score': 0.0,
                'recommendations': ['No chunks generated']
            }
            
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)
        size_variance = sum((len(chunk['text']) - avg_chunk_size) ** 2 for chunk in chunks) / len(chunks)
        
        # Calculate efficiency metrics
        size_efficiency = min(1.0, avg_chunk_size / self.chunk_size)
        variance_penalty = max(0.0, 1.0 - (size_variance / (self.chunk_size ** 2)))
        
        efficiency_score = (size_efficiency + variance_penalty) / 2
        
        recommendations = []
        if size_efficiency < 0.7:
            recommendations.append("Consider increasing chunk size")
        if variance_penalty < 0.5:
            recommendations.append("High size variance - review chunking boundaries")
        if len(chunks) == 1 and total_chars > self.chunk_size * 1.5:
            recommendations.append("Single large chunk - may need better boundary detection")
            
        return {
            'total_chunks': len(chunks),
            'average_chunk_size': avg_chunk_size,
            'size_variance': size_variance,
            'efficiency_score': efficiency_score,
            'size_efficiency': size_efficiency,
            'variance_penalty': variance_penalty,
            'recommendations': recommendations or ['Chunking efficiency is good']
        }


class TestLanguageDetector:
    """Tests for language detection component."""
    
    @pytest.fixture
    def detector(self):
        return LanguageDetector()
        
    def test_init_basic(self, detector):
        """Test basic initialization."""
        assert detector is not None
        assert hasattr(detector, 'supported_languages')
        assert len(detector.supported_languages) > 0
        
    def test_detect_by_file_extension(self, detector):
        """Test language detection by file extension."""
        test_cases = [
            ('script.py', 'python'),
            ('app.js', 'javascript'),
            ('Main.java', 'java'),
            ('program.cpp', 'cpp'),
            ('README.md', 'markdown'),
            ('config.json', 'json'),
            ('settings.yaml', 'yaml'),
            ('query.sql', 'sql'),
            ('document.txt', 'text')
        ]
        
        for file_path, expected_lang in test_cases:
            result = detector.detect(file_path=file_path)
            assert result == expected_lang, f"Failed for {file_path}"
            
    def test_detect_by_content(self, detector):
        """Test language detection by content."""
        test_cases = [
            ('def hello(): pass', 'python'),
            ('function test() { return true; }', 'javascript'),
            ('public class Test { }', 'java'),
            ('# Header\nContent', 'markdown'),
            ('{"key": "value"}', 'json')
        ]
        
        for content, expected_lang in test_cases:
            result = detector.detect(content=content)
            assert result == expected_lang, f"Failed for content: {content}"
            
    def test_detect_fallback_to_text(self, detector):
        """Test fallback to text for unknown content."""
        unknown_content = "This is some unknown format content."
        result = detector.detect(content=unknown_content)
        assert result == 'text'


class TestCodeChunker:
    """Tests for code-specific chunking."""
    
    @pytest.fixture
    def chunker(self):
        return CodeChunker(chunk_size=300, chunk_overlap=50)
        
    def test_init_basic(self, chunker):
        """Test basic initialization."""
        assert chunker.chunk_size == 300
        assert chunker.chunk_overlap == 50
        
    def test_chunk_python_code(self, chunker):
        """Test chunking Python code."""
        python_code = """
import os
import sys

def function1():
    print("Function 1")
    return True

def function2():
    print("Function 2")
    for i in range(10):
        if i % 2 == 0:
            print(i)

class MyClass:
    def __init__(self):
        self.value = 0
        
    def method1(self):
        return self.value
"""
        
        chunks = chunker.chunk_code(python_code, 'python')
        
        assert len(chunks) >= 1
        assert all('language' in chunk for chunk in chunks)
        assert all(chunk['language'] == 'python' for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        
        # Check metadata extraction
        metadata = chunks[0]['metadata']
        assert 'functions' in metadata
        assert 'classes' in metadata
        assert 'imports' in metadata
        
    def test_chunk_javascript_code(self, chunker):
        """Test chunking JavaScript code."""
        js_code = """
import React from 'react';

function Component1() {
    return <div>Component 1</div>;
}

const Component2 = () => {
    const [state, setState] = useState(0);
    
    return (
        <div onClick={() => setState(state + 1)}>
            {state}
        </div>
    );
};

export { Component1, Component2 };
"""
        
        chunks = chunker.chunk_code(js_code, 'javascript')
        
        assert len(chunks) >= 1
        assert all(chunk['language'] == 'javascript' for chunk in chunks)
        
    def test_chunk_boundary_detection(self, chunker):
        """Test chunk boundary detection."""
        test_cases = [
            ('def test():', 'python', True),
            ('class Test:', 'python', True),
            ('    print("hello")', 'python', False),
            ('function test() {', 'javascript', True),
            ('    console.log("hello");', 'javascript', False),
            ('public class Test {', 'java', True)
        ]
        
        for line, language, expected in test_cases:
            result = chunker._is_chunk_boundary(line, language)
            assert result == expected, f"Failed for: {line} in {language}"
            
    def test_code_overlap_generation(self, chunker):
        """Test smart overlap generation for code."""
        code_lines = [
            "def function1():",
            "    print('test')",
            "    return True",
            "",
            "def function2():",
            "    pass"
        ]
        
        overlap = chunker._get_code_overlap(code_lines, 'python')
        
        assert isinstance(overlap, list)
        assert len(overlap) <= 10
        
    def test_metadata_extraction(self, chunker):
        """Test code metadata extraction."""
        code_lines = [
            "import os",
            "from collections import defaultdict",
            "",
            "def my_function(arg1, arg2):",
            "    if arg1 > arg2:",
            "        for i in range(10):",
            "            print(i)",
            "    return True",
            "",
            "class MyClass:",
            "    def __init__(self):",
            "        pass"
        ]
        
        metadata = chunker._extract_code_metadata(code_lines, 'python')
        
        assert 'functions' in metadata
        assert 'classes' in metadata
        assert 'imports' in metadata
        assert 'complexity' in metadata
        
        assert len(metadata['functions']) >= 1
        assert len(metadata['classes']) >= 1
        assert len(metadata['imports']) >= 2
        assert metadata['complexity'] > 0


class TestDocumentChunker:
    """Tests for document-specific chunking."""
    
    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size=300, chunk_overlap=50)
        
    def test_init_basic(self, chunker):
        """Test basic initialization."""
        assert chunker.chunk_size == 300
        assert chunker.chunk_overlap == 50
        
    def test_chunk_markdown(self, chunker):
        """Test markdown chunking by headers."""
        markdown_text = """
# Main Title

This is the introduction section.

## Section 1

Content for section 1 with some details.

### Subsection 1.1

More detailed content here.

## Section 2

Content for section 2.
"""
        
        chunks = chunker.chunk_document(markdown_text, 'markdown')
        
        assert len(chunks) >= 1
        assert all(chunk['language'] == 'markdown' for chunk in chunks)
        assert all('header' in chunk for chunk in chunks)
        assert all('level' in chunk for chunk in chunks)
        
    def test_chunk_json(self, chunker):
        """Test JSON chunking by keys."""
        json_text = '{"users": [{"name": "John"}, {"name": "Jane"}], "settings": {"debug": true, "version": "1.0"}, "data": {"items": [1, 2, 3, 4, 5]}}'
        
        chunks = chunker.chunk_document(json_text, 'json')
        
        assert len(chunks) >= 1
        assert all(chunk['language'] == 'json' for chunk in chunks)
        
        if len(chunks) > 1:
            # Should have individual keys
            assert any('json_key' in chunk for chunk in chunks)
            
    def test_chunk_yaml(self, chunker):
        """Test YAML chunking by sections."""
        yaml_text = """
version: "3.8"
services:
  web:
    image: nginx
    ports:
      - "80:80"
---
version: "2"
services:
  app:
    build: .
    ports:
      - "3000:3000"
"""
        
        chunks = chunker.chunk_document(yaml_text, 'yaml')
        
        assert len(chunks) >= 2  # Should split on ---
        assert all(chunk['language'] == 'yaml' for chunk in chunks)
        assert all('section_number' in chunk for chunk in chunks)
        
    def test_chunk_plain_text(self, chunker):
        """Test plain text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three. " * 20
        
        chunks = chunker.chunk_document(text, 'text')
        
        assert len(chunks) >= 1
        assert all(chunk['language'] == 'text' for chunk in chunks)


class TestLanguageAwareChunker:
    """Tests for the main language-aware chunker."""
    
    @pytest.fixture
    def chunker(self):
        return LanguageAwareChunker(chunk_size=400, chunk_overlap=80)
        
    def test_init_basic(self, chunker):
        """Test basic initialization."""
        assert chunker.chunk_size == 400
        assert chunker.chunk_overlap == 80
        assert chunker.strategy == 'adaptive'
        assert hasattr(chunker, 'language_detector')
        assert hasattr(chunker, 'code_chunker')
        assert hasattr(chunker, 'document_chunker')
        
    def test_chunk_with_file_path(self, chunker):
        """Test chunking with file path for language detection."""
        python_code = """
def hello_world():
    print("Hello, World!")
    return True

def main():
    result = hello_world()
    print(f"Result: {result}")
"""
        
        chunks = chunker.chunk_text(python_code, file_path="script.py")
        
        assert len(chunks) >= 1
        assert all('detected_language' in chunk for chunk in chunks)
        assert all(chunk['detected_language'] == 'python' for chunk in chunks)
        assert all('file_path' in chunk for chunk in chunks)
        assert chunks[0]['file_path'] == "script.py"
        
    def test_chunk_with_content_detection(self, chunker):
        """Test chunking with content-based language detection."""
        js_code = """
function calculateSum(a, b) {
    return a + b;
}

const result = calculateSum(5, 3);
console.log(result);
"""
        
        chunks = chunker.chunk_text(js_code)
        
        assert len(chunks) >= 1
        assert all(chunk['detected_language'] == 'javascript' for chunk in chunks)
        
    def test_chunk_with_metadata_override(self, chunker):
        """Test language override via metadata."""
        code_text = "def test(): pass"  # Looks like Python
        metadata = {'language': 'text'}  # But treat as text
        
        chunks = chunker.chunk_text(code_text, metadata=metadata)
        
        assert len(chunks) == 1
        assert chunks[0]['detected_language'] == 'text'
        assert 'original_metadata' in chunks[0]
        
    def test_empty_text_handling(self, chunker):
        """Test handling of empty or whitespace-only text."""
        test_cases = ["", "   ", "\n\n\n", "\t\t"]
        
        for text in test_cases:
            chunks = chunker.chunk_text(text)
            assert chunks == []
            
    def test_get_chunking_strategy(self, chunker):
        """Test chunking strategy selection."""
        strategies = {
            'python': 'function_aware',
            'javascript': 'function_aware',
            'java': 'class_aware',
            'markdown': 'header_aware',
            'json': 'key_aware',
            'unknown': 'generic'
        }
        
        for language, expected_strategy in strategies.items():
            result = chunker.get_chunking_strategy_for_language(language)
            assert result == expected_strategy
            
    def test_chunk_metadata_enrichment(self, chunker):
        """Test that chunks are enriched with metadata."""
        text = "Simple test content."
        
        chunks = chunker.chunk_text(text, file_path="test.txt", metadata={'author': 'Test'})
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Check standard metadata
        assert 'chunk_id' in chunk
        assert 'total_chunks' in chunk
        assert 'detected_language' in chunk
        assert 'file_path' in chunk
        assert 'chunking_strategy' in chunk
        assert 'chunk_size_config' in chunk
        assert 'overlap_config' in chunk
        assert 'original_metadata' in chunk
        
        assert chunk['chunk_id'] == 0
        assert chunk['total_chunks'] == 1
        assert chunk['file_path'] == "test.txt"
        assert chunk['original_metadata']['author'] == 'Test'
        
    def test_analyze_chunking_efficiency(self, chunker):
        """Test chunking efficiency analysis."""
        # Create test chunks with known properties
        chunks = [
            {'text': 'A' * 400},  # Optimal size
            {'text': 'B' * 380},  # Near optimal
            {'text': 'C' * 200},  # Small chunk
            {'text': 'D' * 100}   # Very small chunk
        ]
        
        analysis = chunker.analyze_chunking_efficiency(chunks)
        
        assert 'total_chunks' in analysis
        assert 'average_chunk_size' in analysis
        assert 'size_variance' in analysis
        assert 'efficiency_score' in analysis
        assert 'recommendations' in analysis
        
        assert analysis['total_chunks'] == 4
        assert 0.0 <= analysis['efficiency_score'] <= 1.0
        assert isinstance(analysis['recommendations'], list)
        
    def test_analyze_empty_chunks(self, chunker):
        """Test efficiency analysis with no chunks."""
        analysis = chunker.analyze_chunking_efficiency([])
        
        assert analysis['total_chunks'] == 0
        assert analysis['efficiency_score'] == 0.0
        assert 'No chunks generated' in analysis['recommendations']


@pytest.mark.integration
class TestLanguageAwareChunkerIntegration:
    """Integration tests for language-aware chunker."""
    
    def test_multi_language_document(self):
        """Test chunking document with multiple languages."""
        chunker = LanguageAwareChunker(chunk_size=200)
        
        # Test different file types
        test_files = [
            ("script.py", "def hello(): print('world')", "python"),
            ("app.js", "function hello() { console.log('world'); }", "javascript"),
            ("README.md", "# Title\nContent here", "markdown"),
            ("config.json", '{"setting": "value"}', "json")
        ]
        
        all_chunks = []
        for file_path, content, expected_lang in test_files:
            chunks = chunker.chunk_text(content, file_path=file_path)
            all_chunks.extend(chunks)
            
            assert len(chunks) >= 1
            assert all(chunk['detected_language'] == expected_lang for chunk in chunks)
            
        # Verify all chunks have consistent structure
        assert len(all_chunks) == len(test_files)  # One chunk per file (small content)
        assert all('chunk_id' in chunk for chunk in all_chunks)
        assert all('detected_language' in chunk for chunk in all_chunks)
        
    def test_large_code_file_chunking(self):
        """Test chunking large code file."""
        chunker = LanguageAwareChunker(chunk_size=500, chunk_overlap=100)
        
        # Large Python file
        large_python = """
import os
import sys
from collections import defaultdict

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = defaultdict(list)
        
    def process_file(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read()
            return self.parse_content(content)
            
    def parse_content(self, content):
        lines = content.split('\\n')
        results = []
        
        for line_num, line in enumerate(lines):
            if line.strip():
                processed = self.process_line(line, line_num)
                results.append(processed)
                
        return results
        
    def process_line(self, line, line_num):
        # Complex processing logic
        if line.startswith('#'):
            return {'type': 'comment', 'content': line, 'line': line_num}
        elif '=' in line:
            parts = line.split('=')
            return {'type': 'assignment', 'var': parts[0].strip(), 'value': parts[1].strip(), 'line': line_num}
        else:
            return {'type': 'other', 'content': line, 'line': line_num}
            
def main():
    processor = DataProcessor({'debug': True})
    results = processor.process_file('input.txt')
    print(f"Processed {len(results)} lines")

if __name__ == "__main__":
    main()
""" * 2  # Double it to ensure multiple chunks
        
        chunks = chunker.chunk_text(large_python, file_path="processor.py")
        
        assert len(chunks) >= 2  # Should create multiple chunks
        assert all(chunk['detected_language'] == 'python' for chunk in chunks)
        
        # Check that code structure is preserved
        code_chunks = [chunk for chunk in chunks if chunk.get('chunk_type') == 'code']
        assert len(code_chunks) >= 1
        
        # Verify metadata extraction
        for chunk in code_chunks:
            if 'metadata' in chunk:
                metadata = chunk['metadata']
                assert 'functions' in metadata
                assert 'classes' in metadata
                
    def test_performance_with_large_document(self):
        """Test performance with large documents."""
        import time
        
        chunker = LanguageAwareChunker(chunk_size=1000)
        
        # Large markdown document
        large_doc = """
# Chapter 1: Introduction

This is a comprehensive guide to language-aware chunking.

## Section 1.1: Overview

Language-aware chunking is an advanced technique...

## Section 1.2: Benefits

The main benefits include:
1. Better context preservation
2. Language-specific optimization
3. Improved retrieval accuracy

""" * 100  # Repeat to create large document
        
        start_time = time.time()
        chunks = chunker.chunk_text(large_doc, file_path="guide.md")
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process quickly (< 2 seconds)
        assert processing_time < 2.0
        assert len(chunks) > 0
        assert all(chunk['detected_language'] == 'markdown' for chunk in chunks)
        
    def test_efficiency_analysis_comprehensive(self):
        """Test comprehensive efficiency analysis."""
        chunker = LanguageAwareChunker(chunk_size=500, chunk_overlap=100)
        
        # Mix of different content types
        test_contents = [
            ("Small file", "text"),
            ("A" * 600, "text"),  # Slightly over chunk size
            ("A" * 1200, "text"), # Double chunk size
            ("A" * 2000, "text")  # Very large
        ]
        
        all_efficiency_scores = []
        
        for content, language in test_contents:
            chunks = chunker.chunk_text(content, metadata={'language': language})
            analysis = chunker.analyze_chunking_efficiency(chunks)
            
            all_efficiency_scores.append(analysis['efficiency_score'])
            
            # Basic validation
            assert analysis['total_chunks'] >= 1
            assert 0.0 <= analysis['efficiency_score'] <= 1.0
            assert isinstance(analysis['recommendations'], list)
            
        # Should have varying efficiency scores
        assert len(set(all_efficiency_scores)) > 1  # Not all the same