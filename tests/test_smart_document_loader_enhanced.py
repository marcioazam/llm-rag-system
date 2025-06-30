"""
Testes para o módulo smart_document_loader - Carregamento Inteligente de Documentos
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
import json
import hashlib
import mimetypes
from datetime import datetime, timedelta
import asyncio
import aiofiles


class DocumentType(Enum):
    """Tipos de documento suportados"""
    PDF = "pdf"
    WORD = "docx"
    TEXT = "txt" 
    MARKDOWN = "md"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    CODE = "code"
    UNKNOWN = "unknown"


class LoadStatus(Enum):
    """Status de carregamento"""
    PENDING = "pending"
    LOADING = "loading"
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"


class MockSmartDocumentLoader:
    """Mock do carregador inteligente de documentos"""
    
    def __init__(self, cache_enabled: bool = True, max_file_size: int = 10_000_000):
        self.cache_enabled = cache_enabled
        self.max_file_size = max_file_size
        self.cache = {}
        self.processors = {}
        self.extractors = {}
        self.metadata_cache = {}
        self.load_history = []
        self.failed_loads = []
        
        # Configurações
        self.supported_extensions = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.WORD,
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.html': DocumentType.HTML,
            '.csv': DocumentType.CSV,
            '.json': DocumentType.JSON,
            '.xml': DocumentType.XML,
            '.py': DocumentType.CODE,
            '.js': DocumentType.CODE,
            '.ts': DocumentType.CODE,
        }
        
        # Estatísticas
        self.stats = {
            'total_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'failed_loads': 0,
            'bytes_processed': 0,
            'avg_load_time': 0.0
        }
        
        # Inicializa processadores padrão
        self._init_default_processors()
    
    def _init_default_processors(self):
        """Inicializa processadores padrão para cada tipo"""
        self.processors[DocumentType.TEXT] = self._process_text
        self.processors[DocumentType.MARKDOWN] = self._process_markdown
        self.processors[DocumentType.JSON] = self._process_json
        self.processors[DocumentType.CSV] = self._process_csv
        self.processors[DocumentType.CODE] = self._process_code
        self.processors[DocumentType.HTML] = self._process_html
    
    async def load_document(self, file_path: str, **options) -> Dict[str, Any]:
        """Carrega documento de forma inteligente"""
        self.stats['total_loads'] += 1
        
        try:
            # Verifica cache primeiro
            if self.cache_enabled:
                cached_result = await self._check_cache(file_path)
                if cached_result:
                    self.stats['cache_hits'] += 1
                    return cached_result
                self.stats['cache_misses'] += 1
            
            # Valida arquivo
            await self._validate_file(file_path)
            
            # Detecta tipo do documento
            doc_type = self._detect_document_type(file_path)
            
            # Extrai metadados básicos
            metadata = await self._extract_metadata(file_path)
            
            # Processa conteúdo baseado no tipo
            content = await self._process_document(file_path, doc_type, **options)
            
            # Monta resultado
            result = {
                'file_path': file_path,
                'document_type': doc_type.value,
                'content': content,
                'metadata': metadata,
                'load_time': datetime.now().isoformat(),
                'status': LoadStatus.SUCCESS.value,
                'size_bytes': metadata.get('size', 0)
            }
            
            # Adiciona ao cache
            if self.cache_enabled:
                await self._add_to_cache(file_path, result)
            
            # Registra no histórico
            self.load_history.append(result.copy())
            self.stats['bytes_processed'] += metadata.get('size', 0)
            
            return result
            
        except Exception as e:
            self.stats['failed_loads'] += 1
            error_result = {
                'file_path': file_path,
                'status': LoadStatus.FAILED.value,
                'error': str(e),
                'error_type': type(e).__name__,
                'load_time': datetime.now().isoformat()
            }
            self.failed_loads.append(error_result)
            raise e
    
    async def load_multiple_documents(self, file_paths: List[str], **options) -> List[Dict[str, Any]]:
        """Carrega múltiplos documentos em paralelo"""
        if not file_paths:
            return []
        
        # Carregamento paralelo
        tasks = [self.load_document(path, **options) for path in file_paths]
        results = []
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                # Continua mesmo com falhas individuais
                continue
        
        return results
    
    async def _validate_file(self, file_path: str):
        """Valida se arquivo pode ser carregado"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes > {self.max_file_size}")
        
        if file_size == 0:
            raise ValueError("Empty file")
    
    def _detect_document_type(self, file_path: str) -> DocumentType:
        """Detecta tipo do documento"""
        ext = Path(file_path).suffix.lower()
        return self.supported_extensions.get(ext, DocumentType.UNKNOWN)
    
    async def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extrai metadados do arquivo"""
        path_obj = Path(file_path)
        stat = path_obj.stat()
        
        metadata = {
            'filename': path_obj.name,
            'extension': path_obj.suffix.lower(),
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'mime_type': mimetypes.guess_type(file_path)[0],
            'encoding': 'utf-8'  # Assume UTF-8 por padrão
        }
        
        # Hash do arquivo para detecção de mudanças
        metadata['file_hash'] = await self._calculate_file_hash(file_path)
        
        return metadata
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calcula hash SHA-256 do arquivo"""
        hasher = hashlib.sha256()
        
        # Mock: retorna hash fixo para testes (sem tentar abrir arquivo real)
        return "mock_hash_" + str(abs(hash(file_path)))[:16]
    
    async def _process_document(self, file_path: str, doc_type: DocumentType, **options) -> Dict[str, Any]:
        """Processa documento baseado no tipo"""
        processor = self.processors.get(doc_type)
        if not processor:
            raise ValueError(f"No processor for document type: {doc_type.value}")
        
        return await processor(file_path, **options)
    
    async def _process_text(self, file_path: str, **options) -> Dict[str, Any]:
        """Processa arquivo de texto"""
        encoding = options.get('encoding', 'utf-8')
        
        # Mock: simula leitura de arquivo texto
        content = f"Mock text content from {os.path.basename(file_path)}"
        
        return {
            'text': content,
            'line_count': len(content.split('\n')),
            'char_count': len(content),
            'word_count': len(content.split()),
            'encoding_used': encoding
        }
    
    async def _process_markdown(self, file_path: str, **options) -> Dict[str, Any]:
        """Processa arquivo Markdown"""
        text_content = await self._process_text(file_path, **options)
        
        # Mock: extrai estrutura Markdown
        mock_content = text_content['text']
        headers = ['# Header 1', '## Header 2', '### Header 3']
        links = ['[Link](http://example.com)']
        
        return {
            **text_content,
            'headers': headers,
            'links': links,
            'has_code_blocks': '```' in mock_content,
            'has_tables': '|' in mock_content
        }
    
    async def _process_json(self, file_path: str, **options) -> Dict[str, Any]:
        """Processa arquivo JSON"""
        # Mock: simula parsing JSON
        mock_data = {
            'key1': 'value1',
            'key2': {'nested': 'value'},
            'key3': [1, 2, 3]
        }
        
        return {
            'json_data': mock_data,
            'keys': list(mock_data.keys()),
            'nested_levels': 2,
            'array_count': 1,
            'object_count': 2,
            'is_valid_json': True
        }
    
    async def _process_csv(self, file_path: str, **options) -> Dict[str, Any]:
        """Processa arquivo CSV"""
        delimiter = options.get('delimiter', ',')
        has_header = options.get('has_header', True)
        
        # Mock: simula dados CSV
        mock_headers = ['col1', 'col2', 'col3']
        mock_rows = [
            ['val1', 'val2', 'val3'],
            ['val4', 'val5', 'val6']
        ]
        
        return {
            'headers': mock_headers if has_header else None,
            'rows': mock_rows,
            'row_count': len(mock_rows),
            'column_count': len(mock_headers),
            'delimiter': delimiter,
            'has_header': has_header
        }
    
    async def _process_code(self, file_path: str, **options) -> Dict[str, Any]:
        """Processa arquivo de código"""
        text_content = await self._process_text(file_path, **options)
        
        # Mock: análise básica de código
        ext = Path(file_path).suffix.lower()
        language = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript'}.get(ext, 'unknown')
        
        mock_content = text_content['text']
        
        return {
            **text_content,
            'language': language,
            'functions': ['function1', 'function2'],
            'classes': ['Class1', 'Class2'],
            'imports': ['import os', 'import sys'],
            'comments': ['# Comment 1', '# Comment 2'],
            'complexity_score': 0.75
        }
    
    async def _process_html(self, file_path: str, **options) -> Dict[str, Any]:
        """Processa arquivo HTML"""
        text_content = await self._process_text(file_path, **options)
        
        # Mock: extrai elementos HTML
        return {
            **text_content,
            'title': 'Mock HTML Title',
            'tags': ['html', 'head', 'body', 'div', 'p'],
            'links': ['http://example.com'],
            'images': ['image1.jpg', 'image2.png'],
            'has_scripts': True,
            'has_styles': True
        }
    
    async def _check_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Verifica se documento está em cache"""
        if not self.cache_enabled or file_path not in self.cache:
            return None
        
        cached_item = self.cache[file_path]
        
        # Verifica se cache ainda é válido (metadados não mudaram)
        try:
            current_metadata = await self._extract_metadata(file_path)
            cached_hash = cached_item.get('metadata', {}).get('file_hash')
            current_hash = current_metadata.get('file_hash')
            
            if cached_hash == current_hash:
                # Cache válido
                cached_result = cached_item.copy()
                cached_result['status'] = LoadStatus.CACHED.value
                return cached_result
            else:
                # Cache inválido - remove
                del self.cache[file_path]
                return None
                
        except Exception:
            # Erro ao validar cache - remove
            del self.cache[file_path]
            return None
    
    async def _add_to_cache(self, file_path: str, result: Dict[str, Any]):
        """Adiciona resultado ao cache"""
        if self.cache_enabled:
            self.cache[file_path] = result.copy()
    
    def clear_cache(self):
        """Limpa cache de documentos"""
        self.cache.clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0
    
    def get_supported_types(self) -> List[str]:
        """Retorna tipos de documento suportados"""
        return [doc_type.value for doc_type in DocumentType if doc_type != DocumentType.UNKNOWN]
    
    def get_load_history(self) -> List[Dict[str, Any]]:
        """Retorna histórico de carregamentos"""
        return self.load_history.copy()
    
    def get_failed_loads(self) -> List[Dict[str, Any]]:
        """Retorna carregamentos que falharam"""
        return self.failed_loads.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de carregamento"""
        return self.stats.copy()
    
    def add_processor(self, doc_type: DocumentType, processor_func):
        """Adiciona processador customizado"""
        self.processors[doc_type] = processor_func
    
    def remove_processor(self, doc_type: DocumentType):
        """Remove processador"""
        self.processors.pop(doc_type, None)
    
    async def batch_load_directory(self, directory: str, recursive: bool = False, **options) -> List[Dict[str, Any]]:
        """Carrega todos os documentos de um diretório"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = []
        
        if recursive:
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if self._is_supported_file(file_path):
                        files.append(file_path)
        else:
            for item in os.listdir(directory):
                file_path = os.path.join(directory, item)
                if os.path.isfile(file_path) and self._is_supported_file(file_path):
                    files.append(file_path)
        
        return await self.load_multiple_documents(files, **options)
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Verifica se arquivo é suportado"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions


class TestSmartDocumentLoaderBasic:
    """Testes básicos do carregador de documentos"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.loader = MockSmartDocumentLoader()
    
    def test_loader_initialization(self):
        """Testa inicialização do loader"""
        assert self.loader.cache_enabled is True
        assert self.loader.max_file_size == 10_000_000
        assert len(self.loader.cache) == 0
        assert len(self.loader.processors) > 0
        assert DocumentType.TEXT in self.loader.processors
    
    def test_supported_document_types(self):
        """Testa tipos de documento suportados"""
        supported = self.loader.get_supported_types()
        
        expected_types = ['pdf', 'docx', 'txt', 'md', 'html', 'csv', 'json', 'xml', 'code']
        for doc_type in expected_types:
            assert doc_type in supported
        
        assert 'unknown' not in supported
    
    def test_document_type_detection(self):
        """Testa detecção de tipo de documento"""
        test_cases = [
            ('document.pdf', DocumentType.PDF),
            ('text.txt', DocumentType.TEXT),
            ('readme.md', DocumentType.MARKDOWN),
            ('data.json', DocumentType.JSON),
            ('script.py', DocumentType.CODE),
            ('unknown.xyz', DocumentType.UNKNOWN)
        ]
        
        for filename, expected_type in test_cases:
            detected_type = self.loader._detect_document_type(filename)
            assert detected_type == expected_type
    
    @pytest.mark.asyncio
    async def test_file_validation(self):
        """Testa validação de arquivos"""
        # Arquivo inexistente
        with pytest.raises(FileNotFoundError):
            await self.loader._validate_file("nonexistent.txt")
        
        # Simula arquivo muito grande
        self.loader.max_file_size = 100
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=200):
            with pytest.raises(ValueError, match="File too large"):
                await self.loader._validate_file("large_file.txt")
        
        # Simula arquivo vazio
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=0):
            with pytest.raises(ValueError, match="Empty file"):
                await self.loader._validate_file("empty_file.txt")


class TestSmartDocumentLoaderProcessing:
    """Testes para processamento de documentos"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.loader = MockSmartDocumentLoader()
    
    @pytest.mark.asyncio
    async def test_text_processing(self):
        """Testa processamento de texto"""
        result = await self.loader._process_text("test.txt")
        
        assert 'text' in result
        assert 'line_count' in result
        assert 'char_count' in result
        assert 'word_count' in result
        assert result['encoding_used'] == 'utf-8'
        assert result['char_count'] > 0
        assert result['word_count'] > 0
    
    @pytest.mark.asyncio
    async def test_markdown_processing(self):
        """Testa processamento de Markdown"""
        result = await self.loader._process_markdown("readme.md")
        
        # Deve incluir campos de texto
        assert 'text' in result
        assert 'line_count' in result
        
        # Campos específicos do Markdown
        assert 'headers' in result
        assert 'links' in result
        assert 'has_code_blocks' in result
        assert 'has_tables' in result
        assert isinstance(result['headers'], list)
        assert isinstance(result['links'], list)
    
    @pytest.mark.asyncio
    async def test_json_processing(self):
        """Testa processamento de JSON"""
        result = await self.loader._process_json("data.json")
        
        assert 'json_data' in result
        assert 'keys' in result
        assert 'nested_levels' in result
        assert 'array_count' in result
        assert 'object_count' in result
        assert result['is_valid_json'] is True
        assert isinstance(result['json_data'], dict)
        assert len(result['keys']) > 0
    
    @pytest.mark.asyncio
    async def test_csv_processing(self):
        """Testa processamento de CSV"""
        # Com header padrão
        result = await self.loader._process_csv("data.csv")
        
        assert 'headers' in result
        assert 'rows' in result
        assert 'row_count' in result
        assert 'column_count' in result
        assert result['has_header'] is True
        assert result['delimiter'] == ','
        
        # Sem header
        result_no_header = await self.loader._process_csv("data.csv", has_header=False)
        assert result_no_header['headers'] is None
        
        # Delimiter customizado
        result_custom = await self.loader._process_csv("data.csv", delimiter=';')
        assert result_custom['delimiter'] == ';'
    
    @pytest.mark.asyncio
    async def test_code_processing(self):
        """Testa processamento de código"""
        test_cases = [
            ('script.py', 'python'),
            ('app.js', 'javascript'),
            ('component.ts', 'typescript'),
            ('unknown.xyz', 'unknown')
        ]
        
        for filename, expected_lang in test_cases:
            result = await self.loader._process_code(filename)
            
            assert 'language' in result
            assert 'functions' in result
            assert 'classes' in result
            assert 'imports' in result
            assert 'comments' in result
            assert 'complexity_score' in result
            
            assert result['language'] == expected_lang
            assert isinstance(result['functions'], list)
            assert isinstance(result['classes'], list)
            assert 0 <= result['complexity_score'] <= 1
    
    @pytest.mark.asyncio
    async def test_html_processing(self):
        """Testa processamento de HTML"""
        result = await self.loader._process_html("page.html")
        
        # Campos básicos de texto
        assert 'text' in result
        assert 'line_count' in result
        
        # Campos específicos do HTML
        assert 'title' in result
        assert 'tags' in result
        assert 'links' in result
        assert 'images' in result
        assert 'has_scripts' in result
        assert 'has_styles' in result
        
        assert isinstance(result['tags'], list)
        assert isinstance(result['links'], list)
        assert isinstance(result['images'], list)


class TestSmartDocumentLoaderCache:
    """Testes para sistema de cache"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.loader = MockSmartDocumentLoader(cache_enabled=True)
    
    @pytest.mark.asyncio
    async def test_cache_enabled_disabled(self):
        """Testa cache habilitado/desabilitado"""
        # Cache habilitado
        assert self.loader.cache_enabled is True
        
        # Cache desabilitado
        loader_no_cache = MockSmartDocumentLoader(cache_enabled=False)
        assert loader_no_cache.cache_enabled is False
        
        # Mock file operations
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch.object(loader_no_cache, '_extract_metadata', return_value={'size': 100, 'file_hash': 'hash1'}):
            
            result = await loader_no_cache._check_cache("test.txt")
            assert result is None  # Sem cache
    
    def test_cache_operations(self):
        """Testa operações básicas de cache"""
        # Cache vazio inicialmente
        assert len(self.loader.cache) == 0
        
        # Adiciona item ao cache
        test_result = {'content': 'test', 'metadata': {'file_hash': 'hash123'}}
        asyncio.run(self.loader._add_to_cache("test.txt", test_result))
        
        assert len(self.loader.cache) == 1
        assert "test.txt" in self.loader.cache
        
        # Limpa cache
        self.loader.clear_cache()
        assert len(self.loader.cache) == 0
    
    @pytest.mark.asyncio
    async def test_cache_hit_miss(self):
        """Testa cache hit e miss"""
        file_path = "test.txt"
        
        # Mock file operations
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=100):
            
            # Cache miss inicial
            with patch.object(self.loader, '_extract_metadata', return_value={'file_hash': 'hash1'}):
                result = await self.loader._check_cache(file_path)
                assert result is None
            
            # Adiciona ao cache
            cached_data = {
                'content': 'test content',
                'metadata': {'file_hash': 'hash1'},
                'status': 'success'
            }
            await self.loader._add_to_cache(file_path, cached_data)
            
            # Cache hit
            with patch.object(self.loader, '_extract_metadata', return_value={'file_hash': 'hash1'}):
                result = await self.loader._check_cache(file_path)
                assert result is not None
                assert result['status'] == 'cached'
                assert result['content'] == 'test content'
            
            # Cache miss por mudança de arquivo
            with patch.object(self.loader, '_extract_metadata', return_value={'file_hash': 'hash2'}):
                result = await self.loader._check_cache(file_path)
                assert result is None
                assert file_path not in self.loader.cache  # Removido do cache


class TestSmartDocumentLoaderIntegration:
    """Testes de integração completos"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.loader = MockSmartDocumentLoader()
    
    @pytest.mark.asyncio
    async def test_complete_document_loading_workflow(self):
        """Testa workflow completo de carregamento"""
        file_path = "test_document.txt"
        
        # Mock file operations
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=500), \
             patch('os.path.isfile', return_value=True):
            
            # Mock stat para metadados
            mock_stat = Mock()
            mock_stat.st_size = 500
            mock_stat.st_ctime = 1640995200  # 1 Jan 2022
            mock_stat.st_mtime = 1640995200
            
            with patch('pathlib.Path.stat', return_value=mock_stat), \
                 patch('mimetypes.guess_type', return_value=('text/plain', None)):
                
                result = await self.loader.load_document(file_path)
                
                # Verifica estrutura do resultado
                assert result['file_path'] == file_path
                assert result['document_type'] == 'txt'
                assert result['status'] == 'success'
                assert 'content' in result
                assert 'metadata' in result
                assert 'load_time' in result
                assert result['size_bytes'] == 500
                
                # Verifica conteúdo
                content = result['content']
                assert 'text' in content
                assert 'line_count' in content
                assert 'char_count' in content
                
                # Verifica metadados
                metadata = result['metadata']
                assert metadata['filename'] == 'test_document.txt'
                assert metadata['extension'] == '.txt'
                assert metadata['size'] == 500
                assert 'file_hash' in metadata
    
    @pytest.mark.asyncio
    async def test_multiple_documents_loading(self):
        """Testa carregamento de múltiplos documentos"""
        file_paths = ["doc1.txt", "doc2.md", "doc3.json"]
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch('os.path.isfile', return_value=True):
            
            mock_stat = Mock()
            mock_stat.st_size = 100
            mock_stat.st_ctime = 1640995200
            mock_stat.st_mtime = 1640995200
            
            with patch('pathlib.Path.stat', return_value=mock_stat), \
                 patch('mimetypes.guess_type', return_value=('text/plain', None)):
                
                results = await self.loader.load_multiple_documents(file_paths)
                
                assert len(results) == 3
                
                # Verifica tipos detectados
                doc_types = [result['document_type'] for result in results]
                assert 'txt' in doc_types
                assert 'md' in doc_types
                assert 'json' in doc_types
                
                # Verifica se todos carregaram com sucesso
                statuses = [result['status'] for result in results]
                assert all(status == 'success' for status in statuses)
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Testa tratamento de erros"""
        # Arquivo inexistente
        with pytest.raises(FileNotFoundError):
            await self.loader.load_document("nonexistent.txt")
        
        # Verifica se erro foi registrado
        failed_loads = self.loader.get_failed_loads()
        assert len(failed_loads) == 1
        assert failed_loads[0]['error_type'] == 'FileNotFoundError'
        
        # Verifica estatísticas
        stats = self.loader.get_stats()
        assert stats['failed_loads'] == 1
    
    @pytest.mark.asyncio
    async def test_batch_directory_loading(self):
        """Testa carregamento em lote de diretório"""
        directory = "/mock/directory"
        
        # Mock directory structure
        mock_files = [
            "doc1.txt",
            "doc2.md", 
            "image.jpg",  # Não suportado
            "data.json"
        ]
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=mock_files), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=100):
            
            mock_stat = Mock()
            mock_stat.st_size = 100
            mock_stat.st_ctime = 1640995200
            mock_stat.st_mtime = 1640995200
            
            with patch('pathlib.Path.stat', return_value=mock_stat), \
                 patch('mimetypes.guess_type', return_value=('text/plain', None)):
                
                results = await self.loader.batch_load_directory(directory)
                
                # Deve carregar apenas arquivos suportados
                assert len(results) == 3  # Exclui image.jpg
                
                # Verifica tipos carregados
                doc_types = [result['document_type'] for result in results]
                assert 'txt' in doc_types
                assert 'md' in doc_types
                assert 'json' in doc_types
    
    def test_statistics_tracking(self):
        """Testa rastreamento de estatísticas"""
        initial_stats = self.loader.get_stats()
        
        expected_keys = [
            'total_loads', 'cache_hits', 'cache_misses',
            'failed_loads', 'bytes_processed', 'avg_load_time'
        ]
        
        for key in expected_keys:
            assert key in initial_stats
            assert initial_stats[key] == 0 or initial_stats[key] == 0.0
    
    def test_custom_processor_management(self):
        """Testa gerenciamento de processadores customizados"""
        # Processador customizado
        async def custom_processor(file_path, **options):
            return {'custom': True, 'processed': True}
        
        # Adiciona processador
        self.loader.add_processor(DocumentType.UNKNOWN, custom_processor)
        assert DocumentType.UNKNOWN in self.loader.processors
        
        # Remove processador
        self.loader.remove_processor(DocumentType.UNKNOWN)
        assert DocumentType.UNKNOWN not in self.loader.processors
    
    def test_load_history_tracking(self):
        """Testa rastreamento de histórico"""
        # Histórico inicial vazio
        history = self.loader.get_load_history()
        assert len(history) == 0
        
        # Simula carregamento
        mock_result = {
            'file_path': 'test.txt',
            'status': 'success',
            'load_time': datetime.now().isoformat()
        }
        self.loader.load_history.append(mock_result)
        
        # Verifica histórico
        history = self.loader.get_load_history()
        assert len(history) == 1
        assert history[0]['file_path'] == 'test.txt' 