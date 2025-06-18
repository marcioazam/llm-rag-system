"""Testes para o módulo document_loader.py."""
import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
import tempfile
import os
from typing import List, Dict, Any

# Mock das dependências antes do import
with patch.multiple(
    'sys.modules',
    PyPDF2=Mock(),
    docx=Mock(),
    pandas=Mock()
):
    from src.ingestion.document_loader import DocumentLoader


class TestDocumentLoader:
    """Testes para DocumentLoader."""
    
    @pytest.fixture
    def temp_files(self):
        """Cria arquivos temporários para teste."""
        files = {}
        
        # Arquivo de texto
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test text file.\nWith multiple lines.")
            files['txt'] = f.name
        
        # Arquivo markdown
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Markdown\n\nThis is a **test** markdown file.")
            files['md'] = f.name
        
        yield files
        
        # Cleanup
        for file_path in files.values():
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_init_default_parameters(self):
        """Testa inicialização com parâmetros padrão."""
        loader = DocumentLoader()
        
        assert loader.supported_formats == ['.txt', '.md', '.pdf', '.docx', '.csv', '.json']
        assert loader.max_file_size == 10 * 1024 * 1024  # 10MB
    
    def test_init_custom_parameters(self):
        """Testa inicialização com parâmetros customizados."""
        custom_formats = ['.txt', '.md']
        custom_max_size = 5 * 1024 * 1024  # 5MB
        
        loader = DocumentLoader(
            supported_formats=custom_formats,
            max_file_size=custom_max_size
        )
        
        assert loader.supported_formats == custom_formats
        assert loader.max_file_size == custom_max_size
    
    def test_load_text_file_success(self, temp_files):
        """Testa carregamento de arquivo de texto bem-sucedido."""
        loader = DocumentLoader()
        
        result = loader.load_file(temp_files['txt'])
        
        assert result['content'] == "This is a test text file.\nWith multiple lines."
        assert result['metadata']['file_path'] == temp_files['txt']
        assert result['metadata']['file_type'] == '.txt'
        assert 'file_size' in result['metadata']
        assert 'created_at' in result['metadata']
    
    def test_load_markdown_file_success(self, temp_files):
        """Testa carregamento de arquivo markdown bem-sucedido."""
        loader = DocumentLoader()
        
        result = loader.load_file(temp_files['md'])
        
        assert "# Test Markdown" in result['content']
        assert "**test**" in result['content']
        assert result['metadata']['file_type'] == '.md'
    
    def test_load_nonexistent_file(self):
        """Testa erro ao carregar arquivo inexistente."""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
            loader.load_file("nonexistent_file.txt")
    
    def test_load_unsupported_format(self, temp_files):
        """Testa erro com formato não suportado."""
        # Criar arquivo com extensão não suportada
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("test content")
            unsupported_file = f.name
        
        try:
            loader = DocumentLoader()
            
            with pytest.raises(ValueError, match="Formato de arquivo não suportado"):
                loader.load_file(unsupported_file)
        finally:
            os.unlink(unsupported_file)
    
    def test_load_file_too_large(self):
        """Testa erro com arquivo muito grande."""
        loader = DocumentLoader(max_file_size=100)  # 100 bytes
        
        # Criar arquivo grande
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("x" * 200)  # 200 bytes
            large_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Arquivo muito grande"):
                loader.load_file(large_file)
        finally:
            os.unlink(large_file)
    
    @patch('src.ingestion.document_loader.PyPDF2')
    def test_load_pdf_file_success(self, mock_pypdf2):
        """Testa carregamento de arquivo PDF bem-sucedido."""
        # Configurar mock
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF content page 1\n"
        mock_reader.pages = [mock_page, mock_page]  # 2 páginas
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        loader = DocumentLoader()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            pdf_file = f.name
        
        try:
            with patch('builtins.open', mock_open(read_data=b'fake pdf data')):
                result = loader.load_file(pdf_file)
            
            assert "PDF content page 1" in result['content']
            assert result['metadata']['file_type'] == '.pdf'
            assert result['metadata']['pages'] == 2
        finally:
            os.unlink(pdf_file)
    
    @patch('src.ingestion.document_loader.PyPDF2')
    def test_load_pdf_file_error(self, mock_pypdf2):
        """Testa erro ao carregar arquivo PDF."""
        mock_pypdf2.PdfReader.side_effect = Exception("PDF error")
        
        loader = DocumentLoader()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            pdf_file = f.name
        
        try:
            with pytest.raises(Exception, match="Erro ao processar PDF: PDF error"):
                loader.load_file(pdf_file)
        finally:
            os.unlink(pdf_file)
    
    @patch('src.ingestion.document_loader.docx')
    def test_load_docx_file_success(self, mock_docx):
        """Testa carregamento de arquivo DOCX bem-sucedido."""
        # Configurar mock
        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_paragraph.text = "DOCX paragraph content"
        mock_doc.paragraphs = [mock_paragraph, mock_paragraph]
        mock_docx.Document.return_value = mock_doc
        
        loader = DocumentLoader()
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
            docx_file = f.name
        
        try:
            result = loader.load_file(docx_file)
            
            assert "DOCX paragraph content" in result['content']
            assert result['metadata']['file_type'] == '.docx'
            assert result['metadata']['paragraphs'] == 2
        finally:
            os.unlink(docx_file)
    
    @patch('src.ingestion.document_loader.pandas')
    def test_load_csv_file_success(self, mock_pandas):
        """Testa carregamento de arquivo CSV bem-sucedido."""
        # Configurar mock
        mock_df = Mock()
        mock_df.to_string.return_value = "CSV data as string"
        mock_df.shape = (10, 3)  # 10 linhas, 3 colunas
        mock_pandas.read_csv.return_value = mock_df
        
        loader = DocumentLoader()
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_file = f.name
        
        try:
            result = loader.load_file(csv_file)
            
            assert "CSV data as string" in result['content']
            assert result['metadata']['file_type'] == '.csv'
            assert result['metadata']['rows'] == 10
            assert result['metadata']['columns'] == 3
        finally:
            os.unlink(csv_file)
    
    def test_load_json_file_success(self):
        """Testa carregamento de arquivo JSON bem-sucedido."""
        json_content = '{"key": "value", "number": 42, "array": [1, 2, 3]}'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            json_file = f.name
        
        try:
            loader = DocumentLoader()
            result = loader.load_file(json_file)
            
            assert "key" in result['content']
            assert "value" in result['content']
            assert result['metadata']['file_type'] == '.json'
        finally:
            os.unlink(json_file)
    
    def test_load_json_file_invalid(self):
        """Testa erro com arquivo JSON inválido."""
        invalid_json = '{"key": "value", "invalid": }'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(invalid_json)
            json_file = f.name
        
        try:
            loader = DocumentLoader()
            
            with pytest.raises(Exception, match="Erro ao processar JSON"):
                loader.load_file(json_file)
        finally:
            os.unlink(json_file)
    
    def test_load_directory_success(self, temp_files):
        """Testa carregamento de diretório bem-sucedido."""
        loader = DocumentLoader()
        
        # Criar diretório temporário
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copiar arquivos para o diretório
            import shutil
            for file_path in temp_files.values():
                shutil.copy2(file_path, temp_dir)
            
            results = loader.load_directory(temp_dir)
            
            assert len(results) >= 2  # Pelo menos os arquivos txt e md
            assert all('content' in result for result in results)
            assert all('metadata' in result for result in results)
    
    def test_load_directory_nonexistent(self):
        """Testa erro ao carregar diretório inexistente."""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError, match="Diretório não encontrado"):
            loader.load_directory("nonexistent_directory")
    
    def test_load_directory_recursive(self, temp_files):
        """Testa carregamento recursivo de diretório."""
        loader = DocumentLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Criar subdiretório
            sub_dir = os.path.join(temp_dir, "subdir")
            os.makedirs(sub_dir)
            
            # Copiar arquivos
            import shutil
            shutil.copy2(temp_files['txt'], temp_dir)
            shutil.copy2(temp_files['md'], sub_dir)
            
            results = loader.load_directory(temp_dir, recursive=True)
            
            assert len(results) == 2
            file_types = [r['metadata']['file_type'] for r in results]
            assert '.txt' in file_types
            assert '.md' in file_types
    
    def test_load_directory_non_recursive(self, temp_files):
        """Testa carregamento não recursivo de diretório."""
        loader = DocumentLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Criar subdiretório
            sub_dir = os.path.join(temp_dir, "subdir")
            os.makedirs(sub_dir)
            
            # Copiar arquivos
            import shutil
            shutil.copy2(temp_files['txt'], temp_dir)
            shutil.copy2(temp_files['md'], sub_dir)  # No subdiretório
            
            results = loader.load_directory(temp_dir, recursive=False)
            
            assert len(results) == 1  # Apenas arquivo no diretório raiz
            assert results[0]['metadata']['file_type'] == '.txt'
    
    def test_load_multiple_files_success(self, temp_files):
        """Testa carregamento de múltiplos arquivos bem-sucedido."""
        loader = DocumentLoader()
        
        file_paths = list(temp_files.values())
        results = loader.load_files(file_paths)
        
        assert len(results) == len(file_paths)
        assert all('content' in result for result in results)
        assert all('metadata' in result for result in results)
    
    def test_load_multiple_files_with_errors(self, temp_files):
        """Testa carregamento de múltiplos arquivos com alguns erros."""
        loader = DocumentLoader()
        
        file_paths = list(temp_files.values()) + ["nonexistent.txt"]
        results = loader.load_files(file_paths, skip_errors=True)
        
        # Deve carregar apenas os arquivos válidos
        assert len(results) == len(temp_files)
    
    def test_load_multiple_files_fail_on_error(self, temp_files):
        """Testa falha ao carregar múltiplos arquivos com erro."""
        loader = DocumentLoader()
        
        file_paths = list(temp_files.values()) + ["nonexistent.txt"]
        
        with pytest.raises(FileNotFoundError):
            loader.load_files(file_paths, skip_errors=False)
    
    def test_get_file_metadata(self, temp_files):
        """Testa extração de metadados de arquivo."""
        loader = DocumentLoader()
        
        metadata = loader._get_file_metadata(temp_files['txt'])
        
        assert metadata['file_path'] == temp_files['txt']
        assert metadata['file_type'] == '.txt'
        assert metadata['file_size'] > 0
        assert 'created_at' in metadata
        assert 'modified_at' in metadata
    
    def test_validate_file_size_valid(self, temp_files):
        """Testa validação de tamanho de arquivo válido."""
        loader = DocumentLoader(max_file_size=1024 * 1024)  # 1MB
        
        # Não deve levantar exceção
        loader._validate_file_size(temp_files['txt'])
    
    def test_validate_file_size_invalid(self):
        """Testa validação de tamanho de arquivo inválido."""
        loader = DocumentLoader(max_file_size=10)  # 10 bytes
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("x" * 20)  # 20 bytes
            large_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Arquivo muito grande"):
                loader._validate_file_size(large_file)
        finally:
            os.unlink(large_file)
    
    def test_extract_text_encoding_detection(self):
        """Testa detecção de encoding de arquivo."""
        loader = DocumentLoader()
        
        # Criar arquivo com encoding específico
        content = "Texto com acentos: ção, ã, é"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            utf8_file = f.name
        
        try:
            result = loader.load_file(utf8_file)
            assert "acentos" in result['content']
            assert "ção" in result['content']
        finally:
            os.unlink(utf8_file)
    
    def test_performance_with_large_file(self):
        """Testa performance com arquivo grande."""
        import time
        
        loader = DocumentLoader()
        
        # Criar arquivo grande
        large_content = "This is a line of text.\n" * 10000  # ~240KB
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            large_file = f.name
        
        try:
            start_time = time.time()
            result = loader.load_file(large_file)
            end_time = time.time()
            
            # Deve completar rapidamente (menos de 2 segundos)
            assert end_time - start_time < 2.0
            assert len(result['content']) > 200000  # Verificar que conteúdo foi carregado
        finally:
            os.unlink(large_file)
    
    def test_concurrent_file_loading(self, temp_files):
        """Testa carregamento concorrente de arquivos."""
        import threading
        import time
        
        loader = DocumentLoader()
        results = []
        
        def load_worker(file_path):
            result = loader.load_file(file_path)
            results.append(result)
        
        # Criar múltiplas threads
        threads = []
        for file_path in temp_files.values():
            thread = threading.Thread(target=load_worker, args=(file_path,))
            threads.append(thread)
            thread.start()
        
        # Aguardar conclusão
        for thread in threads:
            thread.join()
        
        assert len(results) == len(temp_files)
        assert all('content' in result for result in results)


class TestDocumentLoaderIntegration:
    """Testes de integração para DocumentLoader."""
    
    def test_real_world_document_processing(self):
        """Testa processamento de documentos do mundo real."""
        loader = DocumentLoader()
        
        # Simular diferentes tipos de documentos
        documents = {
            'readme.md': "# Project README\n\nThis is a sample project.\n\n## Features\n\n- Feature 1\n- Feature 2",
            'config.json': '{"database": {"host": "localhost", "port": 5432}, "debug": true}',
            'data.csv': "name,age,city\nJohn,30,NYC\nJane,25,LA\nBob,35,Chicago",
            'notes.txt': "Important notes:\n1. Remember to backup\n2. Update dependencies\n3. Run tests"
        }
        
        temp_files = []
        try:
            # Criar arquivos temporários
            for filename, content in documents.items():
                with tempfile.NamedTemporaryFile(mode='w', suffix=os.path.splitext(filename)[1], delete=False) as f:
                    f.write(content)
                    temp_files.append(f.name)
            
            # Carregar todos os arquivos
            results = loader.load_files(temp_files)
            
            assert len(results) == 4
            
            # Verificar conteúdos específicos
            contents = [r['content'] for r in results]
            assert any('Project README' in content for content in contents)
            assert any('database' in content for content in contents)
            assert any('John,30,NYC' in content for content in contents)
            assert any('Important notes' in content for content in contents)
            
        finally:
            # Cleanup
            for file_path in temp_files:
                if os.path.exists(file_path):
                    os.unlink(file_path)
    
    def test_error_recovery_scenarios(self):
        """Testa cenários de recuperação de erro."""
        loader = DocumentLoader()
        
        # Misturar arquivos válidos e inválidos
        valid_content = "Valid document content"
        
        temp_files = []
        try:
            # Arquivo válido
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(valid_content)
                temp_files.append(f.name)
            
            # Arquivo inexistente
            file_paths = temp_files + ["nonexistent.txt"]
            
            # Com skip_errors=True, deve processar apenas arquivos válidos
            results = loader.load_files(file_paths, skip_errors=True)
            
            assert len(results) == 1
            assert results[0]['content'] == valid_content
            
        finally:
            for file_path in temp_files:
                if os.path.exists(file_path):
                    os.unlink(file_path)