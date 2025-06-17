import pytest
import tempfile
import json
import os
import shutil
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open, Mock
from pathlib import Path
from src.utils.smart_document_loader import SmartDocumentLoader, CSVLoader, JSONLoader, CodeLoader, GitRepoLoader


class TestSmartDocumentLoader:
    """Testes para a classe SmartDocumentLoader."""

    def setup_method(self):
        """Setup para cada teste."""
        self.loader = SmartDocumentLoader()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup após cada teste."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Testa a inicialização do SmartDocumentLoader."""
        loader = SmartDocumentLoader()
        assert loader.basic_loader is not None
        # context_detector pode ser None se dependências opcionais não estiverem disponíveis
        assert loader.context_detector is None or hasattr(loader.context_detector, 'detect_context')
        assert len(loader.loaders) > 0
        
        # Verifica se os loaders esperados estão presentes
        mime_types = [mime for mime, _ in loader.loaders]
        assert "application/pdf" in mime_types
        assert "text/x-python" in mime_types
        assert "application/json" in mime_types
        assert "text/csv" in mime_types

    @patch('magic.from_file')
    @patch('src.utils.smart_document_loader.DocumentLoader')
    def test_detect_and_load_python_file(self, mock_doc_loader, mock_magic):
        """Testa o carregamento de arquivo Python."""
        # Setup mocks
        mock_magic.return_value = "text/x-python"
        mock_loader_instance = MagicMock()
        mock_doc_loader.return_value = mock_loader_instance
        
        # Mock do context detector se disponível
        if self.loader.context_detector:
            with patch.object(self.loader.context_detector, 'detect_context') as mock_context:
                mock_context.return_value = {
                    "language": "python",
                    "symbols": ["function1", "class1"],
                    "imports": ["os", "sys"]
                }
                
                # Mock do CodeLoader
                with patch('builtins.open', mock_open(read_data="def hello(): pass")):
                    result = self.loader.detect_and_load("/path/to/test.py")
                    
                    assert "content" in result
                    assert "metadata" in result
                    assert result["metadata"]["language"] == "python"
                    assert "symbols" in result["metadata"]
        else:
            # Se context_detector não estiver disponível, apenas testa o carregamento básico
            with patch('builtins.open', mock_open(read_data="def hello(): pass")):
                result = self.loader.detect_and_load("/path/to/test.py")
                assert "content" in result
                assert "metadata" in result

    @patch('magic.from_file')
    def test_detect_and_load_json_file(self, mock_magic):
        """Testa o carregamento de arquivo JSON."""
        mock_magic.return_value = "application/json"
        
        test_data = {"key": "value", "number": 42}
        json_content = json.dumps(test_data)
        
        with patch('builtins.open', mock_open(read_data=json_content)):
            if self.loader.context_detector:
                with patch.object(self.loader.context_detector, 'detect_context') as mock_context:
                    mock_context.return_value = {}
                    
                    result = self.loader.detect_and_load("/path/to/test.json")
                    
                    assert "content" in result
                    assert "metadata" in result
                    assert result["metadata"]["loader"] == "JSONLoader"
            else:
                result = self.loader.detect_and_load("/path/to/test.json")
                
                assert "content" in result
                assert "metadata" in result
                assert result["metadata"]["loader"] == "JSONLoader"

    @patch('magic.from_file')
    def test_detect_and_load_csv_file(self, mock_magic):
        """Testa o carregamento de arquivo CSV."""
        mock_magic.return_value = "text/csv"
        
        csv_content = "name,age\nJohn,30\nJane,25"
        
        with patch('pandas.read_csv') as mock_read_csv:
            import pandas as pd
            mock_df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [30, 25]})
            mock_read_csv.return_value = mock_df
            
            if self.loader.context_detector:
                with patch.object(self.loader.context_detector, 'detect_context') as mock_context:
                    mock_context.return_value = {}
                    
                    result = self.loader.detect_and_load("/path/to/test.csv")
                    
                    assert "content" in result
                    assert "metadata" in result
                    assert result["metadata"]["loader"] == "CSVLoader"
                    assert "rows" in result["metadata"]
                    assert "columns" in result["metadata"]
            else:
                result = self.loader.detect_and_load("/path/to/test.csv")
                
                assert "content" in result
                assert "metadata" in result
                assert result["metadata"]["loader"] == "CSVLoader"
                assert "rows" in result["metadata"]
                assert "columns" in result["metadata"]

    @patch('magic.from_file')
    def test_detect_and_load_fallback(self, mock_magic):
        """Testa o fallback quando o tipo MIME não é reconhecido."""
        mock_magic.return_value = "application/unknown"
        
        # Cria um arquivo temporário real para o teste
        test_file = os.path.join(self.temp_dir, "unknown.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        result = self.loader.detect_and_load(test_file)
        
        assert "content" in result
        assert "metadata" in result
        # Deve usar DocumentLoader como fallback
        assert result["metadata"]["loader"] == "DocumentLoader"

    @patch('magic.from_file')
    def test_detect_and_load_magic_exception(self, mock_magic):
        """Testa o comportamento quando magic.from_file falha."""
        mock_magic.side_effect = Exception("Magic failed")
        
        # Cria um arquivo temporário real para o teste
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        result = self.loader.detect_and_load(test_file)
        
        assert "content" in result
        assert "metadata" in result
        # Deve usar DocumentLoader como fallback quando magic falha
        assert result["metadata"]["loader"] == "DocumentLoader"

    def test_detect_and_load_with_context_enrichment(self):
        """Testa o enriquecimento de metadados com contexto de código."""
        with patch('magic.from_file') as mock_magic:
            mock_magic.return_value = "text/x-python"
            
            with patch('builtins.open', mock_open(read_data="import os\ndef main(): pass")):
                if self.loader.context_detector:
                    with patch.object(self.loader.context_detector, 'detect_context') as mock_context:
                        mock_context.return_value = {
                            "language": "python",
                            "symbols": ["main"],
                            "imports": ["os"],
                            "complexity": "low"
                        }
                        
                        result = self.loader.detect_and_load("/path/to/script.py")
                        
                        assert result["metadata"]["language"] == "python"
                        assert "symbols" in result["metadata"]
                        assert "imports" in result["metadata"]
                        assert "complexity" in result["metadata"]
                else:
                    result = self.loader.detect_and_load("/path/to/script.py")
                    assert "content" in result
                    assert "metadata" in result


class TestCSVLoader:
    """Testes para a classe CSVLoader."""

    def test_load_csv(self):
        """Testa o carregamento de arquivo CSV."""
        loader = CSVLoader()
        
        with patch('pandas.read_csv') as mock_read_csv:
            import pandas as pd
            mock_df = pd.DataFrame({
                'name': ['Alice', 'Bob'],
                'age': [25, 30],
                'city': ['NYC', 'LA']
            })
            mock_read_csv.return_value = mock_df
            
            result = loader.load("/path/to/test.csv")
            
            assert "content" in result
            assert "metadata" in result
            assert result["metadata"]["loader"] == "CSVLoader"
            assert result["metadata"]["rows"] == 2
            assert result["metadata"]["columns"] == ['name', 'age', 'city']
            assert result["metadata"]["source"] == "/path/to/test.csv"


class TestJSONLoader:
    """Testes para a classe JSONLoader."""

    def test_load_json(self):
        """Testa o carregamento de arquivo JSON."""
        loader = JSONLoader()
        
        test_data = {
            "users": [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30}
            ],
            "total": 2
        }
        
        json_content = json.dumps(test_data)
        
        with patch('builtins.open', mock_open(read_data=json_content)):
            result = loader.load("/path/to/test.json")
            
            assert "content" in result
            assert "metadata" in result
            assert result["metadata"]["loader"] == "JSONLoader"
            assert result["metadata"]["source"] == "/path/to/test.json"
            
            # Verifica se o conteúdo é JSON válido
            loaded_data = json.loads(result["content"])
            assert loaded_data == test_data


class TestCodeLoader:
    """Testes para a classe CodeLoader."""

    def test_load_python_code(self):
        """Testa o carregamento de código Python."""
        loader = CodeLoader(language="python")
        
        python_code = """import os
import sys

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
"""
        
        with patch('builtins.open', mock_open(read_data=python_code)):
            result = loader.load("/path/to/script.py")
            
            assert "content" in result
            assert "metadata" in result
            assert result["content"] == python_code
            assert result["metadata"]["loader"] == "CodeLoader"
            assert result["metadata"]["language"] == "python"
            assert result["metadata"]["source"] == "/path/to/script.py"

    def test_load_csharp_code(self):
        """Testa o carregamento de código C#."""
        loader = CodeLoader(language="csharp")
        
        csharp_code = """using System;

namespace HelloWorld
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
        }
    }
}
"""
        
        with patch('builtins.open', mock_open(read_data=csharp_code)):
            result = loader.load("/path/to/Program.cs")
            
            assert "content" in result
            assert "metadata" in result
            assert result["content"] == csharp_code
            assert result["metadata"]["loader"] == "CodeLoader"
            assert result["metadata"]["language"] == "csharp"
            assert result["metadata"]["source"] == "/path/to/Program.cs"

    def test_load_default_language(self):
        """Testa o carregamento com linguagem padrão."""
        loader = CodeLoader()  # Deve usar "python" como padrão
        
        code = "print('Hello')"
        
        with patch('builtins.open', mock_open(read_data=code)):
            result = loader.load("/path/to/code.py")
            
            assert result["metadata"]["language"] == "python"


class TestGitRepoLoader:
    """Testes para a classe GitRepoLoader."""

    def test_init_default_extensions(self):
        """Testa a inicialização com extensões padrão."""
        loader = GitRepoLoader()
        
        expected_exts = [".md", ".py", ".cs", ".js", ".tsx", ".yaml", ".json"]
        assert loader.important_exts == expected_exts

    def test_init_custom_extensions(self):
        """Testa a inicialização com extensões customizadas."""
        custom_exts = [".py", ".js", ".html"]
        loader = GitRepoLoader(important_exts=custom_exts)
        
        assert loader.important_exts == custom_exts

    @patch('git.Repo')
    def test_get_repo_info(self, mock_repo_class):
        """Testa a extração de informações do repositório."""
        loader = GitRepoLoader()
        
        # Mock do repositório
        mock_repo = MagicMock()
        mock_repo.heads = [MagicMock(name="main"), MagicMock(name="develop")]
        mock_repo.head.commit.hexsha = "abc123def456"
        mock_repo.head.commit.committed_date = 1640995200  # 2022-01-01
        mock_repo.head.commit.author.name = "Test Author"
        
        repo_info = loader._get_repo_info(mock_repo)
        
        assert "branches" in repo_info
        assert "latest_commit" in repo_info
        assert "commit_date" in repo_info
        assert "author" in repo_info
        assert repo_info["latest_commit"] == "abc123def456"
        assert repo_info["author"] == "Test Author"

    @patch('git.Repo')
    def test_load_repo(self, mock_repo_class):
        """Testa o carregamento de repositório Git."""
        loader = GitRepoLoader(important_exts=[".py", ".md"])
        
        # Mock do repositório e arquivos
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        
        # Mock dos arquivos no repositório
        mock_file1 = MagicMock()
        mock_file1.path = "src/main.py"
        mock_file1.data_stream.read.return_value = b"def main(): pass"
        
        mock_file2 = MagicMock()
        mock_file2.path = "README.md"
        mock_file2.data_stream.read.return_value = b"# Project Title"
        
        mock_file3 = MagicMock()
        mock_file3.path = "config.txt"  # Não deve ser incluído
        
        # Mock do tree e traverse
        mock_tree = MagicMock()
        mock_tree.traverse.return_value = iter([mock_file1, mock_file2, mock_file3])
        mock_repo.tree.return_value = mock_tree
        
        # Mock dos commits
        mock_commit = MagicMock()
        mock_commit.committed_date = 1640995200
        mock_commit.author.name = "Developer"
        mock_commit.message = "Initial commit"
        
        mock_repo.iter_commits.side_effect = lambda paths=None, max_count=None: iter([mock_commit])
        
        # Mock das informações do repo
        mock_repo.heads = [MagicMock(name="main")]
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.head.commit.committed_date = 1640995200
        mock_repo.head.commit.author.name = "Test Author"
        
        result = loader.load("/path/to/repo")
        
        assert "documents" in result
        assert "repo_info" in result
        assert len(result["documents"]) == 2  # Apenas .py e .md
        
        # Verifica se os documentos corretos foram carregados
        doc_paths = [doc["metadata"]["source"] for doc in result["documents"]]
        assert "src/main.py" in doc_paths
        assert "README.md" in doc_paths
        assert "config.txt" not in doc_paths

    @patch('git.Repo')
    def test_load_repo_with_read_error(self, mock_repo_class):
        """Testa o tratamento de erros de leitura de arquivos."""
        loader = GitRepoLoader(important_exts=[".py"])
        
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        
        # Mock de arquivo que falha na leitura
        mock_file = MagicMock()
        mock_file.path = "broken.py"
        mock_file.data_stream.read.side_effect = Exception("Read error")
        
        # Mock do tree e traverse
        mock_tree = MagicMock()
        mock_tree.traverse.return_value = iter([mock_file])
        mock_repo.tree.return_value = mock_tree
        
        # Mock das informações do repo
        mock_repo.heads = []
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.head.commit.committed_date = 1640995200
        mock_repo.head.commit.author.name = "Test Author"
        
        result = loader.load("/path/to/repo")
        
        # Deve retornar resultado vazio mas sem falhar
        assert "documents" in result
        assert "repo_info" in result
        assert len(result["documents"]) == 0


class TestSmartDocumentLoaderEdgeCases:
    """Testes para casos especiais do SmartDocumentLoader."""

    def setup_method(self):
        """Configuração para cada teste."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = SmartDocumentLoader()

    def teardown_method(self):
        """Limpeza após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('magic.from_file')
    @patch('src.utils.smart_document_loader.DocumentLoader')
    def test_detect_and_load_pdf_file(self, mock_doc_loader_class, mock_magic):
        """Testa o carregamento de arquivo PDF."""
        mock_magic.return_value = "application/pdf"
        
        # Mock da instância do DocumentLoader
        mock_doc_loader = MagicMock()
        mock_doc_loader.load.return_value = {
            "content": "PDF content",
            "metadata": {"source": "/path/to/document.pdf", "filename": "document.pdf", "loader": "DocumentLoader"}
        }
        mock_doc_loader_class.return_value = mock_doc_loader
        
        loader = SmartDocumentLoader()
        result = loader.detect_and_load("/path/to/file.pdf")
        
        assert result["content"] == "PDF content"
        assert "source" in result["metadata"]
        mock_magic.assert_called_once_with("/path/to/file.pdf", mime=True)

    @patch('magic.from_file')
    @patch('src.utils.smart_document_loader.DocumentLoader')
    def test_detect_and_load_docx_file(self, mock_doc_loader_class, mock_magic):
        """Testa o carregamento de arquivo DOCX."""
        mock_magic.return_value = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        # Mock da instância do DocumentLoader
        mock_doc_loader = MagicMock()
        mock_doc_loader.load.return_value = {
            "content": "DOCX content",
            "metadata": {"source": "/path/to/document.docx", "filename": "document.docx", "loader": "DocumentLoader"}
        }
        mock_doc_loader_class.return_value = mock_doc_loader
        
        loader = SmartDocumentLoader()
        result = loader.detect_and_load("/path/to/file.docx")
        
        assert result["content"] == "DOCX content"
        assert "source" in result["metadata"]
        mock_magic.assert_called_once_with("/path/to/file.docx", mime=True)

    @patch('magic.from_file')
    @patch('src.utils.smart_document_loader.DocumentLoader')
    def test_detect_and_load_markdown_file(self, mock_doc_loader_class, mock_magic):
        """Testa o carregamento de arquivo Markdown."""
        mock_magic.return_value = "text/markdown"
        
        # Mock da instância do DocumentLoader
        mock_doc_loader = MagicMock()
        mock_doc_loader.load.return_value = {
            "content": "# Título\n\nConteúdo markdown",
            "metadata": {"source": "/path/to/README.md", "filename": "README.md", "loader": "DocumentLoader"}
        }
        mock_doc_loader_class.return_value = mock_doc_loader
        
        loader = SmartDocumentLoader()
        result = loader.detect_and_load("/path/to/file.md")
        
        assert result["content"] == "# Título\n\nConteúdo markdown"
        assert "source" in result["metadata"]
        mock_magic.assert_called_once_with("/path/to/file.md", mime=True)

    @patch('magic.from_file')
    def test_fallback_by_extension_when_magic_fails(self, mock_magic):
        """Testa o fallback por extensão quando magic falha."""
        mock_magic.side_effect = Exception("Magic failed")
        
        # Testa arquivo de texto (extensão suportada pelo DocumentLoader)
        test_file = os.path.join(self.temp_dir, "script.txt")
        with open(test_file, 'w') as f:
            f.write("print('hello')")
        
        result = self.loader.detect_and_load(test_file)
        
        assert "content" in result
        assert "metadata" in result
        assert "source" in result["metadata"]
        # O SmartDocumentLoader adiciona o metadado 'loader' no fallback
        assert result["metadata"]["loader"] == "DocumentLoader"

    def test_empty_file_handling(self):
        """Testa o comportamento com arquivos vazios."""
        # Cria um arquivo vazio
        test_file = os.path.join(self.temp_dir, "empty.txt")
        with open(test_file, 'w') as f:
            pass  # Arquivo vazio
        
        with patch('magic.from_file') as mock_magic:
            mock_magic.return_value = "text/plain"
            
            result = self.loader.detect_and_load(test_file)
            
            assert "content" in result
            assert "metadata" in result
            assert result["content"] == ""
            # O DocumentLoader não adiciona 'loader' automaticamente, 
            # mas o SmartDocumentLoader adiciona no fallback
            assert "source" in result["metadata"]

    def test_different_encodings(self):
        """Testa o carregamento de arquivos com diferentes encodings."""
        # Teste com UTF-8
        test_file_utf8 = os.path.join(self.temp_dir, "utf8.txt")
        with open(test_file_utf8, 'w', encoding='utf-8') as f:
            f.write("Texto com acentos: ção, ã, é")
        
        with patch('magic.from_file') as mock_magic:
            mock_magic.return_value = "text/plain"
            
            result = self.loader.detect_and_load(test_file_utf8)
            
            assert "content" in result
            assert "metadata" in result
            assert "ção" in result["content"]

    def test_json_loader_invalid_json(self):
        """Testa o tratamento de erro no JSONLoader com JSON inválido."""
        loader = JSONLoader()
        
        # Cria arquivo com JSON inválido
        test_file = os.path.join(self.temp_dir, "invalid.json")
        with open(test_file, 'w') as f:
            f.write('{"key": invalid_value}')
        
        with pytest.raises(json.JSONDecodeError):
            loader.load(test_file)

    def test_csv_loader_malformed_csv(self):
        """Testa o CSVLoader com arquivo CSV malformado."""
        loader = CSVLoader()
        
        # Cria arquivo CSV malformado
        test_file = os.path.join(self.temp_dir, "malformed.csv")
        with open(test_file, 'w') as f:
            f.write('name,age\nJohn,30\nJane')
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = pd.errors.ParserError("Error parsing CSV")
            
            with pytest.raises(pd.errors.ParserError):
                loader.load(test_file)

    @patch('os.access')
    def test_file_permission_error(self, mock_access):
        """Testa o comportamento com arquivos sem permissão de leitura."""
        mock_access.return_value = False
        
        test_file = os.path.join(self.temp_dir, "no_permission.txt")
        with open(test_file, 'w') as f:
            f.write("content")
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                self.loader.detect_and_load(test_file)


class TestGitRepoLoaderEdgeCases:
    """Testes para casos especiais do GitRepoLoader."""

    @patch('git.Repo')
    def test_repo_without_commits(self, mock_repo_class):
        """Testa repositório sem commits."""
        loader = GitRepoLoader(important_exts=[".py"])
        
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        
        # Mock de arquivo sem commits
        mock_file = MagicMock()
        mock_file.path = "empty.py"
        mock_file.data_stream.read.return_value = b"# empty file"
        
        # Mock do tree e traverse
        mock_tree = MagicMock()
        mock_tree.traverse.return_value = iter([mock_file])
        mock_repo.tree.return_value = mock_tree
        
        # Simula repositório sem commits - usar StopIteration para simular iterador vazio
        def empty_commits(*args, **kwargs):
            return iter([])
        
        mock_repo.iter_commits = empty_commits
        
        # Mock das informações do repo
        mock_repo.heads = []
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.head.commit.committed_date = 1640995200
        mock_repo.head.commit.author.name = "Test Author"
        
        # O teste deve falhar graciosamente quando não há commits
        with pytest.raises(StopIteration):
            loader.load("/path/to/repo")
    
    @patch('git.Repo')
    def test_repo_with_no_important_files(self, mock_repo_class):
        """Testa repositório sem arquivos importantes."""
        loader = GitRepoLoader(important_exts=[".py"])
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock de arquivo que não tem extensão importante
        mock_file = Mock()
        mock_file.path = "README.txt"  # Não está na lista de extensões importantes
        
        # Mock do tree e traverse
        mock_tree = Mock()
        mock_tree.traverse.return_value = iter([mock_file])
        mock_repo.tree.return_value = mock_tree
        
        # Mock das informações do repo
        mock_repo.heads = [Mock(name="main")]
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.head.commit.committed_date = 1640995200
        mock_repo.head.commit.author.name = "Test Author"
        
        result = loader.load("/path/to/repo")
        
        # Deve retornar lista vazia de documentos
        assert "documents" in result
        assert "repo_info" in result
        assert len(result["documents"]) == 0
        assert result["repo_info"]["latest_commit"] == "abc123"


class TestSmartDocumentLoaderWithCodeContext:
    """Testes para SmartDocumentLoader com CodeContextDetector."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('src.utils.smart_document_loader.CodeContextDetector')
    def test_code_context_enrichment_with_language(self, mock_context_detector_class):
        """Testa enriquecimento de metadados com CodeContextDetector quando há linguagem."""
        # Cria arquivo temporário
        test_file = os.path.join(self.temp_dir, "test_file.py")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("print('hello world')")
        
        # Mock do CodeContextDetector
        mock_detector = Mock()
        mock_context_detector_class.return_value = mock_detector
        mock_detector.detect_context.return_value = {
            "language": "python",
            "framework": "django",
            "complexity": "medium"
        }
        
        # Mock do magic para detectar tipo MIME
        with patch('magic.from_file', return_value="text/x-python"):
            loader = SmartDocumentLoader()
            result = loader.detect_and_load(test_file)
            
            # Verifica se o contexto foi adicionado aos metadados
            assert "language" in result["metadata"]
            assert "framework" in result["metadata"]
            assert "complexity" in result["metadata"]
            assert result["metadata"]["language"] == "python"
            assert result["metadata"]["framework"] == "django"
            
            # Verifica se o detector foi chamado
            mock_detector.detect_context.assert_called_once_with(
                file_path=test_file,
                code="print('hello world')"
            )
    
    @patch('src.utils.smart_document_loader.CodeContextDetector')
    def test_code_context_enrichment_without_language(self, mock_context_detector_class):
        """Testa enriquecimento quando CodeContextDetector não detecta linguagem."""
        # Cria arquivo temporário
        test_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Some text content")
        
        # Mock do CodeContextDetector
        mock_detector = Mock()
        mock_context_detector_class.return_value = mock_detector
        mock_detector.detect_context.return_value = {
            "complexity": "low"
            # Sem 'language'
        }
        
        # Mock do magic para detectar tipo MIME
        with patch('magic.from_file', return_value="text/plain"):
            loader = SmartDocumentLoader()
            result = loader.detect_and_load(test_file)
            
            # Verifica que contexto não foi adicionado (sem language)
            assert "complexity" not in result["metadata"]
            assert "language" not in result["metadata"]
            
            # Verifica se o detector foi chamado
            mock_detector.detect_context.assert_called_once_with(
                file_path=test_file,
                code="Some text content"
            )
    
    def test_without_code_context_detector(self):
        """Testa SmartDocumentLoader quando CodeContextDetector não está disponível."""
        # Cria arquivo temporário de texto simples (não código)
        test_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("This is a simple text file")
        
        # Mock do magic para detectar tipo MIME como texto simples
        with patch('magic.from_file', return_value="text/plain"):
            # Mock do CodeContextDetector para simular que não está disponível
            with patch('src.utils.smart_document_loader.CodeContextDetector', None):
                loader = SmartDocumentLoader()
                result = loader.detect_and_load(test_file)
                
                # Verifica que não há enriquecimento de contexto
                assert "language" not in result["metadata"]
                assert "framework" not in result["metadata"]
                assert result["content"] == "This is a simple text file"
                assert result["metadata"]["source"] == test_file


class TestJSONLoaderSpecific:
    """Testes específicos para JSONLoader."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.loader = JSONLoader()
        
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_load_complex_json(self):
        """Testa carregamento de JSON complexo."""
        test_file = os.path.join(self.temp_dir, "complex.json")
        complex_data = {
            "users": [
                {"name": "John", "age": 30, "active": True},
                {"name": "Jane", "age": 25, "active": False}
            ],
            "metadata": {
                "version": "1.0",
                "created": "2023-01-01"
            }
        }
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(complex_data, f)
        
        result = self.loader.load(test_file)
        
        assert "content" in result
        assert "metadata" in result
        assert result["metadata"]["loader"] == "JSONLoader"
        assert result["metadata"]["source"] == test_file
        
        # Verifica se o conteúdo é JSON formatado
        loaded_data = json.loads(result["content"])
        assert loaded_data == complex_data
    
    def test_load_invalid_json(self):
        """Testa carregamento de JSON inválido."""
        test_file = os.path.join(self.temp_dir, "invalid.json")
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('{"name": "John", "age":}')  # JSON inválido
        
        with pytest.raises(json.JSONDecodeError):
            self.loader.load(test_file)


class TestCodeLoaderSpecific:
    """Testes específicos para CodeLoader."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_code_loader_with_custom_language(self):
        """Testa CodeLoader com linguagem personalizada."""
        loader = CodeLoader(language="javascript")
        test_file = os.path.join(self.temp_dir, "script.js")
        
        code_content = "function hello() { console.log('Hello World'); }"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        result = loader.load(test_file)
        
        assert result["content"] == code_content
        assert result["metadata"]["language"] == "javascript"
        assert result["metadata"]["loader"] == "CodeLoader"
        assert result["metadata"]["source"] == test_file
    
    def test_code_loader_default_language(self):
        """Testa CodeLoader com linguagem padrão (python)."""
        loader = CodeLoader()  # Sem especificar linguagem
        test_file = os.path.join(self.temp_dir, "script.py")
        
        code_content = "def hello():\n    print('Hello World')"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        result = loader.load(test_file)
        
        assert result["content"] == code_content
        assert result["metadata"]["language"] == "python"
        assert result["metadata"]["loader"] == "CodeLoader"
    
    def test_code_loader_with_encoding_errors(self):
        """Testa CodeLoader com erros de encoding (deve ignorar)."""
        loader = CodeLoader(language="python")
        test_file = os.path.join(self.temp_dir, "bad_encoding.py")
        
        # Escreve conteúdo com caracteres problemáticos
        with open(test_file, 'wb') as f:
            f.write(b'# -*- coding: utf-8 -*-\nprint("\xff\xfe")')  # Bytes inválidos
        
        result = loader.load(test_file)
        
        # Deve carregar mesmo com erros de encoding (errors="ignore")
        assert "content" in result
        assert result["metadata"]["language"] == "python"


class TestSmartDocumentLoaderMimeMatching:
    """Testes para correspondência de MIME types no SmartDocumentLoader."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('magic.from_file')
    def test_partial_mime_matching(self, mock_magic):
        """Testa correspondência parcial de MIME types."""
        # Simula MIME type que contém o padrão
        mock_magic.return_value = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        with patch('src.utils.smart_document_loader.DocumentLoader') as mock_doc_loader_class:
            mock_doc_loader = Mock()
            mock_doc_loader_class.return_value = mock_doc_loader
            mock_doc_loader.load.return_value = {
                "content": "Document content",
                "metadata": {"source": "/test/file.docx"}
            }
            
            loader = SmartDocumentLoader()
            result = loader.detect_and_load("/test/file.docx")
            
            # Deve usar o DocumentLoader para DOCX
            mock_doc_loader.load.assert_called_once_with("/test/file.docx")
            assert result["content"] == "Document content"
    
    @patch('magic.from_file')
    def test_no_mime_match_fallback(self, mock_magic):
        """Testa fallback quando nenhum MIME type corresponde."""
        # Simula MIME type que não corresponde a nenhum loader
        mock_magic.return_value = "application/unknown-type"
        
        with patch('src.utils.smart_document_loader.DocumentLoader') as mock_doc_loader_class:
            mock_doc_loader = Mock()
            mock_doc_loader_class.return_value = mock_doc_loader
            mock_doc_loader.load.return_value = {
                "content": "Unknown content",
                "metadata": {"source": "/test/file.unknown"}
            }
            
            loader = SmartDocumentLoader()
            result = loader.detect_and_load("/test/file.unknown")
            
            # Deve usar fallback (DocumentLoader)
            mock_doc_loader.load.assert_called_once_with("/test/file.unknown")
            assert result["metadata"]["loader"] == "DocumentLoader"
    
    @patch('magic.from_file')
    def test_git_repo_mime_detection(self, mock_magic):
        """Testa detecção de repositório Git."""
        mock_magic.return_value = "application/x-git"
        
        with patch('src.utils.smart_document_loader.GitRepoLoader') as mock_git_loader_class:
            mock_git_loader = Mock()
            mock_git_loader_class.return_value = mock_git_loader
            mock_git_loader.load.return_value = {
                "documents": [],
                "repo_info": {"latest_commit": "abc123"}
            }
            
            loader = SmartDocumentLoader()
            result = loader.detect_and_load("/path/to/repo")
            
            # Deve usar GitRepoLoader
            mock_git_loader.load.assert_called_once_with("/path/to/repo")
            assert "documents" in result
            assert "repo_info" in result
 
    @patch('git.Repo')
    def test_repo_with_binary_files(self, mock_repo_class):
        """Testa repositório com arquivos binários."""
        loader = GitRepoLoader(important_exts=[".py", ".png"])
        
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        
        # Mock de arquivo binário
        mock_binary_file = MagicMock()
        mock_binary_file.path = "image.png"
        mock_binary_file.data_stream.read.return_value = b"\x89PNG\r\n\x1a\n"  # PNG header
        
        # Mock de arquivo texto
        mock_text_file = MagicMock()
        mock_text_file.path = "script.py"
        mock_text_file.data_stream.read.return_value = b"print('hello')"
        
        # Mock do tree e traverse
        mock_tree = MagicMock()
        mock_tree.traverse.return_value = iter([mock_binary_file, mock_text_file])
        mock_repo.tree.return_value = mock_tree
        
        # Mock dos commits
        mock_commit = MagicMock()
        mock_commit.committed_date = 1640995200
        mock_commit.author.name = "Developer"
        mock_commit.message = "Add files"
        
        mock_repo.iter_commits.side_effect = lambda paths=None, max_count=None: iter([mock_commit])
        
        # Mock das informações do repo
        mock_repo.heads = [MagicMock(name="main")]
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.head.commit.committed_date = 1640995200
        mock_repo.head.commit.author.name = "Test Author"
        
        result = loader.load("/path/to/repo")
        
        assert "documents" in result
        assert "repo_info" in result
        # Deve carregar ambos os arquivos (texto e binário)
        assert len(result["documents"]) == 2
        
        # Verifica se os documentos corretos foram carregados
        doc_paths = [doc["metadata"]["source"] for doc in result["documents"]]
        assert "image.png" in doc_paths
        assert "script.py" in doc_paths

    @patch('git.Repo')
    def test_repo_with_submodules(self, mock_repo_class):
        """Testa repositório com submodules."""
        loader = GitRepoLoader(important_exts=[".py"])
        
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        
        # Mock de arquivo no submodule
        mock_submodule_file = MagicMock()
        mock_submodule_file.path = "submodule/lib.py"
        mock_submodule_file.data_stream.read.return_value = b"def helper(): pass"
        
        # Mock de arquivo principal
        mock_main_file = MagicMock()
        mock_main_file.path = "main.py"
        mock_main_file.data_stream.read.return_value = b"import submodule.lib"
        
        # Mock do tree e traverse
        mock_tree = MagicMock()
        mock_tree.traverse.return_value = iter([mock_submodule_file, mock_main_file])
        mock_repo.tree.return_value = mock_tree
        
        # Mock dos commits
        mock_commit = MagicMock()
        mock_commit.committed_date = 1640995200
        mock_commit.author.name = "Developer"
        mock_commit.message = "Add submodule"
        
        mock_repo.iter_commits.side_effect = lambda paths=None, max_count=None: iter([mock_commit])
        
        # Mock das informações do repo
        mock_repo.heads = [MagicMock(name="main")]
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.head.commit.committed_date = 1640995200
        mock_repo.head.commit.author.name = "Test Author"
        
        result = loader.load("/path/to/repo")
        
        assert "documents" in result
        assert "repo_info" in result
        assert len(result["documents"]) == 2
        
        # Verifica se os documentos corretos foram carregados
        doc_paths = [doc["metadata"]["source"] for doc in result["documents"]]
        assert "submodule/lib.py" in doc_paths
        assert "main.py" in doc_paths