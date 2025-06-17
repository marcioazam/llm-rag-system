import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.metadata.sqlite_store import SQLiteMetadataStore


class TestSQLiteMetadataStore:
    """Testes para a classe SQLiteMetadataStore."""

    def setup_method(self):
        """Setup para cada teste - cria um arquivo temporário."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_file.close()
        self.db_path = self.temp_file.name

    def teardown_method(self):
        """Cleanup após cada teste - remove o arquivo temporário."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_init_default(self):
        """Testa a inicialização com parâmetros padrão."""
        store = SQLiteMetadataStore(self.db_path)
        assert store.db_path == self.db_path
        assert store.conn is not None
        store.close()

    def test_init_custom_path(self):
        """Testa a inicialização com caminho customizado."""
        custom_path = self.db_path
        store = SQLiteMetadataStore(custom_path)
        assert store.db_path == custom_path
        assert os.path.exists(custom_path)
        store.close()

    def test_upsert_metadata(self):
        """Testa o armazenamento de metadados de um chunk."""
        store = SQLiteMetadataStore(self.db_path)
        
        metadata = {
            "id": "test_chunk_1",
            "file_path": "/path/to/test.py",
            "language": "python",
            "symbols": ["function1", "class1"],
            "relations": ["imports", "calls"],
            "coverage": "high",
            "source": "test",
            "chunk_hash": "abc123",
            "project_id": "test_project"
        }
        
        store.upsert_metadata(metadata)
        
        # Verifica se foi armazenado consultando por ID
        result_id = store.get_id_by_filepath("/path/to/test.py")
        assert result_id == "test_chunk_1"
        
        store.close()

    def test_upsert_multiple_metadata(self):
        """Testa o armazenamento de múltiplos metadados."""
        store = SQLiteMetadataStore(self.db_path)
        
        metadatas = [
            {
                "id": f"chunk_{i}",
                "file_path": f"/path/to/file{i}.py",
                "language": "python",
                "symbols": [f"function{i}"],
                "relations": ["imports"],
                "coverage": "medium",
                "source": "test",
                "chunk_hash": f"hash{i}",
                "project_id": "multi_project"
            }
            for i in range(3)
        ]
        
        for metadata in metadatas:
            store.upsert_metadata(metadata)
        
        # Verifica se todos foram armazenados
        for i in range(3):
            result_id = store.get_id_by_filepath(f"/path/to/file{i}.py")
            assert result_id == f"chunk_{i}"
        
        store.close()

    def test_get_id_by_filepath_nonexistent(self):
        """Testa a busca por arquivo inexistente."""
        store = SQLiteMetadataStore(self.db_path)
        
        result = store.get_id_by_filepath("/nonexistent/path.py")
        assert result is None
        
        store.close()

    def test_delete_by_id(self):
        """Testa a exclusão de metadados por ID."""
        store = SQLiteMetadataStore(self.db_path)
        
        metadata = {
            "id": "delete_me",
            "file_path": "/path/to/delete.py",
            "language": "python",
            "symbols": ["temp_function"],
            "relations": [],
            "coverage": "low",
            "source": "test",
            "chunk_hash": "temp_hash",
            "project_id": "temp_project"
        }
        
        store.upsert_metadata(metadata)
        assert store.get_id_by_filepath("/path/to/delete.py") == "delete_me"
        
        store.delete_by_id("delete_me")
        assert store.get_id_by_filepath("/path/to/delete.py") is None
        
        store.close()

    def test_query_by_language(self):
        """Testa a consulta por linguagem."""
        store = SQLiteMetadataStore(self.db_path)
        
        # Armazena chunks de diferentes linguagens
        metadatas = [
            {
                "id": "python_chunk",
                "file_path": "/path/to/script.py",
                "language": "python",
                "symbols": ["main"],
                "relations": [],
                "coverage": "high",
                "source": "test",
                "chunk_hash": "py_hash",
                "project_id": "lang_project"
            },
            {
                "id": "js_chunk",
                "file_path": "/path/to/script.js",
                "language": "javascript",
                "symbols": ["function"],
                "relations": [],
                "coverage": "medium",
                "source": "test",
                "chunk_hash": "js_hash",
                "project_id": "lang_project"
            },
            {
                "id": "python_chunk2",
                "file_path": "/path/to/module.py",
                "language": "python",
                "symbols": ["class"],
                "relations": [],
                "coverage": "low",
                "source": "test",
                "chunk_hash": "py_hash2",
                "project_id": "lang_project"
            }
        ]
        
        for metadata in metadatas:
            store.upsert_metadata(metadata)
        
        # Consulta chunks Python
        python_chunks = list(store.query_by_language("python"))
        assert len(python_chunks) == 2
        
        python_ids = [chunk["id"] for chunk in python_chunks]
        assert "python_chunk" in python_ids
        assert "python_chunk2" in python_ids
        assert "js_chunk" not in python_ids
        
        store.close()

    def test_query_by_project(self):
        """Testa a consulta por projeto."""
        store = SQLiteMetadataStore(self.db_path)
        
        # Armazena chunks de diferentes projetos
        metadatas = [
            {
                "id": "proj1_chunk1",
                "file_path": "/proj1/file1.py",
                "language": "python",
                "symbols": [],
                "relations": [],
                "coverage": "high",
                "source": "test",
                "chunk_hash": "hash1",
                "project_id": "project1"
            },
            {
                "id": "proj1_chunk2",
                "file_path": "/proj1/file2.py",
                "language": "python",
                "symbols": [],
                "relations": [],
                "coverage": "medium",
                "source": "test",
                "chunk_hash": "hash2",
                "project_id": "project1"
            },
            {
                "id": "proj2_chunk1",
                "file_path": "/proj2/file1.py",
                "language": "python",
                "symbols": [],
                "relations": [],
                "coverage": "low",
                "source": "test",
                "chunk_hash": "hash3",
                "project_id": "project2"
            }
        ]
        
        for metadata in metadatas:
            store.upsert_metadata(metadata)
        
        # Consulta chunks do projeto1
        proj1_chunks = list(store.query_by_project("project1"))
        assert len(proj1_chunks) == 2
        
        proj1_ids = [chunk["id"] for chunk in proj1_chunks]
        assert "proj1_chunk1" in proj1_ids
        assert "proj1_chunk2" in proj1_ids
        assert "proj2_chunk1" not in proj1_ids
        
        store.close()

    def test_distinct_coverage(self):
        """Testa a obtenção de valores distintos de coverage."""
        store = SQLiteMetadataStore(self.db_path)
        
        # Armazena chunks com diferentes níveis de coverage
        metadatas = [
            {
                "id": "high_chunk",
                "file_path": "/path/high.py",
                "language": "python",
                "symbols": [],
                "relations": [],
                "coverage": "high",
                "source": "test",
                "chunk_hash": "hash_high",
                "project_id": "coverage_project"
            },
            {
                "id": "medium_chunk",
                "file_path": "/path/medium.py",
                "language": "python",
                "symbols": [],
                "relations": [],
                "coverage": "medium",
                "source": "test",
                "chunk_hash": "hash_medium",
                "project_id": "coverage_project"
            },
            {
                "id": "high_chunk2",
                "file_path": "/path/high2.py",
                "language": "python",
                "symbols": [],
                "relations": [],
                "coverage": "high",
                "source": "test",
                "chunk_hash": "hash_high2",
                "project_id": "coverage_project"
            }
        ]
        
        for metadata in metadatas:
            store.upsert_metadata(metadata)
        
        # Obtém valores distintos de coverage
        coverages = store.distinct_coverage()
        assert isinstance(coverages, list)
        assert "high" in coverages
        assert "medium" in coverages
        assert len(set(coverages)) == len(coverages)  # Verifica se são únicos
        
        store.close()

    def test_close_connection(self):
        """Testa o fechamento da conexão."""
        store = SQLiteMetadataStore(self.db_path)
        assert store.conn is not None
        
        store.close()
        # Após fechar, tentativas de usar a conexão devem falhar
        with pytest.raises(Exception):
            store.conn.execute("SELECT 1")

    def test_upsert_update_existing(self):
        """Testa a atualização de metadados existentes."""
        store = SQLiteMetadataStore(self.db_path)
        
        # Insere metadados iniciais
        original_metadata = {
            "id": "update_test",
            "file_path": "/path/to/update.py",
            "language": "python",
            "symbols": ["old_function"],
            "relations": [],
            "coverage": "low",
            "source": "test",
            "chunk_hash": "old_hash",
            "project_id": "update_project"
        }
        
        store.upsert_metadata(original_metadata)
        
        # Atualiza os metadados
        updated_metadata = {
            "id": "update_test",
            "file_path": "/path/to/update.py",
            "language": "python",
            "symbols": ["new_function"],
            "relations": ["calls"],
            "coverage": "high",
            "source": "test",
            "chunk_hash": "new_hash",
            "project_id": "update_project"
        }
        
        store.upsert_metadata(updated_metadata)
        
        # Verifica se ainda existe apenas um registro
        result_id = store.get_id_by_filepath("/path/to/update.py")
        assert result_id == "update_test"
        
        store.close()

    def test_empty_database_queries(self):
        """Testa consultas em banco vazio."""
        store = SQLiteMetadataStore(self.db_path)
        
        # Consultas em banco vazio devem retornar resultados vazios
        assert store.get_id_by_filepath("/any/path.py") is None
        assert list(store.query_by_language("python")) == []
        assert list(store.query_by_project("any_project")) == []
        assert store.distinct_coverage() == []
        
        store.close()

    def test_special_characters_in_data(self):
        """Testa o armazenamento de dados com caracteres especiais."""
        store = SQLiteMetadataStore(self.db_path)
        
        metadata = {
            "id": "special_chars",
            "file_path": "/path/with spaces/file-name_with.special@chars.py",
            "language": "python",
            "symbols": ["função_com_acentos", "class_with_ñ"],
            "relations": ["imports", "calls"],
            "coverage": "médium",
            "source": "test",
            "chunk_hash": "hash_with_特殊字符",
            "project_id": "projeto_especial"
        }
        
        store.upsert_metadata(metadata)
        
        result_id = store.get_id_by_filepath("/path/with spaces/file-name_with.special@chars.py")
        assert result_id == "special_chars"
        
        store.close()

    def test_none_and_empty_values(self):
        """Testa o tratamento de valores None e vazios."""
        store = SQLiteMetadataStore(self.db_path)
        
        metadata = {
            "id": "none_test",
            "file_path": "/path/to/none.py",
            "language": None,
            "symbols": [],
            "relations": None,
            "coverage": "",
            "source": None,
            "chunk_hash": None,
            "project_id": ""
        }
        
        # Deve conseguir armazenar sem erro
        store.upsert_metadata(metadata)
        
        result_id = store.get_id_by_filepath("/path/to/none.py")
        assert result_id == "none_test"
        
        store.close()