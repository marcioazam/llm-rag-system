"""
Testes completos para o SQLite Metadata Store.
Cobertura atual: 0% -> Meta: 100%
"""

import pytest
import tempfile
import shutil
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any

from src.metadata.sqlite_store import SQLiteMetadataStore


class TestSQLiteMetadataStore:
    """Testes para o SQLite Metadata Store."""

    @pytest.fixture
    def temp_dir(self):
        """Diretório temporário para testes."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def db_path(self, temp_dir):
        """Caminho do banco de teste."""
        return str(Path(temp_dir) / "test_chunks.db")

    @pytest.fixture
    def store(self, db_path):
        """Instância do store para testes."""
        store = SQLiteMetadataStore(db_path)
        yield store
        store.close()

    @pytest.fixture
    def sample_metadata(self):
        """Metadata de exemplo para testes."""
        return {
            "id": "chunk_001",
            "file_path": "/path/to/file.py",
            "language": "python",
            "symbols": ["function1", "Class1"],
            "relations": [{"from": "function1", "to": "Class1", "type": "calls"}],
            "coverage": "function",
            "source": "code_analysis",
            "chunk_hash": "abc123",
            "project_id": "project_1"
        }

    def test_init_creates_database(self, db_path):
        """Testar que a inicialização cria o banco de dados."""
        assert not Path(db_path).exists()
        
        store = SQLiteMetadataStore(db_path)
        
        assert Path(db_path).exists()
        assert store.db_path == db_path
        assert isinstance(store.conn, sqlite3.Connection)
        
        store.close()

    def test_init_creates_parent_directories(self, temp_dir):
        """Testar criação de diretórios pais."""
        nested_path = str(Path(temp_dir) / "nested" / "dir" / "test.db")
        
        store = SQLiteMetadataStore(nested_path)
        
        assert Path(nested_path).exists()
        assert Path(nested_path).parent.exists()
        
        store.close()

    def test_create_schema(self, store):
        """Testar criação do schema do banco."""
        cursor = store.conn.cursor()
        
        # Verificar se a tabela foi criada
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "chunks"
        
        # Verificar colunas
        cursor.execute("PRAGMA table_info(chunks)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        expected_columns = [
            "id", "file_path", "language", "symbols", "relations", 
            "coverage", "chunk_hash", "project_id", "source", "created_at"
        ]
        
        for col in expected_columns:
            assert col in column_names

    def test_upsert_metadata_insert(self, store, sample_metadata):
        """Testar inserção de metadata."""
        store.upsert_metadata(sample_metadata)
        
        cursor = store.conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE id = ?", (sample_metadata["id"],))
        result = cursor.fetchone()
        
        assert result is not None
        # Verificar alguns campos principais
        assert result[0] == sample_metadata["id"]  # id
        assert result[1] == sample_metadata["file_path"]  # file_path
        assert result[2] == sample_metadata["language"]  # language

    def test_upsert_metadata_update(self, store, sample_metadata):
        """Testar atualização de metadata existente."""
        # Inserir primeiro
        store.upsert_metadata(sample_metadata)
        
        # Atualizar
        updated_metadata = sample_metadata.copy()
        updated_metadata["file_path"] = "/updated/path/file.py"
        updated_metadata["language"] = "javascript"
        
        store.upsert_metadata(updated_metadata)
        
        # Verificar que foi atualizado
        cursor = store.conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE id = ?", (sample_metadata["id"],))
        result = cursor.fetchone()
        
        assert result[1] == "/updated/path/file.py"  # file_path
        assert result[2] == "javascript"  # language
        
        # Verificar que não duplicou
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", (sample_metadata["id"],))
        count = cursor.fetchone()[0]
        assert count == 1

    def test_upsert_metadata_with_complex_data(self, store):
        """Testar inserção com dados complexos (listas, dicts)."""
        complex_metadata = {
            "id": "chunk_complex",
            "file_path": "/path/complex.py",
            "language": "python",
            "symbols": ["func1", "func2", "Class1", "Class2"],
            "relations": [
                {"from": "func1", "to": "Class1", "type": "instantiates"},
                {"from": "func2", "to": "func1", "type": "calls"}
            ],
            "coverage": "module",
            "source": "ast_analysis",
            "chunk_hash": "def456",
            "project_id": "complex_project"
        }
        
        store.upsert_metadata(complex_metadata)
        
        cursor = store.conn.cursor()
        cursor.execute("SELECT symbols, relations FROM chunks WHERE id = ?", (complex_metadata["id"],))
        result = cursor.fetchone()
        
        # Verificar que JSON foi serializado/deserializado corretamente
        symbols = json.loads(result[0])
        relations = json.loads(result[1])
        
        assert symbols == complex_metadata["symbols"]
        assert relations == complex_metadata["relations"]

    def test_query_by_language(self, store):
        """Testar query por linguagem."""
        # Inserir dados de diferentes linguagens
        python_data = {
            "id": "py_chunk", "file_path": "/file.py", "language": "python",
            "symbols": [], "relations": [], "coverage": "", "source": "", 
            "chunk_hash": "", "project_id": ""
        }
        
        js_data = {
            "id": "js_chunk", "file_path": "/file.js", "language": "javascript",
            "symbols": [], "relations": [], "coverage": "", "source": "", 
            "chunk_hash": "", "project_id": ""
        }
        
        store.upsert_metadata(python_data)
        store.upsert_metadata(js_data)
        
        # Query por Python
        python_results = list(store.query_by_language("python"))
        assert len(python_results) == 1
        assert python_results[0]["id"] == "py_chunk"
        assert python_results[0]["language"] == "python"
        
        # Query por JavaScript
        js_results = list(store.query_by_language("javascript"))
        assert len(js_results) == 1
        assert js_results[0]["id"] == "js_chunk"

    def test_query_by_language_empty_result(self, store):
        """Testar query por linguagem que não existe."""
        results = list(store.query_by_language("nonexistent"))
        assert len(results) == 0

    def test_get_id_by_filepath(self, store, sample_metadata):
        """Testar busca de ID por file path."""
        store.upsert_metadata(sample_metadata)
        
        found_id = store.get_id_by_filepath(sample_metadata["file_path"])
        assert found_id == sample_metadata["id"]

    def test_get_id_by_filepath_not_found(self, store):
        """Testar busca de ID por file path inexistente."""
        found_id = store.get_id_by_filepath("/nonexistent/path.py")
        assert found_id is None

    def test_delete_by_id(self, store, sample_metadata):
        """Testar deleção por ID."""
        # Inserir
        store.upsert_metadata(sample_metadata)
        
        # Verificar que existe
        cursor = store.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", (sample_metadata["id"],))
        assert cursor.fetchone()[0] == 1
        
        # Deletar
        store.delete_by_id(sample_metadata["id"])
        
        # Verificar que foi deletado
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", (sample_metadata["id"],))
        assert cursor.fetchone()[0] == 0

    def test_delete_by_id_nonexistent(self, store):
        """Testar deleção de ID que não existe."""
        # Não deve dar erro
        store.delete_by_id("nonexistent_id")
        
        # Verificar que nada foi afetado
        cursor = store.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        assert cursor.fetchone()[0] == 0

    def test_query_by_project(self, store):
        """Testar query por projeto."""
        # Inserir dados de diferentes projetos
        proj1_data = {
            "id": "proj1_chunk", "file_path": "/proj1/file.py", "language": "python",
            "symbols": [], "relations": [], "coverage": "", "source": "", 
            "chunk_hash": "", "project_id": "project1"
        }
        
        proj2_data = {
            "id": "proj2_chunk", "file_path": "/proj2/file.py", "language": "python",
            "symbols": [], "relations": [], "coverage": "", "source": "", 
            "chunk_hash": "", "project_id": "project2"
        }
        
        store.upsert_metadata(proj1_data)
        store.upsert_metadata(proj2_data)
        
        # Query por project1
        proj1_results = list(store.query_by_project("project1"))
        assert len(proj1_results) == 1
        assert proj1_results[0]["id"] == "proj1_chunk"
        assert proj1_results[0]["project_id"] == "project1"

    def test_distinct_coverage(self, store):
        """Testar busca de valores únicos de coverage."""
        # Inserir dados com diferentes coverages
        coverages_data = [
            {"id": "1", "coverage": "function", "file_path": "", "language": "", 
             "symbols": [], "relations": [], "source": "", "chunk_hash": "", "project_id": ""},
            {"id": "2", "coverage": "class", "file_path": "", "language": "", 
             "symbols": [], "relations": [], "source": "", "chunk_hash": "", "project_id": ""},
            {"id": "3", "coverage": "function", "file_path": "", "language": "", 
             "symbols": [], "relations": [], "source": "", "chunk_hash": "", "project_id": ""},
            {"id": "4", "coverage": "module", "file_path": "", "language": "", 
             "symbols": [], "relations": [], "source": "", "chunk_hash": "", "project_id": ""},
        ]
        
        for data in coverages_data:
            store.upsert_metadata(data)
        
        distinct_coverages = store.distinct_coverage()
        
        assert isinstance(distinct_coverages, list)
        assert len(distinct_coverages) == 3  # function, class, module
        assert "function" in distinct_coverages
        assert "class" in distinct_coverages
        assert "module" in distinct_coverages

    def test_distinct_coverage_empty(self, store):
        """Testar distinct_coverage com banco vazio."""
        result = store.distinct_coverage()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_distinct_coverage_with_nulls_and_empty(self, store):
        """Testar distinct_coverage com valores null e vazios."""
        test_data = [
            {"id": "1", "coverage": "function", "file_path": "", "language": "", 
             "symbols": [], "relations": [], "source": "", "chunk_hash": "", "project_id": ""},
            {"id": "2", "coverage": "", "file_path": "", "language": "", 
             "symbols": [], "relations": [], "source": "", "chunk_hash": "", "project_id": ""},
            {"id": "3", "coverage": None, "file_path": "", "language": "", 
             "symbols": [], "relations": [], "source": "", "chunk_hash": "", "project_id": ""},
        ]
        
        for data in test_data:
            store.upsert_metadata(data)
        
        distinct_coverages = store.distinct_coverage()
        
        # Deve retornar apenas valores não-null e não-vazios
        assert "function" in distinct_coverages
        assert "" not in distinct_coverages
        assert None not in distinct_coverages

    def test_close_connection(self, store):
        """Testar fechamento da conexão."""
        # Verificar que conexão está ativa
        assert store.conn is not None
        
        # Fechar
        store.close()
        
        # Verificar que não consegue mais executar queries
        with pytest.raises(sqlite3.ProgrammingError):
            cursor = store.conn.cursor()
            cursor.execute("SELECT * FROM chunks")

    def test_migration_coverage_column(self, temp_dir):
        """Testar migração automática da coluna coverage."""
        db_path = str(Path(temp_dir) / "migration_test.db")
        
        # Criar banco sem coluna coverage
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                language TEXT,
                symbols TEXT,
                relations TEXT,
                chunk_hash TEXT,
                project_id TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        
        # Inicializar store (deve fazer migração)
        store = SQLiteMetadataStore(db_path)
        
        # Verificar que coluna coverage foi adicionada
        cursor = store.conn.cursor()
        cursor.execute("PRAGMA table_info(chunks)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        assert "coverage" in column_names
        
        store.close()

    def test_multiple_operations(self, store):
        """Testar múltiplas operações em sequência."""
        # Inserir múltiplos itens
        items = []
        for i in range(10):
            item = {
                "id": f"chunk_{i}",
                "file_path": f"/path/file_{i}.py",
                "language": "python",
                "symbols": [f"func_{i}"],
                "relations": [],
                "coverage": "function",
                "source": "test",
                "chunk_hash": f"hash_{i}",
                "project_id": "test_project"
            }
            items.append(item)
            store.upsert_metadata(item)
        
        # Verificar contagem total
        cursor = store.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        assert cursor.fetchone()[0] == 10
        
        # Query por projeto
        project_results = list(store.query_by_project("test_project"))
        assert len(project_results) == 10
        
        # Deletar alguns
        for i in range(0, 5):
            store.delete_by_id(f"chunk_{i}")
        
        # Verificar contagem após deleções
        cursor.execute("SELECT COUNT(*) FROM chunks")
        assert cursor.fetchone()[0] == 5

    def test_error_handling_json_serialization(self, store):
        """Testar tratamento de erro na serialização JSON."""
        # Usar dados que podem causar problemas na serialização
        metadata = {
            "id": "test_json",
            "file_path": "/path/file.py",
            "language": "python",
            "symbols": ["func1", "func2"],  # Lista normal
            "relations": [{"type": "calls"}],  # Dict normal
            "coverage": "function",
            "source": "test",
            "chunk_hash": "hash",
            "project_id": "project"
        }
        
        # Deve funcionar normalmente
        store.upsert_metadata(metadata)
        
        results = list(store.query_by_language("python"))
        assert len(results) == 1

    def test_empty_metadata_values(self, store):
        """Testar com valores de metadata vazios ou None."""
        metadata = {
            "id": "empty_test",
            "file_path": None,
            "language": "",
            "symbols": [],
            "relations": [],
            "coverage": None,
            "source": "",
            "chunk_hash": "",
            "project_id": None
        }
        
        # Deve funcionar sem erros
        store.upsert_metadata(metadata)
        
        cursor = store.conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE id = ?", ("empty_test",))
        result = cursor.fetchone()
        assert result is not None

    def test_default_database_path(self, temp_dir):
        """Testar criação com caminho padrão."""
        # Mudar para diretório temporário para não criar arquivos no sistema
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            store = SQLiteMetadataStore()
            
            # Verificar que criou no local esperado
            expected_path = Path("./data/metadata/chunks.db")
            assert expected_path.exists()
            
            store.close()
        finally:
            os.chdir(original_cwd)

    def test_concurrent_access(self, db_path):
        """Testar acesso concorrente (básico)."""
        # Criar duas instâncias do store
        store1 = SQLiteMetadataStore(db_path)
        store2 = SQLiteMetadataStore(db_path)
        
        # Inserir via store1
        metadata1 = {
            "id": "concurrent_1", "file_path": "/path1.py", "language": "python",
            "symbols": [], "relations": [], "coverage": "", "source": "", 
            "chunk_hash": "", "project_id": ""
        }
        store1.upsert_metadata(metadata1)
        
        # Ler via store2
        results = list(store2.query_by_language("python"))
        assert len(results) == 1
        assert results[0]["id"] == "concurrent_1"
        
        store1.close()
        store2.close() 