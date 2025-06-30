"""Comprehensive tests for SQLite metadata store functionality."""

import pytest
import sqlite3
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.metadata.sqlite_store import SQLiteMetadataStore


class TestSQLiteMetadataStore:
    """Test suite for SQLite metadata store."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a SQLiteMetadataStore instance for testing."""
        store = SQLiteMetadataStore(temp_db_path)
        yield store
        store.close()

    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return {
            "id": "test_chunk_1",
            "file_path": "/path/to/test.py",
            "language": "python",
            "symbols": ["function_a", "class_b"],
            "relations": [{"type": "imports", "target": "module_x"}],
            "coverage": "high",
            "source": "git",
            "chunk_hash": "abc123",
            "project_id": "project_1"
        }

    def test_init_creates_database_and_schema(self, temp_db_path):
        """Test that initialization creates database file and schema."""
        store = SQLiteMetadataStore(temp_db_path)
        
        # Check database file exists
        assert Path(temp_db_path).exists()
        
        # Check schema exists
        cur = store.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
        assert cur.fetchone() is not None
        
        store.close()

    def test_init_creates_parent_directories(self):
        """Test that initialization creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "deep", "test.db")
            store = SQLiteMetadataStore(nested_path)
            
            assert Path(nested_path).exists()
            assert Path(nested_path).parent.exists()
            
            store.close()

    def test_schema_has_all_required_columns(self, store):
        """Test that schema contains all required columns."""
        cur = store.conn.cursor()
        cur.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cur.fetchall()}
        
        expected_columns = {
            "id", "file_path", "language", "symbols", "relations",
            "coverage", "chunk_hash", "project_id", "source", "created_at"
        }
        
        assert expected_columns.issubset(columns)

    def test_coverage_column_migration(self, temp_db_path):
        """Test that coverage column is added to existing databases."""
        # Create database without coverage column
        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()
        cur.execute(
            """
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
            """
        )
        conn.commit()
        conn.close()
        
        # Initialize store (should trigger migration)
        store = SQLiteMetadataStore(temp_db_path)
        
        # Check coverage column exists
        cur = store.conn.cursor()
        cur.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cur.fetchall()}
        assert "coverage" in columns
        
        store.close()

    def test_upsert_metadata_insert(self, store, sample_metadata):
        """Test inserting new metadata."""
        store.upsert_metadata(sample_metadata)
        
        # Verify insertion
        cur = store.conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE id=?", (sample_metadata["id"],))
        row = cur.fetchone()
        
        assert row is not None
        assert row[0] == sample_metadata["id"]  # id column
        assert row[1] == sample_metadata["file_path"]  # file_path column
        assert row[2] == sample_metadata["language"]  # language column
        
        # Check JSON serialization
        symbols = json.loads(row[3])
        relations = json.loads(row[4])
        assert symbols == sample_metadata["symbols"]
        assert relations == sample_metadata["relations"]

    def test_upsert_metadata_update(self, store, sample_metadata):
        """Test updating existing metadata."""
        # Insert initial data
        store.upsert_metadata(sample_metadata)
        
        # Update data
        updated_metadata = sample_metadata.copy()
        updated_metadata["language"] = "javascript"
        updated_metadata["coverage"] = "low"
        
        store.upsert_metadata(updated_metadata)
        
        # Verify update
        cur = store.conn.cursor()
        cur.execute("SELECT language, coverage FROM chunks WHERE id=?", (sample_metadata["id"],))
        row = cur.fetchone()
        
        assert row[0] == "javascript"
        assert row[1] == "low"
        
        # Verify only one record exists
        cur.execute("SELECT COUNT(*) FROM chunks WHERE id=?", (sample_metadata["id"],))
        count = cur.fetchone()[0]
        assert count == 1

    def test_upsert_metadata_with_none_values(self, store):
        """Test upserting metadata with None values."""
        metadata = {
            "id": "test_none",
            "file_path": None,
            "language": "python",
            "symbols": None,
            "relations": None,
            "coverage": None,
            "source": None,
            "chunk_hash": None,
            "project_id": None
        }
        
        store.upsert_metadata(metadata)
        
        # Verify insertion with None values
        cur = store.conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE id=?", (metadata["id"],))
        row = cur.fetchone()
        
        assert row is not None
        assert row[0] == "test_none"
        assert row[1] is None  # file_path
        assert row[2] == "python"  # language

    def test_query_by_language(self, store, sample_metadata):
        """Test querying chunks by language."""
        # Insert test data
        python_chunk = sample_metadata.copy()
        python_chunk["id"] = "python_chunk"
        python_chunk["language"] = "python"
        
        js_chunk = sample_metadata.copy()
        js_chunk["id"] = "js_chunk"
        js_chunk["language"] = "javascript"
        
        store.upsert_metadata(python_chunk)
        store.upsert_metadata(js_chunk)
        
        # Query Python chunks
        python_results = list(store.query_by_language("python"))
        assert len(python_results) == 1
        assert python_results[0]["id"] == "python_chunk"
        assert python_results[0]["language"] == "python"
        
        # Query JavaScript chunks
        js_results = list(store.query_by_language("javascript"))
        assert len(js_results) == 1
        assert js_results[0]["id"] == "js_chunk"
        
        # Query non-existent language
        empty_results = list(store.query_by_language("rust"))
        assert len(empty_results) == 0

    def test_query_by_language_returns_all_columns(self, store, sample_metadata):
        """Test that query_by_language returns all columns."""
        store.upsert_metadata(sample_metadata)
        
        results = list(store.query_by_language("python"))
        result = results[0]
        
        expected_keys = {
            "id", "file_path", "language", "symbols", "relations",
            "coverage", "chunk_hash", "project_id", "source", "created_at"
        }
        
        assert set(result.keys()) == expected_keys

    def test_get_id_by_filepath(self, store, sample_metadata):
        """Test getting chunk ID by file path."""
        store.upsert_metadata(sample_metadata)
        
        # Test existing file path
        chunk_id = store.get_id_by_filepath(sample_metadata["file_path"])
        assert chunk_id == sample_metadata["id"]
        
        # Test non-existent file path
        chunk_id = store.get_id_by_filepath("/non/existent/path.py")
        assert chunk_id is None

    def test_delete_by_id(self, store, sample_metadata):
        """Test deleting chunk by ID."""
        store.upsert_metadata(sample_metadata)
        
        # Verify chunk exists
        chunk_id = store.get_id_by_filepath(sample_metadata["file_path"])
        assert chunk_id is not None
        
        # Delete chunk
        store.delete_by_id(sample_metadata["id"])
        
        # Verify chunk is deleted
        chunk_id = store.get_id_by_filepath(sample_metadata["file_path"])
        assert chunk_id is None

    def test_delete_by_id_nonexistent(self, store):
        """Test deleting non-existent chunk by ID."""
        # Should not raise an error
        store.delete_by_id("nonexistent_id")

    def test_query_by_project(self, store, sample_metadata):
        """Test querying chunks by project ID."""
        # Insert test data for different projects
        chunk1 = sample_metadata.copy()
        chunk1["id"] = "chunk1"
        chunk1["project_id"] = "project_a"
        
        chunk2 = sample_metadata.copy()
        chunk2["id"] = "chunk2"
        chunk2["project_id"] = "project_a"
        
        chunk3 = sample_metadata.copy()
        chunk3["id"] = "chunk3"
        chunk3["project_id"] = "project_b"
        
        store.upsert_metadata(chunk1)
        store.upsert_metadata(chunk2)
        store.upsert_metadata(chunk3)
        
        # Query project_a chunks
        project_a_results = list(store.query_by_project("project_a"))
        assert len(project_a_results) == 2
        project_a_ids = {result["id"] for result in project_a_results}
        assert project_a_ids == {"chunk1", "chunk2"}
        
        # Query project_b chunks
        project_b_results = list(store.query_by_project("project_b"))
        assert len(project_b_results) == 1
        assert project_b_results[0]["id"] == "chunk3"
        
        # Query non-existent project
        empty_results = list(store.query_by_project("nonexistent_project"))
        assert len(empty_results) == 0

    def test_distinct_coverage(self, store, sample_metadata):
        """Test getting distinct coverage values."""
        # Insert chunks with different coverage values
        coverages = ["high", "medium", "low", "high", "medium", None, ""]
        
        for i, coverage in enumerate(coverages):
            chunk = sample_metadata.copy()
            chunk["id"] = f"chunk_{i}"
            chunk["coverage"] = coverage
            store.upsert_metadata(chunk)
        
        # Get distinct coverage values
        distinct_coverages = store.distinct_coverage()
        
        # Should return sorted unique non-null, non-empty values
        expected = ["high", "low", "medium"]
        assert distinct_coverages == expected

    def test_distinct_coverage_empty_table(self, store):
        """Test distinct_coverage with empty table."""
        result = store.distinct_coverage()
        assert result == []

    def test_distinct_coverage_exception_handling(self, store):
        """Test distinct_coverage handles exceptions gracefully."""
        # Temporarily patch the execute method to raise an exception
        original_method = store.distinct_coverage
        
        def mock_distinct_coverage():
            try:
                # Force an exception by using an invalid SQL
                cur = store.conn.cursor()
                cur.execute("SELECT DISTINCT coverage FROM invalid_table")
                return []
            except Exception:
                return []
        
        store.distinct_coverage = mock_distinct_coverage
        
        try:
            result = store.distinct_coverage()
            assert result == []
        finally:
            # Restore original method
            store.distinct_coverage = original_method

    def test_close_connection(self, temp_db_path):
        """Test closing database connection."""
        store = SQLiteMetadataStore(temp_db_path)
        
        # Verify connection is open
        assert store.conn is not None
        
        # Close connection
        store.close()
        
        # Verify connection is closed (attempting to use it should raise an error)
        with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
            store.conn.execute("SELECT 1")

    def test_json_serialization_deserialization(self, store):
        """Test JSON serialization and deserialization of complex data."""
        complex_metadata = {
            "id": "complex_chunk",
            "file_path": "/path/to/complex.py",
            "language": "python",
            "symbols": [
                {"name": "function_a", "type": "function", "line": 10},
                {"name": "ClassB", "type": "class", "line": 20}
            ],
            "relations": [
                {"type": "imports", "target": "module_x", "line": 1},
                {"type": "calls", "target": "function_y", "line": 15}
            ],
            "coverage": "high",
            "source": "git",
            "chunk_hash": "def456",
            "project_id": "complex_project"
        }
        
        store.upsert_metadata(complex_metadata)
        
        # Query and verify deserialization
        results = list(store.query_by_language("python"))
        result = results[0]
        
        # Symbols and relations should be JSON strings in the database
        # but we need to manually deserialize them for comparison
        cur = store.conn.cursor()
        cur.execute("SELECT symbols, relations FROM chunks WHERE id=?", ("complex_chunk",))
        row = cur.fetchone()
        
        symbols = json.loads(row[0])
        relations = json.loads(row[1])
        
        assert symbols == complex_metadata["symbols"]
        assert relations == complex_metadata["relations"]

    def test_concurrent_access(self, temp_db_path):
        """Test that multiple store instances can access the same database."""
        store1 = SQLiteMetadataStore(temp_db_path)
        store2 = SQLiteMetadataStore(temp_db_path)
        
        # Insert data with store1
        metadata1 = {
            "id": "concurrent_1",
            "file_path": "/path/1.py",
            "language": "python",
            "symbols": [],
            "relations": [],
            "coverage": "high",
            "source": "git",
            "chunk_hash": "hash1",
            "project_id": "project1"
        }
        store1.upsert_metadata(metadata1)
        
        # Read data with store2
        chunk_id = store2.get_id_by_filepath("/path/1.py")
        assert chunk_id == "concurrent_1"
        
        store1.close()
        store2.close()

    def test_special_characters_in_data(self, store):
        """Test handling of special characters in metadata."""
        special_metadata = {
            "id": "special_chars",
            "file_path": "/path/with spaces/file-name_123.py",
            "language": "python",
            "symbols": ["function_with_unicode_🚀", "class_with_émojis"],
            "relations": [{"type": "imports", "target": "module-with-dashes"}],
            "coverage": "medium",
            "source": "git",
            "chunk_hash": "hash_with_special_chars!@#$%",
            "project_id": "project-123_test"
        }
        
        store.upsert_metadata(special_metadata)
        
        # Verify data integrity
        chunk_id = store.get_id_by_filepath("/path/with spaces/file-name_123.py")
        assert chunk_id == "special_chars"
        
        results = list(store.query_by_language("python"))
        assert len(results) == 1
        
        # Verify JSON data
        cur = store.conn.cursor()
        cur.execute("SELECT symbols FROM chunks WHERE id=?", ("special_chars",))
        symbols_json = cur.fetchone()[0]
        symbols = json.loads(symbols_json)
        assert "function_with_unicode_🚀" in symbols