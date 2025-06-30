from uuid import uuid4

from src.metadata.sqlite_store import SQLiteMetadataStore


def test_sqlite_store_basic(tmp_path):
    db_path = tmp_path / "test.db"
    store = SQLiteMetadataStore(db_path=str(db_path))

    item_id = str(uuid4())
    item = {
        "id": item_id,
        "file_path": "src/example.py",
        "language": "python",
        "symbols": {"functions": ["foo"]},
        "relations": {},
        "coverage": "core",
        "source": "unit_test",
        "chunk_hash": "abc123",
        "project_id": "proj1",
    }
    # Insert
    store.upsert_metadata(item)

    # Query by language
    results = list(store.query_by_language("python"))
    assert len(results) == 1
    assert results[0]["id"] == item_id

    # get_id_by_filepath
    assert store.get_id_by_filepath("src/example.py") == item_id

    # distinct coverage
    assert store.distinct_coverage() == ["core"]

    # delete
    store.delete_by_id(item_id)
    assert store.get_id_by_filepath("src/example.py") is None

    store.close() 