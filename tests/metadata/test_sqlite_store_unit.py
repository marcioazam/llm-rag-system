import uuid
from src.metadata.sqlite_store import SQLiteMetadataStore


def _sample_item() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "file_path": "src/example.py",
        "language": "python",
        "symbols": ["func_a", "ClassB"],
        "relations": {"imports": ["os", "sys"]},
        "coverage": "full",
        "source": "unit-test",
        "chunk_hash": "abc123",
        "project_id": "proj-1",
    }


def test_upsert_and_query():
    store = SQLiteMetadataStore(db_path=":memory:")
    
    # Criar o projeto primeiro (nova validação obrigatória)
    store.create_project(
        project_id="proj-1",
        name="Test Project",
        description="Projeto de teste para validação"
    )
    
    item = _sample_item()

    # Inserir
    store.upsert_metadata(item)

    # get_id_by_filepath
    assert store.get_id_by_filepath(item["file_path"]) == item["id"]

    # query_by_language
    results = list(store.query_by_language("python"))
    assert len(results) == 1
    assert results[0]["id"] == item["id"]

    # distinct_coverage
    assert store.distinct_coverage() == ["full"]

    # query_by_project
    proj_results = list(store.query_by_project("proj-1"))
    assert proj_results[0]["id"] == item["id"]

    # delete
    store.delete_by_id(item["id"])
    assert store.get_id_by_filepath(item["file_path"]) is None

    store.close() 