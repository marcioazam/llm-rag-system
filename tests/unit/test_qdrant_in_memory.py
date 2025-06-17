from src.vectordb.qdrant_store import QdrantVectorStore


def _create_in_memory_store():
    """Força o fallback em memória apontando para host/porta inválidos."""
    return QdrantVectorStore(host="localhost", port=9999)  # porta improvável


def test_qdrant_fallback_flag():
    store = _create_in_memory_store()
    assert getattr(store, "_in_memory", False) is True


def test_qdrant_add_documents_in_memory():
    store = _create_in_memory_store()

    docs = ["hello world"]
    vecs = [[0.1] * 768]
    success = store.add_documents(docs, embeddings=vecs, metadata=[{}], ids=["id1"])

    assert success is True
    # Verifica se o documento foi armazenado internamente
    assert len(store._mem_store) == 1  # type: ignore[attr-defined]


def test_qdrant_search_in_memory_returns_empty():
    store = _create_in_memory_store()
    result = store.search(query_embedding=[0.1] * 768, k=1)
    assert result == [] 