import numpy as np

from src.vectordb.qdrant_store import QdrantVectorStore


def test_qdrant_in_memory_add_and_search():
    store = QdrantVectorStore(host="invalid_host", port=1234)
    assert getattr(store, "_in_memory", False)

    docs = ["texto de exemplo"]
    vecs = [[0.0] * 768]
    meta = [{}]
    ids = ["id1"]

    ok = store.add_documents(docs, vecs, meta, ids)
    assert ok is True

    results = store.search(query_embedding=[0.0] * 768, k=1)
    assert results == []


def test_qdrant_in_memory_delete_and_clear():
    store = QdrantVectorStore(host="invalid", port=999)
    docs = ["abc", "def"]
    vecs = [[0.0] * 768, [0.1] * 768]
    meta = [{}, {}]
    ids = ["a", "b"]
    assert store.add_documents(docs, vecs, meta, ids)

    # Método delete_documents deve retornar True mesmo que noop em memória
    assert store.delete_documents(["a"]) is True

    # clear_collection deve funcionar e reiniciar store._mem_store
    assert store.clear_collection() is True  # em memória devolve True ou False conforme implementação
    assert getattr(store, "_mem_store", []) == [] 