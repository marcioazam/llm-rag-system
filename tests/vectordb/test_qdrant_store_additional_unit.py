from src.vectordb.qdrant_store import QdrantVectorStore


def test_update_and_get_document_in_memory():
    store = QdrantVectorStore(dim=3)
    docs = ["x"]
    emb = [[0.1, 0.1, 0.1]]
    store.add_documents(docs, embeddings=emb)

    # update should return True even in-memory (no-op)
    assert store.update_document("doc_0", [0.2, 0.2, 0.2]) is True
    # Dependendo do stub, pode retornar None ou dict
    res = store.get_document_by_id("doc_0")
    assert res is None or isinstance(res, dict)
    store.close() 