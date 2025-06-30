from src.vectordb.qdrant_store import QdrantVectorStore


def test_add_search_clear():
    store = QdrantVectorStore(dim=3)
    docs = ["foo", "bar"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert store.add_documents(docs, embeddings=embeddings)

    results = store.search(query=embeddings[0], k=2)
    assert isinstance(results, list)

    assert store.clear_collection()
    store.close() 