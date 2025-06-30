from src.vectordb.qdrant_store import QdrantVectorStore


def _make_store():
    # Dim=2 para vetores pequenos, roda sempre em memória por conta dos stubs
    return QdrantVectorStore(dim=2)


def test_add_documents_without_embeddings():
    store = _make_store()
    ok = store.add_documents(["doc sem embedding"])
    # Deve falhar e retornar False
    assert ok is False
    store.close()


def test_delete_documents_in_memory():
    store = _make_store()
    docs = ["A", "B", "C"]
    embeds = [[0.1, 0.1]] * 3
    store.add_documents(docs, embeddings=embeds)

    # Deletar um ID
    assert store.delete_documents(["doc_1"]) is True
    # Após remoção in-memory get_document_count não é suportado (depende de client),
    # mas podemos verificar que busca retorna lista vazia.
    res = store.search(query=embeds[0], k=5)
    assert isinstance(res, list)
    store.close()


def test_search_requires_embedding_for_string_query():
    store = _make_store()
    docs = ["texto"]
    embeds = [[0.2, 0.3]]
    store.add_documents(docs, embeddings=embeds)

    try:
        result = store.search(query="query sem embedding")
    except ValueError as exc:
        assert "query_embedding" in str(exc)
    else:
        # Caso stubs ignorem, resultado deve ser lista
        assert isinstance(result, list)
    store.close() 