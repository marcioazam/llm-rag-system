import numpy as np
from typing import List, Dict, Any

from src.retrieval.retriever import HybridRetriever


class _StubEmbedding:
    """Embedding service mínimo para testes."""

    def embed_query(self, _):
        return np.array([1.0, 0.0])

    def embed_texts(self, texts, show_progress=False):  # noqa: D401
        return [np.array([1.0, 0.0]) for _ in texts]


class _StubVectorStore:
    """Vector store simples em memória para testar semantic_search."""

    def __init__(self):
        self.docs = [
            {"id": "d1", "content": "Exemplo de código em Python", "distance": 0.1},
            {"id": "d2", "content": "Guia de viagem para França", "distance": 0.3},
        ]

    def search(self, query_embedding, k=5, filter=None):  # noqa: D401
        return self.docs[:k]


def _make_retriever():
    vs = _StubVectorStore()
    emb = _StubEmbedding()
    return HybridRetriever(vector_store=vs, embedding_service=emb, rerank=False)


def test_semantic_search_basic():
    retriever = _make_retriever()
    results = retriever.retrieve("python", k=2, search_type="semantic")
    assert len(results) == 2
    # Scores devem existir e estar ordenados decrescentemente
    assert results[0]["score"] >= results[1]["score"]


def test_keyword_search_after_index():
    retriever = _make_retriever()
    # Indexa corpus para BM25
    corpus = [
        {"id": "a", "content": "optimizar performance de código"},
        {"id": "b", "content": "documentação de viagem"},
    ]
    retriever.index_bm25(corpus)
    res = retriever.retrieve("performance", k=1, search_type="keyword")
    assert res
    assert res[0]["id"] == "a"


def test_semantic_search_filters_by_threshold():
    retriever = _make_retriever()
    results = retriever.retrieve("python", k=2, similarity_threshold=0.5, search_type="semantic")
    # Deve trazer apenas documentos com distance <= 0.5 (score >= 0.5)
    assert len(results) == 2
    assert all(r["score"] >= 0.5 for r in results)


def test_keyword_search_after_index():
    retriever = _make_retriever()

    corpus = [
        {"content": "RAG tutorial", "metadata": {}, "id": "c1"},
        {"content": "Python tips", "metadata": {}, "id": "c2"},
    ]
    retriever.index_bm25(corpus)

    kw_results = retriever.retrieve("python", k=1, search_type="keyword")
    assert kw_results
    assert kw_results[0]["content"].lower().startswith("python")


def test_hybrid_search_merges_scores():
    retriever = _make_retriever()
    corpus = [
        {"content": "RAG tutorial", "metadata": {}, "id": "c1"},
        {"content": "Python tips", "metadata": {}, "id": "c2"},
    ]
    retriever.index_bm25(corpus)

    hybrid = retriever.retrieve("python", k=2, search_type="hybrid")
    assert hybrid
    # Score deve estar presente e <=1
    assert all("score" in r for r in hybrid) 