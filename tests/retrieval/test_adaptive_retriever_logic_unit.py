import asyncio
import pytest

from types import SimpleNamespace


class _StubHybridRetriever:
    async def retrieve(self, query: str, limit: int, strategy: str = "auto", **kwargs):
        # Retorna N documentos fictícios
        return [{"content": f"doc{i}", "strategy": strategy} for i in range(limit)]


@pytest.fixture
def retriever():
    from src.retrieval.adaptive_retriever import AdaptiveRetriever

    return AdaptiveRetriever(base_retriever=_StubHybridRetriever())


@pytest.mark.parametrize(
    "query, expected_type",
    [
        ("O que é API?", "definition"),
        ("Liste frameworks python", "list"),
        ("Diferença entre REST e GraphQL", "comparison"),
        ("Como implementar quicksort em Python", "implementation"),
        ("Analise vantagens do PostgreSQL", "analysis"),
    ],
)
def test_identify_query_type(retriever, query, expected_type):
    analysis = retriever.analyze_query(query)
    assert analysis.query_type == expected_type
    assert 0 <= analysis.complexity_score <= 1
    assert 3 <= analysis.optimal_k <= 15


@pytest.mark.asyncio
async def test_retrieve_adaptive_returns_expected_counts(retriever):
    query = "O que é API REST?"
    result = await retriever.retrieve_adaptive(query)

    analysis = result["query_analysis"]
    k = analysis["optimal_k"]

    # Deve retornar exatamente k documentos após processamento (sem filtros extras)
    assert result["retrieval_metadata"]["after_processing"] == k

    # Estratégia deve vir do analysis
    assert analysis["strategy"] in {"dense", "sparse", "hybrid"} 