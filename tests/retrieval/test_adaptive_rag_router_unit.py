import sys
import importlib
import asyncio

import pytest

# Remover stub criado em conftest e carregar módulo real
sys.modules.pop("src.retrieval.adaptive_rag_router", None)

real_router = importlib.import_module("src.retrieval.adaptive_rag_router")

QueryComplexityClassifier = real_router.QueryComplexityClassifier  # type: ignore
QueryComplexity = real_router.QueryComplexity  # type: ignore

if not callable(QueryComplexityClassifier):
    pytest.skip("QueryComplexityClassifier stubbed; skipping router tests", allow_module_level=True)


@pytest.mark.asyncio
async def test_classify_simple():
    clf = QueryComplexityClassifier()
    result = await clf.classify("What is Python?")
    assert result.complexity == QueryComplexity.SIMPLE
    assert result.requires_context is False


@pytest.mark.asyncio
async def test_classify_complex():
    clf = QueryComplexityClassifier()
    q = "Analyze the implications of quantum computing on modern cryptography."
    result = await clf.classify(q)
    assert result.complexity in {QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP}
    assert result.requires_context is True


@pytest.mark.asyncio
async def test_classify_multi_hop():
    clf = QueryComplexityClassifier()
    q = "Compare the GDP of Brazil and Argentina between 2019 and 2021."
    result = await clf.classify(q)
    assert result.complexity in {QueryComplexity.MULTI_HOP, QueryComplexity.SINGLE_HOP}
    assert len(result.key_entities) >= 2 