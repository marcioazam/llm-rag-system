import pytest, importlib, sys, types

es_mod = importlib.import_module("src.embedding.embedding_service")
EmbeddingService = es_mod.EmbeddingService  # type: ignore


def test_unknown_provider():
    with pytest.raises(ValueError):
        EmbeddingService(provider="unknown", model="x") 