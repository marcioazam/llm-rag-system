import types
from typing import List

import pytest

from src.vectordb.qdrant_store import QdrantVectorStore


class _FakePoint:
    """Estrutura mínima para simular retorno do QdrantClient.search."""

    def __init__(self, id: str, vector: List[float], payload: dict | None = None, score: float = 0.42):
        self.id = id
        self.vector = vector
        self.payload = payload or {}
        self.score = score


def _make_remote_store(monkeypatch):  # pragma: no cover
    """Cria *store* forçando modo *remote* com client falso."""
    store = QdrantVectorStore(dim=3)
    # Forçar modo remoto
    store._in_memory = False  # type: ignore[attr-defined]

    # Client falso – apenas métodos utilizados nos testes
    fake_client = types.SimpleNamespace()

    def _upsert(*_, **__):  # noqa: D401
        # Retorna algo genérico – real client devolve dict
        return {"status": "ok"}

    def _search(*_, **__):  # noqa: D401
        pt = _FakePoint(
            id="unit1",
            vector=[0.1, 0.2, 0.3],
            payload={"document": "texto", "source": "test"},
            score=0.99,
        )
        return [pt]

    def _get_collection(*_, **__):  # noqa: D401
        return types.SimpleNamespace(points_count=5)

    def _delete(*_, **__):  # noqa: D401
        return True

    def _retrieve(*_, **__):  # noqa: D401
        pt = _FakePoint(id="unit1", vector=[0.1, 0.2, 0.3], payload={})
        return [pt]

    fake_client.upsert = _upsert  # type: ignore[attr-defined]
    fake_client.search = _search  # type: ignore[attr-defined]
    fake_client.get_collection = _get_collection  # type: ignore[attr-defined]
    fake_client.delete = _delete  # type: ignore[attr-defined]
    fake_client.retrieve = _retrieve  # type: ignore[attr-defined]

    store.client = fake_client  # type: ignore
    return store


def test_add_documents_remote_success(monkeypatch):
    store = _make_remote_store(monkeypatch)
    docs = ["A", "B"]
    embeds = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ok = store.add_documents(docs, embeddings=embeds)
    assert ok is True
    store.close()


def test_search_remote_format(monkeypatch):
    store = _make_remote_store(monkeypatch)
    res = store.search(query=[0.1, 0.2, 0.3], k=1)
    assert isinstance(res, list)
    if res:
        item = res[0]
        # Conteúdo formatado conforme esperado
        assert item["content"] == "texto"
        assert item["metadata"].get("source") == "test"
        assert "distance" in item and "id" in item
    store.close()


def test_get_document_count(monkeypatch):
    store = _make_remote_store(monkeypatch)
    assert store.get_document_count() == 5
    store.close()


def test_update_document_error(monkeypatch):
    store = _make_remote_store(monkeypatch)

    # Forçar erro no upsert para cobrir caminho de exceção
    def _boom(*_, **__):  # noqa: D401
        raise RuntimeError("upsert failed")

    store.client.upsert = _boom  # type: ignore

    ok = store.update_document("unit1", embedding=[0.1, 0.2, 0.3])
    assert ok is False
    store.close() 