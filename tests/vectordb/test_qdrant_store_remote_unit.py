import importlib
import sys
import types
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Stub completo que simula ``qdrant_client`` operando remotamente.
# ---------------------------------------------------------------------------

def _inject_full_qdrant_stub() -> None:
    """Injeta stub de ``qdrant_client`` nos módulos do sistema."""
    # Remover qualquer implementação pré-existente
    for mod in list(sys.modules):
        if mod.startswith("qdrant_client"):
            sys.modules.pop(mod, None)

    qc_mod = types.ModuleType("qdrant_client")

    # ----------------- Submódulo http.models -----------------
    http_mod = types.ModuleType("qdrant_client.http")
    rest_mod = types.ModuleType("qdrant_client.http.models")

    class Distance:  # type: ignore
        COSINE = "Cosine"
        EUCLID = "Euclid"

        @classmethod
        def __getattr__(cls, _):  # noqa: D401
            return "Cosine"

    class VectorParams:  # type: ignore
        def __init__(self, size: int, distance: str):
            self.size = size
            self.distance = distance

    class MatchValue:  # type: ignore
        def __init__(self, value):
            self.value = value

    class FieldCondition:  # type: ignore
        def __init__(self, key, match):
            self.key = key
            self.match = match

    class Filter:  # type: ignore
        def __init__(self, must=None):
            self.must = must or []

    class PointStruct:  # type: ignore
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload
            self.score = 0.0

    # Registrar stubs
    rest_mod.Distance = Distance  # type: ignore
    rest_mod.VectorParams = VectorParams  # type: ignore
    rest_mod.MatchValue = MatchValue  # type: ignore
    rest_mod.FieldCondition = FieldCondition  # type: ignore
    rest_mod.Filter = Filter  # type: ignore
    rest_mod.PointStruct = PointStruct  # type: ignore

    http_mod.models = rest_mod  # type: ignore
    qc_mod.http = http_mod  # type: ignore

    # ----------------- QdrantClient Stub -----------------
    class _DummyQdrantClient:  # type: ignore
        def __init__(self, *_, **__):
            self._points: dict[str, List[PointStruct]] = {}
            self._config = types.SimpleNamespace(host="stub", port=6333)

        # Coleções --------------------------------------------------------
        def get_collections(self):  # noqa: D401
            class _Resp:  # noqa: D401
                def __init__(self, names):
                    self.collections = [types.SimpleNamespace(name=n) for n in names]

            return _Resp(self._points.keys())

        def create_collection(self, collection_name, vectors_config):  # noqa: D401
            self._points.setdefault(collection_name, [])

        # Pontos ----------------------------------------------------------
        def upsert(self, collection_name, points):  # noqa: D401
            self._points.setdefault(collection_name, [])
            self._points[collection_name].extend(points)

        def search(self, collection_name, query_vector, limit=5, query_filter=None):  # noqa: D401
            pts = self._points.get(collection_name, [])
            results = []
            for p in pts:
                score = sum(a * b for a, b in zip(p.vector, query_vector))
                results.append(types.SimpleNamespace(id=p.id, payload=p.payload, score=score))
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:limit]

        def get_collection(self, collection_name):  # noqa: D401
            class _Info:  # noqa: D401
                def __init__(self, count):
                    self.points_count = count

            return _Info(len(self._points.get(collection_name, [])))

        def delete(self, collection_name, points_selector):  # noqa: D401
            ids = set(points_selector.points)
            pts = self._points.get(collection_name, [])
            self._points[collection_name] = [p for p in pts if p.id not in ids]
            return True

        def delete_collection(self, collection_name):  # noqa: D401
            self._points.pop(collection_name, None)
            return True

        def retrieve(self, collection_name, ids):  # noqa: D401
            return [p for p in self._points.get(collection_name, []) if p.id in ids]

    qc_mod.QdrantClient = _DummyQdrantClient  # type: ignore

    # Registrar módulos
    sys.modules.update(
        {
            "qdrant_client": qc_mod,
            "qdrant_client.http": http_mod,
            "qdrant_client.http.models": rest_mod,
        }
    )


# ---------------------------------------------------------------------------
# Fixtures e utilidades
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _remote_stub():
    """Ativa stub antes de cada teste e recarrega implementação."""
    _inject_full_qdrant_stub()
    if "src.vectordb.qdrant_store" in sys.modules:
        importlib.reload(sys.modules["src.vectordb.qdrant_store"])
    yield


def _make_remote_store():
    from src.vectordb.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore(collection_name="unit_collection", dim=3)
    # Certificar que não estamos em modo memória
    assert getattr(store, "_in_memory", False) is False
    return store


def _rand_vec():
    return [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Testes principais
# ---------------------------------------------------------------------------

def test_remote_full_flow():
    store = _make_remote_store()

    docs = ["alpha", "beta", "gamma"]
    embeds = [_rand_vec() for _ in docs]
    metas = [{"idx": i} for i in range(len(docs))]

    # add_documents -----------------------------------------------------
    assert store.add_documents(docs, embeddings=embeds, metadata=metas)
    assert store.get_document_count() == 3

    # search (embedding) -----------------------------------------------
    res = store.search(query=embeds[0], k=2)
    assert len(res) == 2 and all("content" in r for r in res)

    # search (string + query_embedding) ---------------------------------
    res_txt = store.search(query="texto", query_embedding=embeds[1], k=1)
    assert isinstance(res_txt, list)

    # update & get ------------------------------------------------------
    assert store.update_document("doc_0", embedding=[0.0, 0.0, 1.0])
    doc = store.get_document_by_id("doc_0")
    assert doc is None or doc["id"] == "doc_0"

    # delete ------------------------------------------------------------
    assert store.delete_documents(["doc_2"])
    assert store.get_document_count() == 2

    # info --------------------------------------------------------------
    info = store.get_collection_info()
    assert info.get("name") == "unit_collection" and info.get("count") == 2

    # clear_collection --------------------------------------------------
    assert store.clear_collection()
    assert store.get_document_count() == 0

    store.close()


def test_error_paths(monkeypatch):
    """Valida caminhos de erro quando o stub levanta exceções."""
    store = _make_remote_store()

    def _boom(*_, **__):
        raise RuntimeError("fail")

    # search -----------------------------------------------------------
    monkeypatch.setattr(store.client, "search", _boom)
    assert store.search(query=_rand_vec()) == []

    # delete_documents --------------------------------------------------
    monkeypatch.setattr(store.client, "delete", _boom)
    assert store.delete_documents(["x"]) is False

    store.close() 