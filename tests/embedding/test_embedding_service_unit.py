import sys, types, importlib

# Stub sentence_transformers before import
st_mod = types.ModuleType("sentence_transformers")
class _DummyST:
    def __init__(self, *a, **k):
        pass
    def encode(self, texts):  # noqa: D401
        return [[0.0] * 3 for _ in texts]

st_mod.SentenceTransformer = _DummyST  # type: ignore
sys.modules["sentence_transformers"] = st_mod

es_mod = importlib.import_module("src.embedding.embedding_service")
EmbeddingService = es_mod.EmbeddingService  # type: ignore


def test_embed_texts_sentence_transformers():
    svc = EmbeddingService(provider="sentence-transformers", model="dummy-model")
    embeddings = svc.embed_texts(["a", "b"])
    assert embeddings  # retorna lista ou objeto iterável


def test_embed_text_single():
    svc = EmbeddingService(provider="sentence-transformers", model="dummy-model")
    vec = svc.embed_text("hello")
    assert vec 