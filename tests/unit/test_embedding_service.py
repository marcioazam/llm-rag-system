import numpy as np

from src.embeddings.embedding_service import EmbeddingService


def _make_service():
    # O modelo 'any' será interceptado pelo stub definido em sitecustomize
    return EmbeddingService(model_name="any", device="cpu", batch_size=2)


def test_embed_query_array():
    svc = _make_service()
    vec = svc.embed_query("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (3,)


def test_embed_texts_shape():
    svc = _make_service()
    arr = svc.embed_texts(["a", "b", "c"], show_progress=False)
    assert arr.shape == (3, 3) 