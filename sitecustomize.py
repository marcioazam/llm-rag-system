"""Auto-executado pelo Python; injeta stubs leves para dependências pesadas.

Este arquivo é importado automaticamente em cada inicialização do interpretador
quando reside no PYTHONPATH. Ele cria versões falsas de bibliotecas que
requerem downloads de modelos ou GPU, garantindo que testes rodem offline.
"""

import sys
import types
import numpy as np

# ------------------------------------------------------------------
# Stub sentence_transformers (SentenceTransformer & CrossEncoder)
# ------------------------------------------------------------------
st_mod = types.ModuleType("sentence_transformers")

class _DummyModel:  # pylint: disable=too-few-public-methods
    def __init__(self, *_, **__):
        pass

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):  # noqa: D401
        if isinstance(texts, str):
            return np.zeros(3)
        return [np.zeros(3) for _ in texts]

class _DummyCross:  # pylint: disable=too-few-public-methods
    def __init__(self, *_, **__):
        pass

    def predict(self, pairs):  # noqa: D401
        return np.zeros(len(pairs))

st_mod.SentenceTransformer = _DummyModel
st_mod.CrossEncoder = _DummyCross
sys.modules["sentence_transformers"] = st_mod

# ------------------------------------------------------------------
# Stub torch (somente atributos utilizados)
# ------------------------------------------------------------------
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _Ctx:  # pylint: disable=too-few-public-methods
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def inference_mode():  # noqa: D401
        return _Ctx()

    torch_stub.inference_mode = inference_mode
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_stub 