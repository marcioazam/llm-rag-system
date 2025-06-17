import sys
from pathlib import Path
import types
import numpy as np
import prometheus_client

# Garantir que a pasta raiz do projeto esteja no sys.path antes de imports locais
root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.pipeline_dependency import get_pipeline

# ------------------------------------------------------------------
# Stub sentence_transformers
# ------------------------------------------------------------------
stub_st = types.ModuleType("sentence_transformers")

class _StubSentenceTransformer:  # pylint: disable=too-few-public-methods
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):  # noqa: D401
        if isinstance(texts, str):
            return np.zeros(3)
        return [np.zeros(3) for _ in texts]

class _StubCrossEncoder:  # pylint: disable=too-few-public-methods
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):  # noqa: D401
        return np.zeros(len(pairs))

stub_st.SentenceTransformer = _StubSentenceTransformer
stub_st.CrossEncoder = _StubCrossEncoder
sys.modules["sentence_transformers"] = stub_st

# ------------------------------------------------------------------
# Stub torch (apenas atributos usados)
# ------------------------------------------------------------------
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _dummy_ctx:  # pylint: disable=too-few-public-methods
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def inference_mode():  # noqa: D401
        return _dummy_ctx()

    torch_stub.inference_mode = inference_mode
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_stub

class _DummyPipeline:
    """Pipeline leve para mocks em testes API."""

    def query_llm_only(self, question: str, system_prompt: str | None = None):
        return {"answer": "[mock]", "sources": [], "model": "dummy"}

    def query(self, query_text: str, k: int = 5, use_hybrid: bool = True):
        return {
            "answer": "[mock-hybrid]",
            "sources": [],
            "model": "dummy",
            "strategy": "hybrid",
        }

@pytest.fixture(scope="session", autouse=True)
def override_pipeline_dependency():
    """Substitui o get_pipeline por dummy durante suíte de testes."""
    # Para Depends (FastAPI)
    app.dependency_overrides[get_pipeline] = lambda: _DummyPipeline()
    # Para chamadas diretas em rotas
    import src.api.pipeline_dependency as _dep  # import local para evitar ciclos
    _dep.get_pipeline = lambda: _DummyPipeline()  # type: ignore
    import src.api.main as _api_main
    _api_main.get_pipeline = lambda: _DummyPipeline()  # type: ignore
    yield
    app.dependency_overrides.pop(get_pipeline, None)
    import src.api.pipeline_dependency as _dep
    _dep.get_pipeline = get_pipeline  # restabelecer original
    import src.api.main as _api_main
    _api_main.get_pipeline = get_pipeline

@pytest.fixture()
def api_client():
    """Client FastAPI pronto para uso."""
    with TestClient(app) as client:
        yield client 

# ------------------------------------------------------------------
# Reset global CollectorRegistry entre testes para evitar duplicação de métricas
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_prometheus_registry(monkeypatch):
    """Garante que cada teste tenha um CollectorRegistry limpo.

    Isso evita `ValueError: Duplicated timeseries` quando diferentes testes
    criam várias instâncias de `RAGPipeline`, cada uma registrando as mesmas
    métricas no registry padrão do Prometheus.
    """
    # Substitui o registry global por um novo CollectorRegistry vazio.
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)

    # Reset da flag global dentro de src.rag_pipeline, se existir
    import importlib
    try:
        rp = importlib.import_module("src.rag_pipeline")
        if hasattr(rp, "PROMETHEUS_STARTED"):
            rp.PROMETHEUS_STARTED = False  # type: ignore
    except ModuleNotFoundError:
        pass

    yield 