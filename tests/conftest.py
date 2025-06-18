import sys
import os
from pathlib import Path
import types
import numpy as np
import prometheus_client
from unittest.mock import Mock, patch

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
# Fixtures para mocks de APIs externas
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configura ambiente de teste com variáveis de ambiente."""
    # Salvar valores originais
    original_env = {}
    test_vars = {
        'OPENAI_API_KEY': 'test-key-mock',
        'TESTING': 'true',
        'QDRANT_HOST': 'localhost',
        'QDRANT_PORT': '6333',
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USER': 'neo4j',
        'NEO4J_PASSWORD': 'test'
    }
    
    for key, value in test_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restaurar valores originais
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def mock_openai_client():
    """Mock para cliente OpenAI."""
    with patch('openai.OpenAI') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock para chat completions
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Resposta mock do GPT"
        mock_instance.chat.completions.create.return_value = mock_response
        
        yield mock_instance


@pytest.fixture
def mock_qdrant_client():
    """Mock para Qdrant Client."""
    with patch('qdrant_client.QdrantClient') as mock_qdrant:
        mock_instance = Mock()
        mock_qdrant.return_value = mock_instance
        
        # Mock para operações básicas
        mock_instance.get_collections.return_value = Mock()
        mock_instance.upsert.return_value = Mock()
        mock_instance.search.return_value = []
        mock_instance.count.return_value = Mock(count=0)
        
        yield mock_instance


@pytest.fixture
def mock_neo4j_driver():
    """Mock para Neo4j Driver."""
    with patch('neo4j.GraphDatabase.driver') as mock_driver:
        mock_instance = Mock()
        mock_driver.return_value = mock_instance
        
        # Mock para sessões
        mock_session = Mock()
        mock_instance.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = []
        
        yield mock_instance


@pytest.fixture
def mock_external_apis():
    """Mock abrangente para todas as APIs externas."""
    with patch.multiple(
        'builtins.__import__',
        side_effect=lambda name, *args, **kwargs: {
            'openai': Mock(),
            'qdrant_client': Mock(),
            'neo4j': Mock()
        }.get(name, __import__(name, *args, **kwargs))
    ):
        yield

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