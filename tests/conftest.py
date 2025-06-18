import sys
import os
import sys
from pathlib import Path
import types
import numpy as np
import prometheus_client
import tempfile
import shutil
import time
import gc
from unittest.mock import Mock, patch
from typing import Dict, Any, Generator, List

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

@pytest.fixture(scope="session", autouse=True)
def setup_global_test_environment():
    """
    Configura ambiente global de testes para toda a sessão.
    
    Este fixture:
    - Define variáveis de ambiente necessárias
    - Configura logging para testes
    - Prepara diretórios temporários
    - Limpa recursos após os testes
    """
    # Backup das variáveis de ambiente originais
    original_env = os.environ.copy()
    
    # Configurar variáveis de ambiente para testes
    test_env = {
        'TESTING': 'true',
        'LOG_LEVEL': 'DEBUG',
        'OPENAI_API_KEY': 'test-key-mock-global',
        'QDRANT_HOST': 'localhost',
        'QDRANT_PORT': '6333',
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USER': 'neo4j',
        'NEO4J_PASSWORD': 'test',
        'PYTHONPATH': str(Path(__file__).parent.parent),
        'PYTEST_RUNNING': 'true'
    }
    
    os.environ.update(test_env)
    
    # Configurar logging para testes
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suprimir logs verbosos durante testes
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    yield
    
    # Cleanup: restaurar ambiente original
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configura ambiente de teste para cada teste individual."""
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


# ------------------------------------------------------------------
# Fixtures para diretórios temporários e isolamento
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def temp_test_dir():
    """
    Cria diretório temporário para testes da sessão.
    
    Yields:
        Path: Caminho para diretório temporário
    """
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    temp_path = Path(temp_dir)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def isolated_temp_dir():
    """
    Cria diretório temporário isolado para cada teste.
    
    Yields:
        Path: Caminho para diretório temporário isolado
    """
    temp_dir = tempfile.mkdtemp(prefix="rag_isolated_")
    temp_path = Path(temp_dir)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def clean_environment():
    """
    Garante ambiente limpo para cada teste.
    
    Este fixture:
    - Limpa caches
    - Reseta singletons
    - Limpa registros globais
    """
    # Limpar caches conhecidos
    gc.collect()
    
    yield
    
    # Cleanup pós-teste
    gc.collect()


# ------------------------------------------------------------------
# Fixtures para monitoramento de performance
# ------------------------------------------------------------------

@pytest.fixture
def performance_monitor():
    """
    Monitor de performance para testes.
    
    Yields:
        PerformanceMonitor: Objeto para monitorar performance
    """
    monitor = PerformanceMonitor()
    monitor.start()
    
    yield monitor
    
    monitor.stop()
    
    # Log de performance se teste demorou muito
    if monitor.duration > 5.0:  # 5 segundos
        print(f"\nWarning: Teste demorou {monitor.duration:.2f}s")


class PerformanceMonitor:
    """
    Monitor simples de performance para testes.
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = 0
    
    def start(self):
        """Inicia monitoramento."""
        self.start_time = time.time()
    
    def stop(self):
        """Para monitoramento."""
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time
    
    def get_duration(self) -> float:
        """Retorna duração em segundos."""
        return self.duration


# ------------------------------------------------------------------
# Fixtures para dados de teste comuns
# ------------------------------------------------------------------

@pytest.fixture
def sample_documents():
    """
    Documentos de exemplo para testes.
    
    Returns:
        List[Dict]: Lista de documentos de teste
    """
    return [
        {
            "content": "Este é um documento sobre inteligência artificial. "
                      "Contém informações sobre machine learning e deep learning.",
            "metadata": {
                "source": "ai_doc.txt",
                "category": "technology",
                "author": "AI Expert"
            }
        },
        {
            "content": "Sustentabilidade é importante para o futuro. "
                      "Devemos cuidar do meio ambiente e usar energia renovável.",
            "metadata": {
                "source": "sustainability.txt",
                "category": "environment",
                "author": "Green Advocate"
            }
        },
        {
            "content": "Python é uma linguagem de programação versátil. "
                      "É amplamente usada em ciência de dados e desenvolvimento web.",
            "metadata": {
                "source": "python_guide.txt",
                "category": "programming",
                "author": "Dev Expert"
            }
        }
    ]


@pytest.fixture
def sample_queries():
    """
    Queries de exemplo para testes.
    
    Returns:
        List[str]: Lista de queries de teste
    """
    return [
        "O que é inteligência artificial?",
        "Como funciona machine learning?",
        "Quais são as práticas de sustentabilidade?",
        "Como reduzir o impacto ambiental?",
        "Explique deep learning",
        "Como programar em Python?",
        "Quais são as vantagens do Python?"
    ]


@pytest.fixture
def mock_config():
    """
    Configuração mock para testes.
    
    Returns:
        Dict: Configuração de teste
    """
    return {
        "chunking": {
            "method": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 100
        },
        "embeddings": {
            "model_name": "test-model",
            "device": "cpu"
        },
        "vectordb": {
            "type": "qdrant",
            "collection_name": "test_collection"
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        },
        "rag": {
            "fallback_to_llm": True,
            "min_relevance_score": 0.5
        }
    }


# ------------------------------------------------------------------
# Hooks do pytest
# ------------------------------------------------------------------

def pytest_configure(config):
    """
    Configuração executada no início dos testes.
    
    Args:
        config: Configuração do pytest
    """
    # Registrar marcadores customizados adicionais
    markers = [
        "slow: marca testes lentos",
        "integration: marca testes de integração",
        "unit: marca testes unitários",
        "edge_case: marca testes de casos extremos",
        "regression: marca testes de regressão",
        "smoke: marca testes de smoke",
        "critical: marca testes críticos",
        "external: marca testes que dependem de serviços externos",
        "mock: marca testes que usam mocks",
        "real_data: marca testes que usam dados reais",
        "memory: marca testes de uso de memória",
        "concurrent: marca testes concorrentes"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """
    Modifica itens coletados antes da execução.
    
    Args:
        config: Configuração do pytest
        items: Lista de itens de teste coletados
    """
    # Adicionar marcador 'slow' para testes que demoram
    for item in items:
        # Marcar testes com 'concurrent' como slow
        if 'concurrent' in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Marcar testes com 'performance' como slow
        if 'performance' in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Marcar testes com 'memory' como slow
        if 'memory' in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Marcar testes com 'integration' como slow
        if 'integration' in item.keywords:
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """
    Setup executado antes de cada teste.
    
    Args:
        item: Item de teste a ser executado
    """
    # Skip testes marcados como 'external' se não em ambiente CI
    if 'external' in item.keywords and not os.getenv('CI'):
        pytest.skip("Teste externo pulado em ambiente local")
    
    # Skip testes marcados como 'slow' se flag --fast foi usada
    if 'slow' in item.keywords and item.config.getoption('--fast', default=False):
        pytest.skip("Teste lento pulado com --fast")


def pytest_addoption(parser):
    """
    Adiciona opções customizadas ao pytest.
    
    Args:
        parser: Parser de argumentos do pytest
    """
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Pular testes lentos"
    )
    
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Executar apenas testes de integração"
    )
    
    parser.addoption(
        "--unit",
        action="store_true",
        default=False,
        help="Executar apenas testes unitários"
    )
    
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help="Executar apenas testes de smoke"
    )


@pytest.fixture
def pytest_options(request):
    """
    Fornece acesso às opções do pytest.
    
    Args:
        request: Objeto request do pytest
        
    Returns:
        Dict: Dicionário com opções ativas
    """
    return {
        'fast': request.config.getoption('--fast'),
        'integration': request.config.getoption('--integration'),
        'unit': request.config.getoption('--unit'),
        'smoke': request.config.getoption('--smoke')
    }