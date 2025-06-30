"""
Configuracao global de testes com mocks para dependencias problematicas.
Atualizado para resolver conflitos de importacao.
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock
from pathlib import Path

# Mock de todas as dependencias externas problematicas
EXTERNAL_DEPS = {
    'torch': Mock(),
    'sentence_transformers': Mock(),
    'transformers': Mock(), 
    'qdrant_client': Mock(),
    'openai': Mock(),
    'neo4j': Mock(),
    'redis': Mock(),
    'sklearn': Mock(),
    'numpy': Mock(),
    'pandas': Mock(),
}

# Aplicar mocks antes de qualquer importacao
for name, mock_obj in EXTERNAL_DEPS.items():
    sys.modules[name] = mock_obj

# Mock classes especificas
class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.device = "cpu"
    
    def encode(self, texts, **kwargs):
        import numpy as np
        if isinstance(texts, str):
            return np.random.random(384)
        return np.random.random((len(texts), 384))

class MockQdrantClient:
    def __init__(self, *args, **kwargs):
        pass
    
    def search(self, *args, **kwargs):
        return []
    
    def upsert(self, *args, **kwargs):
        return {"status": "ok"}

# Configurar mocks especificos
sys.modules['sentence_transformers'].SentenceTransformer = MockSentenceTransformer
sys.modules['qdrant_client'].QdrantClient = MockQdrantClient

@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock automatico de dependencias externas para todos os testes."""
    yield

@pytest.fixture
def mock_openai_client():
    """Mock cliente OpenAI."""
    client = Mock()
    client.embeddings.create.return_value.data = [
        Mock(embedding=[0.1] * 1536)
    ]
    return client

@pytest.fixture  
def sample_documents():
    """Documentos de exemplo para testes."""
    return [
        {"id": "1", "content": "Sample document 1", "metadata": {}},
        {"id": "2", "content": "Sample document 2", "metadata": {}},
    ]
