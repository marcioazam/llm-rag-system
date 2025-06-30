"""
Conftest de emergência com mocks básicos
Criado automaticamente pelo QuickCoverageAnalyzer
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session", autouse=True)
def mock_heavy_dependencies():
    """Mock automático para dependências pesadas"""
    mocks = {}
    
    # Mock sentence-transformers
    try:
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_st.return_value.encode.return_value = [[0.1] * 384]
            mocks['sentence_transformer'] = mock_st
            yield mocks
    except:
        pass
    
    # Mock qdrant-client
    try:
        with patch('qdrant_client.QdrantClient') as mock_qc:
            mock_qc.return_value.search.return_value = []
            mocks['qdrant_client'] = mock_qc
    except:
        pass
    
    # Mock openai
    try:
        with patch('openai.OpenAI') as mock_openai:
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_openai.return_value.embeddings.create.return_value = mock_response
            mocks['openai'] = mock_openai
    except:
        pass

@pytest.fixture
def sample_text():
    """Texto de exemplo para testes"""
    return "Este é um texto de exemplo para testes de RAG."

@pytest.fixture
def sample_embedding():
    """Embedding de exemplo"""
    return [0.1] * 384

@pytest.fixture
def sample_query():
    """Query de exemplo"""
    return "Como funciona o sistema RAG?"
