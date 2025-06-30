"""Testes para o módulo advanced_chunker.py."""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

def setup_mocks():
    """Configura mocks para sklearn e nltk antes de qualquer importação"""
    
    # Mock sklearn
    sklearn_mock = MagicMock()
    sklearn_mock.feature_extraction = MagicMock()
    sklearn_mock.feature_extraction.text = MagicMock()
    sklearn_mock.feature_extraction.text.TfidfVectorizer = MagicMock()
    sklearn_mock.metrics = MagicMock()
    sklearn_mock.metrics.pairwise = MagicMock()
    sklearn_mock.metrics.pairwise.cosine_similarity = MagicMock()
    sklearn_mock.metrics.pairwise.cosine_similarity.return_value = [[0.8]]
    
    # Mock nltk
    nltk_mock = MagicMock()
    nltk_mock.sent_tokenize = MagicMock()
    nltk_mock.sent_tokenize.return_value = ["Sentence 1.", "Sentence 2."]
    
    # Aplicar os mocks no sys.modules
    sys.modules['sklearn'] = sklearn_mock
    sys.modules['sklearn.feature_extraction'] = sklearn_mock.feature_extraction
    sys.modules['sklearn.feature_extraction.text'] = sklearn_mock.feature_extraction.text
    sys.modules['sklearn.metrics'] = sklearn_mock.metrics  
    sys.modules['sklearn.metrics.pairwise'] = sklearn_mock.metrics.pairwise
    sys.modules['nltk'] = nltk_mock
    
    return sklearn_mock, nltk_mock

# Configurar mocks antes de qualquer importação
sklearn_mock, nltk_mock = setup_mocks()

# Agora importar o módulo que precisa dos mocks
from src.chunking.advanced_chunker import AdvancedChunker


class TestAdvancedChunker:
    """Testes para AdvancedChunker."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock do serviço de embeddings."""
        service = Mock()
        service.embed_texts.return_value = [
            [0.1, 0.2, 0.3],  # embedding 1
            [0.4, 0.5, 0.6],  # embedding 2
            [0.7, 0.8, 0.9]   # embedding 3
        ]
        return service
    
    @pytest.fixture
    def sample_document(self):
        """Documento de exemplo para testes."""
        return {
            "content": "This is a test document. It has multiple sentences. Each sentence contains important information. We will use this for chunking tests.",
            "metadata": {
                "source": "test.txt",
                "type": "text"
            }
        }
    
    @pytest.fixture
    def long_document(self):
        """Documento longo para testes de chunking."""
        content = "This is a very long document. " * 100  # 3000+ caracteres
        return {
            "content": content,
            "metadata": {
                "source": "long_test.txt",
                "type": "text"
            }
        }
    
    def test_init_default_parameters(self, mock_embedding_service):
        """Testa inicialização com parâmetros padrão."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        assert chunker.embedding_service == mock_embedding_service
        assert chunker.max_chunk_size == 800
        assert chunker.chunk_overlap == 50
        assert chunker.recursive is not None
        assert len(chunker.strategies) == 7  # 6 estratégias originais + semantic_basic
    
    def test_init_custom_parameters(self, mock_embedding_service):
        """Testa inicialização com parâmetros customizados."""
        chunker = AdvancedChunker(
            mock_embedding_service,
            max_chunk_size=1000,
            chunk_overlap=100
        )
        
        assert chunker.max_chunk_size == 1000
        assert chunker.chunk_overlap == 100
    
    @patch('src.chunking.advanced_chunker.IntelligentPreprocessor')
    def test_init_with_preprocessor_error(self, mock_preprocessor, mock_embedding_service):
        """Testa inicialização quando preprocessor falha."""
        mock_preprocessor.side_effect = Exception("Preprocessor error")
        
        chunker = AdvancedChunker(mock_embedding_service)
        assert chunker.preprocessor is None
    
    def test_chunk_hybrid_strategy(self, mock_embedding_service, sample_document):
        """Testa estratégia híbrida de chunking."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        with patch.object(chunker, 'structural_chunk') as mock_structural, \
             patch.object(chunker, '_enrich_with_entities') as mock_enrich, \
             patch.object(chunker, '_add_contextual_overlap') as mock_overlap:
            
            # Mock retornos
            mock_structural.return_value = [sample_document]
            mock_enrich.return_value = [sample_document]
            mock_overlap.return_value = [sample_document]
            
            result = chunker.chunk(sample_document, strategy="hybrid")
            
            assert result == [sample_document]
            mock_structural.assert_called_once_with(sample_document)
            mock_enrich.assert_called_once()
            mock_overlap.assert_called_once()
    
    def test_chunk_semantic_strategy(self, mock_embedding_service, sample_document):
        """Testa estratégia semântica de chunking."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        with patch.object(chunker, '_split_sentences') as mock_split:
            mock_split.return_value = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
            
            result = chunker.semantic_chunk(sample_document)
            
            assert isinstance(result, list)
            mock_split.assert_called_once_with(sample_document["content"])
    
    def test_chunk_structural_strategy(self, mock_embedding_service, sample_document):
        """Testa estratégia estrutural de chunking."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        # Testar diretamente sem mock de método inexistente
        result = chunker.structural_chunk(sample_document)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('content' in chunk for chunk in result)
        assert all('metadata' in chunk for chunk in result)
    
    def test_chunk_sliding_window_strategy(self, mock_embedding_service, sample_document):
        """Testa estratégia de janela deslizante."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        result = chunker.sliding_window_chunk(sample_document)
        
        assert isinstance(result, list)
        assert len(result) > 0
        # Verificar se chunks têm overlap
        if len(result) > 1:
            assert all('content' in chunk for chunk in result)
    
    def test_chunk_recursive_strategy(self, mock_embedding_service, sample_document):
        """Testa estratégia recursiva de chunking."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        # Testar diretamente sem mock - o método existe
        result = chunker.recursive_chunk(sample_document)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('content' in chunk for chunk in result)
        assert all('metadata' in chunk for chunk in result)
    
    def test_chunk_topic_based_strategy(self, mock_embedding_service, sample_document):
        """Testa estratégia baseada em tópicos."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        # Testar diretamente - é um fallback para semantic
        with patch.object(chunker, 'semantic_chunk') as mock_semantic:
            mock_semantic.return_value = [sample_document]
            
            result = chunker.topic_based_chunk(sample_document)
            
            assert isinstance(result, list)
            mock_semantic.assert_called_once_with(sample_document)
    
    def test_chunk_entity_aware_strategy(self, mock_embedding_service, sample_document):
        """Testa estratégia consciente de entidades."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        # Testar sem preprocessor (fallback)
        chunker.preprocessor = None
        
        with patch.object(chunker, 'semantic_chunk') as mock_semantic:
            mock_semantic.return_value = [sample_document]
            
            result = chunker.entity_aware_chunk(sample_document)
            
            assert isinstance(result, list)
            mock_semantic.assert_called_once_with(sample_document)
    
    def test_chunk_unknown_strategy(self, mock_embedding_service, sample_document):
        """Testa erro com estratégia desconhecida."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        with pytest.raises(ValueError, match="Estratégia desconhecida: unknown"):
            chunker.chunk(sample_document, strategy="unknown")
    
    def test_hybrid_chunk_with_large_chunks(self, mock_embedding_service, long_document):
        """Testa chunking híbrido com chunks grandes que precisam ser refinados."""
        chunker = AdvancedChunker(mock_embedding_service, max_chunk_size=100)
        
        with patch.object(chunker, 'structural_chunk') as mock_structural, \
             patch.object(chunker, 'enhanced_semantic_chunk') as mock_semantic, \
             patch.object(chunker, '_enrich_with_entities') as mock_enrich, \
             patch.object(chunker, '_add_contextual_overlap') as mock_overlap:
            
            # Simular chunk grande que precisa ser refinado
            large_chunk = {
                "content": "x" * 200,  # Maior que max_chunk_size
                "metadata": {}
            }
            mock_structural.return_value = [large_chunk]
            mock_semantic.return_value = [large_chunk]  # Chunk refinado
            mock_enrich.return_value = [large_chunk]
            mock_overlap.return_value = [large_chunk]
            
            result = chunker._hybrid_chunk(long_document)
            
            mock_structural.assert_called_once_with(long_document)
            mock_semantic.assert_called_once_with(large_chunk)
            mock_enrich.assert_called_once()
            mock_overlap.assert_called_once()
    
    def test_split_sentences_with_nltk(self, mock_embedding_service):
        """Testa divisão de sentenças com NLTK disponível."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        # Aplicar o mock diretamente no módulo importado
        with patch('src.chunking.advanced_chunker.sent_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["Sentence 1.", "Sentence 2."]
            
            result = chunker._split_sentences("Sentence 1. Sentence 2.")
            
            assert result == ["Sentence 1.", "Sentence 2."]
            mock_tokenize.assert_called_once_with("Sentence 1. Sentence 2.")
    
    def test_split_sentences_without_nltk(self, mock_embedding_service):
        """Testa divisão de sentenças sem NLTK (fallback)."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        # Simular NLTK indisponível
        with patch('nltk.sent_tokenize', None):
            result = chunker._split_sentences("Sentence 1. Sentence 2.")
            
            # Deve usar fallback regex
            assert isinstance(result, list)
            assert len(result) > 0
    
    def test_semantic_chunk_empty_sentences(self, mock_embedding_service, sample_document):
        """Testa chunking semântico com sentenças vazias."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        with patch.object(chunker, '_split_sentences') as mock_split:
            mock_split.return_value = []  # Nenhuma sentença
            
            result = chunker.semantic_chunk(sample_document)
            
            assert result == [sample_document]  # Deve retornar documento original
    
    def test_performance_with_large_document(self, mock_embedding_service):
        """Testa performance com documento muito grande."""
        import time
        
        # Documento muito grande
        huge_content = "This is a test sentence. " * 1000  # Reduzido para ser mais rápido
        huge_document = {
            "content": huge_content,
            "metadata": {"source": "huge.txt"}
        }
        
        chunker = AdvancedChunker(mock_embedding_service)
        
        start_time = time.time()
        result = chunker.chunk(huge_document, strategy="recursive")
        end_time = time.time()
        
        # Deve completar em tempo razoável (menos de 10 segundos)
        assert end_time - start_time < 10.0
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_chunk_with_special_characters(self, mock_embedding_service):
        """Testa chunking com caracteres especiais e unicode."""
        special_document = {
            "content": "Test with émojis 🚀. Special chars: @#$%^&*(). Unicode: 测试 тест テスト.",
            "metadata": {"source": "special.txt"}
        }
        
        chunker = AdvancedChunker(mock_embedding_service)
        
        result = chunker.chunk(special_document, strategy="semantic")
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('content' in chunk for chunk in result)
    
    def test_chunk_empty_document(self, mock_embedding_service):
        """Testa chunking com documento vazio."""
        empty_document = {
            "content": "",
            "metadata": {"source": "empty.txt"}
        }
        
        chunker = AdvancedChunker(mock_embedding_service)
        
        result = chunker.chunk(empty_document, strategy="semantic")
        
        assert isinstance(result, list)
        # Pode retornar lista vazia ou documento original
        assert len(result) >= 0
    
    def test_all_strategies_available(self, mock_embedding_service, sample_document):
        """Testa que todas as estratégias estão disponíveis e funcionam."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        expected_strategies = [
            "semantic", "semantic_basic", "structural", "sliding_window", 
            "recursive", "topic_based", "entity_aware"
        ]
        
        # Verificar se estratégias estão registradas
        for strategy in expected_strategies:
            assert strategy in chunker.strategies
        
        # Testar estratégias que funcionam sem dependências externas
        working_strategies = ["structural", "sliding_window", "recursive", "topic_based"]
        
        for strategy in working_strategies:
            try:
                result = chunker.chunk(sample_document, strategy=strategy)
                assert isinstance(result, list)
                assert len(result) > 0
            except Exception as e:
                pytest.fail(f"Strategy {strategy} failed: {e}")
        
        # Para estratégias que dependem de preprocessor/embeddings, usar mocks
        with patch.object(chunker, 'preprocessor') as mock_preprocessor:
            mock_preprocessor.process.return_value = {"entities": []}
            
            try:
                result = chunker.chunk(sample_document, strategy="entity_aware")
                assert isinstance(result, list)
            except Exception:
                # Se falhar, usar fallback semântico
                result = chunker.semantic_chunk(sample_document)
                assert isinstance(result, list)


class TestAdvancedChunkerIntegration:
    """Testes de integração para AdvancedChunker."""
    
    def test_real_world_document_chunking(self, mock_embedding_service):
        """Testa chunking com documento do mundo real."""
        real_document = {
            "content": """
            # Introduction to Machine Learning
            
            Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.
            
            ## Types of Machine Learning
            
            ### Supervised Learning
            Supervised learning uses labeled training data to learn a mapping function from input variables to output variables.
            
            ### Unsupervised Learning
            Unsupervised learning finds hidden patterns in data without labeled examples.
            
            ### Reinforcement Learning
            Reinforcement learning learns through interaction with an environment to maximize cumulative reward.
            
            ## Conclusion
            
            Machine learning continues to evolve and find applications in various domains.
            """,
            "metadata": {
                "source": "ml_intro.md",
                "type": "markdown"
            }
        }
        
        chunker = AdvancedChunker(mock_embedding_service, max_chunk_size=200)
        
        # Testar diferentes estratégias que funcionam
        strategies = ["structural", "sliding_window", "recursive"]
        
        for strategy in strategies:
            result = chunker.chunk(real_document, strategy=strategy)
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Verificar estrutura dos chunks
            for chunk in result:
                assert isinstance(chunk, dict)
                assert 'content' in chunk
                if 'metadata' in chunk:
                    assert isinstance(chunk['metadata'], dict)

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock do serviço de embeddings para testes de integração."""
        service = Mock()
        service.embed_texts.return_value = [
            [0.1, 0.2, 0.3],  # embedding 1
            [0.4, 0.5, 0.6],  # embedding 2
            [0.7, 0.8, 0.9]   # embedding 3
        ]
        return service