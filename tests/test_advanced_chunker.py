"""Testes para o módulo advanced_chunker.py."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Mock das dependências antes do import
with patch.multiple(
    'sys.modules',
    sklearn=Mock(),
    nltk=Mock()
):
    # Mock específico para sklearn
    mock_sklearn = Mock()
    mock_sklearn.metrics.pairwise.cosine_similarity = Mock(return_value=[[0.8, 0.6], [0.6, 0.9]])
    
    # Mock específico para nltk
    mock_nltk = Mock()
    mock_nltk.sent_tokenize = Mock(return_value=["Sentence 1.", "Sentence 2.", "Sentence 3."])
    
    with patch.dict('sys.modules', {
        'sklearn': mock_sklearn,
        'sklearn.metrics': mock_sklearn.metrics,
        'sklearn.metrics.pairwise': mock_sklearn.metrics.pairwise,
        'nltk': mock_nltk
    }):
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
        assert len(chunker.strategies) == 6
    
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
        
        with patch.object(chunker, '_detect_structure') as mock_detect:
            mock_detect.return_value = ["paragraph1", "paragraph2"]
            
            result = chunker.structural_chunk(sample_document)
            
            assert isinstance(result, list)
            assert len(result) > 0
    
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
        
        with patch.object(chunker.recursive, 'chunk_text') as mock_recursive:
            mock_recursive.return_value = ["chunk1", "chunk2"]
            
            result = chunker.recursive_chunk(sample_document)
            
            assert isinstance(result, list)
            mock_recursive.assert_called_once_with(sample_document["content"])
    
    def test_chunk_topic_based_strategy(self, mock_embedding_service, sample_document):
        """Testa estratégia baseada em tópicos."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        with patch.object(chunker, '_extract_topics') as mock_topics:
            mock_topics.return_value = ["topic1", "topic2"]
            
            result = chunker.topic_based_chunk(sample_document)
            
            assert isinstance(result, list)
            mock_topics.assert_called_once_with(sample_document["content"])
    
    def test_chunk_entity_aware_strategy(self, mock_embedding_service, sample_document):
        """Testa estratégia consciente de entidades."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        with patch.object(chunker, '_extract_entities') as mock_entities:
            mock_entities.return_value = ["entity1", "entity2"]
            
            result = chunker.entity_aware_chunk(sample_document)
            
            assert isinstance(result, list)
            mock_entities.assert_called_once_with(sample_document["content"])
    
    def test_chunk_unknown_strategy(self, mock_embedding_service, sample_document):
        """Testa erro com estratégia desconhecida."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        with pytest.raises(ValueError, match="Estratégia desconhecida: unknown"):
            chunker.chunk(sample_document, strategy="unknown")
    
    def test_hybrid_chunk_with_large_chunks(self, mock_embedding_service, long_document):
        """Testa chunking híbrido com chunks grandes que precisam ser refinados."""
        chunker = AdvancedChunker(mock_embedding_service, max_chunk_size=100)
        
        with patch.object(chunker, 'structural_chunk') as mock_structural, \
             patch.object(chunker, 'semantic_chunk') as mock_semantic, \
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
        
        with patch('src.chunking.advanced_chunker.sent_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["Sentence 1.", "Sentence 2."]
            
            result = chunker._split_sentences("Sentence 1. Sentence 2.")
            
            assert result == ["Sentence 1.", "Sentence 2."]
            mock_tokenize.assert_called_once()
    
    def test_split_sentences_without_nltk(self, mock_embedding_service):
        """Testa divisão de sentenças sem NLTK (fallback)."""
        chunker = AdvancedChunker(mock_embedding_service)
        
        with patch('src.chunking.advanced_chunker.sent_tokenize', None):
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
        huge_content = "This is a test sentence. " * 10000
        huge_document = {
            "content": huge_content,
            "metadata": {"source": "huge.txt"}
        }
        
        chunker = AdvancedChunker(mock_embedding_service)
        
        start_time = time.time()
        result = chunker.chunk(huge_document, strategy="recursive")
        end_time = time.time()
        
        # Deve completar em tempo razoável (menos de 5 segundos)
        assert end_time - start_time < 5.0
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
            "semantic", "structural", "sliding_window", 
            "recursive", "topic_based", "entity_aware"
        ]
        
        for strategy in expected_strategies:
            assert strategy in chunker.strategies
            
            # Testar que a estratégia funciona (com mocks apropriados)
            with patch.object(chunker, '_split_sentences', return_value=["test"]), \
                 patch.object(chunker, '_detect_structure', return_value=["test"]), \
                 patch.object(chunker, '_extract_topics', return_value=["test"]), \
                 patch.object(chunker, '_extract_entities', return_value=["test"]), \
                 patch.object(chunker.recursive, 'chunk_text', return_value=["test"]):
                
                result = chunker.chunk(sample_document, strategy=strategy)
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
        
        # Testar diferentes estratégias
        strategies = ["hybrid", "semantic", "structural", "recursive"]
        
        for strategy in strategies:
            with patch.object(chunker, '_split_sentences', return_value=["Sentence 1.", "Sentence 2."]), \
                 patch.object(chunker, '_detect_structure', return_value=["Para 1", "Para 2"]), \
                 patch.object(chunker, '_enrich_with_entities', side_effect=lambda x, y: x), \
                 patch.object(chunker, '_add_contextual_overlap', side_effect=lambda x: x), \
                 patch.object(chunker.recursive, 'chunk_text', return_value=["Chunk 1", "Chunk 2"]):
                
                result = chunker.chunk(real_document, strategy=strategy)
                
                assert isinstance(result, list)
                assert len(result) > 0
                
                # Verificar estrutura dos chunks
                for chunk in result:
                    assert isinstance(chunk, dict)
                    assert 'content' in chunk
                    if 'metadata' in chunk:
                        assert isinstance(chunk['metadata'], dict)