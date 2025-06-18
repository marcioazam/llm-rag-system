"""Testes para o módulo embedding_service.py."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List, Dict, Any

# Mock das dependências antes do import
with patch.multiple(
    'sys.modules',
    openai=Mock(),
    sentence_transformers=Mock()
):
    from src.embedding.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Testes para EmbeddingService."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock do cliente OpenAI."""
        client = Mock()
        client.embeddings.create.return_value = Mock(
            data=[
                Mock(embedding=[0.1, 0.2, 0.3]),
                Mock(embedding=[0.4, 0.5, 0.6])
            ]
        )
        return client
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock do SentenceTransformer."""
        model = Mock()
        model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        return model
    
    def test_init_openai_provider(self, mock_openai_client):
        """Testa inicialização com provedor OpenAI."""
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            assert service.provider == "openai"
            assert service.model == "text-embedding-ada-002"
            assert service.client == mock_openai_client
    
    def test_init_sentence_transformers_provider(self, mock_sentence_transformer):
        """Testa inicialização com provedor SentenceTransformers."""
        with patch('src.embedding.embedding_service.SentenceTransformer', return_value=mock_sentence_transformer):
            service = EmbeddingService(
                provider="sentence-transformers",
                model="all-MiniLM-L6-v2"
            )
            
            assert service.provider == "sentence-transformers"
            assert service.model == "all-MiniLM-L6-v2"
            assert service.client == mock_sentence_transformer
    
    def test_init_invalid_provider(self):
        """Testa erro com provedor inválido."""
        with pytest.raises(ValueError, match="Provedor não suportado: invalid"):
            EmbeddingService(provider="invalid", model="test")
    
    def test_init_openai_without_api_key(self):
        """Testa erro OpenAI sem API key."""
        with pytest.raises(ValueError, match="API key é obrigatória para OpenAI"):
            EmbeddingService(provider="openai", model="text-embedding-ada-002")
    
    def test_embed_texts_openai_success(self, mock_openai_client):
        """Testa embedding de textos com OpenAI - sucesso."""
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            texts = ["Hello world", "Test text"]
            result = service.embed_texts(texts)
            
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_openai_client.embeddings.create.assert_called_once_with(
                model="text-embedding-ada-002",
                input=texts
            )
    
    def test_embed_texts_openai_api_error(self, mock_openai_client):
        """Testa erro da API OpenAI."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            with pytest.raises(Exception, match="Erro ao gerar embeddings com OpenAI: API Error"):
                service.embed_texts(["test"])
    
    def test_embed_texts_sentence_transformers_success(self, mock_sentence_transformer):
        """Testa embedding de textos com SentenceTransformers - sucesso."""
        with patch('src.embedding.embedding_service.SentenceTransformer', return_value=mock_sentence_transformer):
            service = EmbeddingService(
                provider="sentence-transformers",
                model="all-MiniLM-L6-v2"
            )
            
            texts = ["Hello world", "Test text"]
            result = service.embed_texts(texts)
            
            expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            assert result == expected
            mock_sentence_transformer.encode.assert_called_once_with(texts)
    
    def test_embed_texts_sentence_transformers_error(self, mock_sentence_transformer):
        """Testa erro com SentenceTransformers."""
        mock_sentence_transformer.encode.side_effect = Exception("Model Error")
        
        with patch('src.embedding.embedding_service.SentenceTransformer', return_value=mock_sentence_transformer):
            service = EmbeddingService(
                provider="sentence-transformers",
                model="all-MiniLM-L6-v2"
            )
            
            with pytest.raises(Exception, match="Erro ao gerar embeddings com SentenceTransformers: Model Error"):
                service.embed_texts(["test"])
    
    def test_embed_single_text(self, mock_openai_client):
        """Testa embedding de texto único."""
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1, 0.2, 0.3])]
        )
        
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            result = service.embed_text("Hello world")
            
            assert result == [0.1, 0.2, 0.3]
            mock_openai_client.embeddings.create.assert_called_once_with(
                model="text-embedding-ada-002",
                input=["Hello world"]
            )
    
    def test_embed_empty_texts(self, mock_openai_client):
        """Testa embedding de lista vazia."""
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            result = service.embed_texts([])
            
            assert result == []
            mock_openai_client.embeddings.create.assert_not_called()
    
    def test_embed_texts_with_special_characters(self, mock_openai_client):
        """Testa embedding com caracteres especiais."""
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            texts = ["Hello 🌍!", "Test with émojis 🚀", "Unicode: 测试 тест テスト"]
            result = service.embed_texts(texts)
            
            assert len(result) == 2  # Mock retorna 2 embeddings
            mock_openai_client.embeddings.create.assert_called_once_with(
                model="text-embedding-ada-002",
                input=texts
            )
    
    def test_embed_very_long_text(self, mock_openai_client):
        """Testa embedding de texto muito longo."""
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            long_text = "This is a very long text. " * 1000
            result = service.embed_text(long_text)
            
            assert result == [0.1, 0.2, 0.3]
    
    def test_batch_processing_large_list(self, mock_openai_client):
        """Testa processamento em lote de lista grande."""
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            # Lista com muitos textos
            texts = [f"Text {i}" for i in range(100)]
            result = service.embed_texts(texts)
            
            assert len(result) == 2  # Mock retorna 2 embeddings
            mock_openai_client.embeddings.create.assert_called_once()
    
    def test_different_models_openai(self, mock_openai_client):
        """Testa diferentes modelos OpenAI."""
        models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
        
        for model in models:
            with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
                service = EmbeddingService(
                    provider="openai",
                    model=model,
                    api_key="test-key"
                )
                
                service.embed_text("test")
                
                mock_openai_client.embeddings.create.assert_called_with(
                    model=model,
                    input=["test"]
                )
    
    def test_different_models_sentence_transformers(self, mock_sentence_transformer):
        """Testa diferentes modelos SentenceTransformers."""
        models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        for model in models:
            with patch('src.embedding.embedding_service.SentenceTransformer', return_value=mock_sentence_transformer) as mock_st:
                service = EmbeddingService(
                    provider="sentence-transformers",
                    model=model
                )
                
                mock_st.assert_called_with(model)
    
    def test_performance_measurement(self, mock_openai_client):
        """Testa medição de performance."""
        import time
        
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            start_time = time.time()
            service.embed_texts(["test"] * 10)
            end_time = time.time()
            
            # Deve completar rapidamente (menos de 1 segundo com mocks)
            assert end_time - start_time < 1.0
    
    def test_concurrent_requests_simulation(self, mock_openai_client):
        """Simula requisições concorrentes."""
        import threading
        import time
        
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            results = []
            
            def embed_worker(text):
                result = service.embed_text(f"test {text}")
                results.append(result)
            
            # Criar múltiplas threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=embed_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Aguardar conclusão
            for thread in threads:
                thread.join()
            
            assert len(results) == 5
            assert all(result == [0.1, 0.2, 0.3] for result in results)


class TestEmbeddingServiceIntegration:
    """Testes de integração para EmbeddingService."""
    
    def test_real_world_usage_pattern(self, mock_openai_client):
        """Testa padrão de uso do mundo real."""
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            # Simular uso típico: documentos, queries, etc.
            documents = [
                "This is a research paper about machine learning.",
                "The methodology section describes the experimental setup.",
                "Results show significant improvement over baseline methods."
            ]
            
            queries = [
                "machine learning research",
                "experimental methodology",
                "performance results"
            ]
            
            # Embeddings de documentos
            doc_embeddings = service.embed_texts(documents)
            assert len(doc_embeddings) == 2  # Mock retorna 2
            
            # Embeddings de queries
            for query in queries:
                query_embedding = service.embed_text(query)
                assert query_embedding == [0.1, 0.2, 0.3]
    
    def test_error_recovery_scenarios(self, mock_openai_client):
        """Testa cenários de recuperação de erro."""
        # Primeiro erro, depois sucesso
        mock_openai_client.embeddings.create.side_effect = [
            Exception("Temporary error"),
            Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
        ]
        
        with patch('src.embedding.embedding_service.OpenAI', return_value=mock_openai_client):
            service = EmbeddingService(
                provider="openai",
                model="text-embedding-ada-002",
                api_key="test-key"
            )
            
            # Primeira chamada deve falhar
            with pytest.raises(Exception):
                service.embed_text("test")
            
            # Segunda chamada deve funcionar
            result = service.embed_text("test")
            assert result == [0.1, 0.2, 0.3]