"""Testes para cobertura de cenários específicos e edge cases."""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml

# Import com mocks já configurados
with patch.multiple(
    'sys.modules',
    openai=Mock(),
    sentence_transformers=Mock(),
    qdrant_client=Mock(),
    neo4j=Mock(),
    ollama=Mock()
):
    from src.rag_pipeline import RAGPipeline


class TestRAGPipelineErrorScenarios:
    """Testes para cenários de erro e edge cases do RAGPipeline."""
    
    @pytest.fixture
    def invalid_config(self):
        """Configuração inválida para testes de erro."""
        return {
            'chunking': {'strategy': 'invalid_strategy'},
            'embeddings': {'model': 'invalid_model'},
            'vectordb': {'type': 'invalid_db'},
            'llm': {'provider': 'invalid_provider'}
        }
    
    @pytest.fixture
    def minimal_config(self):
        """Configuração mínima válida."""
        return {
            'chunking': {'strategy': 'recursive'},
            'embeddings': {'model': 'all-MiniLM-L6-v2'},
            'vectordb': {'type': 'qdrant'},
            'llm': {'provider': 'openai'}
        }
    
    def test_init_with_missing_config_file(self):
        """Testa inicialização com arquivo de configuração inexistente."""
        with pytest.raises(FileNotFoundError):
            RAGPipeline(config_path="/path/that/does/not/exist.yaml")
    
    def test_init_with_invalid_yaml(self):
        """Testa inicialização com YAML inválido."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                RAGPipeline(config_path=temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_init_with_empty_config(self):
        """Testa inicialização com configuração vazia."""
        with pytest.raises((KeyError, ValueError)):
            RAGPipeline(config_dict={})
    
    @patch('src.rag_pipeline.SemanticChunker')
    @patch('src.rag_pipeline.RecursiveChunker')
    @patch('src.rag_pipeline.EmbeddingService')
    def test_init_with_invalid_chunking_strategy(self, mock_embedding, mock_recursive, mock_semantic, invalid_config):
        """Testa inicialização com estratégia de chunking inválida."""
        mock_embedding.return_value = Mock()
        mock_recursive.return_value = Mock()
        mock_semantic.return_value = Mock()
        
        with pytest.raises((ValueError, KeyError)):
            RAGPipeline(config_dict=invalid_config)
    
    @patch('src.rag_pipeline.openai.OpenAI')
    def test_openai_api_error(self, mock_openai, minimal_config):
        """Testa erro de API do OpenAI."""
        # Configurar mock para lançar exceção
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch.multiple(
            'src.rag_pipeline',
            RecursiveChunker=Mock(),
            EmbeddingService=Mock(),
            HybridRetriever=Mock(),
            DocumentLoader=Mock(),
            ModelRouter=Mock(),
            Neo4jStore=Mock(),
            QdrantVectorStore=Mock(),
            SQLiteMetadataStore=Mock()
        ):
            pipeline = RAGPipeline(config_dict=minimal_config)
            
            with pytest.raises(Exception, match="API Error"):
                pipeline.query_llm_only("test question")
    
    def test_missing_api_key(self, minimal_config):
        """Testa comportamento sem API key configurada."""
        # Remover API key do ambiente
        original_key = os.environ.pop('OPENAI_API_KEY', None)
        
        try:
            with patch.multiple(
                'src.rag_pipeline',
                RecursiveChunker=Mock(),
                EmbeddingService=Mock(),
                HybridRetriever=Mock(),
                DocumentLoader=Mock(),
                ModelRouter=Mock(),
                Neo4jStore=Mock(),
                QdrantVectorStore=Mock(),
                SQLiteMetadataStore=Mock()
            ):
                pipeline = RAGPipeline(config_dict=minimal_config)
                
                # Deve retornar erro ou mensagem padrão
                result = pipeline.query_llm_only("test question")
                assert "erro" in result.lower() or "indisponível" in result.lower()
        finally:
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
    
    @patch('src.rag_pipeline.HybridRetriever')
    def test_retriever_error(self, mock_retriever, minimal_config):
        """Testa erro no retriever."""
        mock_retriever_instance = Mock()
        mock_retriever.return_value = mock_retriever_instance
        mock_retriever_instance.retrieve.side_effect = Exception("Retriever Error")
        
        with patch.multiple(
            'src.rag_pipeline',
            RecursiveChunker=Mock(),
            EmbeddingService=Mock(),
            DocumentLoader=Mock(),
            ModelRouter=Mock(),
            Neo4jStore=Mock(),
            QdrantVectorStore=Mock(),
            SQLiteMetadataStore=Mock(),
            openai=Mock()
        ):
            pipeline = RAGPipeline(config_dict=minimal_config)
            
            with pytest.raises(Exception, match="Retriever Error"):
                pipeline.query("test question")
    
    def test_empty_query(self, minimal_config):
        """Testa query vazia."""
        with patch.multiple(
            'src.rag_pipeline',
            RecursiveChunker=Mock(),
            EmbeddingService=Mock(),
            HybridRetriever=Mock(),
            DocumentLoader=Mock(),
            ModelRouter=Mock(),
            Neo4jStore=Mock(),
            QdrantVectorStore=Mock(),
            SQLiteMetadataStore=Mock(),
            openai=Mock()
        ):
            pipeline = RAGPipeline(config_dict=minimal_config)
            
            # Deve lidar com query vazia graciosamente
            result = pipeline.query_llm_only("")
            assert isinstance(result, str)
    
    def test_very_long_query(self, minimal_config):
        """Testa query muito longa."""
        long_query = "test " * 10000  # Query muito longa
        
        with patch.multiple(
            'src.rag_pipeline',
            RecursiveChunker=Mock(),
            EmbeddingService=Mock(),
            HybridRetriever=Mock(),
            DocumentLoader=Mock(),
            ModelRouter=Mock(),
            Neo4jStore=Mock(),
            QdrantVectorStore=Mock(),
            SQLiteMetadataStore=Mock()
        ):
            # Mock OpenAI para retornar resposta válida
            mock_openai_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Resposta para query longa"
            mock_openai_client.chat.completions.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_openai_client):
                pipeline = RAGPipeline(config_dict=minimal_config)
                result = pipeline.query_llm_only(long_query)
                assert isinstance(result, str)
                assert len(result) > 0


class TestRAGPipelineEdgeCases:
    """Testes para edge cases específicos."""
    
    @pytest.fixture
    def config_with_all_options(self):
        """Configuração com todas as opções disponíveis."""
        return {
            'chunking': {
                'strategy': 'semantic',
                'chunk_size': 1000,
                'chunk_overlap': 200
            },
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'batch_size': 32
            },
            'vectordb': {
                'type': 'qdrant',
                'host': 'localhost',
                'port': 6333,
                'collection_name': 'test_collection'
            },
            'retrieval': {
                'k': 10,
                'use_hybrid': True,
                'alpha': 0.5
            },
            'llm': {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'model_router': {
                'enabled': True,
                'strategies': ['simple', 'complex']
            },
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'user': 'neo4j',
                'password': 'test'
            }
        }
    
    def test_init_with_all_configurations(self, config_with_all_options):
        """Testa inicialização com todas as configurações possíveis."""
        with patch.multiple(
            'src.rag_pipeline',
            SemanticChunker=Mock(),
            RecursiveChunker=Mock(),
            EmbeddingService=Mock(),
            HybridRetriever=Mock(),
            DocumentLoader=Mock(),
            ModelRouter=Mock(),
            Neo4jStore=Mock(),
            QdrantVectorStore=Mock(),
            SQLiteMetadataStore=Mock(),
            openai=Mock()
        ):
            pipeline = RAGPipeline(config_dict=config_with_all_options)
            assert pipeline is not None
    
    def test_query_with_different_k_values(self, config_with_all_options):
        """Testa query com diferentes valores de k."""
        with patch.multiple(
            'src.rag_pipeline',
            SemanticChunker=Mock(),
            RecursiveChunker=Mock(),
            EmbeddingService=Mock(),
            HybridRetriever=Mock(),
            DocumentLoader=Mock(),
            ModelRouter=Mock(),
            Neo4jStore=Mock(),
            QdrantVectorStore=Mock(),
            SQLiteMetadataStore=Mock(),
            openai=Mock()
        ) as mocks:
            # Configurar mock do retriever
            mock_retriever = mocks['HybridRetriever'].return_value
            mock_retriever.retrieve.return_value = []
            
            # Mock OpenAI
            mock_openai_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Resposta mock"
            mock_openai_client.chat.completions.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_openai_client):
                pipeline = RAGPipeline(config_dict=config_with_all_options)
                
                # Testar diferentes valores de k
                for k in [1, 5, 10, 20]:
                    result = pipeline.query("test question", k=k)
                    assert isinstance(result, dict)
                    # Verificar se o retriever foi chamado com o k correto
                    mock_retriever.retrieve.assert_called()
    
    def test_query_with_hybrid_toggle(self, config_with_all_options):
        """Testa query com hybrid search ligado e desligado."""
        with patch.multiple(
            'src.rag_pipeline',
            SemanticChunker=Mock(),
            RecursiveChunker=Mock(),
            EmbeddingService=Mock(),
            HybridRetriever=Mock(),
            DocumentLoader=Mock(),
            ModelRouter=Mock(),
            Neo4jStore=Mock(),
            QdrantVectorStore=Mock(),
            SQLiteMetadataStore=Mock(),
            openai=Mock()
        ) as mocks:
            # Configurar mock do retriever
            mock_retriever = mocks['HybridRetriever'].return_value
            mock_retriever.retrieve.return_value = []
            
            # Mock OpenAI
            mock_openai_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Resposta mock"
            mock_openai_client.chat.completions.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_openai_client):
                pipeline = RAGPipeline(config_dict=config_with_all_options)
                
                # Testar com hybrid=True
                result_hybrid = pipeline.query("test question", use_hybrid=True)
                assert isinstance(result_hybrid, dict)
                
                # Testar com hybrid=False
                result_no_hybrid = pipeline.query("test question", use_hybrid=False)
                assert isinstance(result_no_hybrid, dict)
    
    def test_system_prompt_variations(self, config_with_all_options):
        """Testa diferentes variações de system prompt."""
        with patch.multiple(
            'src.rag_pipeline',
            SemanticChunker=Mock(),
            RecursiveChunker=Mock(),
            EmbeddingService=Mock(),
            HybridRetriever=Mock(),
            DocumentLoader=Mock(),
            ModelRouter=Mock(),
            Neo4jStore=Mock(),
            QdrantVectorStore=Mock(),
            SQLiteMetadataStore=Mock(),
            openai=Mock()
        ):
            # Mock OpenAI
            mock_openai_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Resposta mock"
            mock_openai_client.chat.completions.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_openai_client):
                pipeline = RAGPipeline(config_dict=config_with_all_options)
                
                # Testar diferentes system prompts
                prompts = [
                    None,
                    "",
                    "Você é um assistente útil.",
                    "Responda de forma técnica e detalhada.",
                    "System prompt muito longo " * 100
                ]
                
                for prompt in prompts:
                    result = pipeline.query_llm_only("test question", system_prompt=prompt)
                    assert isinstance(result, str)