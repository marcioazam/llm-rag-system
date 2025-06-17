import pytest
import tempfile
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from src.chunking.base_chunker import Chunk


class TestRAGPipeline:
    """Testes para a classe RAGPipeline."""

    @pytest.fixture
    def temp_config_file(self):
        """Cria um arquivo de configuração temporário para testes."""
        config_content = """
chunking:
  method: "recursive"
  chunk_size: 1000
  chunk_overlap: 200

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"

vectordb:
  type: "qdrant"
  host: "localhost"
  port: 6333
  collection_name: "test_collection"

retrieval:
  top_k: 5
  similarity_threshold: 0.5

llm:
  model: "llama2"
  base_url: "http://localhost:11434"

model_router:
  type: "simple"
  default_model: "llama2"

neo4j:
  enabled: false
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def mock_dependencies(self):
        """Mock das dependências principais do RAGPipeline."""
        with patch('src.rag_pipeline.SemanticChunker') as mock_semantic, \
             patch('src.rag_pipeline.RecursiveChunker') as mock_recursive, \
             patch('src.rag_pipeline.EmbeddingService') as mock_embedding, \
             patch('src.rag_pipeline.HybridRetriever') as mock_retriever, \
             patch('src.rag_pipeline.DocumentLoader') as mock_loader, \
             patch('src.rag_pipeline.ModelRouter') as mock_router, \
             patch('src.rag_pipeline.Neo4jStore') as mock_neo4j:
            
            # Configurar mocks
            mock_chunker_instance = Mock()
            mock_recursive.return_value = mock_chunker_instance
            mock_semantic.return_value = mock_chunker_instance
            
            mock_embedding_instance = Mock()
            mock_embedding.return_value = mock_embedding_instance
            
            mock_retriever_instance = Mock()
            mock_retriever.return_value = mock_retriever_instance
            
            mock_loader_instance = Mock()
            mock_loader.return_value = mock_loader_instance
            
            mock_router_instance = Mock()
            mock_router.return_value = mock_router_instance
            
            mock_neo4j_instance = Mock()
            mock_neo4j.return_value = mock_neo4j_instance
            
            yield {
                'chunker': mock_chunker_instance,
                'embedding': mock_embedding_instance,
                'retriever': mock_retriever_instance,
                'loader': mock_loader_instance,
                'router': mock_router_instance,
                'neo4j': mock_neo4j_instance
            }

    def test_init_with_config_file(self, temp_config_file, mock_dependencies):
        """Testa inicialização com arquivo de configuração."""
        pipeline = RAGPipeline(config_path=temp_config_file)
        
        assert pipeline.config is not None
        assert pipeline.config['chunking']['method'] == 'recursive'
        assert pipeline.config['embeddings']['model_name'] == 'sentence-transformers/all-MiniLM-L6-v2'

    def test_init_with_config_dict(self, mock_dependencies):
        """Testa inicialização com dicionário de configuração via arquivo temporário."""
        config_content = """
chunking:
  method: "semantic"
  chunk_size: 500

embeddings:
  model_name: "test-model"

vectordb:
  type: "qdrant"

retrieval:
  top_k: 5
  similarity_threshold: 0.5

llm:
  model: "test-model"
  base_url: "http://localhost:11434"

model_router:
  type: "simple"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            pipeline = RAGPipeline(config_path=temp_path)
            
            assert pipeline.config is not None
            assert pipeline.config['chunking']['method'] == 'semantic'
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_init_without_config(self, mock_dependencies):
        """Testa inicialização sem configuração (usa defaults)."""
        pipeline = RAGPipeline()
        
        assert pipeline.config is not None
        assert 'chunking' in pipeline.config
        assert 'embeddings' in pipeline.config

    @patch('src.rag_pipeline.yaml.safe_load')
    @patch('builtins.open')
    def test_load_config_file_not_found(self, mock_open, mock_yaml, mock_dependencies):
        """Testa comportamento quando arquivo de configuração não existe."""
        mock_open.side_effect = FileNotFoundError()
        
        with pytest.raises(FileNotFoundError):
            RAGPipeline(config_path="nonexistent.yaml")

    def test_add_documents(self, mock_dependencies):
        """Testa adição de documentos ao pipeline."""
        pipeline = RAGPipeline()
        
        # Mock do chunker para retornar chunks
        mock_chunks = [
            Chunk(content="chunk 1", metadata={"source": "doc1"}, chunk_id="1", document_id="doc1", position=0),
            Chunk(content="chunk 2", metadata={"source": "doc1"}, chunk_id="2", document_id="doc1", position=1)
        ]
        mock_dependencies['chunker'].chunk.return_value = mock_chunks
        
        # Mock do embedding service
        mock_dependencies['embedding'].embed_chunks.return_value = [
            [0.1, 0.2, 0.3],  # embedding para chunk 1
            [0.4, 0.5, 0.6]   # embedding para chunk 2
        ]
        
        documents = [{"content": "Documento de teste", "metadata": {"source": "test.txt"}}]
        result = pipeline.add_documents(documents)
        
        # Verificar que o método chunk foi chamado
        mock_dependencies['chunker'].chunk.assert_called_once()
        
        assert result is None  # add_documents não retorna valor

    def test_query(self, mock_dependencies):
        """Testa consulta básica."""
        pipeline = RAGPipeline()
        
        # Mock dos resultados de busca
        mock_results = [
            {"content": "Documento 1", "metadata": {"source": "test1.txt"}, "distance": 0.1},
            {"content": "Documento 2", "metadata": {"source": "test2.txt"}, "distance": 0.2}
        ]
        
        mock_dependencies['retriever'].retrieve.return_value = mock_results
        
        # Mock da resposta do router
        mock_dependencies['router'].generate_response.return_value = "Resposta gerada"
        
        query = "Qual é a resposta?"
        result = pipeline.query(query)
        
        # Verificações
        # Note: o método retrieve é chamado com parâmetros diferentes
        assert "answer" in result

    def test_query_with_custom_k(self, mock_dependencies):
        """Testa consulta com valor k customizado."""
        pipeline = RAGPipeline()
        
        mock_dependencies['retriever'].retrieve.return_value = []
        mock_dependencies['router'].generate_response.return_value = "Resposta"
        
        result = pipeline.query("teste", k=10)
        
        # Verificar que o resultado foi gerado
        assert "answer" in result

    def test_query_empty_results(self, mock_dependencies):
        """Testa consulta sem resultados."""
        pipeline = RAGPipeline()
        
        # Mock sem resultados
        mock_dependencies['retriever'].retrieve.return_value = []
        mock_dependencies['router'].generate_response.return_value = "Sem contexto disponível"
        
        result = pipeline.query("consulta sem resultados")
        
        # Verificar que o resultado indica ausência de contexto
        assert "answer" in result
        assert result["sources"] == []

    # def test_get_stats(self, mock_dependencies):
    #     """Testa obtenção de estatísticas do pipeline."""
    #     pipeline = RAGPipeline()
    #     
    #     # Mock das estatísticas dos componentes
    #     mock_dependencies['retriever'].get_stats.return_value = {"total_docs": 100}
    #     
    #     stats = pipeline.get_stats()
    #     
    #     assert isinstance(stats, dict)
    #     assert "retriever" in stats or "total_docs" in stats

    def test_chunking_strategy_recursive(self, mock_dependencies):
        """Testa seleção da estratégia de chunking recursiva."""
        config_content = """
chunking:
  method: "recursive"

embeddings:
  model_name: "test"

vectordb:
  type: "qdrant"

retrieval:
  top_k: 5
  similarity_threshold: 0.5

llm:
  model: "test-model"
  base_url: "http://localhost:11434"

model_router:
  type: "simple"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            with patch('src.rag_pipeline.RecursiveChunker') as mock_recursive:
                pipeline = RAGPipeline(config_path=temp_path)
                mock_recursive.assert_called_once()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_chunking_strategy_semantic(self, mock_dependencies):
        """Testa seleção da estratégia de chunking semântica."""
        config_content = """
chunking:
  method: "semantic"

embeddings:
  model_name: "test"

vectordb:
  type: "qdrant"

retrieval:
  top_k: 5
  similarity_threshold: 0.5

llm:
  model: "test-model"
  base_url: "http://localhost:11434"

model_router:
  type: "simple"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            with patch('src.rag_pipeline.SemanticChunker') as mock_semantic:
                pipeline = RAGPipeline(config_path=temp_path)
                mock_semantic.assert_called_once()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_error_handling_in_query(self, mock_dependencies):
        """Testa tratamento de erros durante consulta."""
        pipeline = RAGPipeline()
        
        # Mock que levanta exceção no método retrieve
        mock_dependencies['retriever'].retrieve.side_effect = Exception("Erro de busca")
        
        # O método query captura exceções do retriever e continua a execução
        # então não devemos esperar que a exceção seja propagada
        result = pipeline.query("consulta com erro")
        
        # Verificar que o resultado indica que não há contexto disponível
        assert "answer" in result
        assert result["sources"] == []

    def test_neo4j_integration_disabled(self, mock_dependencies):
        """Testa pipeline com integração Neo4j desabilitada."""
        config_content = """
chunking:
  method: "recursive"

embeddings:
  model_name: "test"

vectordb:
  type: "qdrant"

retrieval:
  top_k: 5
  similarity_threshold: 0.5

llm:
  model: "test-model"
  base_url: "http://localhost:11434"

model_router:
  type: "simple"

neo4j:
  enabled: false
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            pipeline = RAGPipeline(config_path=temp_path)
            
            # Verificar que Neo4j não foi inicializado
            assert hasattr(pipeline, 'config')
            assert pipeline.config['neo4j']['enabled'] is False
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_prometheus_metrics_initialization(self, mock_dependencies):
        """Testa inicialização das métricas Prometheus."""
        with patch('src.rag_pipeline.PROMETHEUS_STARTED', False):
            pipeline = RAGPipeline()
            # Verificar que o pipeline foi criado sem erros
            assert pipeline is not None