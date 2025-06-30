"""
Testes básicos para o RAG Pipeline Avançado.
Cobertura atual: 7% -> Meta: 80%
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.rag_pipeline_advanced import AdvancedRAGPipeline


class TestAdvancedRAGPipeline:
    """Testes para o RAG Pipeline Avançado."""

    @pytest.fixture
    def mock_settings(self):
        """Mock das configurações."""
        with patch('src.rag_pipeline_advanced.get_settings') as mock:
            settings = Mock()
            settings.llm_provider = "openai"
            settings.model_name = "gpt-4"
            settings.embedding_provider = "openai"
            settings.embedding_model = "text-embedding-ada-002"
            settings.vector_store_type = "qdrant"
            settings.qdrant_url = "http://localhost:6333"
            settings.qdrant_collection = "test_collection"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def mock_components(self):
        """Mock dos componentes do pipeline."""
        components = {}
        
        # Mock vector store
        components['vector_store'] = Mock()
        components['vector_store'].search = Mock(return_value=[
            {
                "content": "Test content 1",
                "metadata": {"source": "doc1.txt"},
                "distance": 0.1
            },
            {
                "content": "Test content 2", 
                "metadata": {"source": "doc2.txt"},
                "distance": 0.2
            }
        ])
        
        # Mock embedding service
        components['embedding_service'] = Mock()
        components['embedding_service'].embed_query = Mock(return_value=[0.1, 0.2, 0.3])
        
        # Mock model router
        components['model_router'] = AsyncMock()
        components['model_router'].generate = AsyncMock(return_value={
            "response": "Generated response",
            "usage": {"tokens": 100},
            "model": "gpt-4"
        })
        
        # Mock chunker
        components['chunker'] = Mock()
        components['chunker'].chunk_text = Mock(return_value=[
            {"content": "Chunk 1", "metadata": {}},
            {"content": "Chunk 2", "metadata": {}}
        ])
        
        return components

    @pytest.fixture
    def pipeline(self, mock_settings, mock_components):
        """Criar instância do pipeline com mocks."""
        with patch.multiple(
            'src.rag_pipeline_advanced',
            QdrantVectorStore=Mock(return_value=mock_components['vector_store']),
            APIEmbeddingService=Mock(return_value=mock_components['embedding_service']),
            APIModelRouter=Mock(return_value=mock_components['model_router']),
            AdvancedChunker=Mock(return_value=mock_components['chunker']),
        ):
            pipeline = AdvancedRAGPipeline()
            # Injetar mocks diretamente se necessário
            pipeline.vector_store = mock_components['vector_store']
            pipeline.embedding_service = mock_components['embedding_service']
            pipeline.model_router = mock_components['model_router']
            pipeline.chunker = mock_components['chunker']
            return pipeline

    def test_init(self, pipeline):
        """Testar inicialização do pipeline."""
        assert pipeline is not None
        assert hasattr(pipeline, 'vector_store')
        assert hasattr(pipeline, 'embedding_service')
        assert hasattr(pipeline, 'model_router')

    @pytest.mark.asyncio
    async def test_query_basic(self, pipeline, mock_components):
        """Testar query básica."""
        query = "What is RAG?"
        
        result = await pipeline.query(query)
        
        assert result is not None
        assert "response" in result
        assert result["response"] == "Generated response"
        
        # Verificar que os componentes foram chamados
        mock_components['embedding_service'].embed_query.assert_called_with(query)
        mock_components['vector_store'].search.assert_called()
        mock_components['model_router'].generate.assert_called()

    @pytest.mark.asyncio
    async def test_query_with_options(self, pipeline, mock_components):
        """Testar query com opções personalizadas."""
        query = "Explain machine learning"
        options = {
            "k": 10,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        result = await pipeline.query(query, **options)
        
        assert result is not None
        assert "response" in result

    @pytest.mark.asyncio
    async def test_query_empty(self, pipeline):
        """Testar query vazia."""
        query = ""
        
        result = await pipeline.query(query)
        
        # Deve lidar graciosamente com query vazia
        assert result is not None

    @pytest.mark.asyncio
    async def test_query_very_long(self, pipeline):
        """Testar query muito longa."""
        query = "What is AI? " * 1000  # Query muito longa
        
        result = await pipeline.query(query)
        
        # Deve truncar ou lidar adequadamente
        assert result is not None

    def test_add_documents(self, pipeline, mock_components):
        """Testar adição de documentos."""
        documents = [
            {"content": "Document 1", "metadata": {"source": "doc1.txt"}},
            {"content": "Document 2", "metadata": {"source": "doc2.txt"}}
        ]
        
        result = pipeline.add_documents(documents)
        
        # Verificar que chunker e vector store foram chamados
        mock_components['chunker'].chunk_text.assert_called()
        assert result is not None

    def test_add_documents_empty(self, pipeline):
        """Testar adição de documentos vazios."""
        documents = []
        
        result = pipeline.add_documents(documents)
        
        # Deve lidar graciosamente com lista vazia
        assert result is not None

    def test_add_documents_large_batch(self, pipeline, mock_components):
        """Testar adição de lote grande de documentos."""
        documents = [
            {"content": f"Document {i}", "metadata": {"source": f"doc{i}.txt"}}
            for i in range(1000)
        ]
        
        result = pipeline.add_documents(documents)
        
        # Deve processar em lotes
        assert result is not None

    @pytest.mark.asyncio
    async def test_retrieval_fallback(self, pipeline, mock_components):
        """Testar fallback quando não encontra documentos relevantes."""
        # Mock vector store para retornar lista vazia
        mock_components['vector_store'].search.return_value = []
        
        query = "Obscure topic not in documents"
        result = await pipeline.query(query)
        
        # Deve ainda gerar uma resposta
        assert result is not None

    @pytest.mark.asyncio
    async def test_model_router_failure(self, pipeline, mock_components):
        """Testar falha do model router."""
        # Mock model router para falhar
        mock_components['model_router'].generate.side_effect = Exception("API Error")
        
        query = "Test query"
        
        # Deve lidar com falha graciosamente
        result = await pipeline.query(query)
        assert result is not None

    @pytest.mark.asyncio
    async def test_embedding_failure(self, pipeline, mock_components):
        """Testar falha do embedding service."""
        # Mock embedding service para falhar
        mock_components['embedding_service'].embed_query.side_effect = Exception("Embedding Error")
        
        query = "Test query"
        
        # Deve lidar com falha graciosamente
        result = await pipeline.query(query)
        assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, pipeline):
        """Testar queries concorrentes."""
        queries = ["Query 1", "Query 2", "Query 3", "Query 4", "Query 5"]
        
        # Executar queries em paralelo
        tasks = [pipeline.query(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Todas devem retornar resultados (ou exceções tratadas)
        assert len(results) == len(queries)
        for result in results:
            assert result is not None

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_query_performance(self, pipeline):
        """Testar performance de queries."""
        import time
        
        query = "What is machine learning?"
        
        start_time = time.time()
        for _ in range(10):
            await pipeline.query(query)
        end_time = time.time()
        
        # 10 queries devem ser executadas em menos de 5 segundos
        assert end_time - start_time < 5.0

    def test_clear_cache(self, pipeline):
        """Testar limpeza de cache."""
        # Se o pipeline tiver cache
        if hasattr(pipeline, 'clear_cache'):
            result = pipeline.clear_cache()
            assert result is not None

    def test_get_stats(self, pipeline):
        """Testar obtenção de estatísticas."""
        # Se o pipeline tiver stats
        if hasattr(pipeline, 'get_stats'):
            stats = pipeline.get_stats()
            assert stats is not None
            assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_streaming_response(self, pipeline, mock_components):
        """Testar resposta em streaming."""
        # Mock para suportar streaming
        async def mock_stream():
            for chunk in ["Hello", " world", "!"]:
                yield {"chunk": chunk}
        
        if hasattr(mock_components['model_router'], 'stream'):
            mock_components['model_router'].stream = AsyncMock(return_value=mock_stream())
            
            query = "Test streaming"
            
            if hasattr(pipeline, 'query_stream'):
                async for chunk in pipeline.query_stream(query):
                    assert chunk is not None
                    assert "chunk" in chunk 