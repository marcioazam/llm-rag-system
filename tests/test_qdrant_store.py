"""Comprehensive tests for Qdrant vector store functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.vectordb.qdrant_store import QdrantVectorStore


class TestQdrantVectorStore:
    """Test suite for Qdrant vector store."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock QdrantClient for testing."""
        with patch('src.vectordb.qdrant_store.QdrantClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock collections response
            mock_collection = Mock()
            mock_collection.name = "test_collection"
            mock_collections_response = Mock()
            mock_collections_response.collections = [mock_collection]
            mock_client.get_collections.return_value = mock_collections_response
            
            yield mock_client

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "This is the first document about machine learning.",
            "The second document discusses natural language processing.",
            "Third document covers computer vision topics."
        ]

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7]
        ]

    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return [
            {"source": "doc1.txt", "category": "ml"},
            {"source": "doc2.txt", "category": "nlp"},
            {"source": "doc3.txt", "category": "cv"}
        ]

    def test_init_successful_connection(self, mock_qdrant_client):
        """Test successful initialization with Qdrant connection."""
        # Collection doesn't exist, should be created
        mock_qdrant_client.get_collections.return_value.collections = []
        
        store = QdrantVectorStore(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            distance="Cosine",
            dim=768
        )
        
        assert store.collection_name == "test_collection"
        assert store.dim == 768
        assert not store._in_memory
        mock_qdrant_client.create_collection.assert_called_once()

    def test_init_existing_collection(self, mock_qdrant_client):
        """Test initialization with existing collection."""
        # Collection already exists
        mock_collection = Mock()
        mock_collection.name = "existing_collection"
        mock_qdrant_client.get_collections.return_value.collections = [mock_collection]
        
        store = QdrantVectorStore(collection_name="existing_collection")
        
        assert not store._in_memory
        mock_qdrant_client.create_collection.assert_not_called()

    def test_init_fallback_to_memory(self):
        """Test fallback to in-memory mode when Qdrant is unavailable."""
        with patch('src.vectordb.qdrant_store.QdrantClient') as mock_client_class:
            # Simula falha na criação do cliente
            mock_client_instance = Mock()
            mock_client_instance.get_collections.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client_instance
            
            store = QdrantVectorStore()
            
            assert store._in_memory
            assert store.client is None
            assert hasattr(store, '_mem_store')
            assert store._mem_store == []

    def test_add_documents_success(self, mock_qdrant_client, sample_documents, sample_embeddings, sample_metadata):
        """Test successful document addition."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        result = store.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            ids=["doc1", "doc2", "doc3"]
        )
        
        assert result is True
        mock_qdrant_client.upsert.assert_called_once()
        
        # Verify the points structure
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]['points']
        assert len(points) == 3
        assert points[0].id == "doc1"
        assert points[0].payload["document"] == sample_documents[0]

    def test_add_documents_no_documents(self, mock_qdrant_client):
        """Test adding empty document list."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        result = store.add_documents(documents=[])
        
        assert result is False
        mock_qdrant_client.upsert.assert_not_called()

    def test_add_documents_no_embeddings(self, mock_qdrant_client, sample_documents):
        """Test adding documents without embeddings."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        result = store.add_documents(documents=sample_documents)
        
        assert result is False
        mock_qdrant_client.upsert.assert_not_called()

    def test_add_documents_auto_generate_ids(self, mock_qdrant_client, sample_documents, sample_embeddings):
        """Test automatic ID generation when not provided."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        result = store.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]['points']
        assert points[0].id == "doc_0"
        assert points[1].id == "doc_1"
        assert points[2].id == "doc_2"

    def test_add_documents_auto_generate_metadata(self, mock_qdrant_client, sample_documents, sample_embeddings):
        """Test automatic metadata generation when not provided."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        result = store.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]['points']
        assert points[0].payload["source"] == "unknown"

    def test_add_documents_memory_mode(self, sample_documents, sample_embeddings, sample_metadata):
        """Test adding documents in memory mode."""
        with patch('src.vectordb.qdrant_store.QdrantClient') as mock_client_class:
            # Simula falha na criação do cliente
            mock_client_instance = Mock()
            mock_client_instance.get_collections.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client_instance
            
            store = QdrantVectorStore()
            
            result = store.add_documents(
                documents=sample_documents,
                embeddings=sample_embeddings,
                metadata=sample_metadata,
                ids=["doc1", "doc2", "doc3"]
            )
            
            assert result is True
            assert len(store._mem_store) == 3
            assert store._mem_store[0]["id"] == "doc1"
            assert store._mem_store[0]["content"] == sample_documents[0]

    def test_add_documents_exception_handling(self, mock_qdrant_client, sample_documents, sample_embeddings):
        """Test exception handling during document addition."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        mock_qdrant_client.upsert.side_effect = Exception("Qdrant error")
        
        result = store.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        assert result is False

    def test_search_with_query_embedding(self, mock_qdrant_client):
        """Test search with pre-computed query embedding."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        # Mock search results
        mock_hit = Mock()
        mock_hit.payload = {"document": "Test document", "source": "test.txt"}
        mock_hit.score = 0.95
        mock_hit.id = "doc1"
        mock_qdrant_client.search.return_value = [mock_hit]
        
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = store.search(
            query="test query",
            query_embedding=query_embedding,
            k=5
        )
        
        assert len(results) == 1
        assert results[0]["content"] == "Test document"
        assert results[0]["metadata"]["source"] == "test.txt"
        assert results[0]["distance"] == 0.95
        assert results[0]["id"] == "doc1"
        
        mock_qdrant_client.search.assert_called_once()

    def test_search_with_numpy_array(self, mock_qdrant_client):
        """Test search with numpy array as query."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        mock_qdrant_client.search.return_value = []
        
        query_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        results = store.search(query=query_vector, k=3)
        
        assert results == []
        mock_qdrant_client.search.assert_called_once()

    def test_search_string_without_embedding(self, mock_qdrant_client):
        """Test search with string query but no embedding provided."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        # O código captura a ValueError e retorna lista vazia
        results = store.search(query="test query", k=5)
        assert results == []

    def test_search_with_filter(self, mock_qdrant_client):
        """Test search with metadata filter."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        mock_qdrant_client.search.return_value = []
        
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        filter_dict = {"category": "ml", "source": "doc1.txt"}
        
        store.search(
            query="test",
            query_embedding=query_embedding,
            filter=filter_dict,
            k=5
        )
        
        # Verify filter was applied
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]['query_filter'] is not None

    def test_search_memory_mode(self):
        """Test search in memory mode returns empty list."""
        with patch('src.vectordb.qdrant_store.QdrantClient') as mock_client_class:
            # Simula falha na criação do cliente
            mock_client_instance = Mock()
            mock_client_instance.get_collections.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client_instance
            
            store = QdrantVectorStore()
            
            results = store.search(query=np.array([0.1, 0.2, 0.3]), k=5)
            
            assert results == []

    def test_search_exception_handling(self, mock_qdrant_client):
        """Test search exception handling."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        mock_qdrant_client.search.side_effect = Exception("Search error")
        
        results = store.search(
            query="test",
            query_embedding=[0.1, 0.2, 0.3],
            k=5
        )
        
        assert results == []

    def test_get_document_count(self, mock_qdrant_client):
        """Test getting document count."""
        store = QdrantVectorStore()
        
        mock_collection_info = Mock()
        mock_collection_info.points_count = 42
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        count = store.get_document_count()
        
        assert count == 42

    def test_get_document_count_exception(self, mock_qdrant_client):
        """Test document count with exception."""
        store = QdrantVectorStore()
        
        mock_qdrant_client.get_collection.side_effect = Exception("Error")
        
        count = store.get_document_count()
        
        assert count == 0

    def test_delete_documents(self, mock_qdrant_client):
        """Test deleting documents by IDs."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        result = store.delete_documents(["doc1", "doc2", "doc3"])
        
        assert result is True
        mock_qdrant_client.delete.assert_called_once()

    def test_delete_documents_empty_list(self, mock_qdrant_client):
        """Test deleting with empty ID list."""
        store = QdrantVectorStore()
        
        result = store.delete_documents([])
        
        assert result is True
        mock_qdrant_client.delete.assert_not_called()

    def test_delete_documents_memory_mode(self):
        """Test deleting documents in memory mode."""
        with patch('src.vectordb.qdrant_store.QdrantClient') as mock_client_class:
            # Simula falha na criação do cliente
            mock_client_instance = Mock()
            mock_client_instance.get_collections.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client_instance
            
            store = QdrantVectorStore()
            
            # Add some documents first
            store._mem_store = [
                {"id": "doc1", "content": "test1"},
                {"id": "doc2", "content": "test2"},
                {"id": "doc3", "content": "test3"}
            ]
            
            result = store.delete_documents(["doc1", "doc3"])
            
            assert result is True
            assert len(store._mem_store) == 1
            assert store._mem_store[0]["id"] == "doc2"

    def test_delete_documents_exception(self, mock_qdrant_client):
        """Test delete documents exception handling."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        mock_qdrant_client.delete.side_effect = Exception("Delete error")
        
        result = store.delete_documents(["doc1"])
        
        assert result is False

    def test_clear_collection(self, mock_qdrant_client):
        """Test clearing collection."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        result = store.clear_collection()
        
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once()
        mock_qdrant_client.create_collection.assert_called()

    def test_clear_collection_memory_mode(self):
        """Test clearing collection in memory mode."""
        with patch('src.vectordb.qdrant_store.QdrantClient') as mock_client_class:
            # Simula falha na criação do cliente
            mock_client_instance = Mock()
            mock_client_instance.get_collections.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client_instance
            
            store = QdrantVectorStore()
            store._mem_store = [{"id": "doc1", "content": "test"}]
            
            result = store.clear_collection()
            
            assert result is True
            assert store._mem_store == []

    def test_clear_collection_exception(self, mock_qdrant_client):
        """Test clear collection exception handling."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        mock_qdrant_client.delete_collection.side_effect = Exception("Clear error")
        
        result = store.clear_collection()
        
        assert result is False

    def test_get_collection_info(self, mock_qdrant_client):
        """Test getting collection information."""
        store = QdrantVectorStore(collection_name="test_collection")
        
        mock_info = Mock()
        mock_info.points_count = 100
        mock_qdrant_client.get_collection.return_value = mock_info
        mock_qdrant_client._config.host = "localhost"
        mock_qdrant_client._config.port = 6333
        
        info = store.get_collection_info()
        
        assert info["name"] == "test_collection"
        assert info["count"] == 100
        assert info["host"] == "localhost"
        assert info["port"] == 6333

    def test_get_collection_info_exception(self, mock_qdrant_client):
        """Test get collection info exception handling."""
        store = QdrantVectorStore()
        
        mock_qdrant_client.get_collection.side_effect = Exception("Info error")
        
        info = store.get_collection_info()
        
        assert info == {}

    def test_update_document(self, mock_qdrant_client):
        """Test updating a document."""
        store = QdrantVectorStore()
        
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {"source": "updated.txt", "category": "updated"}
        
        result = store.update_document("doc1", embedding, metadata)
        
        assert result is True
        mock_qdrant_client.upsert.assert_called_once()
        
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]['points']
        assert len(points) == 1
        assert points[0].id == "doc1"
        assert points[0].vector == embedding
        assert points[0].payload == metadata

    def test_update_document_no_metadata(self, mock_qdrant_client):
        """Test updating document without metadata."""
        store = QdrantVectorStore()
        
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        result = store.update_document("doc1", embedding)
        
        assert result is True
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]['points']
        assert points[0].payload == {}

    def test_update_document_exception(self, mock_qdrant_client):
        """Test update document exception handling."""
        store = QdrantVectorStore()
        
        mock_qdrant_client.upsert.side_effect = Exception("Update error")
        
        result = store.update_document("doc1", [0.1, 0.2, 0.3])
        
        assert result is False

    def test_get_document_by_id(self, mock_qdrant_client):
        """Test retrieving document by ID."""
        store = QdrantVectorStore()
        
        mock_record = Mock()
        mock_record.id = "doc1"
        mock_record.payload = {"source": "test.txt", "category": "ml"}
        mock_record.vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_qdrant_client.retrieve.return_value = [mock_record]
        
        result = store.get_document_by_id("doc1")
        
        assert result is not None
        assert result["id"] == "doc1"
        assert result["metadata"] == {"source": "test.txt", "category": "ml"}
        assert result["vector"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_get_document_by_id_not_found(self, mock_qdrant_client):
        """Test retrieving non-existent document."""
        store = QdrantVectorStore()
        
        mock_qdrant_client.retrieve.return_value = []
        
        result = store.get_document_by_id("nonexistent")
        
        assert result is None

    def test_get_document_by_id_exception(self, mock_qdrant_client):
        """Test get document by ID exception handling."""
        store = QdrantVectorStore()
        
        mock_qdrant_client.retrieve.side_effect = Exception("Retrieve error")
        
        result = store.get_document_by_id("doc1")
        
        assert result is None

    def test_close(self, mock_qdrant_client):
        """Test closing the store (no-op for Qdrant)."""
        store = QdrantVectorStore()
        
        # Should not raise any exception
        store.close()

    def test_distance_enum_mapping(self, mock_qdrant_client):
        """Test distance enum mapping."""
        with patch('src.vectordb.qdrant_store.rest') as mock_rest:
            mock_rest.Distance.COSINE = "COSINE"
            mock_rest.Distance.EUCLIDEAN = "EUCLIDEAN"
            mock_rest.VectorParams = Mock()
            
            # Test different distance metrics
            distances = ["Cosine", "Euclidean", "InvalidDistance"]
            
            for distance in distances:
                store = QdrantVectorStore(distance=distance)
                # Should not raise an error
                assert store is not None

    def test_complex_search_scenario(self, mock_qdrant_client):
        """Test complex search scenario with multiple results."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        # Mock multiple search results
        mock_hits = []
        for i in range(3):
            hit = Mock()
            hit.payload = {
                "document": f"Document {i}",
                "source": f"doc{i}.txt",
                "category": "test"
            }
            hit.score = 0.9 - (i * 0.1)
            hit.id = f"doc{i}"
            mock_hits.append(hit)
        
        mock_qdrant_client.search.return_value = mock_hits
        
        results = store.search(
            query="test",
            query_embedding=[0.1, 0.2, 0.3],
            k=3,
            filter={"category": "test"}
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["content"] == f"Document {i}"
            assert result["metadata"]["source"] == f"doc{i}.txt"
            assert result["distance"] == 0.9 - (i * 0.1)
            assert result["id"] == f"doc{i}"

    def test_payload_handling_edge_cases(self, mock_qdrant_client):
        """Test edge cases in payload handling."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        # Mock hit with minimal payload
        mock_hit = Mock()
        mock_hit.payload = None
        mock_hit.score = 0.8
        mock_hit.id = "minimal_doc"
        mock_qdrant_client.search.return_value = [mock_hit]
        
        results = store.search(
            query="test",
            query_embedding=[0.1, 0.2, 0.3],
            k=1
        )
        
        assert len(results) == 1
        assert results[0]["content"] == ""
        assert results[0]["metadata"] == {}
        assert results[0]["id"] == "minimal_doc"

    def test_large_batch_operations(self, mock_qdrant_client):
        """Test operations with large batches of documents."""
        store = QdrantVectorStore()
        store._in_memory = False
        
        # Create large batch
        num_docs = 1000
        documents = [f"Document {i}" for i in range(num_docs)]
        embeddings = [[0.1] * 5 for _ in range(num_docs)]
        metadata = [{"index": i} for i in range(num_docs)]
        ids = [f"doc_{i}" for i in range(num_docs)]
        
        result = store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadata=metadata,
            ids=ids
        )
        
        assert result is True
        mock_qdrant_client.upsert.assert_called_once()
        
        # Verify batch size
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]['points']
        assert len(points) == num_docs