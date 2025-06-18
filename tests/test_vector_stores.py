"""Testes para os módulos de vector stores."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List, Dict, Any

# Mock das dependências antes do import
with patch.multiple(
    'sys.modules',
    qdrant_client=Mock(),
    sqlite3=Mock()
):
    from src.storage.qdrant_store import QdrantVectorStore
    from src.storage.sqlite_metadata_store import SQLiteMetadataStore


class TestQdrantVectorStore:
    """Testes para QdrantVectorStore."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock do cliente Qdrant."""
        client = Mock()
        client.search.return_value = [
            Mock(
                payload={"content": "Test result 1", "metadata": {"source": "doc1.txt"}},
                score=0.95
            ),
            Mock(
                payload={"content": "Test result 2", "metadata": {"source": "doc2.txt"}},
                score=0.87
            )
        ]
        client.upsert.return_value = Mock(status="completed")
        client.delete.return_value = Mock(status="completed")
        return client
    
    def test_init_default_parameters(self, mock_qdrant_client):
        """Testa inicialização com parâmetros padrão."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            assert store.client == mock_qdrant_client
            assert store.collection_name == "documents"
            assert store.vector_size == 1536
    
    def test_init_custom_parameters(self, mock_qdrant_client):
        """Testa inicialização com parâmetros customizados."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore(
                host="custom-host",
                port=6334,
                collection_name="custom_collection",
                vector_size=768
            )
            
            assert store.collection_name == "custom_collection"
            assert store.vector_size == 768
    
    def test_search_success(self, mock_qdrant_client):
        """Testa busca bem-sucedida."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            query_vector = [0.1, 0.2, 0.3]
            results = store.search(query_vector, k=5)
            
            # Verificar chamada ao cliente
            mock_qdrant_client.search.assert_called_once_with(
                collection_name="documents",
                query_vector=query_vector,
                limit=5
            )
            
            # Verificar formato dos resultados
            assert len(results) == 2
            assert results[0]["content"] == "Test result 1"
            assert results[0]["score"] == 0.95
            assert results[1]["content"] == "Test result 2"
            assert results[1]["score"] == 0.87
    
    def test_search_with_filters(self, mock_qdrant_client):
        """Testa busca com filtros."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            query_vector = [0.1, 0.2, 0.3]
            filters = {"source": "specific_doc.txt"}
            
            results = store.search(query_vector, k=3, filters=filters)
            
            # Verificar que filtros foram incluídos na chamada
            call_args = mock_qdrant_client.search.call_args
            assert "query_filter" in call_args.kwargs or len(call_args.args) > 3
    
    def test_search_empty_vector(self, mock_qdrant_client):
        """Testa busca com vetor vazio."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            results = store.search([], k=5)
            
            assert results == []
            mock_qdrant_client.search.assert_not_called()
    
    def test_search_client_error(self, mock_qdrant_client):
        """Testa erro do cliente Qdrant."""
        mock_qdrant_client.search.side_effect = Exception("Qdrant error")
        
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            with pytest.raises(Exception, match="Erro na busca vetorial: Qdrant error"):
                store.search([0.1, 0.2, 0.3], k=5)
    
    def test_add_documents_success(self, mock_qdrant_client):
        """Testa adição de documentos bem-sucedida."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            documents = [
                {
                    "id": "doc1",
                    "content": "Document 1 content",
                    "vector": [0.1, 0.2, 0.3],
                    "metadata": {"source": "doc1.txt"}
                },
                {
                    "id": "doc2",
                    "content": "Document 2 content",
                    "vector": [0.4, 0.5, 0.6],
                    "metadata": {"source": "doc2.txt"}
                }
            ]
            
            store.add_documents(documents)
            
            # Verificar chamada ao cliente
            mock_qdrant_client.upsert.assert_called_once()
            call_args = mock_qdrant_client.upsert.call_args
            assert call_args.kwargs["collection_name"] == "documents"
    
    def test_add_documents_empty_list(self, mock_qdrant_client):
        """Testa adição de lista vazia de documentos."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            store.add_documents([])
            
            mock_qdrant_client.upsert.assert_not_called()
    
    def test_add_documents_missing_fields(self, mock_qdrant_client):
        """Testa adição de documentos com campos obrigatórios faltando."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            invalid_documents = [
                {"id": "doc1", "content": "Content without vector"},
                {"content": "Content without id", "vector": [0.1, 0.2, 0.3]}
            ]
            
            with pytest.raises(ValueError, match="Documento deve conter"):
                store.add_documents(invalid_documents)
    
    def test_delete_documents_success(self, mock_qdrant_client):
        """Testa exclusão de documentos bem-sucedida."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            document_ids = ["doc1", "doc2", "doc3"]
            store.delete_documents(document_ids)
            
            # Verificar chamada ao cliente
            mock_qdrant_client.delete.assert_called_once()
            call_args = mock_qdrant_client.delete.call_args
            assert call_args.kwargs["collection_name"] == "documents"
    
    def test_delete_documents_empty_list(self, mock_qdrant_client):
        """Testa exclusão de lista vazia de IDs."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            store.delete_documents([])
            
            mock_qdrant_client.delete.assert_not_called()
    
    def test_create_collection_success(self, mock_qdrant_client):
        """Testa criação de coleção bem-sucedida."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            store.create_collection()
            
            # Verificar chamada de criação
            mock_qdrant_client.create_collection.assert_called_once()
    
    def test_collection_exists_check(self, mock_qdrant_client):
        """Testa verificação de existência de coleção."""
        mock_qdrant_client.collection_exists.return_value = True
        
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            exists = store.collection_exists()
            
            assert exists is True
            mock_qdrant_client.collection_exists.assert_called_once_with("documents")


class TestSQLiteMetadataStore:
    """Testes para SQLiteMetadataStore."""
    
    @pytest.fixture
    def mock_sqlite_connection(self):
        """Mock da conexão SQLite."""
        conn = Mock()
        cursor = Mock()
        conn.cursor.return_value = cursor
        cursor.fetchall.return_value = [
            ("doc1", "Document 1", '{"source": "doc1.txt"}', "2023-01-01 10:00:00"),
            ("doc2", "Document 2", '{"source": "doc2.txt"}', "2023-01-01 11:00:00")
        ]
        cursor.fetchone.return_value = ("doc1", "Document 1", '{"source": "doc1.txt"}', "2023-01-01 10:00:00")
        return conn
    
    def test_init_default_parameters(self, mock_sqlite_connection):
        """Testa inicialização com parâmetros padrão."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            assert store.db_path == "metadata.db"
            assert store.connection == mock_sqlite_connection
    
    def test_init_custom_parameters(self, mock_sqlite_connection):
        """Testa inicialização com parâmetros customizados."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore(db_path="custom.db")
            
            assert store.db_path == "custom.db"
    
    def test_create_tables(self, mock_sqlite_connection):
        """Testa criação de tabelas."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            store.create_tables()
            
            # Verificar que comandos SQL foram executados
            cursor = mock_sqlite_connection.cursor.return_value
            assert cursor.execute.called
            mock_sqlite_connection.commit.assert_called()
    
    def test_add_document_success(self, mock_sqlite_connection):
        """Testa adição de documento bem-sucedida."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            document = {
                "id": "doc1",
                "content": "Document content",
                "metadata": {"source": "doc1.txt", "type": "text"}
            }
            
            store.add_document(document)
            
            # Verificar execução SQL
            cursor = mock_sqlite_connection.cursor.return_value
            cursor.execute.assert_called()
            mock_sqlite_connection.commit.assert_called()
    
    def test_add_document_missing_id(self, mock_sqlite_connection):
        """Testa erro ao adicionar documento sem ID."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            document = {"content": "Document without ID"}
            
            with pytest.raises(ValueError, match="Documento deve conter um ID"):
                store.add_document(document)
    
    def test_get_document_success(self, mock_sqlite_connection):
        """Testa recuperação de documento bem-sucedida."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            document = store.get_document("doc1")
            
            assert document["id"] == "doc1"
            assert document["content"] == "Document 1"
            assert document["metadata"]["source"] == "doc1.txt"
    
    def test_get_document_not_found(self, mock_sqlite_connection):
        """Testa recuperação de documento não encontrado."""
        cursor = mock_sqlite_connection.cursor.return_value
        cursor.fetchone.return_value = None
        
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            document = store.get_document("nonexistent")
            
            assert document is None
    
    def test_search_documents_success(self, mock_sqlite_connection):
        """Testa busca de documentos bem-sucedida."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            results = store.search_documents("Document", limit=10)
            
            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[1]["id"] == "doc2"
    
    def test_search_documents_with_filters(self, mock_sqlite_connection):
        """Testa busca com filtros de metadata."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            filters = {"source": "doc1.txt"}
            results = store.search_documents("Document", filters=filters, limit=5)
            
            # Verificar que filtros foram aplicados na query
            cursor = mock_sqlite_connection.cursor.return_value
            cursor.execute.assert_called()
    
    def test_delete_document_success(self, mock_sqlite_connection):
        """Testa exclusão de documento bem-sucedida."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            store.delete_document("doc1")
            
            # Verificar execução SQL
            cursor = mock_sqlite_connection.cursor.return_value
            cursor.execute.assert_called()
            mock_sqlite_connection.commit.assert_called()
    
    def test_update_document_success(self, mock_sqlite_connection):
        """Testa atualização de documento bem-sucedida."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            updates = {
                "content": "Updated content",
                "metadata": {"source": "updated.txt"}
            }
            
            store.update_document("doc1", updates)
            
            # Verificar execução SQL
            cursor = mock_sqlite_connection.cursor.return_value
            cursor.execute.assert_called()
            mock_sqlite_connection.commit.assert_called()
    
    def test_get_all_documents(self, mock_sqlite_connection):
        """Testa recuperação de todos os documentos."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            documents = store.get_all_documents()
            
            assert len(documents) == 2
            assert all("id" in doc for doc in documents)
    
    def test_count_documents(self, mock_sqlite_connection):
        """Testa contagem de documentos."""
        cursor = mock_sqlite_connection.cursor.return_value
        cursor.fetchone.return_value = (42,)
        
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            count = store.count_documents()
            
            assert count == 42
    
    def test_database_error_handling(self, mock_sqlite_connection):
        """Testa tratamento de erros de banco de dados."""
        cursor = mock_sqlite_connection.cursor.return_value
        cursor.execute.side_effect = Exception("Database error")
        
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            with pytest.raises(Exception, match="Erro no banco de dados: Database error"):
                store.add_document({"id": "test", "content": "test"})
    
    def test_close_connection(self, mock_sqlite_connection):
        """Testa fechamento de conexão."""
        with patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            store = SQLiteMetadataStore()
            
            store.close()
            
            mock_sqlite_connection.close.assert_called_once()


class TestVectorStoresIntegration:
    """Testes de integração para vector stores."""
    
    def test_qdrant_sqlite_integration(self, mock_qdrant_client, mock_sqlite_connection):
        """Testa integração entre Qdrant e SQLite stores."""
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client), \
             patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            
            vector_store = QdrantVectorStore()
            metadata_store = SQLiteMetadataStore()
            
            # Documento de teste
            document = {
                "id": "test_doc",
                "content": "Test document content",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"source": "test.txt", "type": "text"}
            }
            
            # Adicionar aos dois stores
            vector_store.add_documents([document])
            metadata_store.add_document(document)
            
            # Verificar que ambos foram chamados
            mock_qdrant_client.upsert.assert_called_once()
            cursor = mock_sqlite_connection.cursor.return_value
            cursor.execute.assert_called()
    
    def test_performance_with_large_dataset(self, mock_qdrant_client, mock_sqlite_connection):
        """Testa performance com dataset grande."""
        import time
        
        # Configurar muitos resultados
        large_results = [
            Mock(
                payload={"content": f"Document {i}", "metadata": {"id": str(i)}},
                score=0.9 - (i * 0.001)
            )
            for i in range(1000)
        ]
        mock_qdrant_client.search.return_value = large_results
        
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client):
            store = QdrantVectorStore()
            
            start_time = time.time()
            results = store.search([0.1, 0.2, 0.3], k=100)
            end_time = time.time()
            
            # Deve completar rapidamente
            assert end_time - start_time < 1.0
            assert len(results) == 1000
    
    def test_concurrent_operations(self, mock_qdrant_client, mock_sqlite_connection):
        """Testa operações concorrentes."""
        import threading
        
        with patch('src.storage.qdrant_store.QdrantClient', return_value=mock_qdrant_client), \
             patch('src.storage.sqlite_metadata_store.sqlite3.connect', return_value=mock_sqlite_connection):
            
            vector_store = QdrantVectorStore()
            metadata_store = SQLiteMetadataStore()
            
            def search_worker(worker_id):
                vector_store.search([0.1, 0.2, 0.3], k=5)
                metadata_store.search_documents(f"query {worker_id}")
            
            # Criar múltiplas threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=search_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Aguardar conclusão
            for thread in threads:
                thread.join()
            
            # Verificar que operações foram executadas
            assert mock_qdrant_client.search.call_count == 5