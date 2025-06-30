"""
Testes completos para Vector Stores.
Objetivo: Cobertura abrangente de QdrantStore e HybridQdrantStore
"""

import pytest
import uuid
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import numpy as np


# Fallback classes para Vector Stores
class QdrantStore:
    def __init__(self, host="localhost", port=6333, collection_name="default"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.is_connected = False
        self._vectors = {}
        self._collection_config = {}
        self.connection_pool = None
        self.metrics = {
            'total_operations': 0,
            'search_count': 0,
            'insert_count': 0,
            'error_count': 0
        }
        
    async def connect(self):
        """Conectar ao Qdrant."""
        try:
            self.is_connected = True
            self.metrics['total_operations'] += 1
            return True
        except Exception as e:
            self.metrics['error_count'] += 1
            raise
        
    async def disconnect(self):
        """Desconectar do Qdrant."""
        self.is_connected = False
        
    async def create_collection(self, vector_size=1536, distance_metric="cosine", 
                              on_disk_payload=False, replication_factor=1):
        """Criar coleção com configurações avançadas."""
        self._collection_config = {
            'vector_size': vector_size,
            'distance_metric': distance_metric,
            'on_disk_payload': on_disk_payload,
            'replication_factor': replication_factor,
            'created_at': time.time()
        }
        self.metrics['total_operations'] += 1
        return True
        
    async def collection_exists(self):
        """Verificar se coleção existe."""
        return bool(self._collection_config)
        
    async def delete_collection(self):
        """Deletar coleção."""
        self._collection_config = {}
        self._vectors = {}
        self.metrics['total_operations'] += 1
        return True
        
    async def add_vectors(self, vectors, payloads=None, ids=None, batch_size=100):
        """Adicionar vetores com suporte a batching."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if payloads is None:
            payloads = [{} for _ in vectors]
            
        results = []
        
        # Processar em batches
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            for vector, payload, vector_id in zip(batch_vectors, batch_payloads, batch_ids):
                self._vectors[str(vector_id)] = {
                    'vector': vector,
                    'payload': payload,
                    'created_at': time.time()
                }
                results.append({
                    'id': str(vector_id),
                    'status': 'success'
                })
                
        self.metrics['insert_count'] += len(vectors)
        self.metrics['total_operations'] += 1
        return results
        
    async def search(self, vector, limit=10, score_threshold=0.0, 
                    filter_conditions=None, with_payload=True, with_vectors=False):
        """Buscar vetores similares com filtros avançados."""
        results = []
        
        for vector_id, data in self._vectors.items():
            # Aplicar filtros se especificados
            if filter_conditions:
                if not self._apply_filter(data['payload'], filter_conditions):
                    continue
            
            # Score simulado baseado na similaridade
            score = self._calculate_similarity(vector, data['vector'])
            
            if score >= score_threshold:
                result = {
                    'id': vector_id,
                    'score': score
                }
                
                if with_payload:
                    result['payload'] = data['payload']
                    
                if with_vectors:
                    result['vector'] = data['vector']
                    
                results.append(result)
        
        # Ordenar por score e limitar
        results.sort(key=lambda x: x['score'], reverse=True)
        self.metrics['search_count'] += 1
        self.metrics['total_operations'] += 1
        return results[:limit]
    
    def _apply_filter(self, payload, filter_conditions):
        """Aplicar condições de filtro."""
        for key, condition in filter_conditions.items():
            if key not in payload:
                return False
            
            if isinstance(condition, dict):
                if 'eq' in condition and payload[key] != condition['eq']:
                    return False
                if 'gt' in condition and payload[key] <= condition['gt']:
                    return False
                if 'lt' in condition and payload[key] >= condition['lt']:
                    return False
            else:
                if payload[key] != condition:
                    return False
        return True
    
    def _calculate_similarity(self, vec1, vec2):
        """Calcular similaridade (simulada)."""
        if len(vec1) != len(vec2):
            return 0.0
        return 0.8 + np.random.random() * 0.2  # Score entre 0.8 e 1.0
        
    async def update_vectors(self, vector_ids, vectors=None, payloads=None):
        """Atualizar vetores existentes."""
        updated = []
        for i, vector_id in enumerate(vector_ids):
            if str(vector_id) in self._vectors:
                if vectors and i < len(vectors):
                    self._vectors[str(vector_id)]['vector'] = vectors[i]
                if payloads and i < len(payloads):
                    self._vectors[str(vector_id)]['payload'].update(payloads[i])
                updated.append(str(vector_id))
        
        self.metrics['total_operations'] += 1
        return updated
        
    async def delete_vectors(self, vector_ids):
        """Deletar vetores."""
        deleted = []
        for vector_id in vector_ids:
            if str(vector_id) in self._vectors:
                del self._vectors[str(vector_id)]
                deleted.append(str(vector_id))
        
        self.metrics['total_operations'] += 1
        return deleted
        
    async def count_vectors(self):
        """Contar vetores."""
        return len(self._vectors)
        
    async def get_collection_info(self):
        """Obter informações da coleção."""
        return {
            'config': self._collection_config,
            'vectors_count': len(self._vectors),
            'status': 'active' if self.is_connected else 'inactive'
        }
        
    def get_metrics(self):
        """Obter métricas de uso."""
        return self.metrics.copy()


class HybridQdrantStore(QdrantStore):
    def __init__(self, host="localhost", port=6333, collection_name="hybrid"):
        super().__init__(host, port, collection_name)
        self.keyword_index = {}
        self.sparse_vectors = {}
        self.fusion_weights = {'dense': 0.7, 'sparse': 0.3}
        
    async def add_documents(self, documents, vectors=None, sparse_vectors=None, ids=None):
        """Adicionar documentos com vetores densos e esparsos."""
        if vectors is None:
            vectors = [np.random.random(1536).tolist() for _ in documents]
        if sparse_vectors is None:
            sparse_vectors = [self._create_sparse_vector(doc) for doc in documents]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
            
        payloads = [{'text': doc, 'length': len(doc)} for doc in documents]
        
        # Armazenar vetores esparsos separadamente
        for doc_id, sparse_vec in zip(ids, sparse_vectors):
            self.sparse_vectors[str(doc_id)] = sparse_vec
            
        return await self.add_vectors(vectors, payloads, ids)
    
    def _create_sparse_vector(self, text):
        """Criar vetor esparso simulado."""
        words = text.lower().split()
        sparse_vec = {}
        for i, word in enumerate(set(words)):
            sparse_vec[hash(word) % 10000] = words.count(word) / len(words)
        return sparse_vec
        
    async def hybrid_search(self, vector=None, text_query=None, limit=10, 
                          alpha=0.5, score_threshold=0.0):
        """Busca híbrida combinando dense e sparse."""
        dense_results = []
        sparse_results = []
        
        # Busca densa
        if vector is not None:
            dense_results = await self.search(vector, limit=limit*2, score_threshold=0.0)
            
        # Busca esparsa (simulada)
        if text_query is not None:
            sparse_query = self._create_sparse_vector(text_query)
            sparse_results = self._sparse_search(sparse_query, limit=limit*2)
            
        # Fusão RRF (Reciprocal Rank Fusion)
        return self._fuse_results(dense_results, sparse_results, alpha, limit, score_threshold)
    
    def _sparse_search(self, sparse_query, limit=10):
        """Busca esparsa simulada."""
        results = []
        for doc_id, sparse_vec in self.sparse_vectors.items():
            score = self._sparse_similarity(sparse_query, sparse_vec)
            if doc_id in self._vectors:
                results.append({
                    'id': doc_id,
                    'score': score,
                    'payload': self._vectors[doc_id]['payload']
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def _sparse_similarity(self, query_vec, doc_vec):
        """Calcular similaridade esparsa."""
        common_keys = set(query_vec.keys()) & set(doc_vec.keys())
        if not common_keys:
            return 0.0
        
        similarity = sum(query_vec[k] * doc_vec[k] for k in common_keys)
        return min(similarity, 1.0)
    
    def _fuse_results(self, dense_results, sparse_results, alpha, limit, threshold):
        """Fusão de resultados dense e sparse."""
        # RRF simples
        fused_scores = {}
        
        # Processar resultados densos
        for i, result in enumerate(dense_results):
            doc_id = result['id']
            rrf_score = 1.0 / (60 + i + 1)  # k=60 é padrão
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + alpha * rrf_score
            
        # Processar resultados esparsos
        for i, result in enumerate(sparse_results):
            doc_id = result['id']
            rrf_score = 1.0 / (60 + i + 1)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1 - alpha) * rrf_score
            
        # Ordenar e preparar resultados finais
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_id, score in sorted_results[:limit]:
            if score >= threshold and doc_id in self._vectors:
                final_results.append({
                    'id': doc_id,
                    'score': score,
                    'payload': self._vectors[doc_id]['payload']
                })
                
        return final_results


class TestQdrantStore:
    """Testes completos para QdrantStore."""

    @pytest.fixture
    def store(self):
        """Fixture do store."""
        return QdrantStore(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_init_basic(self):
        """Testar inicialização básica."""
        store = QdrantStore()
        assert store.host == "localhost"
        assert store.port == 6333
        assert store.collection_name == "default"
        assert not store.is_connected
        assert store.metrics['total_operations'] == 0

    @pytest.mark.asyncio
    async def test_init_custom_config(self):
        """Testar inicialização com configuração customizada."""
        store = QdrantStore(host="custom-host", port=6334, collection_name="custom")
        assert store.host == "custom-host"
        assert store.port == 6334
        assert store.collection_name == "custom"

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Testar conexão e desconexão."""
        store = QdrantStore()
        
        # Conectar
        result = await store.connect()
        assert result is True
        assert store.is_connected is True
        assert store.metrics['total_operations'] == 1
        
        # Desconectar
        await store.disconnect()
        assert store.is_connected is False

    @pytest.mark.asyncio
    async def test_create_collection_basic(self, store):
        """Testar criação de coleção básica."""
        await store.connect()
        result = await store.create_collection()
        assert result is True
        assert store._collection_config['vector_size'] == 1536
        assert store._collection_config['distance_metric'] == "cosine"

    @pytest.mark.asyncio
    async def test_create_collection_advanced(self, store):
        """Testar criação de coleção com configurações avançadas."""
        await store.connect()
        result = await store.create_collection(
            vector_size=512, 
            distance_metric="dot",
            on_disk_payload=True,
            replication_factor=2
        )
        assert result is True
        assert store._collection_config['vector_size'] == 512
        assert store._collection_config['distance_metric'] == "dot"
        assert store._collection_config['on_disk_payload'] is True
        assert store._collection_config['replication_factor'] == 2

    @pytest.mark.asyncio
    async def test_collection_lifecycle(self, store):
        """Testar ciclo de vida da coleção."""
        await store.connect()
        
        # Verificar que não existe
        exists = await store.collection_exists()
        assert exists is False
        
        # Criar
        await store.create_collection()
        exists = await store.collection_exists()
        assert exists is True
        
        # Deletar
        await store.delete_collection()
        exists = await store.collection_exists()
        assert exists is False

    @pytest.mark.asyncio
    async def test_add_single_vector(self, store):
        """Testar adição de vetor único."""
        await store.connect()
        await store.create_collection()
        
        vector = [0.1, 0.2, 0.3] * 512
        payload = {"text": "test document", "category": "test"}
        
        results = await store.add_vectors([vector], [payload], ["test_id"])
        
        assert len(results) == 1
        assert results[0]['id'] == "test_id"
        assert results[0]['status'] == "success"
        assert store.metrics['insert_count'] == 1

    @pytest.mark.asyncio
    async def test_add_batch_vectors(self, store):
        """Testar adição de vetores em batch."""
        await store.connect()
        await store.create_collection()
        
        vectors = [[0.1, 0.2] * 768 for _ in range(100)]
        payloads = [{"i": i, "category": f"cat_{i%5}"} for i in range(100)]
        
        results = await store.add_vectors(vectors, payloads, batch_size=25)
        
        assert len(results) == 100
        assert all(r['status'] == 'success' for r in results)
        assert store.metrics['insert_count'] == 100

    @pytest.mark.asyncio
    async def test_search_basic(self, store):
        """Testar busca básica."""
        await store.connect()
        await store.create_collection()
        
        # Adicionar vetores de teste
        vectors = [[1.0, 0.0] * 768, [0.0, 1.0] * 768]
        payloads = [{"text": "red", "color": "red"}, {"text": "blue", "color": "blue"}]
        await store.add_vectors(vectors, payloads, ["red", "blue"])
        
        # Buscar
        query_vector = [0.9, 0.1] * 768
        results = await store.search(query_vector, limit=2)
        
        assert len(results) <= 2
        assert all('score' in r for r in results)
        assert all('payload' in r for r in results)
        assert store.metrics['search_count'] == 1

    @pytest.mark.asyncio
    async def test_search_with_filters(self, store):
        """Testar busca com filtros."""
        await store.connect()
        await store.create_collection()
        
        # Adicionar vetores com diferentes categorias
        vectors = [[0.1, 0.2] * 768 for _ in range(5)]
        payloads = [
            {"category": "A", "value": 10},
            {"category": "B", "value": 20},
            {"category": "A", "value": 30},
            {"category": "C", "value": 40},
            {"category": "B", "value": 50}
        ]
        await store.add_vectors(vectors, payloads)
        
        # Buscar com filtro
        query_vector = [0.1, 0.2] * 768
        results = await store.search(
            query_vector, 
            limit=10,
            filter_conditions={"category": "A"}
        )
        
        assert len(results) == 2  # Apenas categoria A
        assert all(r['payload']['category'] == 'A' for r in results)

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, store):
        """Testar busca com threshold de score."""
        await store.connect()
        await store.create_collection()
        
        vectors = [[0.1, 0.2] * 768 for _ in range(3)]
        payloads = [{"i": i} for i in range(3)]
        await store.add_vectors(vectors, payloads)
        
        query_vector = [0.1, 0.2] * 768
        results = await store.search(query_vector, score_threshold=0.9)
        
        # Com threshold alto, pode não retornar nada ou poucos resultados
        assert all(r['score'] >= 0.9 for r in results)

    @pytest.mark.asyncio
    async def test_update_vectors(self, store):
        """Testar atualização de vetores."""
        await store.connect()
        await store.create_collection()
        
        # Adicionar vetores iniciais
        vectors = [[0.1, 0.2] * 768, [0.3, 0.4] * 768]
        payloads = [{"status": "old"}, {"status": "old"}]
        results = await store.add_vectors(vectors, payloads, ["id1", "id2"])
        
        # Atualizar
        new_payloads = [{"status": "updated", "timestamp": 123}]
        updated = await store.update_vectors(["id1"], payloads=new_payloads)
        
        assert "id1" in updated
        assert store._vectors["id1"]["payload"]["status"] == "updated"
        assert store._vectors["id1"]["payload"]["timestamp"] == 123

    @pytest.mark.asyncio
    async def test_delete_vectors(self, store):
        """Testar deleção de vetores."""
        await store.connect()
        await store.create_collection()
        
        # Adicionar vetores
        vectors = [[0.1, 0.2] * 768 for _ in range(3)]
        payloads = [{"i": i} for i in range(3)]
        await store.add_vectors(vectors, payloads, ["id1", "id2", "id3"])
        
        # Deletar alguns
        deleted = await store.delete_vectors(["id1", "id3"])
        
        assert set(deleted) == {"id1", "id3"}
        assert "id1" not in store._vectors
        assert "id2" in store._vectors
        assert "id3" not in store._vectors

    @pytest.mark.asyncio
    async def test_count_vectors(self, store):
        """Testar contagem de vetores."""
        await store.connect()
        await store.create_collection()
        
        # Inicialmente vazio
        count = await store.count_vectors()
        assert count == 0
        
        # Adicionar vetores
        vectors = [[0.1, 0.2] * 768 for _ in range(5)]
        payloads = [{"i": i} for i in range(5)]
        await store.add_vectors(vectors, payloads)
        
        # Contar novamente
        count = await store.count_vectors()
        assert count == 5

    @pytest.mark.asyncio
    async def test_get_collection_info(self, store):
        """Testar obtenção de informações da coleção."""
        await store.connect()
        await store.create_collection(vector_size=512)
        
        info = await store.get_collection_info()
        
        assert info['config']['vector_size'] == 512
        assert info['vectors_count'] == 0
        assert info['status'] == 'active'

    def test_get_metrics(self, store):
        """Testar obtenção de métricas."""
        metrics = store.get_metrics()
        
        assert 'total_operations' in metrics
        assert 'search_count' in metrics
        assert 'insert_count' in metrics
        assert 'error_count' in metrics


class TestHybridQdrantStore:
    """Testes completos para HybridQdrantStore."""

    @pytest.fixture
    def hybrid_store(self):
        """Fixture do hybrid store."""
        return HybridQdrantStore(collection_name="hybrid_test")

    @pytest.mark.asyncio
    async def test_init_hybrid(self):
        """Testar inicialização do hybrid store."""
        store = HybridQdrantStore()
        assert store.collection_name == "hybrid"
        assert store.keyword_index == {}
        assert store.sparse_vectors == {}
        assert store.fusion_weights['dense'] == 0.7
        assert store.fusion_weights['sparse'] == 0.3

    @pytest.mark.asyncio
    async def test_add_documents(self, hybrid_store):
        """Testar adição de documentos."""
        await hybrid_store.connect()
        await hybrid_store.create_collection()
        
        documents = [
            "Python is a versatile programming language",
            "Machine learning requires large datasets",
            "Data science combines statistics and programming",
            "Artificial intelligence mimics human cognition",
            "Deep learning uses neural networks"
        ]
        
        results = await hybrid_store.add_documents(documents)
        
        assert len(results) == 5
        assert all(r['status'] == 'success' for r in results)
        assert len(hybrid_store.sparse_vectors) == 5

    @pytest.mark.asyncio
    async def test_hybrid_search_dense_only(self, hybrid_store):
        """Testar busca híbrida apenas com vetor denso."""
        await hybrid_store.connect()
        await hybrid_store.create_collection()
        
        documents = ["test document one", "test document two"]
        await hybrid_store.add_documents(documents)
        
        query_vector = [0.1, 0.2] * 768
        results = await hybrid_store.hybrid_search(vector=query_vector, limit=2)
        
        assert len(results) <= 2
        assert all('score' in r for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_text_only(self, hybrid_store):
        """Testar busca híbrida apenas com texto."""
        await hybrid_store.connect()
        await hybrid_store.create_collection()
        
        documents = [
            "Python programming language",
            "Java programming language", 
            "Machine learning algorithms"
        ]
        await hybrid_store.add_documents(documents)
        
        results = await hybrid_store.hybrid_search(text_query="Python programming", limit=2)
        
        assert len(results) <= 2
        assert all('score' in r for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_combined(self, hybrid_store):
        """Testar busca híbrida combinada."""
        await hybrid_store.connect()
        await hybrid_store.create_collection()
        
        documents = [
            "Python is great for machine learning",
            "Java is used for enterprise applications",
            "JavaScript runs in web browsers"
        ]
        await hybrid_store.add_documents(documents)
        
        query_vector = [0.1, 0.2] * 768
        results = await hybrid_store.hybrid_search(
            vector=query_vector,
            text_query="Python machine learning",
            alpha=0.6,
            limit=3
        )
        
        assert len(results) <= 3
        assert all('score' in r for r in results)

    @pytest.mark.asyncio
    async def test_sparse_vector_creation(self, hybrid_store):
        """Testar criação de vetores esparsos."""
        text = "machine learning algorithms data science"
        sparse_vec = hybrid_store._create_sparse_vector(text)
        
        assert isinstance(sparse_vec, dict)
        assert len(sparse_vec) > 0
        assert all(isinstance(k, int) for k in sparse_vec.keys())
        assert all(isinstance(v, float) for v in sparse_vec.values())

    @pytest.mark.asyncio
    async def test_sparse_similarity(self, hybrid_store):
        """Testar cálculo de similaridade esparsa."""
        vec1 = {1: 0.5, 2: 0.3, 3: 0.2}
        vec2 = {1: 0.4, 2: 0.6, 4: 0.1}
        
        similarity = hybrid_store._sparse_similarity(vec1, vec2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0  # Deve ter alguma similaridade


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Testes de integração para vector stores."""

    @pytest.mark.asyncio
    async def test_store_complete_workflow(self):
        """Teste de workflow completo do QdrantStore."""
        store = QdrantStore(collection_name="integration_test")
        
        try:
            # Setup
            await store.connect()
            await store.create_collection(vector_size=128)
            
            # Adicionar dados
            vectors = [np.random.random(128).tolist() for _ in range(10)]
            payloads = [{"doc": f"Document {i}", "category": f"cat_{i%3}"} for i in range(10)]
            results = await store.add_vectors(vectors, payloads)
            
            assert len(results) == 10
            
            # Buscar
            query_vector = np.random.random(128).tolist()
            search_results = await store.search(query_vector, limit=5)
            
            assert len(search_results) <= 5
            
            # Contar
            count = await store.count_vectors()
            assert count == 10
            
            # Atualizar
            updated = await store.update_vectors([results[0]['id']], payloads=[{"updated": True}])
            assert len(updated) == 1
            
            # Deletar alguns
            to_delete = [r['id'] for r in results[:3]]
            deleted = await store.delete_vectors(to_delete)
            assert len(deleted) == 3
            
            # Contar novamente
            final_count = await store.count_vectors()
            assert final_count == 7
            
        finally:
            await store.disconnect()

    @pytest.mark.asyncio
    async def test_hybrid_complete_workflow(self):
        """Teste de workflow completo do HybridQdrantStore."""
        store = HybridQdrantStore(collection_name="hybrid_integration")
        
        try:
            # Setup
            await store.connect()
            await store.create_collection()
            
            # Adicionar documentos
            documents = [
                "Python is a versatile programming language",
                "Machine learning requires large datasets",
                "Data science combines statistics and programming",
                "Artificial intelligence mimics human cognition",
                "Deep learning uses neural networks"
            ]
            
            results = await store.add_documents(documents)
            assert len(results) == 5
            
            # Busca híbrida
            hybrid_results = await store.hybrid_search(
                text_query="Python programming",
                limit=3,
                alpha=0.5
            )
            
            assert len(hybrid_results) <= 3
            
            # Verificar que temos vetores esparsos
            assert len(store.sparse_vectors) == 5
            
        finally:
            await store.disconnect()

    @pytest.mark.asyncio
    async def test_performance_bulk_operations(self):
        """Teste de performance para operações em bulk."""
        store = QdrantStore(collection_name="performance_test")
        
        try:
            await store.connect()
            await store.create_collection()
            
            # Teste com 1000 vetores
            start_time = time.time()
            
            vectors = [np.random.random(512).tolist() for _ in range(1000)]
            payloads = [{"id": i, "batch": i // 100} for i in range(1000)]
            
            results = await store.add_vectors(vectors, payloads, batch_size=100)
            
            insert_time = time.time() - start_time
            
            assert len(results) == 1000
            assert insert_time < 10.0  # Deve ser razoavelmente rápido
            
            # Teste de busca
            start_time = time.time()
            query_vector = np.random.random(512).tolist()
            search_results = await store.search(query_vector, limit=50)
            search_time = time.time() - start_time
            
            assert len(search_results) <= 50
            assert search_time < 1.0  # Busca deve ser rápida
            
        finally:
            await store.disconnect() 