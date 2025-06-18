"""
Hybrid Vector Store para Qdrant 1.8.0+
Combina dense vectors (semânticos) + sparse vectors (BM25) 
Performance otimizada: 16x improvement em sparse vector search
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SearchRequest, QueryRequest,
    SparseVector as QdrantSparseVector,
    NamedVector, NamedSparseVector
)
import yaml
import uuid
from datetime import datetime

from ..embeddings.sparse_vector_service import SparseVector, AdvancedSparseVectorService
from ..embeddings.api_embedding_service import APIEmbeddingService

logger = logging.getLogger(__name__)

@dataclass
class HybridSearchResult:
    """Resultado de busca híbrida com scores combinados"""
    id: str
    payload: Dict[str, Any]
    dense_score: float
    sparse_score: float
    combined_score: float
    content: str
    metadata: Dict[str, Any]

@dataclass
class HybridDocument:
    """Documento com vetores densos e esparsos"""
    id: str
    content: str
    dense_vector: List[float]
    sparse_vector: SparseVector
    metadata: Dict[str, Any]

class RRFFusion:
    """
    Reciprocal Rank Fusion para combinar resultados
    Baseado no paper de Cormack et al. (2009)
    """
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse_results(
        self, 
        dense_results: List[Tuple[str, float]], 
        sparse_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combina resultados usando RRF
        """
        # Criar rankings
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_results)}
        sparse_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sparse_results)}
        
        # Coletar todos os documentos únicos
        all_docs = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calcular scores RRF
        rrf_scores = {}
        for doc_id in all_docs:
            dense_rank = dense_ranks.get(doc_id, len(dense_results) + 1)
            sparse_rank = sparse_ranks.get(doc_id, len(sparse_results) + 1)
            
            rrf_score = (1 / (self.k + dense_rank)) + (1 / (self.k + sparse_rank))
            rrf_scores[doc_id] = rrf_score
        
        # Ordenar por score RRF
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

class HybridQdrantStore:
    """
    Vector Store híbrido para Qdrant 1.8.0+
    Combina dense vectors (OpenAI) + sparse vectors (BM25)
    Otimizado para máxima performance
    """
    
    def __init__(
        self, 
        config_path: str = "config/hybrid_search_config.yaml",
        client: Optional[QdrantClient] = None
    ):
        self.config = self._load_config(config_path)
        self.collection_name = self.config["hybrid_search"]["collection_name"]
        
        # Clientes Qdrant
        if client:
            self.client = client
        else:
            self.client = QdrantClient(":memory:")  # Para desenvolvimento
        
        self.async_client = AsyncQdrantClient(":memory:")  # Para operações assíncronas
        
        # Serviços de embedding
        self.dense_embedding_service = APIEmbeddingService()
        self.sparse_vector_service = AdvancedSparseVectorService(config_path)
        
        # Fusion strategy
        self.rrf_fusion = RRFFusion(k=self.config["hybrid_search"]["search_strategy"]["rrf_k"])
        
        # Cache para performance
        self._query_cache: Dict[str, List[HybridSearchResult]] = {}
        
        self.collection_exists = False
    
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config não encontrado: {config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Configuração padrão"""
        return {
            "hybrid_search": {
                "collection_name": "hybrid_rag_collection",
                "dense_vectors": {
                    "vector_name": "dense",
                    "dimension": 1536,
                    "distance_metric": "Cosine"
                },
                "sparse_vectors": {
                    "vector_name": "sparse",
                    "modifier": "idf"
                },
                "search_strategy": {
                    "dense_weight": 0.7,
                    "sparse_weight": 0.3,
                    "rrf_k": 60,
                    "dense_limit": 50,
                    "sparse_limit": 50,
                    "final_limit": 20
                },
                "indexing": {
                    "optimizer_cpu_budget": 0,
                    "hnsw_config": {
                        "m": 16,
                        "ef_construct": 128
                    }
                }
            }
        }
    
    async def create_collection(self) -> None:
        """
        Cria collection com configuração otimizada para Qdrant 1.8.0
        Suporte para dense + sparse vectors
        """
        if self.collection_exists:
            return
        
        logger.info(f"Criando collection híbrida: {self.collection_name}")
        
        # Configuração de dense vectors
        dense_config = self.config["hybrid_search"]["dense_vectors"]
        vectors_config = {
            dense_config["vector_name"]: VectorParams(
                size=dense_config["dimension"],
                distance=getattr(Distance, dense_config["distance_metric"].upper()),
                hnsw_config=models.HnswConfigDiff(
                    m=self.config["hybrid_search"]["indexing"]["hnsw_config"]["m"],
                    ef_construct=self.config["hybrid_search"]["indexing"]["hnsw_config"]["ef_construct"]
                )
            )
        }
        
        # Configuração de sparse vectors (Qdrant 1.8.0)
        sparse_config = self.config["hybrid_search"]["sparse_vectors"]
        sparse_vectors_config = {
            sparse_config["vector_name"]: SparseVectorParams(
                modifier=getattr(models.Modifier, sparse_config["modifier"].upper())
            )
        }
        
        # Criar collection
        await self.async_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=2,
                max_segment_size=None,
                memmap_threshold=None,
                indexing_threshold=20000,
                flush_interval_sec=5,
                max_optimization_threads=self.config["hybrid_search"]["indexing"]["optimizer_cpu_budget"] or None
            )
        )
        
        self.collection_exists = True
        logger.info("Collection híbrida criada com sucesso")
    
    async def fit_sparse_encoders(self, documents: List[str]) -> None:
        """
        Treina encoders sparse no corpus
        Necessário antes de indexar documentos
        """
        logger.info("Treinando sparse encoders no corpus")
        await self.sparse_vector_service.fit(documents)
        logger.info("Sparse encoders treinados")
    
    async def add_documents(self, documents: List[HybridDocument]) -> None:
        """
        Adiciona documentos com vetores híbridos
        Otimizado para batch processing
        """
        if not self.collection_exists:
            await self.create_collection()
        
        logger.info(f"Indexando {len(documents)} documentos híbridos")
        
        # Preparar pontos para Qdrant
        points = []
        for doc in documents:
            # Preparar vetores nomeados
            dense_vector = NamedVector(
                name=self.config["hybrid_search"]["dense_vectors"]["vector_name"],
                vector=doc.dense_vector
            )
            
            sparse_vector = NamedSparseVector(
                name=self.config["hybrid_search"]["sparse_vectors"]["vector_name"],
                vector=QdrantSparseVector(
                    indices=doc.sparse_vector.indices,
                    values=doc.sparse_vector.values
                )
            )
            
            # Criar ponto
            point = PointStruct(
                id=doc.id,
                vector={
                    self.config["hybrid_search"]["dense_vectors"]["vector_name"]: doc.dense_vector,
                },
                sparse_vector={
                    self.config["hybrid_search"]["sparse_vectors"]["vector_name"]: QdrantSparseVector(
                        indices=doc.sparse_vector.indices,
                        values=doc.sparse_vector.values
                    )
                },
                payload={
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "indexed_at": datetime.now().isoformat()
                }
            )
            points.append(point)
        
        # Upload em batch
        await self.async_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Indexação concluída: {len(documents)} documentos")
    
    async def hybrid_search(
        self, 
        query: str, 
        limit: int = None
    ) -> List[HybridSearchResult]:
        """
        Busca híbrida combinando dense + sparse vectors
        Utiliza RRF para fusion dos resultados
        Performance otimizada para Qdrant 1.8.0
        """
        limit = limit or self.config["hybrid_search"]["search_strategy"]["final_limit"]
        
        # Cache check
        cache_key = f"{query}:{limit}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Preparar vetores de query
        dense_vector = await self.dense_embedding_service.embed_text(query)
        sparse_vector = self.sparse_vector_service.encode_text(query)
        
        # Configurações de busca
        search_config = self.config["hybrid_search"]["search_strategy"]
        dense_limit = search_config["dense_limit"]
        sparse_limit = search_config["sparse_limit"]
        
        # Busca paralela: dense + sparse
        dense_task = self._search_dense(dense_vector, dense_limit)
        sparse_task = self._search_sparse(sparse_vector, sparse_limit)
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Fusion com RRF
        fused_results = self.rrf_fusion.fuse_results(
            [(r.id, r.score) for r in dense_results],
            [(r.id, r.score) for r in sparse_results]
        )
        
        # Construir resultados finais
        final_results = []
        dense_scores = {r.id: r.score for r in dense_results}
        sparse_scores = {r.id: r.score for r in sparse_results}
        
        for doc_id, combined_score in fused_results[:limit]:
            # Buscar payload do documento
            doc_data = await self._get_document(doc_id)
            if doc_data:
                result = HybridSearchResult(
                    id=doc_id,
                    payload=doc_data.payload,
                    dense_score=dense_scores.get(doc_id, 0.0),
                    sparse_score=sparse_scores.get(doc_id, 0.0),
                    combined_score=combined_score,
                    content=doc_data.payload.get("content", ""),
                    metadata=doc_data.payload.get("metadata", {})
                )
                final_results.append(result)
        
        # Cache resultado
        if len(self._query_cache) < 1000:  # Limitar cache
            self._query_cache[cache_key] = final_results
        
        return final_results
    
    async def _search_dense(self, vector: List[float], limit: int):
        """Busca por dense vectors"""
        return await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=(
                self.config["hybrid_search"]["dense_vectors"]["vector_name"],
                vector
            ),
            limit=limit,
            with_payload=True
        )
    
    async def _search_sparse(self, sparse_vector: SparseVector, limit: int):
        """Busca por sparse vectors (otimizada Qdrant 1.8.0)"""
        return await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=(
                self.config["hybrid_search"]["sparse_vectors"]["vector_name"],
                QdrantSparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values
                )
            ),
            limit=limit,
            with_payload=True
        )
    
    async def _get_document(self, doc_id: str):
        """Recupera documento por ID"""
        try:
            result = await self.async_client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True
            )
            return result[0] if result else None
        except Exception as e:
            logger.warning(f"Erro ao recuperar documento {doc_id}: {e}")
            return None
    
    async def get_collection_info(self) -> Dict:
        """Retorna informações da collection"""
        try:
            info = await self.async_client.get_collection(self.collection_name)
            return {
                "status": info.status,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "config": {
                    "dense_vectors": bool(info.config.params.vectors),
                    "sparse_vectors": bool(info.config.params.sparse_vectors)
                }
            }
        except Exception as e:
            logger.error(f"Erro ao obter info da collection: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_cache(self) -> None:
        """Limpa cache de queries"""
        self._query_cache.clear()
        logger.info("Cache de queries limpo")

# Factory function
async def create_hybrid_store(
    config_path: str = "config/hybrid_search_config.yaml",
    client: Optional[QdrantClient] = None
) -> HybridQdrantStore:
    """Cria e inicializa hybrid store"""
    store = HybridQdrantStore(config_path, client)
    await store.create_collection()
    return store 