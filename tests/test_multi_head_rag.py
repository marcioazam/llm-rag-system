"""Testes para o Multi-Head RAG."""
import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Dict, Any
import time

from src.retrieval.multi_head_rag import (
    create_multi_head_retriever,
    MultiHeadRetriever,
    AttentionHead
)


class MockEmbeddingService:
    """Mock do serviço de embeddings."""
    
    async def aembed_query(self, text: str) -> List[float]:
        """Retorna embedding simulado."""
        await asyncio.sleep(0.001)
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768).tolist()
    
    async def embed_query(self, text: str) -> List[float]:
        """Versão síncrona do embed."""
        return await self.aembed_query(text)


class MockVectorStore:
    """Mock do vector store."""
    
    def __init__(self):
        self.documents = [
            {"content": "Python é uma linguagem de programação", "metadata": {"type": "definition"}},
            {"content": "Para programar em Python, use funções", "metadata": {"type": "tutorial"}},
            {"content": "O processo de desenvolvimento envolve etapas", "metadata": {"type": "process"}},
            {"content": "Python foi criado em 1991", "metadata": {"type": "history"}},
            {"content": "Comparado com Java, Python é mais simples", "metadata": {"type": "comparison"}}
        ]
    
    async def similarity_search_with_score(self, embedding: List[float], k: int = 5):
        """Retorna documentos com scores simulados."""
        await asyncio.sleep(0.001)
        
        class Document:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        
        results = []
        for i, doc in enumerate(self.documents[:k]):
            doc_obj = Document(doc["content"], doc["metadata"])
            score = np.random.uniform(0.5, 0.95)
            results.append((doc_obj, score))
        
        return results


class TestMultiHeadRetriever:
    """Testes para o MultiHeadRetriever."""
    
    @pytest.fixture
    def embedding_service(self):
        """Fixture para serviço de embeddings."""
        return MockEmbeddingService()
    
    @pytest.fixture
    def vector_store(self):
        """Fixture para vector store."""
        return MockVectorStore()
    
    @pytest.fixture
    def multi_head_retriever(self, embedding_service, vector_store):
        """Fixture para multi-head retriever."""
        return create_multi_head_retriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            config={
                "num_heads": 5,
                "attention_dim": 768,
                "voting_strategy": "weighted_majority"
            }
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, multi_head_retriever):
        """Testa inicialização do multi-head retriever."""
        assert multi_head_retriever is not None
        assert multi_head_retriever.num_heads == 5
        assert multi_head_retriever.attention_dim == 768
        assert len(multi_head_retriever.attention_heads) == 5
        
        # Verificar tipos de heads
        head_names = [head.name for head in multi_head_retriever.attention_heads]
        assert "factual" in head_names
        assert "conceptual" in head_names
        assert "procedural" in head_names
        assert "contextual" in head_names
        assert "temporal" in head_names
    
    @pytest.mark.asyncio
    async def test_retrieve_multi_head(self, multi_head_retriever):
        """Testa retrieval com múltiplas heads."""
        query = "Como programar em Python?"
        documents, metadata = await multi_head_retriever.retrieve_multi_head(query, k=5)
        
        assert isinstance(documents, list)
        assert len(documents) <= 5
        assert isinstance(metadata, dict)
        
        # Verificar metadados
        assert "retrieval_method" in metadata
        assert metadata["retrieval_method"] == "multi_head"
        assert "num_heads" in metadata
        assert "head_contributions" in metadata
        assert "diversity_score" in metadata
        assert "semantic_coverage" in metadata
    
    @pytest.mark.asyncio
    async def test_voting_strategies(self, embedding_service, vector_store):
        """Testa diferentes estratégias de voting."""
        strategies = ["weighted_majority", "borda_count", "coverage_optimization"]
        
        for strategy in strategies:
            retriever = create_multi_head_retriever(
                embedding_service=embedding_service,
                vector_store=vector_store,
                config={
                    "num_heads": 3,
                    "voting_strategy": strategy
                }
            )
            
            query = "Ensine Python"
            documents, metadata = await retriever.retrieve_multi_head(query, k=5)
            
            assert isinstance(documents, list)
            assert metadata["voting_strategy"] == strategy
    
    def test_get_stats(self, multi_head_retriever):
        """Testa estatísticas do retriever."""
        stats = multi_head_retriever.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_queries" in stats
        assert "head_performance" in stats
        assert "voting_strategies_used" in stats
        assert "average_diversity" in stats
        assert "heads_config" in stats
    
    @pytest.mark.asyncio
    async def test_performance(self, multi_head_retriever):
        """Testa performance com múltiplas queries."""
        queries = [
            "O que é Python?",
            "Como funciona machine learning?",
            "Explique programação orientada a objetos"
        ]
        
        start_time = time.time()
        
        for query in queries:
            documents, metadata = await multi_head_retriever.retrieve_multi_head(query, k=5)
            assert isinstance(documents, list)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Deve processar 3 queries em menos de 1 segundo com mocks
        assert total_time < 1.0


class TestAttentionHead:
    """Testes para a classe AttentionHead."""
    
    def test_attention_head_initialization(self):
        """Testa inicialização de AttentionHead."""
        head = AttentionHead(
            name="test_head",
            semantic_focus="test focus",
            temperature=0.8,
            top_k=5
        )
        
        assert head.name == "test_head"
        assert head.semantic_focus == "test focus"
        assert head.temperature == 0.8
        assert head.top_k == 5
        assert head.weight_matrix is not None
        assert head.weight_matrix.shape == (768, 768) 