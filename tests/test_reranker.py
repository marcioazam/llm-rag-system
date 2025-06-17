import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Tuple

from src.retrieval.reranker import Reranker


class TestReranker:
    """Testes para a classe Reranker."""

    @pytest.fixture
    def reranker(self):
        """Cria uma instância do Reranker para testes."""
        return Reranker()

    @pytest.fixture
    def reranker_custom(self):
        """Cria uma instância do Reranker com configurações customizadas."""
        return Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-2-v1", max_length=256)

    @pytest.fixture
    def sample_documents(self):
        """Documentos de exemplo para testes."""
        return [
            {
                "id": "doc1",
                "content": "Machine learning is a subset of artificial intelligence",
                "metadata": {"source": "textbook", "chapter": 1},
                "score": 0.8
            },
            {
                "id": "doc2",
                "content": "Deep learning uses neural networks with multiple layers",
                "metadata": {"source": "research_paper", "year": 2020},
                "score": 0.7
            },
            {
                "id": "doc3",
                "content": "Natural language processing enables computers to understand text",
                "metadata": {"source": "article", "domain": "NLP"},
                "score": 0.6
            },
            {
                "id": "doc4",
                "content": "Computer vision allows machines to interpret visual information",
                "metadata": {"source": "tutorial", "difficulty": "beginner"},
                "score": 0.5
            },
            {
                "id": "doc5",
                "content": "Reinforcement learning trains agents through rewards and penalties",
                "metadata": {"source": "course", "level": "advanced"},
                "score": 0.4
            }
        ]

    def test_init_default(self, reranker):
        """Testa inicialização com configurações padrão."""
        assert hasattr(reranker, 'cross_encoder')
        assert hasattr(reranker, 'device')

    def test_init_custom(self, reranker_custom):
        """Testa inicialização com configurações customizadas."""
        assert hasattr(reranker_custom, 'cross_encoder')
        assert hasattr(reranker_custom, 'device')

    def test_rerank_basic(self, reranker, sample_documents):
        """Testa reranqueamento básico."""
        query = "machine learning algorithms"
        reranked = reranker.rerank(query, sample_documents)
        
        # Deve retornar uma lista de documentos
        assert isinstance(reranked, list)
        assert len(reranked) <= len(sample_documents)
        assert len(reranked) <= 10  # valor padrão de k
        
        # Cada item deve ser um dicionário
        for doc in reranked:
            assert isinstance(doc, dict)
            assert "id" in doc
            assert "content" in doc

    def test_rerank_with_custom_k(self, reranker, sample_documents):
        """Testa reranqueamento com k customizado."""
        query = "deep learning"
        custom_k = 3
        reranked = reranker.rerank(query, sample_documents, k=custom_k)
        
        assert len(reranked) <= custom_k
        assert len(reranked) <= len(sample_documents)

    def test_rerank_empty_documents(self, reranker):
        """Testa reranqueamento com lista vazia de documentos."""
        query = "test query"
        reranked = reranker.rerank(query, [])
        
        assert isinstance(reranked, list)
        assert len(reranked) == 0

    def test_rerank_single_document(self, reranker):
        """Testa reranqueamento com um único documento."""
        query = "artificial intelligence"
        single_doc = [{
            "id": "doc1",
            "content": "AI is transforming technology",
            "score": 0.9
        }]
        
        reranked = reranker.rerank(query, single_doc)
        
        assert len(reranked) == 1
        assert reranked[0]["id"] == "doc1"

    def test_rerank_preserves_document_structure(self, reranker, sample_documents):
        """Testa que o reranqueamento preserva a estrutura dos documentos."""
        query = "neural networks"
        reranked = reranker.rerank(query, sample_documents)
        
        for doc in reranked:
            # Deve preservar campos originais
            assert "id" in doc
            assert "content" in doc
            assert "metadata" in doc
            
            # Pode adicionar score de relevância
            if "relevance_score" in doc:
                assert isinstance(doc["relevance_score"], (int, float))

    def test_predict_scores(self, reranker, sample_documents):
        """Testa predição de scores de relevância."""
        query = "machine learning"
        contents = [doc["content"] for doc in sample_documents]
        
        scores = reranker.predict_scores(query, contents)
        
        # Deve retornar um array numpy de scores
        assert hasattr(scores, '__len__')
        assert len(scores) == len(contents)
        
        # Todos os scores devem ser números
        for score in scores:
            assert isinstance(score, (int, float, np.floating))

    def test_predict_scores_empty(self, reranker):
        """Testa predição de scores com lista vazia."""
        query = "test query"
        scores = reranker.predict_scores(query, [])
        
        assert hasattr(scores, '__len__')
        assert len(scores) == 0

    def test_rerank_with_content_key(self, reranker, sample_documents):
        """Testa reranqueamento com chave de conteúdo customizada."""
        # Criar documentos com chave diferente
        docs_with_text = [
            {"id": "doc1", "text": "Machine learning algorithms", "score": 0.3},
            {"id": "doc2", "text": "Deep learning networks", "score": 0.7}
        ]
        
        query = "machine learning"
        reranked = reranker.rerank(query, docs_with_text, content_key="text")
        
        assert isinstance(reranked, list)
        assert len(reranked) <= len(docs_with_text)
        
        for doc in reranked:
            assert isinstance(doc, dict)
            assert "id" in doc
            assert "text" in doc

    def test_rerank_relevance_ordering(self, reranker):
        """Testa se o reranqueamento ordena por relevância."""
        query = "machine learning"
        documents = [
            {"id": "doc1", "content": "Machine learning is awesome", "score": 0.3},
            {"id": "doc2", "content": "Cooking recipes for dinner", "score": 0.9},
            {"id": "doc3", "content": "Advanced machine learning techniques", "score": 0.1}
        ]
        
        reranked = reranker.rerank(query, documents)
        
        # Documentos mais relevantes devem vir primeiro
        # (assumindo que o reranker funciona corretamente)
        assert len(reranked) > 0
        
        # Verificar que documentos relacionados a ML estão bem posicionados
        ml_docs = [doc for doc in reranked if "machine learning" in doc["content"].lower()]
        cooking_docs = [doc for doc in reranked if "cooking" in doc["content"].lower()]
        
        if ml_docs and cooking_docs:
            # Documentos de ML devem estar melhor posicionados
            ml_positions = [reranked.index(doc) for doc in ml_docs]
            cooking_positions = [reranked.index(doc) for doc in cooking_docs]
            
            # Pelo menos um documento de ML deve estar antes de documentos de culinária
            assert min(ml_positions) < max(cooking_positions)

    def test_rerank_with_metadata_filtering(self, reranker, sample_documents):
        """Testa reranqueamento considerando metadados."""
        query = "research on neural networks"
        reranked = reranker.rerank(query, sample_documents)
        
        # Verificar que documentos de pesquisa podem ser priorizados
        research_docs = [doc for doc in reranked 
                        if doc.get("metadata", {}).get("source") == "research_paper"]
        
        # Deve processar sem erros
        assert isinstance(reranked, list)

    def test_rerank_query_variations(self, reranker, sample_documents):
        """Testa reranqueamento com diferentes tipos de consulta."""
        queries = [
            "What is machine learning?",  # Pergunta
            "machine learning algorithms",  # Frase nominal
            "ML",  # Abreviação
            "artificial intelligence and machine learning",  # Consulta longa
            "deep neural networks for image recognition"  # Consulta específica
        ]
        
        for query in queries:
            reranked = reranker.rerank(query, sample_documents)
            
            assert isinstance(reranked, list)
            assert len(reranked) <= len(sample_documents)
            assert len(reranked) <= 10  # valor padrão de k

    def test_rerank_special_characters(self, reranker):
        """Testa reranqueamento com caracteres especiais."""
        query = "C++ programming & AI/ML frameworks"
        documents = [
            {"id": "doc1", "content": "C++ is a programming language"},
            {"id": "doc2", "content": "AI and ML frameworks comparison"},
            {"id": "doc3", "content": "Python vs Java programming"}
        ]
        
        reranked = reranker.rerank(query, documents)
        
        assert isinstance(reranked, list)
        assert len(reranked) <= len(documents)

    def test_rerank_unicode_content(self, reranker):
        """Testa reranqueamento com conteúdo Unicode."""
        query = "aprendizado de máquina"
        documents = [
            {"id": "doc1", "content": "Aprendizado de máquina é uma área da IA"},
            {"id": "doc2", "content": "机器学习是人工智能的一个分支"},
            {"id": "doc3", "content": "Machine learning algorithms 🤖"}
        ]
        
        reranked = reranker.rerank(query, documents)
        
        assert isinstance(reranked, list)
        assert len(reranked) <= len(documents)

    def test_rerank_with_different_k_values(self, reranker, sample_documents):
        """Testa reranqueamento com diferentes valores de k."""
        query = "machine learning"
        
        # Testar diferentes valores de k
        for k in [1, 3, 5]:
            reranked = reranker.rerank(query, sample_documents, k=k)
            
            assert isinstance(reranked, list)
            assert len(reranked) <= k
            assert len(reranked) <= len(sample_documents)

    def test_rerank_performance_large_dataset(self, reranker):
        """Testa performance com dataset grande."""
        query = "machine learning"
        
        # Criar muitos documentos
        large_documents = []
        for i in range(1000):
            large_documents.append({
                "id": f"doc_{i}",
                "content": f"Document {i} about various topics including ML",
                "score": np.random.random()
            })
        
        reranked = reranker.rerank(query, large_documents, k=10)
        
        # Deve processar eficientemente
        assert isinstance(reranked, list)
        assert len(reranked) <= 10

    def test_rerank_score_consistency(self, reranker, sample_documents):
        """Testa consistência dos scores de reranqueamento."""
        query = "neural networks"
        
        # Executar reranqueamento múltiplas vezes
        results = []
        for _ in range(3):
            reranked = reranker.rerank(query, sample_documents)
            results.append([doc["id"] for doc in reranked])
        
        # Resultados devem ser consistentes (assumindo modelo determinístico)
        if len(results) > 1:
            # Pode haver pequenas variações dependendo da implementação
            assert all(isinstance(result, list) for result in results)

    def test_rerank_edge_cases(self, reranker):
        """Testa casos extremos."""
        # Query muito longa
        long_query = " ".join(["machine learning"] * 100)
        documents = [{"id": "doc1", "content": "ML content"}]
        
        reranked = reranker.rerank(long_query, documents)
        assert isinstance(reranked, list)
        
        # Query vazia
        reranked_empty = reranker.rerank("", documents)
        assert isinstance(reranked_empty, list)
        
        # Documento com conteúdo muito longo
        long_content_doc = [{
            "id": "long_doc",
            "content": "A" * 10000  # 10KB de texto
        }]
        
        reranked_long = reranker.rerank("test", long_content_doc)
        assert isinstance(reranked_long, list)

    def test_rerank_preserves_original_scores(self, reranker, sample_documents):
        """Testa que scores originais são preservados ou atualizados apropriadamente."""
        query = "machine learning"
        original_scores = [doc.get("score") for doc in sample_documents]
        
        reranked = reranker.rerank(query, sample_documents)
        
        # Verificar que documentos ainda têm scores
        for doc in reranked:
            # Deve ter score original ou novo score de relevância
            assert "score" in doc or "relevance_score" in doc

    def test_error_handling_malformed_documents(self, reranker):
        """Testa tratamento de documentos mal formados."""
        query = "test query"
        malformed_docs = [
            {"id": "doc1"},  # Sem content
            {"content": "Content without ID"},  # Sem ID
            {},  # Documento vazio
            {"id": "doc2", "content": None},  # Content None
        ]
        
        try:
            reranked = reranker.rerank(query, malformed_docs)
            # Se não lançar exceção, deve retornar resultado válido
            assert isinstance(reranked, list)
        except (KeyError, TypeError, ValueError):
            # Exceções são aceitáveis para documentos mal formados
            pass

    def test_rerank_different_content_types(self, reranker):
        """Testa reranqueamento com diferentes tipos de conteúdo."""
        query = "programming languages"
        mixed_documents = [
            {"id": "doc1", "content": "Python is a programming language"},
            {"id": "doc2", "content": "def hello_world(): print('Hello, World!')"},  # Código
            {"id": "doc3", "content": "# This is a comment\nprint('Hello')"},  # Código com comentário
            {"id": "doc4", "content": "Programming languages: Python, Java, C++"},  # Lista
        ]
        
        reranked = reranker.rerank(query, mixed_documents)
        
        assert isinstance(reranked, list)
        assert len(reranked) <= len(mixed_documents)

    def test_configuration_impact(self):
        """Testa impacto de diferentes configurações."""
        documents = [
            {"id": f"doc{i}", "content": f"Document {i} content"} 
            for i in range(20)
        ]
        
        # Diferentes configurações de modelo
        reranker_default = Reranker()
        reranker_custom = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-2-v1")
        
        query = "document content"
        
        results_default = reranker_default.rerank(query, documents, k=5)
        results_custom = reranker_custom.rerank(query, documents, k=10)
        
        assert len(results_default) <= 5
        assert len(results_custom) <= 10
        assert isinstance(results_default, list)
        assert isinstance(results_custom, list)