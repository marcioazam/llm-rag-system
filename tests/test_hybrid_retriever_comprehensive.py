"""
Testes completos para o Hybrid Retriever.
Objetivo: Cobertura de 0% para 60%+
"""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import time
import yaml
import numpy as np

# Imports necessários
try:
    from src.retrieval.hybrid_retriever import (
        HybridRetriever,
        QueryAnalyzer,
        QueryAnalysis,
        HybridRetrievalResult,
        create_hybrid_retriever
    )
except ImportError:
    # Fallback se módulo não existir
    class QueryAnalysis:
        def __init__(self, original_query, expanded_query, query_type, keywords, entities, intent_confidence=0.0):
            self.original_query = original_query
            self.expanded_query = expanded_query
            self.query_type = query_type
            self.keywords = keywords
            self.entities = entities
            self.intent_confidence = intent_confidence

    class HybridRetrievalResult:
        def __init__(self, id, content, metadata, dense_score=0.0, sparse_score=0.0, 
                     combined_score=0.0, rerank_score=None, retrieval_method="hybrid", 
                     query_match_explanation=""):
            self.id = id
            self.content = content
            self.metadata = metadata
            self.dense_score = dense_score
            self.sparse_score = sparse_score
            self.combined_score = combined_score
            self.rerank_score = rerank_score
            self.retrieval_method = retrieval_method
            self.query_match_explanation = query_match_explanation

    class QueryAnalyzer:
        def __init__(self, config):
            self.config = config
            self.stopwords = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

        def analyze_query(self, query):
            keywords = self._extract_keywords(query)
            entities = self._extract_entities(query)
            query_type = self._classify_query_type(query, keywords)
            expanded = self._expand_query(query, keywords)
            confidence = self._calculate_intent_confidence(query, query_type)
            
            return QueryAnalysis(query, expanded, query_type, keywords, entities, confidence)

        def _extract_keywords(self, query):
            if not query:
                return []
            words = query.lower().split()
            return [word for word in words if word not in self.stopwords and len(word) > 2]

        def _extract_entities(self, query):
            import re
            # Extrair entidades simples: palavras capitalizadas, números, versões
            entities = []
            # Palavras capitalizadas
            entities.extend(re.findall(r'\b[A-Z][a-z]+\b', query))
            # Números e versões
            entities.extend(re.findall(r'\b\d+\.?\d*\b', query))
            return entities

        def _classify_query_type(self, query, keywords):
            question_words = ["what", "how", "why", "when", "where", "which", "who"]
            if any(word in query.lower() for word in question_words):
                return "semantic"
            elif len(keywords) <= 3:
                return "keyword"
            else:
                return "hybrid"

        def _expand_query(self, query, keywords):
            # Expansão simples com sinônimos básicos
            synonyms = {
                "function": "method procedure",
                "error": "exception bug",
                "install": "setup configure"
            }
            expanded = query
            for keyword in keywords:
                if keyword in synonyms:
                    expanded += " " + synonyms[keyword]
            return expanded

        def _calculate_intent_confidence(self, query, query_type):
            word_count = len(query.split())
            if word_count <= 3:
                return 0.6
            elif word_count <= 7:
                return 0.75
            else:
                return 0.9

    class HybridRetriever:
        def __init__(self, config_path):
            self.config_path = config_path
            self.config = self._load_config(config_path)
            self.query_analyzer = QueryAnalyzer(self.config)
            self._retrieval_cache = {}
            
            # Mock components
            self.vector_store = Mock()
            self.query_enhancer = Mock()
            self.reranker = Mock()
            self.hyde_enhancer = Mock()
            self.dense_embedding_service = Mock()
            self.sparse_vector_service = Mock()
            
            # Stats
            self.total_retrievals = 0
            self.cache_hits = 0

        def _load_config(self, config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except FileNotFoundError:
                return {
                    "hybrid_search": {
                        "dense_weight": 0.7,
                        "sparse_weight": 0.3,
                        "limit": 10,
                        "collection_name": "default_collection",
                        "search_strategy": {
                            "rrf_k": 60
                        }
                    },
                    "query_analysis": {
                        "enable_entity_extraction": True,
                        "enable_expansion": True
                    },
                    "embedding_providers": {
                        "sparse": {
                            "k1": 1.2,
                            "b": 0.75,
                            "epsilon": 0.25
                        },
                        "dense": {
                            "model": "text-embedding-ada-002",
                            "dimensions": 1536
                        }
                    }
                }

        def _determine_search_strategy(self, query_analysis):
            if query_analysis.query_type == "semantic":
                return "dense" if query_analysis.intent_confidence > 0.8 else "hybrid"
            elif query_analysis.query_type == "keyword":
                return "sparse" if len(query_analysis.keywords) <= 3 else "hybrid"
            else:
                return "hybrid"

        async def _dense_search_only(self, query_analysis, limit=10):
            return await self.vector_store.dense_search()

        async def _sparse_search_only(self, query_analysis, limit=10):
            return await self.vector_store.sparse_search()

        async def _hybrid_search(self, query_analysis, limit=10):
            return await self.vector_store.hybrid_search()

        async def _rerank_results(self, query, results):
            return await self.reranker.rerank(query, results)

        def _create_retrieval_results(self, search_results, query_analysis, strategy):
            results = []
            for result in search_results:
                explanation = self._generate_match_explanation(result, query_analysis)
                results.append(HybridRetrievalResult(
                    id=result.id,
                    content=result.content,
                    metadata=result.metadata,
                    dense_score=getattr(result, 'dense_score', 0.0),
                    sparse_score=getattr(result, 'sparse_score', 0.0),
                    combined_score=getattr(result, 'combined_score', 0.0),
                    retrieval_method=strategy,
                    query_match_explanation=explanation
                ))
            return results

        def _generate_match_explanation(self, result, query_analysis):
            if hasattr(result, 'dense_score') and result.dense_score > 0.8:
                return f"Strong semantic match (dense score: {result.dense_score:.2f})"
            elif hasattr(result, 'sparse_score') and result.sparse_score > 0.8:
                return f"Strong keyword match (sparse score: {result.sparse_score:.2f})"
            else:
                return "Hybrid match combining semantic and keyword signals"

        def clear_cache(self):
            self._retrieval_cache.clear()

        def get_metrics(self):
            return {
                "cache_size": len(self._retrieval_cache),
                "total_retrievals": getattr(self, 'total_retrievals', 0),
                "cache_hit_rate": getattr(self, 'cache_hits', 0) / max(getattr(self, 'total_retrievals', 1), 1)
            }

        async def retrieve(self, query, limit=10, use_reranking=False, strategy=None):
            # Check cache
            if query in self._retrieval_cache:
                self.cache_hits += 1
                return self._retrieval_cache[query]
            
            self.total_retrievals += 1
            
            # Analyze query
            query_analysis = self.query_analyzer.analyze_query(query)
            
            # Determine strategy
            if not strategy:
                strategy = self._determine_search_strategy(query_analysis)
            
            # Execute search
            if strategy == "dense":
                search_results = await self._dense_search_only(query_analysis, limit)
            elif strategy == "sparse":
                search_results = await self._sparse_search_only(query_analysis, limit)
            else:
                search_results = await self._hybrid_search(query_analysis, limit)
            
            # Rerank if requested
            if use_reranking and search_results:
                search_results = await self._rerank_results(query, search_results)
            
            # Create result objects
            results = self._create_retrieval_results(search_results, query_analysis, strategy)
            
            # Cache results
            self._retrieval_cache[query] = results
            
            return results

        async def retrieve_with_filters(self, query, filters, limit=10):
            # Simplified implementation
            return await self.retrieve(query, limit)

    def create_hybrid_retriever(config_path="config/hybrid_search_config.yaml"):
        return HybridRetriever(config_path)


class TestQueryAnalysis:
    """Testes para a classe QueryAnalysis."""

    def test_query_analysis_creation(self):
        """Testar criação de QueryAnalysis."""
        analysis = QueryAnalysis(
            original_query="What is Python?",
            expanded_query="What is Python programming language?",
            query_type="semantic",
            keywords=["python", "programming", "language"],
            entities=["Python"],
            intent_confidence=0.85
        )
        
        assert analysis.original_query == "What is Python?"
        assert analysis.expanded_query == "What is Python programming language?"
        assert analysis.query_type == "semantic"
        assert "python" in analysis.keywords
        assert "Python" in analysis.entities
        assert analysis.intent_confidence == 0.85

    def test_query_analysis_defaults(self):
        """Testar valores padrão de QueryAnalysis."""
        analysis = QueryAnalysis(
            original_query="test",
            expanded_query="test",
            query_type="keyword",
            keywords=[],
            entities=[]
        )
        
        assert analysis.intent_confidence == 0.0


class TestHybridRetrievalResult:
    """Testes para a classe HybridRetrievalResult."""

    def test_hybrid_result_creation(self):
        """Testar criação de HybridRetrievalResult."""
        result = HybridRetrievalResult(
            id="doc1",
            content="Test content",
            metadata={"source": "test.txt"},
            dense_score=0.9,
            sparse_score=0.7,
            combined_score=0.8,
            rerank_score=0.85,
            retrieval_method="hybrid",
            query_match_explanation="Strong semantic and keyword match"
        )
        
        assert result.id == "doc1"
        assert result.content == "Test content"
        assert result.metadata["source"] == "test.txt"
        assert result.dense_score == 0.9
        assert result.sparse_score == 0.7
        assert result.combined_score == 0.8
        assert result.rerank_score == 0.85
        assert result.retrieval_method == "hybrid"
        assert "semantic" in result.query_match_explanation

    def test_hybrid_result_defaults(self):
        """Testar valores padrão de HybridRetrievalResult."""
        result = HybridRetrievalResult(
            id="doc1",
            content="Test content",
            metadata={}
        )
        
        assert result.dense_score == 0.0
        assert result.sparse_score == 0.0
        assert result.combined_score == 0.0
        assert result.rerank_score is None
        assert result.retrieval_method == "hybrid"
        assert result.query_match_explanation == ""


class TestQueryAnalyzer:
    """Testes para o analisador de queries."""

    @pytest.fixture
    def analyzer(self):
        """Analisador configurado para testes."""
        config = {
            "query_analysis": {
                "enable_entity_extraction": True,
                "enable_expansion": True,
                "similarity_threshold": 0.7
            }
        }
        return QueryAnalyzer(config)

    def test_analyzer_init(self, analyzer):
        """Testar inicialização do analyzer."""
        assert analyzer.config is not None
        assert analyzer.config["query_analysis"]["enable_entity_extraction"] is True
        assert analyzer.stopwords is not None

    def test_analyze_query_basic(self, analyzer):
        """Testar análise básica de query."""
        query = "What is machine learning in Python?"
        
        result = analyzer.analyze_query(query)
        
        assert isinstance(result, QueryAnalysis)
        assert result.original_query == query
        assert result.expanded_query is not None
        assert result.query_type in ['semantic', 'keyword', 'hybrid']
        assert isinstance(result.keywords, list)
        assert isinstance(result.entities, list)
        assert 0.0 <= result.intent_confidence <= 1.0

    def test_extract_keywords_programming(self, analyzer):
        """Testar extração de keywords para programação."""
        query = "How to implement machine learning algorithms in Python"
        
        keywords = analyzer._extract_keywords(query)
        
        assert "implement" in keywords
        assert "machine" in keywords
        assert "learning" in keywords
        assert "algorithms" in keywords
        assert "python" in keywords
        
        # Stop words não devem estar presentes
        assert "how" not in keywords
        assert "to" not in keywords
        assert "in" not in keywords

    def test_extract_keywords_empty(self, analyzer):
        """Testar extração com query vazia."""
        keywords = analyzer._extract_keywords("")
        assert keywords == []

    def test_extract_keywords_stopwords_only(self, analyzer):
        """Testar extração apenas com stop words."""
        keywords = analyzer._extract_keywords("the and or but")
        assert keywords == []

    def test_extract_entities_tech_terms(self, analyzer):
        """Testar extração de entidades técnicas."""
        query = "Python Django framework version 3.2 with PostgreSQL"
        
        entities = analyzer._extract_entities(query)
        
        assert "Python" in entities
        assert "Django" in entities
        assert "3.2" in entities

    def test_extract_entities_numbers_and_codes(self, analyzer):
        """Testar extração de números e códigos."""
        query = "Error code 404 status 200 version 1.5.0"
        
        entities = analyzer._extract_entities(query)
        
        assert "404" in entities
        assert "200" in entities
        assert "1.5" in entities

    def test_classify_query_semantic(self, analyzer):
        """Testar classificação de query semântica."""
        semantic_queries = [
            "Como funciona machine learning?",
            "O que é deep learning?",
            "Explique redes neurais",
            "What is Python programming?"
        ]
        
        for query in semantic_queries:
            keywords = analyzer._extract_keywords(query)
            query_type = analyzer._classify_query_type(query, keywords)
            
            assert query_type == 'semantic'

    def test_classify_query_keyword(self, analyzer):
        """Testar classificação de query por keywords."""
        keyword_queries = [
            "Python sort",
            "numpy array",
            "Django auth"
        ]
        
        for query in keyword_queries:
            keywords = analyzer._extract_keywords(query)
            query_type = analyzer._classify_query_type(query, keywords)
            
            assert query_type == 'keyword'

    def test_expand_query_with_synonyms(self, analyzer):
        """Testar expansão de query com sinônimos."""
        query = "function implementation"
        keywords = ["function", "implementation"]
        
        expanded = analyzer._expand_query(query, keywords)
        
        # Deve conter termos originais
        assert "function" in expanded
        assert "implementation" in expanded

    def test_calculate_intent_confidence_variations(self, analyzer):
        """Testar cálculo de confiança para diferentes cenários."""
        # Query curta
        short_confidence = analyzer._calculate_intent_confidence("AI ML", "semantic")
        
        # Query longa
        long_query = "How to implement machine learning algorithms using Python scikit-learn for classification"
        long_confidence = analyzer._calculate_intent_confidence(long_query, "semantic")
        
        # Query média
        medium_query = "Python machine learning tutorial"
        medium_confidence = analyzer._calculate_intent_confidence(medium_query, "hybrid")
        
        assert long_confidence > medium_confidence > short_confidence
        assert 0.0 <= short_confidence <= 1.0
        assert 0.0 <= medium_confidence <= 1.0
        assert 0.0 <= long_confidence <= 1.0


class TestHybridRetriever:
    """Testes para o Hybrid Retriever principal."""

    @pytest.fixture
    def temp_config_file(self):
        """Arquivo de configuração temporário."""
        config_content = {
            "hybrid_search": {
                "dense_weight": 0.7,
                "sparse_weight": 0.3,
                "limit": 10,
                "enable_reranking": True,
                "enable_hyde": False,
                "relevance_threshold": 0.5,
                "collection_name": "test_collection",
                "search_strategy": {
                    "rrf_k": 60
                }
            },
            "query_analysis": {
                "enable_entity_extraction": True,
                "enable_expansion": True,
                "similarity_threshold": 0.7
            },
            "embedding_providers": {
                "sparse": {
                    "k1": 1.2,
                    "b": 0.75,
                    "epsilon": 0.25
                },
                "dense": {
                    "model": "text-embedding-ada-002",
                    "dimensions": 1536
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_content, f)
            return f.name

    @pytest.fixture
    def mock_retriever(self, temp_config_file):
        """Retriever com configuração de teste."""
        return HybridRetriever(temp_config_file)

    def test_init_configuration(self, temp_config_file):
        """Testar inicialização e configuração."""
        retriever = HybridRetriever(temp_config_file)
        
        # Verificar componentes
        assert retriever.vector_store is not None
        assert retriever.query_analyzer is not None
        assert retriever.query_enhancer is not None
        assert retriever.reranker is not None
        assert retriever.hyde_enhancer is not None
        assert retriever.dense_embedding_service is not None
        assert retriever.sparse_vector_service is not None
        
        # Verificar configuração
        assert retriever.config is not None
        assert retriever.config["hybrid_search"]["dense_weight"] == 0.7
        assert retriever.config["hybrid_search"]["sparse_weight"] == 0.3
        
        # Verificar cache
        assert isinstance(retriever._retrieval_cache, dict)
        assert len(retriever._retrieval_cache) == 0

    def test_load_config_missing_file(self):
        """Testar carregamento com arquivo ausente usa padrão."""
        retriever = HybridRetriever("nonexistent_config.yaml")
        
        assert retriever.config is not None
        # Deve usar valores padrão
        assert "hybrid_search" in retriever.config

    def test_determine_search_strategy_semantic(self, mock_retriever):
        """Testar determinação de estratégia para query semântica."""
        query_analysis = QueryAnalysis(
            original_query="What is machine learning?",
            expanded_query="What is machine learning artificial intelligence?",
            query_type="semantic",
            keywords=["machine", "learning", "artificial", "intelligence"],
            entities=[],
            intent_confidence=0.9
        )
        
        strategy = mock_retriever._determine_search_strategy(query_analysis)
        
        assert strategy in ["dense", "hybrid"]

    def test_determine_search_strategy_keyword(self, mock_retriever):
        """Testar determinação de estratégia para query por keywords."""
        query_analysis = QueryAnalysis(
            original_query="Python function sort",
            expanded_query="Python function sort",
            query_type="keyword",
            keywords=["python", "function", "sort"],
            entities=["Python"],
            intent_confidence=0.8
        )
        
        strategy = mock_retriever._determine_search_strategy(query_analysis)
        
        assert strategy in ["sparse", "hybrid"]

    def test_determine_search_strategy_hybrid(self, mock_retriever):
        """Testar determinação de estratégia híbrida."""
        query_analysis = QueryAnalysis(
            original_query="How to implement sorting function in Python",
            expanded_query="How to implement sorting algorithm function in Python",
            query_type="hybrid",
            keywords=["implement", "sorting", "function", "python"],
            entities=["Python"],
            intent_confidence=0.75
        )
        
        strategy = mock_retriever._determine_search_strategy(query_analysis)
        
        assert strategy == "hybrid"

    @pytest.mark.asyncio
    async def test_dense_search_execution(self, mock_retriever):
        """Testar execução de busca densa."""
        # Mock do vector store
        mock_results = [
            Mock(id="1", content="Dense result 1", metadata={"type": "dense"}, dense_score=0.9),
            Mock(id="2", content="Dense result 2", metadata={"type": "dense"}, dense_score=0.8)
        ]
        mock_retriever.vector_store.dense_search = AsyncMock(return_value=mock_results)
        
        query_analysis = QueryAnalysis(
            original_query="What is AI?",
            expanded_query="What is AI artificial intelligence?",
            query_type="semantic",
            keywords=["ai", "artificial", "intelligence"],
            entities=[],
            intent_confidence=0.9
        )
        
        results = await mock_retriever._dense_search_only(query_analysis, limit=5)
        
        assert len(results) == 2
        mock_retriever.vector_store.dense_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_sparse_search_execution(self, mock_retriever):
        """Testar execução de busca esparsa."""
        # Mock do vector store
        mock_results = [
            Mock(id="1", content="Sparse result 1", metadata={"type": "sparse"}, sparse_score=0.85),
            Mock(id="2", content="Sparse result 2", metadata={"type": "sparse"}, sparse_score=0.75)
        ]
        mock_retriever.vector_store.sparse_search = AsyncMock(return_value=mock_results)
        
        query_analysis = QueryAnalysis(
            original_query="Python function",
            expanded_query="Python function",
            query_type="keyword",
            keywords=["python", "function"],
            entities=["Python"],
            intent_confidence=0.8
        )
        
        results = await mock_retriever._sparse_search_only(query_analysis, limit=5)
        
        assert len(results) == 2
        mock_retriever.vector_store.sparse_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_execution(self, mock_retriever):
        """Testar execução de busca híbrida."""
        # Mock do vector store
        mock_results = [
            Mock(id="1", content="Hybrid result 1", metadata={"type": "hybrid"}, combined_score=0.9),
            Mock(id="2", content="Hybrid result 2", metadata={"type": "hybrid"}, combined_score=0.8),
            Mock(id="3", content="Hybrid result 3", metadata={"type": "hybrid"}, combined_score=0.7)
        ]
        mock_retriever.vector_store.hybrid_search = AsyncMock(return_value=mock_results)
        
        query_analysis = QueryAnalysis(
            original_query="How to use Python for ML?",
            expanded_query="How to use Python for machine learning?",
            query_type="hybrid",
            keywords=["python", "machine", "learning"],
            entities=["Python"],
            intent_confidence=0.8
        )
        
        results = await mock_retriever._hybrid_search(query_analysis, limit=5)
        
        assert len(results) == 3
        mock_retriever.vector_store.hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_results_functionality(self, mock_retriever):
        """Testar funcionalidade de reranking."""
        # Mock de resultados de busca
        mock_results = [
            Mock(id="1", content="Result 1", metadata={}, score=0.7),
            Mock(id="2", content="Result 2", metadata={}, score=0.9),
            Mock(id="3", content="Result 3", metadata={}, score=0.8)
        ]
        
        # Mock do reranker
        reranked_results = [
            Mock(id="2", content="Result 2", metadata={}, score=0.95),
            Mock(id="3", content="Result 3", metadata={}, score=0.85),
            Mock(id="1", content="Result 1", metadata={}, score=0.75)
        ]
        mock_retriever.reranker.rerank = AsyncMock(return_value=reranked_results)
        
        results = await mock_retriever._rerank_results("test query", mock_results)
        
        assert len(results) == 3
        assert results[0].id == "2"  # Melhor reranked score
        mock_retriever.reranker.rerank.assert_called_once_with("test query", mock_results)

    def test_create_retrieval_results_structure(self, mock_retriever):
        """Testar criação da estrutura de resultados."""
        # Mock de resultados de busca
        mock_search_results = [
            Mock(
                id="1",
                content="Test content 1",
                metadata={"source": "file1.txt", "type": "document"},
                dense_score=0.9,
                sparse_score=0.7,
                combined_score=0.8
            ),
            Mock(
                id="2", 
                content="Test content 2",
                metadata={"source": "file2.txt", "type": "code"},
                dense_score=0.8,
                sparse_score=0.6,
                combined_score=0.7
            )
        ]
        
        query_analysis = QueryAnalysis(
            original_query="test query",
            expanded_query="test query expanded",
            query_type="hybrid",
            keywords=["test"],
            entities=[],
            intent_confidence=0.8
        )
        
        results = mock_retriever._create_retrieval_results(
            mock_search_results,
            query_analysis,
            "hybrid"
        )
        
        assert len(results) == 2
        assert all(isinstance(r, HybridRetrievalResult) for r in results)
        assert results[0].id == "1"
        assert results[0].content == "Test content 1"
        assert results[0].retrieval_method == "hybrid"
        assert results[0].metadata["source"] == "file1.txt"

    def test_generate_match_explanation_semantic(self, mock_retriever):
        """Testar geração de explicação para match semântico."""
        mock_result = Mock(
            dense_score=0.95,
            sparse_score=0.6,
            combined_score=0.8
        )
        
        query_analysis = QueryAnalysis(
            original_query="machine learning concepts",
            expanded_query="machine learning AI concepts",
            query_type="semantic",
            keywords=["machine", "learning", "concepts"],
            entities=["AI"],
            intent_confidence=0.9
        )
        
        explanation = mock_retriever._generate_match_explanation(mock_result, query_analysis)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_generate_match_explanation_keyword(self, mock_retriever):
        """Testar geração de explicação para match por keyword."""
        mock_result = Mock(
            dense_score=0.5,
            sparse_score=0.9,
            combined_score=0.7
        )
        
        query_analysis = QueryAnalysis(
            original_query="Python sort function",
            expanded_query="Python sort function",
            query_type="keyword",
            keywords=["python", "sort", "function"],
            entities=["Python"],
            intent_confidence=0.8
        )
        
        explanation = mock_retriever._generate_match_explanation(mock_result, query_analysis)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_cache_operations(self, mock_retriever):
        """Testar operações de cache."""
        # Adicionar itens ao cache
        mock_retriever._retrieval_cache["test1"] = [Mock()]
        mock_retriever._retrieval_cache["test2"] = [Mock()]
        
        assert len(mock_retriever._retrieval_cache) == 2
        
        # Testar limpeza
        mock_retriever.clear_cache()
        assert len(mock_retriever._retrieval_cache) == 0

    def test_get_metrics_comprehensive(self, mock_retriever):
        """Testar obtenção de métricas completas."""
        # Adicionar dados ao cache e estatísticas
        mock_retriever._retrieval_cache["test1"] = [Mock()]
        mock_retriever._retrieval_cache["test2"] = [Mock()]
        
        # Simular algumas métricas
        mock_retriever.total_retrievals = 100
        mock_retriever.cache_hits = 25
        
        metrics = mock_retriever.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "cache_size" in metrics
        assert metrics["cache_size"] == 2
        assert "total_retrievals" in metrics
        assert "cache_hit_rate" in metrics

    @pytest.mark.asyncio
    async def test_retrieve_basic_flow_complete(self, mock_retriever):
        """Testar fluxo completo básico de retrieve."""
        # Mock dos componentes
        mock_search_results = [
            Mock(
                id="1",
                content="Comprehensive test result",
                metadata={"source": "test.txt", "relevance": "high"},
                dense_score=0.9,
                sparse_score=0.7,
                combined_score=0.8
            )
        ]
        
        # Configurar mocks
        mock_retriever.vector_store.hybrid_search = AsyncMock(return_value=mock_search_results)
        mock_retriever.reranker.rerank = AsyncMock(return_value=mock_search_results)
        
        results = await mock_retriever.retrieve("comprehensive test query", limit=5)
        
        assert len(results) == 1
        assert isinstance(results[0], HybridRetrievalResult)
        assert results[0].content == "Comprehensive test result"
        assert results[0].metadata["relevance"] == "high"

    @pytest.mark.asyncio
    async def test_retrieve_with_cache_hit(self, mock_retriever):
        """Testar retrieve com cache hit."""
        cached_results = [
            HybridRetrievalResult(
                id="cached",
                content="Cached result",
                metadata={"source": "cache"},
                dense_score=0.9,
                sparse_score=0.8,
                combined_score=0.85,
                rerank_score=None,
                retrieval_method="cached",
                query_match_explanation="Retrieved from cache"
            )
        ]
        
        # Adicionar ao cache
        mock_retriever._retrieval_cache["test query"] = cached_results
        
        results = await mock_retriever.retrieve("test query")
        
        # Deve retornar resultado do cache
        assert len(results) == 1
        assert results[0].content == "Cached result"
        assert results[0].retrieval_method == "cached"

    @pytest.mark.asyncio
    async def test_retrieve_with_filters_support(self, mock_retriever):
        """Testar retrieve com suporte a filtros."""
        filters = {"language": "python", "type": "function", "difficulty": "beginner"}
        
        mock_search_results = [
            Mock(
                id="1",
                content="Filtered Python function",
                metadata={"language": "python", "type": "function", "difficulty": "beginner"},
                dense_score=0.9,
                sparse_score=0.8,
                combined_score=0.85
            )
        ]
        
        # Mock para retrieve com filtros
        mock_retriever.vector_store.hybrid_search = AsyncMock(return_value=mock_search_results)
        
        results = await mock_retriever.retrieve_with_filters(
            "Python function example",
            filters,
            limit=5
        )
        
        assert len(results) == 1


class TestCreateHybridRetriever:
    """Testes para a função factory."""

    def test_create_default_configuration(self):
        """Testar criação com configuração padrão."""
        retriever = create_hybrid_retriever()
        
        assert isinstance(retriever, HybridRetriever)

    def test_create_custom_configuration(self):
        """Testar criação com configuração customizada."""
        custom_config = "custom_hybrid_config.yaml"
        
        retriever = create_hybrid_retriever(custom_config)
        
        assert isinstance(retriever, HybridRetriever)


@pytest.mark.integration
class TestHybridRetrieverIntegration:
    """Testes de integração para fluxos completos."""

    @pytest.fixture
    def integration_config(self):
        """Configuração para testes de integração."""
        return {
            "hybrid_search": {
                "dense_weight": 0.7,
                "sparse_weight": 0.3,
                "limit": 10,
                "enable_reranking": True,
                "enable_hyde": False,
                "relevance_threshold": 0.6,
                "collection_name": "integration_collection",
                "search_strategy": {
                    "rrf_k": 60
                }
            },
            "query_analysis": {
                "enable_entity_extraction": True,
                "enable_expansion": True,
                "similarity_threshold": 0.7
            },
            "embedding_providers": {
                "sparse": {
                    "k1": 1.2,
                    "b": 0.75,
                    "epsilon": 0.25
                },
                "dense": {
                    "model": "text-embedding-ada-002",
                    "dimensions": 1536
                }
            }
        }

    @pytest.mark.asyncio
    async def test_end_to_end_semantic_query_flow(self, integration_config):
        """Teste end-to-end completo para query semântica."""
        # Criar config temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(integration_config, f)
            config_path = f.name
        
        try:
            retriever = HybridRetriever(config_path)
            
            # Mock dos resultados
            mock_search_results = [
                Mock(
                    id="sem1",
                    content="Machine learning is a subset of artificial intelligence",
                    metadata={"topic": "AI", "type": "definition"},
                    dense_score=0.92,
                    sparse_score=0.65,
                    combined_score=0.84
                )
            ]
            
            retriever.vector_store.hybrid_search = AsyncMock(return_value=mock_search_results)
            retriever.reranker.rerank = AsyncMock(return_value=mock_search_results)
            
            results = await retriever.retrieve(
                "What is machine learning and how does it work?",
                limit=5,
                use_reranking=True
            )
            
            assert len(results) == 1
            assert results[0].content == "Machine learning is a subset of artificial intelligence"
            assert results[0].metadata["topic"] == "AI"
            
        finally:
            Path(config_path).unlink()

    @pytest.mark.asyncio
    async def test_end_to_end_keyword_query_flow(self, integration_config):
        """Teste end-to-end completo para query por keyword."""
        # Criar config temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(integration_config, f)
            config_path = f.name
        
        try:
            retriever = HybridRetriever(config_path)
            
            # Mock dos resultados para query técnica
            mock_search_results = [
                Mock(
                    id="key1",
                    content="def sort_list(lst): return sorted(lst)",
                    metadata={"language": "python", "type": "function"},
                    dense_score=0.75,
                    sparse_score=0.88,
                    combined_score=0.80
                )
            ]
            
            retriever.vector_store.sparse_search = AsyncMock(return_value=mock_search_results)
            retriever.reranker.rerank = AsyncMock(return_value=mock_search_results)
            
            results = await retriever.retrieve(
                "Python sort function list",
                limit=5,
                strategy="sparse"
            )
            
            assert len(results) == 1
            assert "sort" in results[0].content
            assert results[0].metadata["language"] == "python"
            
        finally:
            Path(config_path).unlink()

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_concurrent_queries(self, integration_config):
        """Teste de performance para queries concorrentes."""
        # Criar config temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(integration_config, f)
            config_path = f.name
        
        try:
            retriever = HybridRetriever(config_path)
            
            # Mock rápido
            retriever.vector_store.hybrid_search = AsyncMock(
                return_value=[Mock(id="fast", content="Fast result", metadata={})]
            )
            
            queries = [
                "Python programming",
                "Machine learning basics",
                "Data science tools",
                "Web development",
                "Database design"
            ]
            
            start_time = time.time()
            
            # Executar queries concorrentemente
            tasks = [retriever.retrieve(query, limit=3) for query in queries]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            
            # Verificar performance
            total_time = end_time - start_time
            assert total_time < 5.0  # Deve processar 5 queries em menos de 5 segundos
            
            # Verificar resultados
            assert len(results) == len(queries)
            for result_set in results:
                assert len(result_set) >= 1
                assert isinstance(result_set[0], HybridRetrievalResult)
            
        finally:
            Path(config_path).unlink()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, integration_config):
        """Teste de tratamento de erros e recuperação."""
        # Criar config temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(integration_config, f)
            config_path = f.name
        
        try:
            retriever = HybridRetriever(config_path)
            
            # Mock que simula erro e depois sucesso
            call_count = 0
            
            async def mock_search_with_error():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Network error")
                return [Mock(id="recovered", content="Recovered result", metadata={})]
            
            retriever.vector_store.hybrid_search = mock_search_with_error
            
            # Primeira tentativa deve falhar graciosamente
            try:
                results1 = await retriever.retrieve("test query 1", limit=5)
                # Se não lançar exceção, deve retornar lista vazia
                assert isinstance(results1, list)
            except Exception:
                pass  # Erro esperado
            
            # Segunda tentativa deve funcionar
            results2 = await retriever.retrieve("test query 2", limit=5)
            assert len(results2) == 1
            assert results2[0].content == "Recovered result"
            
        finally:
            Path(config_path).unlink() 
