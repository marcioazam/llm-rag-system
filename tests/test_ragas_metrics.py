"""
Testes para o sistema RAGAS Metrics - Avaliação de qualidade RAG
Cobertura completa de métricas, fact score, BERT score e detecção de alucinações
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Imports do sistema RAGAS
from src.monitoring.ragas_metrics import (
    RAGASMetrics,
    FactScoreEvaluator,
    BERTScoreEvaluator,
    HallucinationDetector,
    RAGASEvaluationSystem,
    create_ragas_evaluator
)


class TestRAGASMetrics:
    """Testes para a classe RAGASMetrics (dataclass)."""
    
    def test_ragas_metrics_initialization(self):
        """Test de inicialização do RAGASMetrics."""
        metrics = RAGASMetrics(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.85,
            fact_score=0.75,
            bert_score={"precision": 0.8, "recall": 0.85, "f1": 0.82},
            hallucination_score=0.1,
            semantic_similarity=0.88,
            coherence_score=0.92,
            completeness_score=0.78
        )
        
        assert metrics.faithfulness == 0.8
        assert metrics.answer_relevancy == 0.9
        assert metrics.context_precision == 0.7
        assert metrics.context_recall == 0.85
        assert metrics.fact_score == 0.75
        assert metrics.bert_score["f1"] == 0.82
        assert metrics.hallucination_score == 0.1
        assert metrics.semantic_similarity == 0.88
        assert metrics.coherence_score == 0.92
        assert metrics.completeness_score == 0.78
        assert isinstance(metrics.timestamp, datetime)
        assert isinstance(metrics.metadata, dict)


class TestFactScoreEvaluator:
    """Testes para o avaliador de precisão factual."""
    
    @pytest.fixture
    def fact_evaluator(self):
        """Instância do FactScoreEvaluator para testes."""
        return FactScoreEvaluator(knowledge_base={"Python": "linguagem"})
    
    @pytest.mark.asyncio
    async def test_evaluate_fact_score_basic(self, fact_evaluator):
        """Test básico de avaliação de fact score."""
        response = "Python é uma linguagem de programação popular."
        context = "Python é uma linguagem de programação interpretada."
        
        fact_score, details = await fact_evaluator.evaluate_fact_score(response, context)
        
        assert isinstance(fact_score, float)
        assert 0 <= fact_score <= 1
        assert isinstance(details, dict)
        
        # Verificar estrutura dos detalhes
        assert "total_facts" in details
        assert "verified" in details
        assert "unverified" in details
        assert "contradicted" in details
    
    @pytest.mark.asyncio
    async def test_evaluate_fact_score_no_facts(self, fact_evaluator):
        """Test com resposta sem fatos específicos."""
        response = "Olá, como posso ajudar?"
        context = "Sistema de ajuda"
        
        fact_score, details = await fact_evaluator.evaluate_fact_score(response, context)
        
        assert fact_score == 1.0
        assert "message" in details
    
    def test_text_similarity(self, fact_evaluator):
        """Test de cálculo de similaridade de texto."""
        text1 = "Python é uma linguagem"
        text2 = "Python é linguagem"
        
        similarity = fact_evaluator._text_similarity(text1, text2)
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
    
    def test_extract_facts_fallback(self, fact_evaluator):
        """Test de extração de fatos com fallback (sem spacy)."""
        # Forçar uso do fallback
        fact_evaluator.nlp = None
        
        text = "Python é uma linguagem de programação. Java foi criado pela Sun."
        facts = fact_evaluator._extract_facts(text)
        
        assert isinstance(facts, list)
        assert len(facts) >= 1  # Deve encontrar pelo menos "Python é"


class TestBERTScoreEvaluator:
    """Testes para o avaliador BERT Score."""
    
    @pytest.fixture
    def bert_evaluator(self):
        """Mock do BERTScoreEvaluator."""
        evaluator = Mock(spec=BERTScoreEvaluator)
        
        # Mock compute_bert_score
        evaluator.compute_bert_score = Mock(return_value={
            "precision": 0.85,
            "recall": 0.82,
            "f1": 0.835
        })
        
        # Mock batch_bert_score
        evaluator.batch_bert_score = Mock(return_value=[
            {"precision": 0.85, "recall": 0.82, "f1": 0.835},
            {"precision": 0.90, "recall": 0.88, "f1": 0.89}
        ])
        
        return evaluator
    
    def test_compute_bert_score(self, bert_evaluator):
        """Test de cálculo de BERT Score."""
        candidate = "Python é uma linguagem de programação"
        reference = "Python é linguagem de programação"
        
        score = bert_evaluator.compute_bert_score(candidate, reference)
        
        assert isinstance(score, dict)
        assert "precision" in score
        assert "recall" in score
        assert "f1" in score
        assert 0 <= score["f1"] <= 1
    
    def test_batch_bert_score(self, bert_evaluator):
        """Test de BERT Score em lote."""
        candidates = ["Texto 1", "Texto 2"]
        references = ["Referência 1", "Referência 2"]
        
        scores = bert_evaluator.batch_bert_score(candidates, references)
        
        assert isinstance(scores, list)
        assert len(scores) == 2
        for score in scores:
            assert "f1" in score


class TestHallucinationDetector:
    """Testes para o detector de alucinações."""
    
    @pytest.fixture
    def mock_fact_evaluator(self):
        """Mock do FactScoreEvaluator."""
        evaluator = Mock()
        evaluator.evaluate_fact_score = AsyncMock(return_value=(0.8, {}))
        return evaluator
    
    @pytest.fixture
    def mock_bert_evaluator(self):
        """Mock do BERTScoreEvaluator."""
        evaluator = Mock()
        evaluator.compute_bert_score = Mock(return_value={"f1": 0.85})
        return evaluator
    
    @pytest.fixture
    def hallucination_detector(self, mock_fact_evaluator, mock_bert_evaluator):
        """Instância do HallucinationDetector."""
        return HallucinationDetector(
            fact_evaluator=mock_fact_evaluator,
            bert_evaluator=mock_bert_evaluator
        )
    
    @pytest.mark.asyncio
    async def test_detect_hallucination(self, hallucination_detector):
        """Test de detecção de alucinações."""
        response = "Python é uma linguagem de programação criada em 1991"
        context = "Python é uma linguagem de programação interpretada"
        query = "Quando Python foi criado?"
        
        hallucination_score, details = await hallucination_detector.detect_hallucination(
            response, context, query
        )
        
        assert isinstance(hallucination_score, float)
        assert 0 <= hallucination_score <= 1
        assert isinstance(details, dict)
        assert "classification" in details
        assert "signals" in details
    
    def test_classify_hallucination(self, hallucination_detector):
        """Test de classificação de alucinação."""
        assert hallucination_detector._classify_hallucination(0.1) == "very_low"
        assert hallucination_detector._classify_hallucination(0.3) == "low"
        assert hallucination_detector._classify_hallucination(0.8) == "very_high"


class TestRAGASEvaluationSystem:
    """Testes para o sistema completo de avaliação RAGAS."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock do serviço LLM."""
        service = Mock()
        service.generate = AsyncMock(return_value="Resposta gerada")
        service.agenerate = AsyncMock(return_value=Mock(
            generations=[[Mock(text="Resposta gerada")]]
        ))
        return service
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock do serviço de embedding."""
        service = Mock()
        service.aembed_query = AsyncMock(return_value=np.random.randn(768).tolist())
        return service
    
    @pytest.fixture
    def ragas_system(self, mock_llm_service, mock_embedding_service):
        """Instância do RAGASEvaluationSystem para testes."""
        return RAGASEvaluationSystem(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service
        )
    
    def test_ragas_system_initialization(self, ragas_system):
        """Test de inicialização do sistema RAGAS."""
        assert ragas_system.llm is not None
        assert isinstance(ragas_system.evaluation_history, type(ragas_system.evaluation_history))
        assert isinstance(ragas_system.aggregate_stats, dict)
        assert ragas_system.fact_evaluator is not None
        assert ragas_system.bert_evaluator is not None
        assert ragas_system.hallucination_detector is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_rag_response(self, ragas_system):
        """Test de avaliação completa de resposta RAG."""
        # Mock dos métodos de avaliação
        ragas_system._evaluate_faithfulness = AsyncMock(return_value=0.8)
        ragas_system._evaluate_answer_relevancy = AsyncMock(return_value=0.9)
        ragas_system._evaluate_context_quality = AsyncMock(return_value=(0.85, 0.82))
        ragas_system._evaluate_semantic_similarity = AsyncMock(return_value=0.88)
        ragas_system._evaluate_coherence = AsyncMock(return_value=0.92)
        ragas_system._evaluate_completeness = AsyncMock(return_value=0.78)
        
        # Mock fact score e BERT score
        ragas_system.fact_evaluator.evaluate_fact_score = AsyncMock(return_value=(0.85, {}))
        ragas_system.bert_evaluator.compute_bert_score = Mock(return_value={
            "precision": 0.8, "recall": 0.85, "f1": 0.82
        })
        ragas_system.hallucination_detector.detect_hallucination = AsyncMock(return_value=(0.1, {}))
        
        query = "O que é Python?"
        response = "Python é uma linguagem de programação interpretada"
        context = "Python é uma linguagem popular para desenvolvimento"
        
        metrics = await ragas_system.evaluate_rag_response(
            query=query,
            response=response,
            context=context
        )
        
        assert isinstance(metrics, RAGASMetrics)
        assert 0 <= metrics.faithfulness <= 1
        assert 0 <= metrics.answer_relevancy <= 1
        assert 0 <= metrics.context_precision <= 1
        assert 0 <= metrics.context_recall <= 1
        assert 0 <= metrics.fact_score <= 1
        assert isinstance(metrics.bert_score, dict)
        assert 0 <= metrics.hallucination_score <= 1
    
    def test_get_evaluation_report(self, ragas_system):
        """Test de geração de relatório de avaliação."""
        # Adicionar algumas métricas ao histórico
        test_metrics = RAGASMetrics(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.85,
            fact_score=0.75,
            bert_score={"precision": 0.8, "recall": 0.85, "f1": 0.82},
            hallucination_score=0.1,
            semantic_similarity=0.88,
            coherence_score=0.92,
            completeness_score=0.78
        )
        
        ragas_system.evaluation_history.append(test_metrics)
        ragas_system._update_aggregate_stats(test_metrics)
        
        report = ragas_system.get_evaluation_report()
        
        assert isinstance(report, dict)
        assert "aggregate_stats" in report
        assert "recent_performance" in report
        assert "quality_distribution" in report
        assert "component_stats" in report
    
    def test_get_recommendations(self, ragas_system):
        """Test de geração de recomendações."""
        # Criar métricas com problemas específicos
        low_faithfulness_metrics = RAGASMetrics(
            faithfulness=0.5,  # Baixa fidelidade
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.85,
            fact_score=0.75,
            bert_score={"precision": 0.8, "recall": 0.85, "f1": 0.82},
            hallucination_score=0.1,
            semantic_similarity=0.88,
            coherence_score=0.92,
            completeness_score=0.78
        )
        
        recommendations = ragas_system.get_recommendations(low_faithfulness_metrics)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Deve sugerir melhoria na fidelidade
        assert any("fidelidade" in rec.lower() or "contexto" in rec.lower() 
                  for rec in recommendations)


class TestCreateRAGASEvaluator:
    """Testes para a função factory."""
    
    def test_create_ragas_evaluator_basic(self):
        """Test de criação básica do avaliador."""
        evaluator = create_ragas_evaluator()
        
        assert isinstance(evaluator, RAGASEvaluationSystem)
        assert evaluator.fact_evaluator is not None
        assert evaluator.bert_evaluator is not None
        assert evaluator.hallucination_detector is not None
    
    def test_create_ragas_evaluator_with_services(self):
        """Test de criação com serviços customizados."""
        mock_llm = Mock()
        mock_embedding = Mock()
        mock_kb = Mock()
        
        evaluator = create_ragas_evaluator(
            llm_service=mock_llm,
            embedding_service=mock_embedding,
            knowledge_base=mock_kb
        )
        
        assert evaluator.llm == mock_llm
        assert evaluator.embeddings == mock_embedding


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 