"""
Testes para o Corrective RAG.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.retrieval.corrective_rag import CorrectiveRAG, DocumentWithScore


class TestCorrectiveRAG:
    
    @pytest.fixture
    def mock_retriever(self):
        """Mock do retriever."""
        retriever = Mock()
        retriever.retrieve = AsyncMock()
        return retriever
    
    @pytest.fixture
    def corrective_rag(self, mock_retriever):
        """Instância do CorrectiveRAG para testes."""
        return CorrectiveRAG(retriever=mock_retriever)
    
    @pytest.mark.asyncio
    async def test_retrieve_and_correct_high_relevance(self, corrective_rag, mock_retriever):
        """Testa quando documentos têm alta relevância (sem correção)."""
        # Mock documentos com alta relevância
        mock_retriever.retrieve.return_value = [
            {
                "content": "Machine learning é um campo da IA...",
                "metadata": {"source": "ml_guide.pdf"},
                "score": 0.9
            }
        ]
        
        # Mock avaliação de relevância
        with patch.object(corrective_rag, '_evaluate_relevance') as mock_eval:
            mock_eval.return_value = [
                DocumentWithScore(
                    content="Machine learning é um campo da IA...",
                    metadata={"source": "ml_guide.pdf"},
                    relevance_score=0.85,
                    validation_status="relevant"
                )
            ]
            
            result = await corrective_rag.retrieve_and_correct("O que é machine learning?")
            
            # Verificações
            assert result["correction_applied"] is False
            assert result["reformulation_count"] == 0
            assert len(result["documents"]) > 0
            assert result["avg_relevance_score"] >= 0.7
    
    @pytest.mark.asyncio
    async def test_retrieve_and_correct_low_relevance(self, corrective_rag, mock_retriever):
        """Testa quando documentos têm baixa relevância (com correção)."""
        # Mock documentos com baixa relevância
        mock_retriever.retrieve.side_effect = [
            # Primeira tentativa - baixa relevância
            [
                {
                    "content": "Python é uma linguagem...",
                    "metadata": {"source": "python.pdf"},
                    "score": 0.3
                }
            ],
            # Segunda tentativa após reformulação - alta relevância
            [
                {
                    "content": "Machine learning com Python...",
                    "metadata": {"source": "ml_python.pdf"},
                    "score": 0.8
                }
            ]
        ]
        
        # Mock avaliação e reformulação
        with patch.object(corrective_rag, '_evaluate_relevance') as mock_eval:
            mock_eval.side_effect = [
                # Primeira avaliação - baixa relevância
                [DocumentWithScore(
                    content="Python é uma linguagem...",
                    metadata={"source": "python.pdf"},
                    relevance_score=0.3,
                    validation_status="irrelevant"
                )],
                # Segunda avaliação - alta relevância
                [DocumentWithScore(
                    content="Machine learning com Python...",
                    metadata={"source": "ml_python.pdf"},
                    relevance_score=0.8,
                    validation_status="relevant"
                )]
            ]
            
            with patch.object(corrective_rag, '_reformulate_query') as mock_reform:
                mock_reform.return_value = "Machine learning implementação Python exemplos"
                
                result = await corrective_rag.retrieve_and_correct("O que é ML?")
                
                # Verificações
                assert result["correction_applied"] is True
                assert result["reformulation_count"] == 1
                assert mock_reform.called
                assert result["avg_relevance_score"] >= 0.7
    
    @pytest.mark.asyncio
    async def test_parse_relevance_score(self, corrective_rag):
        """Testa parsing de scores de relevância."""
        # Teste com formato padrão
        assert corrective_rag._parse_relevance_score("Score: 0.8") == 0.8
        assert corrective_rag._parse_relevance_score("Score: 1.0") == 1.0
        
        # Teste com decimal solto
        assert corrective_rag._parse_relevance_score("A relevância é 0.65") == 0.65
        
        # Teste com keywords
        assert corrective_rag._parse_relevance_score("Muito relevante") >= 0.8
        assert corrective_rag._parse_relevance_score("Pouco relevante") <= 0.5
        
        # Teste fallback
        assert corrective_rag._parse_relevance_score("Texto aleatório") == 0.2
    
    @pytest.mark.asyncio
    async def test_max_reformulation_attempts(self, corrective_rag, mock_retriever):
        """Testa limite de tentativas de reformulação."""
        corrective_rag.max_reformulation_attempts = 2
        
        # Mock sempre retorna baixa relevância
        mock_retriever.retrieve.return_value = [
            {
                "content": "Conteúdo irrelevante",
                "metadata": {},
                "score": 0.2
            }
        ]
        
        with patch.object(corrective_rag, '_evaluate_relevance') as mock_eval:
            mock_eval.return_value = [
                DocumentWithScore(
                    content="Conteúdo irrelevante",
                    metadata={},
                    relevance_score=0.2,
                    validation_status="irrelevant"
                )
            ]
            
            with patch.object(corrective_rag, '_reformulate_query') as mock_reform:
                mock_reform.return_value = "Query reformulada"
                
                result = await corrective_rag.retrieve_and_correct("Query original")
                
                # Deve parar após max_attempts
                assert result["reformulation_count"] == 2
                assert mock_reform.call_count == 2 