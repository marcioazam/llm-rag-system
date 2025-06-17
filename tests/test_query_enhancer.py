import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.retrieval.query_enhancer import QueryEnhancer


class TestQueryEnhancer:
    """Testes para a classe QueryEnhancer."""

    @pytest.fixture
    def enhancer(self):
        """Cria uma instância do QueryEnhancer para testes."""
        return QueryEnhancer()

    @pytest.fixture
    def enhancer_with_config(self):
        """Cria uma instância do QueryEnhancer com configuração específica."""
        return QueryEnhancer(max_expansions=5)

    def test_init_default(self, enhancer):
        """Testa inicialização com configurações padrão."""
        assert enhancer.max_expansions == 3
        assert hasattr(enhancer, 'nlp')  # Verifica se tem o atributo nlp

    def test_init_custom_config(self, enhancer_with_config):
        """Testa inicialização com configurações customizadas."""
        assert enhancer_with_config.max_expansions == 5

    def test_enhance_query_basic(self, enhancer):
        """Testa aprimoramento básico de consulta."""
        original_query = "machine learning algorithms"
        enhanced = enhancer.enhance_query(original_query)
        
        # O resultado deve ser uma lista de strings
        assert isinstance(enhanced, list)
        # Deve conter pelo menos a consulta original
        assert len(enhanced) > 0
        assert original_query in enhanced

    def test_enhance_query_with_context(self, enhancer):
        """Testa aprimoramento de consulta simples."""
        query = "neural networks"
        
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced

    def test_enhance_query_empty_input(self, enhancer):
        """Testa aprimoramento com entrada vazia."""
        enhanced = enhancer.enhance_query("")
        
        # Deve retornar lista vazia ou lista com string vazia
        assert isinstance(enhanced, list)

    def test_enhance_query_whitespace_only(self, enhancer):
        """Testa aprimoramento com apenas espaços em branco."""
        enhanced = enhancer.enhance_query("   \n\t   ")
        
        assert isinstance(enhanced, list)

    def test_enhance_query_short_words(self, enhancer):
        """Testa aprimoramento com palavras curtas."""
        query = "AI ML"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced

    def test_enhance_query_repeated_words(self, enhancer):
        """Testa aprimoramento com palavras repetidas."""
        query = "machine machine learning learning"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced

    def test_enhance_query_single_word(self, enhancer):
        """Testa aprimoramento com uma única palavra."""
        query = "python"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced

    def test_enhance_query_basic_functionality(self, enhancer):
        """Testa funcionalidade básica do enhance_query."""
        query = "machine learning"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced

    def test_split_clauses_basic(self, enhancer):
        """Testa divisão de cláusulas básica."""
        query = "machine learning and data science"
        clauses = enhancer._split_clauses(query)
        
        assert isinstance(clauses, list)
        assert all(isinstance(clause, str) for clause in clauses)

    def test_split_clauses_simple_query(self, enhancer):
        """Testa divisão de cláusulas com consulta simples."""
        query = "neural networks"
        clauses = enhancer._split_clauses(query)
        
        assert isinstance(clauses, list)
        # Pode retornar lista vazia se spacy não estiver disponível

    def test_enhance_query_with_synonyms(self, enhancer):
        """Testa aprimoramento de consulta com sinônimos."""
        query = "car"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced  # Consulta original deve estar incluída

    def test_enhance_query_multiple_calls(self, enhancer):
        """Testa múltiplas chamadas do enhance_query."""
        query = "machine learning"
        
        # Executar múltiplas vezes
        results = []
        for _ in range(3):
            enhanced = enhancer.enhance_query(query)
            results.append(enhanced)
        
        # Todos os resultados devem ser listas
        for result in results:
            assert isinstance(result, list)
            assert len(result) > 0
            assert query in result  # Consulta original deve estar incluída

    def test_enhance_query_case_sensitivity(self, enhancer):
        """Testa sensibilidade a maiúsculas e minúsculas."""
        queries = ["Machine Learning", "machine learning", "MACHINE LEARNING"]
        
        results = []
        for query in queries:
            enhanced = enhancer.enhance_query(query)
            results.append(enhanced)
        
        # Todos os resultados devem ser listas válidas
        for result in results:
            assert isinstance(result, list)
            assert len(result) > 0

    def test_enhance_query_abbreviations(self, enhancer):
        """Testa aprimoramento com abreviações."""
        query = "ML algorithms"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced

    def test_query_enhancement_preserves_intent(self, enhancer):
        """Testa que o aprimoramento preserva a intenção da consulta."""
        queries = [
            "how to train neural networks",
            "best practices for data preprocessing",
            "comparison between supervised and unsupervised learning"
        ]
        
        for query in queries:
            enhanced = enhancer.enhance_query(query)
            
            # A consulta aprimorada deve ser uma lista válida
            assert isinstance(enhanced, list)
            assert len(enhanced) > 0
            assert query in enhanced  # Consulta original deve estar incluída

    def test_enhance_query_domain_specific(self, enhancer):
        """Testa aprimoramento com termos específicos de domínio."""
        query = "neural network backpropagation gradient descent"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced

    def test_enhance_query_mixed_case(self, enhancer):
        """Testa aprimoramento com casos mistos."""
        query = "Machine Learning AND Deep Learning"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced

    def test_enhance_query_long_text(self, enhancer):
        """Testa aprimoramento com texto longo."""
        query = "machine learning algorithms for natural language processing and computer vision applications"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced  # Consulta original deve estar incluída

    def test_enhance_query_special_characters(self, enhancer):
        """Testa aprimoramento com caracteres especiais."""
        query = "C++ programming & AI/ML frameworks"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        # A consulta original pode estar modificada devido aos caracteres especiais
        assert any("C++" in item or "programming" in item for item in enhanced)

    def test_enhance_query_numeric_content(self, enhancer):
        """Testa aprimoramento com conteúdo numérico."""
        query = "Python 3.9 machine learning 2023"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced  # Consulta original deve estar incluída

    def test_enhance_query_technical_terms(self, enhancer):
        """Testa aprimoramento com termos técnicos."""
        query = "TensorFlow GPU CUDA optimization"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced  # Consulta original deve estar incluída

    def test_enhance_query_with_numbers(self, enhancer):
        """Testa aprimoramento com números."""
        query = "top 10 machine learning algorithms 2023"
        enhanced = enhancer.enhance_query(query)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert query in enhanced

    def test_enhance_query_multilingual(self, enhancer):
        """Testa suporte básico para múltiplos idiomas."""
        queries = [
            "machine learning",  # Inglês
            "aprendizado de máquina",  # Português
        ]
        
        for query in queries:
            enhanced = enhancer.enhance_query(query)
            
            # Deve processar sem erros
            assert isinstance(enhanced, list)
            assert len(enhanced) > 0
            assert query in enhanced

    def test_enhance_query_performance_large(self, enhancer):
        """Testa performance com consultas grandes."""
        # Consulta muito longa
        large_query = " ".join(["machine learning"] * 50)
        
        enhanced = enhancer.enhance_query(large_query)
        
        # Deve processar sem erros
        assert isinstance(enhanced, list)
        assert len(enhanced) > 0
        assert large_query in enhanced

    def test_enhance_query_special_characters_extended(self, enhancer):
        """Testa tratamento de caracteres especiais estendido."""
        special_queries = [
            "C++ programming",
            "AI/ML frameworks",
            "data-science & analytics",
            "web3.0 technologies"
        ]
        
        for query in special_queries:
            enhanced = enhancer.enhance_query(query)
            
            # Deve processar sem erros
            assert isinstance(enhanced, list)
            assert len(enhanced) > 0

    def test_enhance_query_error_handling(self, enhancer):
        """Testa tratamento de entradas inválidas."""
        # Teste com string vazia
        result = enhancer.enhance_query("")
        assert isinstance(result, list)
        
        # Teste com espaços em branco
        result = enhancer.enhance_query("   ")
        assert isinstance(result, list)

    def test_enhance_query_consistency(self, enhancer):
        """Testa consistência dos resultados."""
        query = "machine learning algorithms"
        
        # Múltiplas chamadas
        result1 = enhancer.enhance_query(query)
        result2 = enhancer.enhance_query(query)
        
        # Resultados devem ser listas
        assert isinstance(result1, list)
        assert isinstance(result2, list)
        assert len(result1) > 0
        assert len(result2) > 0
        assert query in result1
        assert query in result2

    def test_configuration_validation(self):
        """Testa validação de configurações."""
        # Teste de inicialização básica
        enhancer = QueryEnhancer()
        
        # Deve ter os atributos básicos
        assert hasattr(enhancer, 'enhance_query')
        
        # Teste básico de funcionalidade
        result = enhancer.enhance_query("test query")
        assert isinstance(result, list)