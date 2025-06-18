"""Testes para o módulo ab_test.py."""
import os
import pytest
from unittest.mock import patch

from src.ab_test import decide_variant, _hash_to_prob


class TestABTest:
    """Testes para funcionalidade de A/B testing."""
    
    def test_hash_to_prob_consistency(self):
        """Testa se a função de hash retorna valores consistentes."""
        # Mesmo input deve retornar mesmo resultado
        query = "test query"
        prob1 = _hash_to_prob(query)
        prob2 = _hash_to_prob(query)
        assert prob1 == prob2
        
        # Resultado deve estar entre 0 e 1
        assert 0.0 <= prob1 <= 1.0
    
    def test_hash_to_prob_different_inputs(self):
        """Testa se inputs diferentes geram probabilidades diferentes."""
        prob1 = _hash_to_prob("query1")
        prob2 = _hash_to_prob("query2")
        prob3 = _hash_to_prob("completely different query")
        
        # Probabilidades devem ser diferentes para inputs diferentes
        assert prob1 != prob2
        assert prob1 != prob3
        assert prob2 != prob3
    
    def test_decide_variant_forced_with(self):
        """Testa variante forçada 'with'."""
        with patch.dict(os.environ, {'RAG_AB_TEST': 'with'}):
            result = decide_variant("any query")
            assert result == "with_prompt"
    
    def test_decide_variant_forced_no(self):
        """Testa variante forçada 'no'."""
        with patch.dict(os.environ, {'RAG_AB_TEST': 'no'}):
            result = decide_variant("any query")
            assert result == "no_prompt"
    
    def test_decide_variant_forced_invalid(self):
        """Testa que valores inválidos não forçam variante."""
        with patch.dict(os.environ, {'RAG_AB_TEST': 'invalid'}):
            # Deve usar lógica normal baseada em hash
            result = decide_variant("test query")
            assert result in ["with_prompt", "no_prompt"]
    
    def test_decide_variant_no_query_random(self):
        """Testa comportamento com query None (fallback random)."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('src.ab_test.random.random', return_value=0.3):
                result = decide_variant(None)
                assert result == "with_prompt"  # 0.3 < 0.5 (default ratio)
            
            with patch('src.ab_test.random.random', return_value=0.7):
                result = decide_variant(None)
                assert result == "no_prompt"  # 0.7 >= 0.5
    
    def test_decide_variant_with_query_deterministic(self):
        """Testa que queries específicas retornam resultados determinísticos."""
        with patch.dict(os.environ, {}, clear=True):
            # Mesmo query deve sempre retornar mesmo resultado
            query = "consistent test query"
            result1 = decide_variant(query)
            result2 = decide_variant(query)
            result3 = decide_variant(query)
            
            assert result1 == result2 == result3
            assert result1 in ["with_prompt", "no_prompt"]
    
    def test_decide_variant_custom_ratio(self):
        """Testa comportamento com ratio customizado."""
        # Ratio 0.0 - sempre no_prompt
        with patch.dict(os.environ, {'RAG_WITH_PROMPT_RATIO': '0.0'}):
            # Usar query que sabemos que gera prob baixa
            result = decide_variant("test")
            # Com ratio 0.0, sempre deve ser no_prompt
            assert result == "no_prompt"
        
        # Ratio 1.0 - sempre with_prompt
        with patch.dict(os.environ, {'RAG_WITH_PROMPT_RATIO': '1.0'}):
            result = decide_variant("test")
            # Com ratio 1.0, sempre deve ser with_prompt
            assert result == "with_prompt"
    
    def test_decide_variant_edge_cases(self):
        """Testa casos extremos."""
        with patch.dict(os.environ, {}, clear=True):
            # Query vazia
            result = decide_variant("")
            assert result in ["with_prompt", "no_prompt"]
            
            # Query muito longa
            long_query = "test " * 1000
            result = decide_variant(long_query)
            assert result in ["with_prompt", "no_prompt"]
            
            # Query com caracteres especiais
            special_query = "test!@#$%^&*()_+{}|:<>?[]\\";'./,"
            result = decide_variant(special_query)
            assert result in ["with_prompt", "no_prompt"]
    
    def test_decide_variant_unicode(self):
        """Testa comportamento com caracteres unicode."""
        with patch.dict(os.environ, {}, clear=True):
            unicode_queries = [
                "测试查询",  # Chinês
                "тестовый запрос",  # Russo
                "テストクエリ",  # Japonês
                "🚀 emoji query 🎯",  # Emojis
                "café naïve résumé",  # Acentos
            ]
            
            for query in unicode_queries:
                result = decide_variant(query)
                assert result in ["with_prompt", "no_prompt"]
                
                # Consistência
                result2 = decide_variant(query)
                assert result == result2
    
    def test_environment_variable_precedence(self):
        """Testa precedência das variáveis de ambiente."""
        # RAG_AB_TEST tem precedência sobre RAG_WITH_PROMPT_RATIO
        with patch.dict(os.environ, {
            'RAG_AB_TEST': 'with',
            'RAG_WITH_PROMPT_RATIO': '0.0'  # Normalmente forçaria no_prompt
        }):
            result = decide_variant("test")
            assert result == "with_prompt"  # RAG_AB_TEST deve ter precedência
    
    def test_hash_distribution(self):
        """Testa se a distribuição de hash é razoavelmente uniforme."""
        queries = [f"query_{i}" for i in range(1000)]
        probs = [_hash_to_prob(q) for q in queries]
        
        # Verificar se temos distribuição razoável
        low_count = sum(1 for p in probs if p < 0.5)
        high_count = sum(1 for p in probs if p >= 0.5)
        
        # Deve ter distribuição aproximadamente 50/50 (com alguma tolerância)
        assert 400 <= low_count <= 600
        assert 400 <= high_count <= 600
    
    def test_decide_variant_distribution(self):
        """Testa distribuição das variantes com ratio padrão."""
        with patch.dict(os.environ, {}, clear=True):
            queries = [f"test_query_{i}" for i in range(100)]
            results = [decide_variant(q) for q in queries]
            
            with_prompt_count = sum(1 for r in results if r == "with_prompt")
            no_prompt_count = sum(1 for r in results if r == "no_prompt")
            
            # Com ratio 0.5, deve ter distribuição aproximadamente 50/50
            assert 30 <= with_prompt_count <= 70  # Tolerância para randomness
            assert 30 <= no_prompt_count <= 70
            assert with_prompt_count + no_prompt_count == 100


class TestABTestIntegration:
    """Testes de integração para A/B testing."""
    
    def test_real_world_scenarios(self):
        """Testa cenários do mundo real."""
        real_queries = [
            "What is the capital of France?",
            "How do I implement a binary search algorithm?",
            "Explain quantum computing in simple terms",
            "What are the best practices for REST API design?",
            "How to optimize database queries?"
        ]
        
        with patch.dict(os.environ, {}, clear=True):
            for query in real_queries:
                result = decide_variant(query)
                assert result in ["with_prompt", "no_prompt"]
                
                # Consistência
                assert decide_variant(query) == result
    
    def test_performance_with_large_queries(self):
        """Testa performance com queries grandes."""
        import time
        
        large_query = "This is a very long query. " * 10000
        
        start_time = time.time()
        result = decide_variant(large_query)
        end_time = time.time()
        
        # Deve ser rápido (menos de 1 segundo)
        assert end_time - start_time < 1.0
        assert result in ["with_prompt", "no_prompt"]
    
    def test_concurrent_calls(self):
        """Testa chamadas concorrentes (simuladas)."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def worker(query_id):
            query = f"concurrent_query_{query_id}"
            result = decide_variant(query)
            results.put((query, result))
        
        # Simular 10 threads concorrentes
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verificar resultados
        collected_results = {}
        while not results.empty():
            query, result = results.get()
            collected_results[query] = result
            assert result in ["with_prompt", "no_prompt"]
        
        assert len(collected_results) == 10
        
        # Verificar consistência - mesma query deve dar mesmo resultado
        for query, result in collected_results.items():
            assert decide_variant(query) == result