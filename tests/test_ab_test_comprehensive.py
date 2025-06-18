"""Comprehensive tests for A/B testing functionality."""

import os
import pytest
from unittest.mock import patch

from src.ab_test import decide_variant, _hash_to_prob


class TestABTest:
    """Test suite for A/B testing functionality."""

    def test_hash_to_prob_consistency(self):
        """Test that hash function produces consistent probabilities."""
        query = "test query"
        prob1 = _hash_to_prob(query)
        prob2 = _hash_to_prob(query)
        assert prob1 == prob2, "Hash function should be deterministic"
        assert 0.0 <= prob1 <= 1.0, "Probability should be between 0 and 1"

    def test_hash_to_prob_different_queries(self):
        """Test that different queries produce different probabilities."""
        query1 = "first query"
        query2 = "second query"
        prob1 = _hash_to_prob(query1)
        prob2 = _hash_to_prob(query2)
        assert prob1 != prob2, "Different queries should produce different probabilities"

    def test_hash_to_prob_range(self):
        """Test that hash function produces values in correct range."""
        test_queries = ["query1", "query2", "query3", "query4", "query5"]
        for query in test_queries:
            prob = _hash_to_prob(query)
            assert 0.0 <= prob < 1.0, f"Probability {prob} for query '{query}' out of range"

    @patch.dict(os.environ, {"RAG_AB_TEST": "with"}, clear=False)
    def test_forced_with_variant(self):
        """Test forced 'with' variant via environment variable."""
        result = decide_variant("any query")
        assert result == "with_prompt"

    @patch.dict(os.environ, {"RAG_AB_TEST": "no"}, clear=False)
    def test_forced_no_variant(self):
        """Test forced 'no' variant via environment variable."""
        result = decide_variant("any query")
        assert result == "no_prompt"

    @patch.dict(os.environ, {"RAG_AB_TEST": "invalid"}, clear=False)
    def test_invalid_forced_variant_falls_back_to_hash(self):
        """Test that invalid forced variant falls back to hash-based decision."""
        query = "test query"
        result = decide_variant(query)
        assert result in ["with_prompt", "no_prompt"]
        
        # Should be consistent for same query
        result2 = decide_variant(query)
        assert result == result2

    @patch.dict(os.environ, {}, clear=True)
    def test_no_environment_variables(self):
        """Test behavior when no environment variables are set."""
        query = "test query"
        result = decide_variant(query)
        assert result in ["with_prompt", "no_prompt"]
        
        # Should be consistent
        result2 = decide_variant(query)
        assert result == result2

    @patch.dict(os.environ, {"RAG_WITH_PROMPT_RATIO": "0.0"}, clear=False)
    def test_zero_ratio(self):
        """Test with 0% ratio (always no_prompt)."""
        query = "test query"
        result = decide_variant(query)
        assert result == "no_prompt"

    @patch.dict(os.environ, {"RAG_WITH_PROMPT_RATIO": "1.0"}, clear=False)
    def test_full_ratio(self):
        """Test with 100% ratio (always with_prompt)."""
        query = "test query"
        result = decide_variant(query)
        assert result == "with_prompt"

    @patch.dict(os.environ, {"RAG_WITH_PROMPT_RATIO": "0.5"}, clear=False)
    def test_half_ratio_consistency(self):
        """Test that 50% ratio produces consistent results for same query."""
        query = "consistent test query"
        results = [decide_variant(query) for _ in range(10)]
        assert all(r == results[0] for r in results), "Results should be consistent for same query"

    def test_none_query_with_random_fallback(self):
        """Test behavior when query is None (should use random)."""
        with patch('src.ab_test.random.random') as mock_random:
            mock_random.return_value = 0.3
            result = decide_variant(None)
            assert result == "with_prompt"  # 0.3 < 0.5 (default ratio)
            
            mock_random.return_value = 0.7
            result = decide_variant(None)
            assert result == "no_prompt"  # 0.7 >= 0.5

    @patch.dict(os.environ, {"RAG_WITH_PROMPT_RATIO": "0.3"}, clear=False)
    def test_custom_ratio(self):
        """Test with custom ratio from environment."""
        # Test multiple queries to verify ratio behavior
        test_queries = [f"query_{i}" for i in range(100)]
        results = [decide_variant(query) for query in test_queries]
        
        with_prompt_count = sum(1 for r in results if r == "with_prompt")
        ratio = with_prompt_count / len(results)
        
        # Should be roughly 30% (allowing some variance due to hashing)
        assert 0.2 <= ratio <= 0.4, f"Ratio {ratio} not close to expected 0.3"

    def test_empty_string_query(self):
        """Test behavior with empty string query."""
        result = decide_variant("")
        assert result in ["with_prompt", "no_prompt"]
        
        # Should be consistent
        result2 = decide_variant("")
        assert result == result2

    def test_unicode_query(self):
        """Test behavior with unicode characters in query."""
        unicode_query = "测试查询 🚀 émojis"
        result = decide_variant(unicode_query)
        assert result in ["with_prompt", "no_prompt"]
        
        # Should be consistent
        result2 = decide_variant(unicode_query)
        assert result == result2

    def test_very_long_query(self):
        """Test behavior with very long query."""
        long_query = "a" * 10000
        result = decide_variant(long_query)
        assert result in ["with_prompt", "no_prompt"]
        
        # Should be consistent
        result2 = decide_variant(long_query)
        assert result == result2

    @patch.dict(os.environ, {"RAG_WITH_PROMPT_RATIO": "invalid"}, clear=False)
    def test_invalid_ratio_falls_back_to_default(self):
        """Test that invalid ratio falls back to default."""
        with patch('src.ab_test._DEFAULT_RATIO', 0.5):
            query = "test query"
            result = decide_variant(query)
            assert result in ["with_prompt", "no_prompt"]

    def test_distribution_over_many_queries(self):
        """Test that distribution is roughly correct over many queries."""
        with patch.dict(os.environ, {"RAG_WITH_PROMPT_RATIO": "0.5"}, clear=False):
            queries = [f"query_{i}" for i in range(1000)]
            results = [decide_variant(query) for query in queries]
            
            with_prompt_count = sum(1 for r in results if r == "with_prompt")
            ratio = with_prompt_count / len(results)
            
            # Should be roughly 50% (allowing 10% variance)
            assert 0.4 <= ratio <= 0.6, f"Distribution ratio {ratio} not close to expected 0.5"

    def test_variant_names_are_correct(self):
        """Test that only valid variant names are returned."""
        valid_variants = {"with_prompt", "no_prompt"}
        
        # Test with various inputs
        test_inputs = ["query1", "query2", None, "", "unicode 🚀"]
        
        for test_input in test_inputs:
            result = decide_variant(test_input)
            assert result in valid_variants, f"Invalid variant '{result}' returned for input '{test_input}'"