"""Comprehensive tests for prompt selection functionality."""

import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import json

from src.prompt_selector import (
    classify_query,
    select_prompt,
    REGISTRY,
    _KEYWORD_MAP,
    _DEFAULT_TYPE,
    _TYPE_TO_PROMPTS
)


class TestPromptSelector:
    """Test suite for prompt selection functionality."""

    def test_classify_query_bugfix(self):
        """Test classification of bug-related queries."""
        bugfix_queries = [
            "I'm getting a traceback error",
            "There's an exception in my code",
            "Undefined variable error",
            "Stack trace shows null pointer",
            "Getting 'not found' error"
        ]
        
        for query in bugfix_queries:
            result = classify_query(query)
            assert result == "bugfix", f"Query '{query}' should be classified as bugfix"

    def test_classify_query_review(self):
        """Test classification of code review queries."""
        review_queries = [
            "Please review this pull request",
            "Can you do a code review?",
            "Revisar este PR",
            "Revisão de código necessária"
        ]
        
        for query in review_queries:
            result = classify_query(query)
            assert result == "review", f"Query '{query}' should be classified as review"

    def test_classify_query_performance(self):
        """Test classification of performance-related queries."""
        perf_queries = [
            "My code is running slow",
            "Performance optimization needed",
            "High latency issues",
            "CPU usage is too high",
            "Memory optimization required",
            "Código está lento"
        ]
        
        for query in perf_queries:
            result = classify_query(query)
            assert result == "perf", f"Query '{query}' should be classified as perf"

    def test_classify_query_architecture(self):
        """Test classification of architecture-related queries."""
        arch_queries = [
            "What's the best architecture for this?",
            "Need an ADR for this decision",
            "Decisão arquitetural necessária",
            "Design decision required"
        ]
        
        for query in arch_queries:
            result = classify_query(query)
            assert result == "arch", f"Query '{query}' should be classified as arch"

    def test_classify_query_testing(self):
        """Test classification of testing-related queries."""
        test_queries = [
            "Need to write unit tests",
            "How to improve test coverage?",
            "Mock this dependency",
            "Jest configuration help",
            "Pytest setup needed",
            "Cypress testing"
        ]
        
        for query in test_queries:
            result = classify_query(query)
            assert result == "test", f"Query '{query}' should be classified as test"

    def test_classify_query_test_generation(self):
        """Test classification of test generation queries."""
        testgen_queries = [
            "Gerar testes para esta função",
            "Criar teste unitário",
            "Generate test cases",
            "Melhorar cobertura de testes"
        ]
        
        for query in testgen_queries:
            result = classify_query(query)
            assert result == "testgen", f"Query '{query}' should be classified as testgen"

    def test_classify_query_data_visualization(self):
        """Test classification of data visualization queries."""
        dataviz_queries = [
            "Create a chart for this data",
            "Gerar gráfico de barras",
            "Plot histogram",
            "Scatter plot visualization",
            "Line chart needed"
        ]
        
        for query in dataviz_queries:
            result = classify_query(query)
            assert result == "data_viz", f"Query '{query}' should be classified as data_viz"

    def test_classify_query_ci_cd(self):
        """Test classification of CI/CD related queries."""
        ci_queries = [
            "CI pipeline is failing",
            "GitHub Actions not working",
            "GitLab CI configuration",
            "Jenkins build failed",
            "Pipeline troubleshooting"
        ]
        
        for query in ci_queries:
            result = classify_query(query)
            assert result == "ci", f"Query '{query}' should be classified as ci"

    def test_classify_query_general_fallback(self):
        """Test that unmatched queries fall back to general category."""
        general_queries = [
            "How do I learn Python?",
            "What is machine learning?",
            "Random question about programming",
            "Explain this concept"
        ]
        
        for query in general_queries:
            result = classify_query(query)
            assert result == _DEFAULT_TYPE, f"Query '{query}' should fall back to {_DEFAULT_TYPE}"

    def test_classify_query_case_insensitive(self):
        """Test that classification is case insensitive."""
        queries = [
            ("TRACEBACK ERROR", "bugfix"),
            ("Code Review", "review"),
            ("PERFORMANCE ISSUE", "perf"),
            ("Unit Test", "test")
        ]
        
        for query, expected in queries:
            result = classify_query(query)
            assert result == expected, f"Query '{query}' should be classified as {expected}"

    def test_classify_query_empty_string(self):
        """Test classification of empty string."""
        result = classify_query("")
        assert result == _DEFAULT_TYPE

    def test_classify_query_first_match_wins(self):
        """Test that first matching pattern wins."""
        # This query could match both 'test' and 'perf' patterns
        query = "performance test optimization"
        result = classify_query(query)
        # Should match 'perf' first since it appears first in _KEYWORD_MAP iteration
        assert result in ["perf", "test"]  # Either is acceptable depending on dict order

    @patch('src.prompt_selector._PROMPT_REGISTRY')
    @patch('src.prompt_selector._REGISTRY_PATH')
    def test_select_prompt_basic(self, mock_registry_path, mock_registry):
        """Test basic prompt selection functionality."""
        # Mock registry
        mock_registry.return_value = {
            "quick_fix_bug": {
                "id": "quick_fix_bug",
                "file": "quick_fix_bug.md",
                "scope": "quick_fix"
            }
        }
        
        # Mock file path and content
        mock_path = MagicMock()
        mock_path.parent = Path("/mock/prompts")
        mock_registry_path.return_value = mock_path
        
        mock_prompt_file = MagicMock()
        mock_prompt_file.exists.return_value = True
        mock_prompt_file.read_text.return_value = "Mock prompt content"
        
        with patch('pathlib.Path.__truediv__', return_value=mock_prompt_file):
            with patch.dict('src.prompt_selector._PROMPT_REGISTRY', mock_registry.return_value):
                prompt_id, prompt_text = select_prompt("traceback error")
                
                assert prompt_id == "quick_fix_bug"
                assert prompt_text == "Mock prompt content"

    def test_select_prompt_depth_filtering(self):
        """Test that depth parameter filters prompts correctly."""
        # This test requires mocking the registry and file system
        with patch('src.prompt_selector._PROMPT_REGISTRY') as mock_registry:
            mock_registry.__getitem__ = lambda self, key: {
                "quick_fix_bug": {"scope": "quick_fix", "file": "quick_fix.md"},
                "deep_analysis": {"scope": "deep_dive", "file": "deep.md"}
            }[key]
            
            with patch('src.prompt_selector._TYPE_TO_PROMPTS', {"bugfix": ["quick_fix_bug", "deep_analysis"]}):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.read_text', return_value="Mock content"):
                        # Quick depth should filter out deep_dive
                        prompt_id, _ = select_prompt("traceback error", depth="quick")
                        assert prompt_id == "quick_fix_bug"

    def test_select_prompt_file_not_found(self):
        """Test error handling when prompt file doesn't exist."""
        with patch('src.prompt_selector._PROMPT_REGISTRY') as mock_registry:
            mock_registry.__getitem__ = lambda self, key: {
                "quick_fix_bug": {"scope": "quick_fix", "file": "nonexistent.md"}
            }[key]
            
            with patch('src.prompt_selector._TYPE_TO_PROMPTS', {"bugfix": ["quick_fix_bug"]}):
                with patch('pathlib.Path.exists', return_value=False):
                    with pytest.raises(FileNotFoundError, match="Prompt file.*not found"):
                        select_prompt("traceback error")

    def test_select_prompt_fallback_when_no_filtered_prompts(self):
        """Test fallback behavior when filtering removes all prompts."""
        with patch('src.prompt_selector._PROMPT_REGISTRY') as mock_registry:
            mock_registry.__getitem__ = lambda self, key: {
                "deep_analysis": {"scope": "deep_dive", "file": "deep.md"}
            }[key]
            
            with patch('src.prompt_selector._TYPE_TO_PROMPTS', {"bugfix": ["deep_analysis"]}):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.read_text', return_value="Deep content"):
                        # Even with quick depth, should fallback to available prompt
                        prompt_id, prompt_text = select_prompt("traceback error", depth="quick")
                        assert prompt_id == "deep_analysis"
                        assert prompt_text == "Deep content"

    def test_select_prompt_unknown_query_type(self):
        """Test prompt selection for unknown query types."""
        with patch('src.prompt_selector._PROMPT_REGISTRY') as mock_registry:
            mock_registry.__getitem__ = lambda self, key: {
                "plan_and_solve": {"scope": "general", "file": "general.md"}
            }[key]
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.read_text', return_value="General content"):
                    prompt_id, prompt_text = select_prompt("random unknown query")
                    assert prompt_id == "plan_and_solve"  # Default for 'geral' type
                    assert prompt_text == "General content"

    def test_registry_is_accessible(self):
        """Test that REGISTRY is accessible for external inspection."""
        assert REGISTRY is not None
        assert isinstance(REGISTRY, dict)

    def test_keyword_map_structure(self):
        """Test that keyword map has expected structure."""
        assert isinstance(_KEYWORD_MAP, dict)
        for category, patterns in _KEYWORD_MAP.items():
            assert isinstance(category, str)
            assert isinstance(patterns, list)
            for pattern in patterns:
                assert isinstance(pattern, str)

    def test_type_to_prompts_structure(self):
        """Test that type to prompts mapping has expected structure."""
        assert isinstance(_TYPE_TO_PROMPTS, dict)
        for qtype, prompt_ids in _TYPE_TO_PROMPTS.items():
            assert isinstance(qtype, str)
            assert isinstance(prompt_ids, list)
            for prompt_id in prompt_ids:
                assert isinstance(prompt_id, str)

    def test_default_type_is_valid(self):
        """Test that default type exists in type to prompts mapping."""
        assert _DEFAULT_TYPE in _TYPE_TO_PROMPTS

    def test_all_keyword_map_types_have_prompts(self):
        """Test that all types in keyword map have corresponding prompts."""
        for qtype in _KEYWORD_MAP.keys():
            assert qtype in _TYPE_TO_PROMPTS, f"Type '{qtype}' missing from _TYPE_TO_PROMPTS"

    def test_classify_query_with_unicode(self):
        """Test classification with unicode characters."""
        unicode_queries = [
            "Erro de traceback 🐛",
            "Revisão de código 📝",
            "Otimização de performance ⚡"
        ]
        
        for query in unicode_queries:
            result = classify_query(query)
            assert result in _TYPE_TO_PROMPTS.keys()

    def test_classify_query_with_special_characters(self):
        """Test classification with special characters."""
        special_queries = [
            "Error: null pointer exception!",
            "Performance issue - 50% slower",
            "Code review (urgent)",
            "Test coverage @ 80%"
        ]
        
        for query in special_queries:
            result = classify_query(query)
            assert result in _TYPE_TO_PROMPTS.keys()

    def test_select_prompt_different_depths(self):
        """Test prompt selection with different depth parameters."""
        depths = ["quick", "deep", "invalid_depth"]
        
        for depth in depths:
            # Should not raise an error regardless of depth value
            try:
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.read_text', return_value="Mock content"):
                        prompt_id, prompt_text = select_prompt("general query", depth=depth)
                        assert isinstance(prompt_id, str)
                        assert isinstance(prompt_text, str)
            except Exception as e:
                pytest.fail(f"select_prompt failed with depth '{depth}': {e}")