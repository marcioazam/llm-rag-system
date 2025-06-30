"""
Testes básicos para o Template Renderer.
Cobertura atual: 0% -> Meta: 90%
"""

import pytest
from src.template_renderer import TemplateRenderer


class TestTemplateRenderer:
    """Testes para o Template Renderer."""

    def test_render_simple_template(self):
        """Testar renderização de template simples."""
        renderer = TemplateRenderer()
        template = "Question: {query}"
        
        result = renderer.render(template, {"query": "What is RAG?"})
        assert result == "Question: What is RAG?"

    def test_render_with_context(self):
        """Testar renderização com contexto."""
        renderer = TemplateRenderer()
        template = "Query: {query}\n\nContext:\n{context}"
        context = {"query": "What is machine learning?", "context": "ML is a subset of AI\n\nIt uses algorithms to learn patterns"}
        
        result = renderer.render(template, context)
        
        assert "What is machine learning?" in result
        assert "ML is a subset of AI" in result
        assert "It uses algorithms to learn patterns" in result

    def test_render_empty_template(self):
        """Testar renderização de template vazio."""
        renderer = TemplateRenderer()
        template = ""
        
        result = renderer.render(template, {"query": "test"})
        assert result == ""

    def test_render_no_variables(self):
        """Testar renderização sem variáveis."""
        renderer = TemplateRenderer()
        template = "Static text without variables"
        
        result = renderer.render(template, {"query": "test"})
        assert result == "Static text without variables"

    def test_render_with_only_query(self):
        """Testar renderização apenas com query."""
        renderer = TemplateRenderer()
        template = "The user asked: {query}"
        
        result = renderer.render(template, {"query": "How does RAG work?"})
        assert result == "The user asked: How does RAG work?"

    def test_render_with_only_context(self):
        """Testar renderização apenas com contexto."""
        renderer = TemplateRenderer()
        template = "Context: {context}"
        
        result = renderer.render(template, {"context": "First snippet\n\nSecond snippet"})
        assert "First snippet" in result
        assert "Second snippet" in result

    def test_render_with_empty_context(self):
        """Testar renderização com contexto vazio."""
        renderer = TemplateRenderer()
        template = "Query: {query} | Context: {context}"
        
        result = renderer.render(template, {"query": "test query", "context": ""})
        assert "test query" in result
        assert "Context: " in result

    def test_render_with_multiple_context_snippets(self):
        """Testar renderização com múltiplos snippets de contexto."""
        renderer = TemplateRenderer()
        template = "{context}"
        
        result = renderer.render(template, {"context": "Snippet 1\n\nSnippet 2\n\nSnippet 3"})
        
        # Snippets devem ser separados por duas quebras de linha
        assert "Snippet 1\n\nSnippet 2\n\nSnippet 3" in result

    def test_render_with_special_characters(self):
        """Testar renderização com caracteres especiais."""
        renderer = TemplateRenderer()
        template = "Query: {query} | Context: {context}"
        
        result = renderer.render(template, {
            "query": "Hello 🌍! <>&'",
            "context": "Context with 🚀 emojis"
        })
        assert "🌍" in result
        assert "🚀" in result
        assert "<>&'" in result

    def test_render_complex_template(self):
        """Testar renderização de template complexo."""
        renderer = TemplateRenderer()
        template = """
User Question: {query}

Relevant Information:
{context}

Please provide a comprehensive answer based on the above context.
        """.strip()
        
        context_text = """Machine learning is a subset of artificial intelligence.

It involves algorithms that can learn from data.

Common types include supervised and unsupervised learning."""
        
        result = renderer.render(template, {
            "query": "What is machine learning?",
            "context": context_text
        })
        
        assert "What is machine learning?" in result
        assert "subset of artificial intelligence" in result
        assert "algorithms that can learn" in result
        assert "supervised and unsupervised" in result

    @pytest.mark.performance
    def test_render_performance(self):
        """Testar performance de renderização."""
        import time
        
        renderer = TemplateRenderer()
        template = "Query: {query} | Context: {context}"
        context = {"query": "Test query", "context": "Context snippet"}
        
        start_time = time.time()
        for _ in range(1000):
            renderer.render(template, context)
        end_time = time.time()
        
        # Deve renderizar 1000 templates em menos de 1 segundo
        assert end_time - start_time < 1.0 