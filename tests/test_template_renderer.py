import pytest
from src.template_renderer import render_template


def test_render_template_basic():
    template = "Pergunta: {{query}}\nContexto:\n{{context}}"
    rendered = render_template(template, query="O que é Python?", context_snippets=["Python é uma linguagem."])
    assert "O que é Python?" in rendered
    assert "Python é uma linguagem." in rendered


def test_render_template_no_context():
    template = "Olá {{query}}!"
    rendered = render_template(template, query="Mundo", context_snippets=None)
    assert rendered == "Olá Mundo!" 