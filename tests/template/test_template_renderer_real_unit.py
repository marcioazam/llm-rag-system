import sys, importlib

import pytest

# Recarregar módulo real, ignorando stub de conftest
sys.modules.pop("src.template_renderer", None)
tr = importlib.import_module("src.template_renderer")

render_template = tr.render_template  # type: ignore


def test_render_with_context_list():
    tpl = "Pergunta: {{query}}\nContexto:\n{{context}}"
    result = render_template(tpl, query="Qual é a cor do céu?", context_snippets=["O céu é azul."])
    assert "Qual é a cor do céu?" in result
    assert "O céu é azul." in result
    assert "{{" not in result  # sem placeholders remanescentes


def test_render_without_context():
    tpl = "{{query}} -> {{context}}"
    result = render_template(tpl, query="2+2?", context_snippets=None)
    assert "2+2?" in result
    # placeholder context removido
    assert "{{context}}" not in result 