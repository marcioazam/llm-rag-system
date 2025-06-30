from src.template_renderer import render_template


def test_render_basic():
    tpl = "Pergunta: {query}\n\nContexto:\n{context}"
    ctx = {"query": "Qual é a capital?", "context": "A capital do Brasil é Brasília."}
    output = render_template(tpl, ctx)
    assert "Qual é a capital?" in output
    assert "Brasília" in output


def test_render_without_context():
    tpl = "P: {query} | C: {context}"
    rendered = render_template(tpl, {"query": "2+2?", "context": ""})
    assert "2+2?" in rendered
    # Placeholder {{context}} deve ser substituído por string vazia
    assert "{{context}}" not in rendered 