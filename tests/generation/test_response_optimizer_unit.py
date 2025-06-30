from src.generation.response_optimizer import ResponseOptimizer


def test_no_sources_returns_answer_intact():
    optimizer = ResponseOptimizer()
    answer = "Resposta simples"
    result = optimizer.add_citations(answer, [])
    assert result == answer  # Deve permanecer igual


def test_add_citations_formats_correctly():
    optimizer = ResponseOptimizer()
    answer = "Conteúdo da resposta."
    sources = [
        {"metadata": {"source": "Doc1"}},
        {"metadata": {"source": "Artigo2"}},
    ]
    result = optimizer.add_citations(answer, sources)

    # Deve conter quebras de linha separando citações
    assert "\n\n" in result

    # Número de citações deve corresponder ao número de fontes
    lines = result.split("\n")
    citation_lines = [ln for ln in lines if ln.startswith("[1]") or ln.startswith("[2]")]
    assert len(citation_lines) == 2

    # Ordem e rótulos corretos
    assert citation_lines[0] == "[1] Doc1"
    assert citation_lines[1] == "[2] Artigo2" 