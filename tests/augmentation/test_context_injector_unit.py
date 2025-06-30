from src.augmentation.context_injector import ContextInjector


def _make_docs():
    return [
        {
            "content": "Python é uma linguagem. Foi criada por Guido. É amplamente usada.",
            "score": 0.9,
            "metadata": {
                "source": "Doc A",
                "symbols": [{"name": "function_a"}],
                "relations": [{"target": "ClassB"}],
            },
            "id": "1",
        },
        {
            "content": "Outro documento irrelevante.",
            "score": 0.5,
            "metadata": {"source": "Doc B"},
            "id": "2",
        },
    ]


def test_inject_context_filters_and_formats():
    inj = ContextInjector(relevance_threshold=0.6, max_tokens=50)
    query = "O que é Python?"

    snippets = inj.inject_context(query, _make_docs())

    # Apenas primeiro doc deve ser incluído (score 0.9)
    assert len(snippets) == 1
    snippet = snippets[0]

    # Deve conter fonte e símbolos/relações formatados
    assert "Fonte: Doc A" in snippet
    assert "Símbolos:" in snippet and "function_a" in snippet
    assert "Relações:" in snippet and "ClassB" in snippet


def test_token_limit():
    inj = ContextInjector(relevance_threshold=0.0, max_tokens=5)  # limite baixo
    docs = _make_docs()
    snippets = inj.inject_context("Python", docs)

    # Deve parar após exceder max_tokens, resultando em lista vazia ou 1 doc truncado
    assert len(snippets) <= 1


def test_split_sentences():
    text = "Primeira frase. Segunda frase? Terceira frase!"
    parts = ContextInjector._split_sentences(text)
    assert parts == ["Primeira frase.", "Segunda frase?", "Terceira frase!"] 