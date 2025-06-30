from src.preprocessing import intelligent_preprocessor as ip

# Desativar libs pesadas para garantir execução rápida
ip.spacy = None  # type: ignore
ip.pipeline = None  # type: ignore

IP = ip.IntelligentPreprocessor()


def test_clean_text_removes_patterns():
    dirty = "Contact me at user@example.com! Visit https://example.com.    Extra\nspaces\tand\rspecial$$$"
    cleaned = IP.clean_text(dirty)
    # Emails removidos
    assert "@" not in cleaned
    # URLs removidas
    assert "http" not in cleaned
    # Caracteres especiais removidos
    assert "$" not in cleaned
    # Espaços normalizados (não contém múltiplos espaços seguidos)
    assert "  " not in cleaned


def test_generate_summary_three_sentences():
    text = "Primeira frase. Segunda frase! Terceira frase? Quarta frase. Quinta frase."
    summary = IP._generate_summary(text)  # type: ignore
    # Deve conter as três primeiras sentenças
    assert summary.count(".") + summary.count("!") + summary.count("?") >= 3
    assert "Quarta" not in summary and "Quinta" not in summary


def test_extract_entities_returns_empty_without_models():
    text = "Barack Obama foi presidente."  # normalmente reconheceria entidade
    ents = IP.extract_entities(text)
    assert ents == []


def test_process_document_classification_none():
    result = IP.process_document("Texto simples sem categoria explícita.")
    # Como classifier está None, classification deve ser None ou estrutura vazia
    assert result["classification"] in (None, {"labels": [], "scores": []}) 