from src.preprocessing import intelligent_preprocessor as ip

# Força desativação de dependências pesadas
ip.spacy = None  # type: ignore
ip.pipeline = None  # type: ignore

IntelligentPreprocessor = ip.IntelligentPreprocessor

IP = IntelligentPreprocessor()


def test_format_entities_dict_to_list():
    """Dict de entidades deve ser convertido corretamente em lista padronizada."""
    raw = {
        "person": ["Alice", "Bob"],
        "org": ["Acme"]
    }
    formatted = IP._format_entities_for_tests(raw)
    # Deve haver três itens
    assert len(formatted) == 3
    labels = {item["label"] for item in formatted}
    assert labels == {"PERSON", "ORG"}
    texts = {item["text"] for item in formatted}
    assert texts == {"Alice", "Bob", "Acme"}


def test_get_language_info_flags():
    info = IP.get_language_info()
    # Como spacy e pipelines foram desativados, flags devem ser falsos
    assert info["spacy_available"] is False
    assert info["classifier_available"] is False
    assert info["ner_available"] is False


def test_classify_content_without_classifier():
    result = IP.classify_content("some text", ["A", "B"])
    assert result == {"labels": [], "scores": []}


def test_process_document_structure():
    doc = "Simple content about AI and ML."
    processed = IP.process_document(doc)
    # Estrutura mínima esperada
    assert "cleaned_text" in processed
    assert "entities" in processed
    assert "classification" in processed
    assert "summary" in processed
    assert "metadata" in processed
    # cleaned_text não contém caracteres especiais removidos
    assert "http" not in processed["cleaned_text"] 