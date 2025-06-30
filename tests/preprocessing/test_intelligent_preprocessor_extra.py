from src.preprocessing import intelligent_preprocessor as ip

# Desativar libs pesadas
ip.spacy = None  # type: ignore
ip.pipeline = None  # type: ignore

IP = ip.IntelligentPreprocessor()

def test_summarize_empty():
    assert IP.summarize_text("") == ""


def test_extract_entities_none():
    ents = IP.extract_entities("Texto simples sem entidades.")
    assert ents == [] 