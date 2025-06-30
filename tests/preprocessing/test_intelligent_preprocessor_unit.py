from src.preprocessing import intelligent_preprocessor as ip

# Desativa spaCy/Pipelines pesados para velocidade e para evitar erros
ip.spacy = None  # type: ignore
ip.pipeline = None  # type: ignore

IntelligentPreprocessor = ip.IntelligentPreprocessor

IP = IntelligentPreprocessor()


def test_clean_text_basic():
    dirty = "Email: test@example.com  Visit http://test.com  Hello!!!"
    cleaned = IP.clean_text(dirty)
    assert "@" not in cleaned
    assert "http" not in cleaned
    assert "Hello" in cleaned


def test_summarize_short():
    text = "Short sentence."
    summary = IP.summarize_text(text, max_length=100)
    assert summary == text


def test_summarize_long():
    text = " ".join(["word"] * 200)
    summary = IP.summarize_text(text, max_length=50)
    assert len(summary) <= 50


def test_process_returns_metadata():
    result = IP.process("Machine learning is awesome. It learns patterns from data.")
    assert result["metadata"]["word_count"] > 0
    assert "cleaned" in result 