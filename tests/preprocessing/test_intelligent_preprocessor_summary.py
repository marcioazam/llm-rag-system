from src.preprocessing.intelligent_preprocessor import IntelligentPreprocessor

IP = IntelligentPreprocessor()

def test_summarize_boundary():
    text = " ".join(["word"] * 10)  # length ~50 when joined with spaces
    summary = IP.summarize_text(text, max_length=len(text))
    assert summary == text 