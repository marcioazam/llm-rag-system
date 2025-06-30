from unittest.mock import MagicMock

from src.code_analysis.code_context_detector import CodeContextDetector


def test_detect_context_python():
    detector = CodeContextDetector()
    sample_code = "def foo():\n    return 1\n"

    # Patch analyzer for deterministic output
    pa = detector.analyzers["python"]
    pa.analyze_content = MagicMock(return_value={"symbols": ["foo"], "relations": []})  # type: ignore

    ctx = detector.detect_context(code=sample_code)
    assert ctx["symbols"] == ["foo"]


def test_detect_context_unknown_language(monkeypatch):
    detector = CodeContextDetector()
    # Forçar detector a retornar linguagem desconhecida
    monkeypatch.setattr(detector.language_detector, "detect", lambda *a, **k: None)
    ctx = detector.detect_context(code="print('hello')")
    assert ctx["symbols"] == [] 