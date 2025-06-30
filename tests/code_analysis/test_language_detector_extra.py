import sys, importlib, pytest

sys.modules.pop("src.code_analysis.language_detector", None)
ld_mod = importlib.import_module("src.code_analysis.language_detector")


def _new_detector():
    return ld_mod.LanguageDetector()


def test_detect_by_extension():
    detector = _new_detector()
    assert detector.detect(path="example.ts") == "typescript"


def test_detect_by_magic(monkeypatch):
    # Patch atributo magic dentro do módulo
    class _DummyMagic:
        @staticmethod
        def from_file(path, mime=True):
            return "text/x-python"

    monkeypatch.setattr(ld_mod, "magic", _DummyMagic, raising=False)
    detector = _new_detector()
    assert detector.detect(path="whatever.txt") == "python"


def test_detect_by_content_fallback():
    detector = _new_detector()
    js_snippet = "function test() { console.log('hi'); }"
    assert detector.detect(content=js_snippet) == "javascript"


def test_unknown_returns_none():
    detector = _new_detector()
    assert detector.detect(content="random text without code") is None 