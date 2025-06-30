from src.code_analysis.language_detector import LanguageDetector


LD = LanguageDetector()


def test_detect_python_by_extension():
    lang = LD.detect(path="example.py")
    assert lang == "python"


def test_detect_javascript_by_content():
    code = "function myFunc() { console.log('hello'); }"
    lang = LD.detect(content=code)
    assert lang == "javascript"


def test_detect_unknown_returns_none():
    lang = LD.detect(content="some random text with no code")
    assert lang is None 