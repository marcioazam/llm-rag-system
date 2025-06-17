import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.code_analysis.language_detector import LanguageDetector


class TestLanguageDetector:
    """Testes para a classe LanguageDetector."""

    def test_init(self):
        """Testa a inicialização da classe."""
        detector = LanguageDetector()
        assert detector is not None

    def test_detect_python_by_extension(self):
        """Testa detecção de Python por extensão de arquivo."""
        detector = LanguageDetector()
        result = detector.detect(path="test.py")
        assert result == "python"

    def test_detect_javascript_by_extension(self):
        """Testa detecção de JavaScript por extensão de arquivo."""
        detector = LanguageDetector()
        result = detector.detect(path="test.js")
        assert result == "javascript"

    def test_detect_typescript_by_extension(self):
        """Testa detecção de TypeScript por extensão de arquivo."""
        detector = LanguageDetector()
        result = detector.detect(path="test.ts")
        assert result == "typescript"

    def test_detect_java_by_extension(self):
        """Testa detecção de Java por extensão de arquivo."""
        detector = LanguageDetector()
        result = detector.detect(path="Test.java")
        assert result == "java"

    def test_detect_csharp_by_extension(self):
        """Testa detecção de C# por extensão de arquivo."""
        detector = LanguageDetector()
        result = detector.detect(path="Test.cs")
        assert result == "csharp"

    def test_detect_go_by_extension(self):
        """Testa detecção de Go por extensão de arquivo."""
        detector = LanguageDetector()
        result = detector.detect(path="main.go")
        assert result == "go"

    def test_detect_ruby_by_extension(self):
        """Testa detecção de Ruby por extensão de arquivo."""
        detector = LanguageDetector()
        result = detector.detect(path="test.rb")
        assert result == "ruby"

    def test_detect_python_by_content(self):
        """Testa detecção de Python por conteúdo."""
        detector = LanguageDetector()
        python_code = "def hello():\n    return 'world'"
        result = detector.detect(content=python_code)
        assert result == "python"

    def test_detect_unknown_extension(self):
        """Testa detecção com extensão desconhecida."""
        detector = LanguageDetector()
        result = detector.detect(path="test.xyz")
        # Deve retornar None para extensões desconhecidas
        assert result is None

    def test_detect_no_content_no_path(self):
        """Testa detecção sem conteúdo nem caminho."""
        detector = LanguageDetector()
        result = detector.detect()
        assert result is None

    def test_detect_empty_content(self):
        """Testa detecção com conteúdo vazio."""
        detector = LanguageDetector()
        result = detector.detect(content="")
        assert result is None

    def test_detect_with_path_object(self):
        """Testa detecção com objeto Path."""
        detector = LanguageDetector()
        path_obj = Path("test.py")
        result = detector.detect(path=path_obj)
        assert result == "python"

    @pytest.mark.parametrize(
        "code,expected",
        [
            ("def foo():\n    pass", "python"),
            ("function bar() {}", "javascript"),
        ],
    )
    def test_detect_language_parametrized(self, code, expected):
        """Testa detecção de linguagem com parâmetros."""
        detector = LanguageDetector()
        assert detector.detect(content=code) == expected