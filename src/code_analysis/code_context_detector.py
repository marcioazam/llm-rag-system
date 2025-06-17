from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from .language_detector import LanguageDetector
from .python_analyzer import PythonAnalyzer
from .tree_sitter_analyzer import TreeSitterAnalyzer
from .base_analyzer import BaseStaticAnalyzer


class CodeContextDetector:
    """Serviço de alto nível que detecta linguagem, escolhe analisador e retorna contexto."""

    def __init__(self) -> None:
        self.language_detector = LanguageDetector()
        # Instanciar analisadores suportados
        self.analyzers: dict[str, BaseStaticAnalyzer] = {}
        for lang, ctor in [
            ("python", lambda: PythonAnalyzer()),
            ("javascript", lambda: TreeSitterAnalyzer("javascript")),
            ("typescript", lambda: TreeSitterAnalyzer("typescript")),
            ("csharp", lambda: TreeSitterAnalyzer("csharp")),
            ("java", lambda: TreeSitterAnalyzer("java")),
            ("go", lambda: TreeSitterAnalyzer("go")),
            ("ruby", lambda: TreeSitterAnalyzer("ruby")),
        ]:
            try:
                self.analyzers[lang] = ctor()
            except Exception as exc:  # pragma: no cover
                import logging
                logging.getLogger(__name__).debug("Analyzer %s indisponível: %s", lang, exc)

    # ------------------------------------------------------------
    def detect_context(self, file_path: str | Path | None = None, code: str | None = None) -> Dict[str, Any]:
        """Retorna dicionário com linguagem, símbolos e relações.

        Args:
            file_path: caminho do arquivo (opcional quando **code** já for fornecido).
            code: conteúdo em texto (opcional; se ausente será lido de *file_path*).
        """
        if code is None and file_path is None:
            raise ValueError("Necessário fornecer *file_path* ou *code*.")

        if code is None and file_path is not None:
            code = Path(file_path).read_text(encoding="utf-8", errors="ignore")

        language = self.language_detector.detect(code, file_path)
        analyzer = self.analyzers.get(language or "")

        if analyzer is None:
            # Linguagem não suportada ainda
            return {
                "language": language,
                "symbols": [],
                "relations": [],
            }

        return analyzer.analyze_content(code) 