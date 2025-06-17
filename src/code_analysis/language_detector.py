from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

try:
    import magic  # type: ignore
except ImportError:  # pragma: no cover
    magic = None  # type: ignore

from pygments.lexers import guess_lexer, get_lexer_for_filename
from pygments.util import ClassNotFound


class LanguageDetector:
    """Detector de linguagem de programação baseado em conteúdo e heurísticas."""

    _EXTENSION_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".cs": "csharp",
        ".java": "java",
        ".rb": "ruby",
        ".go": "go",
        ".php": "php",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
    }

    def detect(self, content: Optional[str] = None, path: Optional[str | Path] = None) -> Optional[str]:
        """Detecta linguagem.

        Args:
            content: código-fonte em texto (opcional, mas recomendado).
            path: caminho do arquivo (opcional).
        Returns:
            String curta da linguagem ou ``None`` se não for possível detectar.
        """
        # 1) MIME via python-magic
        lang = None
        if path and magic is not None:
            try:
                mime = magic.from_file(str(path), mime=True)  # type: ignore[arg-type]
                if mime == "text/x-python":
                    return "python"
                if mime == "text/x-csharp":
                    return "csharp"
                if "javascript" in mime:
                    return "javascript"
            except Exception:
                pass

        # 2) Extensão
        if path:
            ext = Path(path).suffix.lower()
            lang = self._EXTENSION_MAP.get(ext)
            if lang:
                return lang

        # 3) Conteúdo via pygments
        if content:
            try:
                lex = guess_lexer(content)
                lang = lex.name.lower()
                # Normalizar nomes comuns
                if "python" in lang:
                    return "python"
                if "javascript" in lang or "ecmascript" in lang:
                    return "javascript"
                if "typescript" in lang:
                    return "typescript"
                if "c#" in lang or "csharp" in lang:
                    return "csharp"
            except ClassNotFound:
                pass

        # 4) get_lexer_for_filename usando extensão + conteúdo (melhor heurística)
        if path:
            try:
                lex = get_lexer_for_filename(str(path), content or "")
                return lex.name.lower()
            except ClassNotFound:
                pass

        # Falha
        # Heurística adicional para trechos curtos
        if content:
            stripped = content.strip()
            # Detectar Python: começa com 'def' ou contém 'import '
            if stripped.startswith("def ") or "import " in stripped.split("\n")[0]:
                return "python"
            # Detectar JavaScript: começa com 'function' ou contém 'console.log'
            if stripped.startswith("function ") or "console.log" in stripped:
                return "javascript"

        return None 