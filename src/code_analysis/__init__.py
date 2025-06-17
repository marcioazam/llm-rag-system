"""
Pacote de análise de código multilinguagem.
"""

from .language_detector import LanguageDetector  # noqa: F401
try:
    from .code_context_detector import CodeContextDetector  # noqa: F401
except ModuleNotFoundError:
    # Dependências opcionais (ex.: tree_sitter) podem não estar instaladas durante CI/testes
    CodeContextDetector = None  # type: ignore

# ... existing code ... 