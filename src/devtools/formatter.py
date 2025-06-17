from __future__ import annotations

import subprocess
import shutil
from typing import Optional

class FormatterService:
    """Wrapper simples em torno de formatadores populares."""

    def format(self, code: str, language: str) -> str:
        if language == "python" and shutil.which("black"):
            return self._format_with_black(code)
        if language in {"javascript", "typescript"} and shutil.which("prettier"):
            return self._format_with_prettier(code, language)
        # Sem formatter disponível
        return code

    def _format_with_black(self, code: str) -> str:
        try:
            proc = subprocess.run(["black", "-q", "-"], input=code.encode(), capture_output=True, timeout=30)
            return proc.stdout.decode() if proc.stdout else code
        except subprocess.TimeoutExpired:
            return code

    def _format_with_prettier(self, code: str, lang: str) -> str:
        parser = "typescript" if lang == "typescript" else "babel"
        cmd = ["prettier", f"--parser={parser}"]
        try:
            proc = subprocess.run(cmd, input=code.encode(), capture_output=True, timeout=30)
            return proc.stdout.decode() if proc.stdout else code
        except subprocess.TimeoutExpired:
            return code 