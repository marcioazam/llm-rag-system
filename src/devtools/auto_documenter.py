from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from src.code_analysis.code_context_detector import CodeContextDetector

class AutoDocumenter:
    """Gera documentação (docstrings ou README) usando o cliente LLM do RAGPipeline."""

    def __init__(self, llm_client, model: str = None):
        self.client = llm_client
        self.model = model or "llama3.1:8b-instruct-q4_K_M"
        self.context_detector = CodeContextDetector()

    # -------------------------------------------------------------
    # Docstrings
    # -------------------------------------------------------------

    def generate_docstring(self, code: str, style: str = "google") -> str:
        analysis = self.context_detector.detect_context(code=code)
        symbols = analysis.get("symbols", [])

        prompt = f"""
Você é uma IA especialista em documentação de código.
Gere docstrings no estilo {style} para o trecho abaixo. Preserve assinaturas.

Código:
```
{code}
```

Símbolos detectados: {symbols}
"""
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.get("response", "")

    # -------------------------------------------------------------
    # README
    # -------------------------------------------------------------

    def generate_readme(self, project_path: str) -> str:
        structure = self._analyze_project(Path(project_path))
        prompt = f"""
Crie um README.md profissional (Português) contendo Descrição, Instalação, Uso, Arquitetura e Contribuição.
Estrutura do projeto (JSON):
{json.dumps(structure, indent=2)}
"""
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.get("response", "")

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def _analyze_project(self, root: Path) -> Dict:
        files = []
        for p in root.rglob("*.py"):
            files.append(str(p.relative_to(root)))
        return {"files": files, "total_files": len(files)} 