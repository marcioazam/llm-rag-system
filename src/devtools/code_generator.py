from __future__ import annotations

from typing import Optional

class CodeGenerator:
    """Gera código usando um LLM qualquer (Ollama ou OpenAI wrapper)."""

    def __init__(self, llm_client, default_model: Optional[str] = None):
        self.client = llm_client
        self.model = default_model or getattr(llm_client, 'default_model', None)

    def generate(self, task: str, context: str, language: str = "python", style: str = "clean") -> str:
        """Gera código.

        Args:
            task: descrição da tarefa (ex.: "adicionar função que faça X").
            context: trecho de código existente ou explicação.
            language: linguagem desejada.
            style: "clean", "one-liner", etc.
        """
        prompt = (
            f"Você é um gerador de código especializado.
            Linguagem: {language}.
            Estilo: {style}.
            Tarefa: {task}.
            Contexto relevante:
            {context}

            Forneça apenas o código.")
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.get("response") or "" 