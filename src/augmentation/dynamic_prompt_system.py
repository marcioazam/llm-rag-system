from __future__ import annotations

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicPromptSystem:
    """Gera prompts dinâmicos combinando template, exemplos e contexto."""

    def __init__(self):
        self.prompt_templates = {
            "qa": "Pergunta: {query}\nResposta:",
            "code": "Escreva código para: {query}\nCódigo:",
            "analysis": "Tarefa: {query}\nAnálise detalhada:",
        }
        self.base_system_prompts = {
            "qa": "Você é um assistente especialista e responde com base no contexto.",
            "code": "Você é um desenvolvedor experiente que produz código limpo e comentado.",
            "analysis": "Você é um analista sênior que fornece raciocínio aprofundado.",
        }

    # -------------------------------------------------------------
    def generate_prompt(
        self,
        query: str,
        context_chunks: List[str],
        task_type: str = "qa",
        language: str = "Português",
    ) -> str:
        """Retorna prompt final como string."""
        system_prompt = self._generate_system_prompt(task_type, language)
        context_block = self._format_context(context_chunks)
        user_prompt = self.prompt_templates.get(task_type, self.prompt_templates["qa"]).format(query=query)

        parts = [system_prompt]
        if context_block:
            parts.extend(["Contexto:", context_block, "---"])
        parts.append(user_prompt)
        if self._needs_reasoning(query):
            parts.append("\nVamos pensar passo a passo: ")
        return "\n\n".join(parts)

    # -------------------------------------------------------------
    def _generate_system_prompt(self, task_type: str, language: str) -> str:
        base = self.base_system_prompts.get(task_type, self.base_system_prompts["qa"])
        return f"{base} Responda em {language}."

    def _format_context(self, chunks: List[str]) -> str:
        return "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(chunks[:5])])

    def _needs_reasoning(self, query: str) -> bool:
        indicators = ["por que", "como", "explique", "analise", "compare", "projetar", "resolver"]
        ql = query.lower()
        return any(t in ql for t in indicators) 