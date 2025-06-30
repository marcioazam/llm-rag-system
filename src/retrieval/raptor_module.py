"""Módulo stub de integração RAPTOR.

Fornece implementação de *no-op* para manter compatibilidade
quando dependências RAPTOR reais não estão instaladas ou quando
`ENABLE_RAPTOR` está desabilitado.
"""
from __future__ import annotations

from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class RaptorIntegration:  # noqa: D401
    """Integração mínima RAPTOR que age como *no-op*.

    Este stub permite que o `AdvancedRAGPipeline` faça chamadas
    (initialize, build_tree, retrieve) sem falhar quando as
    dependências originais não estão disponíveis.
    """

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.initialized: bool = False

    # ------------------------------------------------------------------
    async def initialize(self) -> None:  # noqa: D401
        """Inicializa componentes RAPTOR (stub)."""
        logger.info("RaptorIntegration stub inicializado (noop)")
        self.initialized = True

    async def build_tree(self, documents: List[str]) -> Dict[str, Any]:  # noqa: D401
        """Constrói árvore RAPTOR (stub) e devolve estatísticas."""
        logger.debug("build_tree chamado com %d documentos (noop)", len(documents))
        return {"built": False, "documents": len(documents)}

    async def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Dict[str, Any]]:  # noqa: D401
        """Realiza recuperação RAPTOR (stub) – sempre lista vazia."""
        logger.debug("retrieve chamado (noop): query='%s' k=%d", query, k)
        return [] 