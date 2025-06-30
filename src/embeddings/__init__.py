# -*- coding: utf-8 -*-

"""Pacote de componentes relacionados a embeddings.

Este pacote fornece implementações simplificadas usadas durante os testes
(unitários e de cobertura). Para produção, substitua pelas
implementações reais que fazem chamadas às APIs de provedores de
embeddings.
"""

from importlib import import_module as _imp
from types import ModuleType as _ModuleType
import sys as _sys

# Compatibilidade: se pacote antigo singular 'embedding' estiver no
# sistema, reexportar seu conteúdo quando apropriado.
if "src.embedding" in _sys.modules:
    _legacy = _sys.modules["src.embedding"]
    _sys.modules[__name__].__dict__.update(_legacy.__dict__)

# Exportar classes principais quando módulo de serviço estiver carregado
try:
    _api_mod = _imp("src.embeddings.api_embedding_service")
    for _name in (
        "APIEmbeddingService",
        "EmbeddingProvider",
        "EmbeddingConfig",
        "EmbeddingResponse",
        "RateLimiter",
        "CostTracker",
    ):
        globals()[_name] = getattr(_api_mod, _name)
except ModuleNotFoundError:
    # Ignorado até api_embedding_service ser importado
    pass
