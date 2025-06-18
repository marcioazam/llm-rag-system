# -*- coding: utf-8 -*-

# FASE 1: Exportar Enhanced Semantic Chunker como padrão
from .semantic_chunker_enhanced import EnhancedSemanticChunker
from .base_chunker import BaseChunker, Chunk
from .advanced_chunker import AdvancedChunker
from .recursive_chunker import RecursiveChunker

# Alias para compatibilidade - usar enhanced por padrão
SemanticChunker = EnhancedSemanticChunker

__all__ = [
    "EnhancedSemanticChunker",
    "SemanticChunker",  # Alias para enhanced
    "BaseChunker", 
    "Chunk",
    "AdvancedChunker",
    "RecursiveChunker"
]
