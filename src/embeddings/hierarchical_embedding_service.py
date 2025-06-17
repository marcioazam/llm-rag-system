from __future__ import annotations

from functools import lru_cache
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Configurações padrão de modelos
# ------------------------------------------------------------------

_MODEL_NAMES = {
    "sentence": "sentence-transformers/all-MiniLM-L6-v2",
    "paragraph": "sentence-transformers/all-mpnet-base-v2",
    "document": "allenai/specter",
    "code": "microsoft/codebert-base",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

# Pesos para vetor composto
_COMPOSITE_WEIGHTS = {
    "sentence_level": 0.3,
    "paragraph_level": 0.5,
    "specialized": 0.2,
}


class HierarchicalEmbeddingService:
    """Gera embeddings multi-nível e devolve vetor composto.

    Método principal `embed_texts` devolve apenas o vetor *composite* para cada texto,
    mantendo compatibilidade com Chroma/VectorStore.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_names: Dict[str, str] | None = None,
        composite_weights: Dict[str, float] | None = None,
        batch_size: int = 16,
    ) -> None:
        if model_names is None:
            model_names = _MODEL_NAMES
        if composite_weights is None:
            composite_weights = _COMPOSITE_WEIGHTS
        self.batch_size = batch_size
        self.composite_weights = composite_weights

        # Resolver device
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        # Carregar modelos
        self.models: Dict[str, SentenceTransformer] = {
            name: SentenceTransformer(model_id, device=device)
            for name, model_id in model_names.items()
        }

    # -------------------------------------------------------------
    # Interface compatível (usada pelo pipeline)
    # -------------------------------------------------------------

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Gera vetor composto para cada texto da lista."""
        outputs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            outputs.extend([self._composite_embedding(t) for t in batch])
        return np.array(outputs)

    def embed_query(self, query: str) -> np.ndarray:
        """Embedding de query usando modelo de parágrafo (rápido) + specialized se aplicável."""
        return self._composite_embedding(query)

    # -------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------

    def _composite_embedding(self, text: str, content_type: str = "general") -> np.ndarray:
        embeddings: Dict[str, np.ndarray] = {}

        # Sentence-level (média das embeddings de sentenças)
        sentences = self._split_sentences(text)
        if sentences:
            sent_embs = self.models["sentence"].encode(sentences, show_progress_bar=False)
            embeddings["sentence_level"] = np.mean(sent_embs, axis=0)

        # Paragraph-level (direto no texto inteiro)
        embeddings["paragraph_level"] = self.models["paragraph"].encode(text, show_progress_bar=False)

        # Specialized
        if content_type == "code":
            embeddings["specialized"] = self.models["code"].encode(text, show_progress_bar=False)
        elif self._is_multilingual(text):
            embeddings["specialized"] = self.models["multilingual"].encode(text, show_progress_bar=False)
        else:
            embeddings["specialized"] = embeddings["paragraph_level"]

        return self._weighted_average(embeddings)

    def _weighted_average(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        composite = np.zeros_like(next(iter(embeddings.values())))
        total_weight = 0.0
        for key, weight in self.composite_weights.items():
            if key in embeddings:
                composite += embeddings[key] * weight
                total_weight += weight
        return composite / total_weight if total_weight else composite

    # -------------------------------------------------------------
    # Utilitários
    # -------------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=10000)
    def _split_sentences(text: str) -> List[str]:
        import re

        return re.split(r"(?<=[.!?]) +", text)

    @staticmethod
    def _is_multilingual(text: str) -> bool:
        # Heurística simples: presença de caracteres não ASCII além de accents comuns
        try:
            text.encode("ascii")
            return False
        except UnicodeEncodeError:
            return True 