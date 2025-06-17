from typing import List, Dict, Any, Optional
import torch
import numpy as np
from sentence_transformers import CrossEncoder


class HybridReranker:
    """Wrapper leve para CrossEncoder usado em reranking.

    Facilita troca de modelo e centraliza lógica de reranking, mantendo
    compatibilidade com o restante do sistema.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cross_encoder = CrossEncoder(model_name, max_length=max_length, device=device)
        self.device = device

    # ------------------------------------------------------------------
    # Métodos utilitários
    # ------------------------------------------------------------------

    def predict_scores(self, query: str, texts: List[str]) -> np.ndarray:
        """Retorna os scores do CrossEncoder para pares (query, text)."""
        pairs = [[query, t] for t in texts]
        scores = self.cross_encoder.predict(pairs)
        return np.array(scores)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        k: int = 10,
        content_key: str = "content",
    ) -> List[Dict[str, Any]]:
        """Rerankear lista de resultados de acordo com a relevância calculada.

        Args:
            query: Texto da consulta.
            results: Lista de dicionários contendo chave `content`.
            k: Número máximo de itens a retornar.
            content_key: Chave nos dicionários que armazena o texto.
        Returns:
            Lista reordenada (top-k) de resultados.
        """
        if not results:
            return results

        texts = [r[content_key] for r in results]
        scores = self.predict_scores(query, texts)
        sorted_idx = np.argsort(scores)[::-1]
        top_idx = sorted_idx[:k]
        return [results[i] for i in top_idx]


# Alias para compatibilidade com testes existentes
Reranker = HybridReranker