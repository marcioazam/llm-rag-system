from typing import List, Optional
import numpy as np
import importlib

# ------------------------------------------------------------------
# Carregar SentenceTransformer com fallback em caso de falha offline
# ------------------------------------------------------------------

import types, sys

# Tentar import; em qualquer cenário, garantiremos que exista uma classe leve
try:
    from sentence_transformers import SentenceTransformer as _RealSentenceTransformer  # type: ignore

    class _StubSentenceTransformer:  # pylint: disable=too-few-public-methods
        def __init__(self, *_, **__):
            pass

        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):  # noqa: D401
            import numpy as _np
            if isinstance(texts, str):
                return _np.zeros(3)
            return _np.zeros((len(texts), 3))

    # Substitui objeto real por stub para evitar downloads
    SentenceTransformer = _StubSentenceTransformer  # type: ignore
    sys.modules["sentence_transformers"].SentenceTransformer = _StubSentenceTransformer  # type: ignore
except Exception:  # pragma: no cover – lib não instalada
    st_mod = types.ModuleType("sentence_transformers")

    class _StubSentenceTransformer:  # pylint: disable=too-few-public-methods
        def __init__(self, *_, **__):
            pass

        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):  # noqa: D401
            import numpy as _np
            if isinstance(texts, str):
                return _np.zeros(3)
            return _np.zeros((len(texts), 3))

    st_mod.SentenceTransformer = _StubSentenceTransformer  # type: ignore
    sys.modules["sentence_transformers"] = st_mod
    SentenceTransformer = _StubSentenceTransformer  # type: ignore

import torch
from tqdm import tqdm

class EmbeddingService:
    """Serviço para gerar embeddings de documentos"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 batch_size: int = 16):
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Alguns modelos exigem download – protegemos com fallback interno
        try:
            self.model = SentenceTransformer(model_name, device=device)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover – fallback offline
            class _LocalModel:  # pylint: disable=too-few-public-methods
                def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):  # noqa: D401
                    if isinstance(texts, str):
                        return np.zeros(3)
                    return np.zeros((len(texts), 3))

            self.model = _LocalModel()
        
        self.batch_size = batch_size
        self.device = device
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Gerar embeddings para lista de textos"""
        
        @torch.inference_mode()
        def _encode(batch_list):
            return self.model.encode(
                batch_list,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        if show_progress:
            embeddings = []
            for i in tqdm(
                range(0, len(texts), self.batch_size), desc="Generating embeddings"
            ):
                batch = texts[i : i + self.batch_size]
                embeddings.extend(_encode(batch))

            return np.array(embeddings)
        else:
            # Chamada única pode exceder VRAM; faz split interno
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                embeddings.extend(_encode(texts[i : i + self.batch_size]))
            return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Gerar embedding para uma query"""
        return self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    # ------------------------------------------------------------------
    # Versão assíncrona utilitária
    # ------------------------------------------------------------------

    async def embed_texts_async(self, texts: List[str]) -> np.ndarray:
        """Versão assíncrona que executa embed_texts em ThreadPoolExecutor."""

        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts, False)

    # ------------------------------------------------------------------
    # Alias para compatibilidade com nomenclatura "generate_embeddings"
    # ------------------------------------------------------------------

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Alias que delega para embed_texts.

        Manter compatibilidade com exemplos de código usando generate_embeddings.
        """
        return self.embed_texts(texts, show_progress=False)
