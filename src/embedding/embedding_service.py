from __future__ import annotations

"""Serviço de geração de embeddings unificado.

Esta implementação foi criada para permitir a execução dos testes de cobertura
em ambientes sem dependências externas pesadas. Ela oferece uma interface
mínima porém compatível com os cenários cobertos pelos arquivos de teste em
``tests/``.

Principais características atendidas:
1. Suporte a dois provedores: ``openai`` e ``sentence-transformers``.
2. Validações de parâmetros (API-key obrigatória para OpenAI).
3. Métodos ``embed_text`` e ``embed_texts`` para uso batelado.
4. Tratamento de erros encapsulando exceções do provedor.

Para ambientes de produção, recomenda-se substituir por uma versão que
implemente cache, paralelismo, back-off e demais requisitos de resiliência.
"""

from typing import List, Sequence, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Try import providers – serão mockados nos testes se inexistentes
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


@dataclass
class EmbeddingService:
    """Serviço de geração de embeddings minimalista."""

    provider: str
    model: str
    api_key: str | None = None
    _client: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        provider_normalized = self.provider.lower().replace("_", "-")
        self.provider = provider_normalized
        logger.debug("Inicializando EmbeddingService - provider=%s model=%s", self.provider, self.model)

        if self.provider == "openai":
            if not self.api_key:
                raise ValueError("API key é obrigatória para OpenAI")
            if OpenAI is None:  # pragma: no cover
                raise ImportError("Pacote 'openai' não disponível. Instale ou forneça mock nos testes.")
            self._client = OpenAI(api_key=self.api_key)  # type: ignore[arg-type]

        elif self.provider in {"sentence-transformers", "sentencetransformers", "sentence_transformers"}:
            if SentenceTransformer is None:  # pragma: no cover
                raise ImportError(
                    "Pacote 'sentence-transformers' não disponível. Instale ou forneça mock nos testes."
                )
            self._client = SentenceTransformer(self.model)
        else:
            raise ValueError(f"Provedor não suportado: {self.provider}")

    # aliases de compatibilidade com testes
    @property
    def client(self):  # noqa: D401
        """Retorna o cliente/provedor interno."""
        return self._client

    # -------------------------------------------------------------------------------------
    # Métodos públicos
    # -------------------------------------------------------------------------------------

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Gera embeddings para uma lista de textos."""
        if not texts:
            return []

        try:
            if self.provider == "openai":
                response = self._client.embeddings.create(model=self.model, input=list(texts))  # type: ignore[attr-defined]
                return [item.embedding for item in response.data]
            else:  # sentence-transformers
                import numpy as np  # local import para reduzir peso quando mockado
                embeddings = self._client.encode(list(texts))  # type: ignore[attr-defined]
                # Converte numpy array para lista nativa para simplicidade
                return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover
            msg = (
                "Erro ao gerar embeddings com OpenAI: " if self.provider == "openai" else
                "Erro ao gerar embeddings com SentenceTransformers: "
            ) + str(exc)
            logger.error(msg)
            raise Exception(msg) from exc

    def embed_text(self, text: str) -> List[float]:
        """Gera embedding para um texto único.

        Este método aproveita :py:meth:`embed_texts` para reduzir código duplicado.
        """
        if not text:
            return []
        return self.embed_texts([text])[0] 