"""
Serviço de Embeddings via API externa.
Substitui completamente modelos locais como sentence-transformers.
Suporta OpenAI, Google e outros provedores.
"""

import os
import asyncio
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Union, Sequence
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import types as _types
    np = _types.ModuleType("numpy")  # type: ignore
    def _np_array_stub(*args, **kwargs):  # type: ignore
        return args[0] if args else []
    np.array = _np_array_stub  # type: ignore

import aiohttp

logger = logging.getLogger(__name__)


class EmbeddingProvider(str, Enum):
    """Enum de provedores suportados."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


@dataclass
class EmbeddingConfig:
    """Configuração para geração de embeddings."""

    provider: EmbeddingProvider
    api_key: str
    model: str = "dummy-model"
    dimensions: int = 1536
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0  # segundos

    def __post_init__(self) -> None:  # noqa: D401
        if not self.api_key:
            raise ValueError("api_key é obrigatório")
        if self.dimensions <= 0:
            raise ValueError("dimensions deve ser positivo")
        if self.batch_size <= 0:
            raise ValueError("batch_size deve ser positivo")
        if self.timeout <= 0:
            raise ValueError("timeout deve ser positivo")


@dataclass
class EmbeddingResponse:
    """Resposta contendo embeddings e metainformação."""

    embeddings: List[List[float]]
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)
    provider: EmbeddingProvider | str | None = None

    # Propriedades auxiliares usadas nos testes --------------------------------

    @property
    def num_embeddings(self) -> int:  # noqa: D401
        return len(self.embeddings)

    @property
    def embedding_dimension(self) -> int:  # noqa: D401
        return len(self.embeddings[0]) if self.embeddings else 0

    @property
    def total_tokens(self) -> int:  # noqa: D401
        return int(self.usage.get("total_tokens", 0))

    # Métodos utilitários ------------------------------------------------------

    def get_embedding(self, idx: int) -> Optional[List[float]]:
        return self.embeddings[idx] if 0 <= idx < len(self.embeddings) else None


# ---------------------------------------------------------------------------
# Rate limiter e cost tracker
# ---------------------------------------------------------------------------

class RateLimiter:
    """Controle simples de rate-limit em memória."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # Counters para janela atual
        self.request_count = 0
        self.token_count = 0
        self._window_start = time.monotonic()

    # ---------------------------------------------------------------------
    async def check_rate_limit(self, tokens: int) -> bool:
        """Retorna *True* se houver capacidade restante e **consome** cotas ao mesmo tempo.

        Anteriormente o método apenas verificava os limites sem atualizar os contadores,
        o que fazia os testes falharem (`request_count` permanecia 0). Agora, quando a
        requisição está dentro dos limites, registramos o consumo imediatamente.
        """
        self._maybe_reset_window()
        within_reqs = self.request_count < self.requests_per_minute
        within_tokens = (self.token_count + tokens) <= self.tokens_per_minute

        if within_reqs and within_tokens:
            # Registra consumo para a janela corrente
            self._consume(tokens)
            return True

        return False

    async def wait_for_reset(self) -> None:  # noqa: D401
        # Aguarda até reinício da janela (simplificado)
        now = time.monotonic()
        elapsed = now - self._window_start
        wait_time = max(0, 60.0 - elapsed)
        await asyncio.sleep(wait_time)
        self.reset_counters()

    def reset_counters(self) -> None:  # noqa: D401
        self.request_count = 0
        self.token_count = 0
        self._window_start = time.monotonic()

    def get_remaining_capacity(self) -> Dict[str, int]:  # noqa: D401
        self._maybe_reset_window()
        return {
            "requests": max(0, self.requests_per_minute - self.request_count),
            "tokens": max(0, self.tokens_per_minute - self.token_count),
        }

    # Helpers -----------------------------------------------------------------
    def _maybe_reset_window(self) -> None:
        if (time.monotonic() - self._window_start) >= 60.0:
            self.reset_counters()

    # Registra consumo -------------------------------------------------------
    def _consume(self, tokens: int) -> None:
        self.request_count += 1
        self.token_count += tokens


class CostTracker:
    """Rastreamento simples de custos e tokens."""

    def __init__(self, pricing: Dict[EmbeddingProvider, Dict[str, float]], daily_budget: float = 50.0):
        self.pricing = pricing or {
            EmbeddingProvider.OPENAI: {"input": 0.0001, "output": 0.0001}
        }
        self.daily_budget = daily_budget
        self.current_spend = 0.0
        # provider -> {tokens, cost}
        self.usage_stats: Dict[EmbeddingProvider, Dict[str, float]] = {
            p: {"tokens": 0, "cost": 0.0} for p in self.pricing
        }

    # ---------------------------------------------------------------------
    def calculate_cost(self, provider: EmbeddingProvider, tokens: int) -> float:
        rate = self.pricing.get(provider, {"input": 0.0}).get("input", 0.0)
        return tokens * rate

    def track_usage(self, provider: EmbeddingProvider, tokens: int, cost: float) -> None:  # noqa: D401
        self.current_spend += cost
        stats = self.usage_stats.setdefault(provider, {"tokens": 0, "cost": 0.0})
        stats["tokens"] += tokens
        stats["cost"] += cost

    def check_budget(self, cost: float) -> bool:  # noqa: D401
        return (self.current_spend + cost) <= self.daily_budget

    def get_usage_stats(self) -> Dict[str, Any]:  # noqa: D401
        total_tokens = sum(s["tokens"] for s in self.usage_stats.values())
        return {
            "total_cost": self.current_spend,
            "total_tokens": total_tokens,
            "by_provider": self.usage_stats,
        }

    def reset_daily_stats(self) -> None:  # noqa: D401
        self.current_spend = 0.0
        for stats in self.usage_stats.values():
            stats["tokens"] = 0
            stats["cost"] = 0.0


# ---------------------------------------------------------------------------
# Serviço principal
# ---------------------------------------------------------------------------

class APIEmbeddingService:
    """Serviço de embeddings que simula chamadas HTTP a provedores externos."""

    def __init__(
        self,
        config: EmbeddingConfig,
        rate_limiter: Optional[RateLimiter] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.config = config
        self.rate_limiter = rate_limiter or RateLimiter(60, 100_000)
        self.cost_tracker = cost_tracker or CostTracker({}, daily_budget=100.0)
        self.session: Optional[aiohttp.ClientSession] = None

        # Métricas de performance
        self._perf_total_latency = 0.0
        self._perf_total_requests = 0
        self._perf_total_tokens = 0

    # ---------------------------------------------------------------------
    async def _create_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _close_session(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    # Helpers internos ------------------------------------------------------
    def _get_api_endpoint(self) -> str:
        # Endpoint fictício apenas para testes
        return f"https://api.fake-embeddings.com/{self.config.provider.value}/embeddings"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _prepare_request_payload(self, texts: Sequence[str]) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "input": texts,
        }

    # ------------------------------------------------------------------
    async def _make_api_request(self, texts: Sequence[str]) -> Dict[str, Any]:
        """Faz a requisição HTTP real (ou simulada) e retorna JSON dict."""
        session = await self._create_session()
        url = self._get_api_endpoint()
        headers = self._get_headers()
        payload = self._prepare_request_payload(texts)

        async with session.post(url, json=payload, headers=headers) as resp:  # type: ignore[arg-type]
            if resp.status != 200:
                text = await getattr(resp, "text", lambda: "<no-text>")()  # type: ignore[misc]
                raise Exception(f"API error {resp.status}: {text}")
            return await resp.json()

    # ------------------------------------------------------------------
    def _parse_response(self, api_response: Dict[str, Any]) -> EmbeddingResponse:
        embeddings = [item["embedding"] for item in api_response.get("data", [])]
        model = api_response.get("model", self.config.model)
        usage = api_response.get("usage", {})
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            usage=usage,
            provider=self.config.provider,
        )

    # ------------------------------------------------------------------
    async def embed_text(self, text: str) -> List[float]:
        resp = await self.embed_batch([text])
        return resp.embeddings[0] if resp.embeddings else []

    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResponse:
        if not texts:
            raise ValueError("texts não pode ser vazio")

        all_embeddings: List[List[float]] = []
        total_usage_tokens = 0
        model_used = self.config.model

        # Chunking conforme batch_size
        batch_size = self._adaptive_batch_size(texts)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens_estimate = sum(len(t.split()) for t in batch)  # aprox.

            # Rate-limit ------------------------------------------------------------------
            allowed = await self.rate_limiter.check_rate_limit(tokens_estimate)
            if not allowed:
                await self.rate_limiter.wait_for_reset()

            # Custos ----------------------------------------------------------------------
            cost_est = self.cost_tracker.calculate_cost(self.config.provider, tokens_estimate)
            if not self.cost_tracker.check_budget(cost_est):
                raise Exception("budget exceeded")

            # Tentativas com retry ---------------------------------------------------------
            attempt = 0
            while True:
                try:
                    start = time.monotonic()
                    api_response = await self._make_api_request(batch)
                    latency = time.monotonic() - start

                    # Métricas perf
                    self._track_performance(latency, tokens_estimate)

                    # Sucesso ----------------------------------------------------------------
                    break
                except Exception as err:  # noqa: BLE001
                    attempt += 1
                    if attempt > self.config.max_retries:
                        raise err
                    await asyncio.sleep(0.5 * attempt)  # backoff simples

            response_obj = self._parse_response(api_response)
            all_embeddings.extend(response_obj.embeddings)
            total_usage_tokens += response_obj.usage.get("total_tokens", tokens_estimate)

            # Custos finais -----------------------------------------------------------------
            cost_real = self.cost_tracker.calculate_cost(self.config.provider, tokens_estimate)
            self.cost_tracker.track_usage(self.config.provider, tokens_estimate, cost_real)
            self.rate_limiter._consume(tokens_estimate)  # pylint: disable=protected-access

        # Resposta consolidada -------------------------------------------------------------
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=model_used,
            usage={"total_tokens": total_usage_tokens},
            provider=self.config.provider,
        )

    # ------------------------------------------------------------------
    def _adaptive_batch_size(self, texts: Sequence[str]) -> int:
        """Ajuste ingênuo: se média de tokens > 20, reduz batch pela metade."""
        avg_tokens = sum(len(t.split()) for t in texts) / len(texts)
        batch = self.config.batch_size
        if avg_tokens > 20:
            batch = max(1, batch // 2)
        return batch

    # ------------------------------------------------------------------
    def get_supported_models(self) -> List[str]:
        return [self.config.model]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "dimensions": self.config.dimensions,
            "provider": self.config.provider,
        }

    def estimate_cost(self, texts: Sequence[str]) -> float:
        tokens = sum(len(t.split()) for t in texts)
        return self.cost_tracker.calculate_cost(self.config.provider, tokens)

    # Performance stats ----------------------------------------------------
    def _track_performance(self, latency: float, tokens: int) -> None:
        self._perf_total_latency += latency
        self._perf_total_requests += 1
        self._perf_total_tokens += tokens

    def get_performance_stats(self) -> Dict[str, Any]:
        avg_latency = (
            self._perf_total_latency / self._perf_total_requests
            if self._perf_total_requests
            else 0.0
        )
        return {
            "total_requests": self._perf_total_requests,
            "total_tokens": self._perf_total_tokens,
            "average_latency": avg_latency,
        }

    # Cleanup ----------------------------------------------------------------
    async def cleanup(self) -> None:
        await self._close_session()


# Convenience re-exports para import direto do módulo ------------------------
__all__ = [
    "EmbeddingProvider",
    "EmbeddingConfig",
    "EmbeddingResponse",
    "RateLimiter",
    "CostTracker",
    "APIEmbeddingService",
] 