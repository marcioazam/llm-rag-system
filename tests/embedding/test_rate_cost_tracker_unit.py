import asyncio
import pytest
import math

from src.embeddings.api_embedding_service import (
    RateLimiter,
    CostTracker,
    EmbeddingProvider,
    EmbeddingConfig,
    APIEmbeddingService,
)


# ---------------------------------------------------------------------------
# RateLimiter ----------------------------------------------------------------
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limiter_basic_flow():
    """Verifica consumo de requisições e tokens."""
    limiter = RateLimiter(requests_per_minute=2, tokens_per_minute=10)

    # Primeiro call dentro dos limites -----------------------------------
    allowed_1 = await limiter.check_rate_limit(tokens=5)
    assert allowed_1 is True
    assert limiter.request_count == 1
    assert limiter.token_count == 5

    # Segundo call excede tokens -----------------------------------------
    allowed_2 = await limiter.check_rate_limit(tokens=6)
    assert allowed_2 is False  # Ultrapassa limite de tokens
    # Contadores não devem mudar após falha
    assert limiter.request_count == 1
    assert limiter.token_count == 5

    # Após reset, deve aceitar novamente ---------------------------------
    limiter.reset_counters()
    allowed_3 = await limiter.check_rate_limit(tokens=4)
    assert allowed_3 is True
    assert limiter.request_count == 1
    assert limiter.token_count == 4


# ---------------------------------------------------------------------------
# CostTracker ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_cost_tracker_usage_and_budget():
    """Valida cálculo de custos, rastreamento e verificação de orçamento."""
    pricing = {EmbeddingProvider.OPENAI: {"input": 0.01}}
    tracker = CostTracker(pricing=pricing, daily_budget=1.0)

    # Cálculo simples ------------------------------------------------------
    cost = tracker.calculate_cost(EmbeddingProvider.OPENAI, tokens=20)
    assert math.isclose(cost, 0.20, rel_tol=1e-9)

    # Dentro do orçamento --------------------------------------------------
    assert tracker.check_budget(cost) is True

    # Registrar uso --------------------------------------------------------
    tracker.track_usage(EmbeddingProvider.OPENAI, tokens=20, cost=cost)
    stats = tracker.get_usage_stats()
    assert stats["total_tokens"] == 20
    assert math.isclose(stats["total_cost"], 0.20, rel_tol=1e-9)

    # Excedendo orçamento --------------------------------------------------
    assert tracker.check_budget(1.0) is False  # 0.2 já gasto + 1.0 > 1.0


# ---------------------------------------------------------------------------
# APIEmbeddingService: batch size adaptação ----------------------------------
# ---------------------------------------------------------------------------

def test_api_embedding_service_adaptive_batch_size():
    """Garante ajuste de batch_size quando textos são longos."""
    cfg = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        api_key="dummy-key",
        batch_size=100,
    )
    service = APIEmbeddingService(cfg)

    # Textos curtos (média <= 20 tokens) ----------------------------------
    short_texts = ["palavra"] * 30  # 1 token cada
    assert service._adaptive_batch_size(short_texts) == 100  # tamanho original

    # Textos longos (média > 20 tokens) -----------------------------------
    long_sentence = " ".join(["token"] * 25)  # 25 tokens
    long_texts = [long_sentence] * 30
    assert service._adaptive_batch_size(long_texts) == 50  # metade do batch 