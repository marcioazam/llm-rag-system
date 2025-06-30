import asyncio
from datetime import timedelta

from src.embeddings.api_embedding_service import (
    RateLimiter,
    CostTracker,
    EmbeddingProvider,
    EmbeddingConfig,
    APIEmbeddingService,
)


def test_rate_limiter_basic():
    rl = RateLimiter(requests_per_minute=3, tokens_per_minute=20)

    async def _run():
        # 3 requisições de 5 tokens cada (total 15) → todas permitidas
        for _ in range(3):
            allowed = await rl.check_rate_limit(5)
            assert allowed is True
        # Quarta requisição excede limite de requests
        allowed = await rl.check_rate_limit(1)
        assert allowed is False
    asyncio.run(_run())


def test_cost_tracker_flow():
    pricing = {EmbeddingProvider.OPENAI: {"input": 0.002}}
    ct = CostTracker(pricing, daily_budget=0.05)
    cost = ct.calculate_cost(EmbeddingProvider.OPENAI, 10)
    assert cost == 0.02
    assert ct.check_budget(cost) is True
    ct.track_usage(EmbeddingProvider.OPENAI, 10, cost)
    stats = ct.get_usage_stats()
    assert stats["total_tokens"] == 10 and stats["total_cost"] == cost
    # Exceder orçamento
    assert ct.check_budget(0.04) is False


def test_adaptive_batch_size():
    cfg = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        api_key="dummy",
        batch_size=100,
    )
    svc = APIEmbeddingService(cfg, rate_limiter=RateLimiter(60, 1000), cost_tracker=CostTracker({}, 10))
    texts_small = ["word"] * 10  # avg tokens 1
    texts_large = ["palavra " * 25] * 10  # avg tokens 25
    assert svc._adaptive_batch_size(texts_small) == 100  # não reduz
    assert svc._adaptive_batch_size(texts_large) == 50  # reduz pela metade 