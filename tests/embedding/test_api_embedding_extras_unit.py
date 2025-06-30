import pytest

from src.embeddings.api_embedding_service import (
    EmbeddingConfig,
    EmbeddingProvider,
    APIEmbeddingService,
    RateLimiter,
    CostTracker,
)


@pytest.fixture
def svc():
    cfg = EmbeddingConfig(provider=EmbeddingProvider.OPENAI, api_key="key", batch_size=10)
    return APIEmbeddingService(cfg, rate_limiter=RateLimiter(60, 1000), cost_tracker=CostTracker({EmbeddingProvider.OPENAI: {"input": 0.001}}, 1))


def test_embed_batch_empty_raises(svc):
    with pytest.raises(ValueError):
        import asyncio
        asyncio.run(svc.embed_batch([]))


def test_estimate_cost(svc):
    texts = ["palavra"] * 5  # 5 tokens aprox
    cost = svc.estimate_cost(texts)
    assert cost == 5 * 0.001 