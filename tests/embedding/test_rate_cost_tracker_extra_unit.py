import asyncio
from unittest.mock import patch
import pytest
from src.embeddings.api_embedding_service import RateLimiter, CostTracker, EmbeddingProvider


# ---------------------------------------------------------------------------
# RateLimiter.wait_for_reset -------------------------------------------------
# ---------------------------------------------------------------------------

@patch("asyncio.sleep", autospec=True)
@patch("time.monotonic", autospec=True)
@pytest.mark.asyncio
async def test_rate_limiter_wait_for_reset(mock_monotonic, mock_sleep):
    # Configura monotonic para simular janela cheia e quase expirada
    start = 100.0
    mock_monotonic.side_effect = [start, start + 30, start + 61]  # third call after sleep
    limiter = RateLimiter(1, 10)
    limiter.request_count = 1  # saturado
    # Deve chamar sleep com 30s (60 - 30)
    await limiter.wait_for_reset()
    mock_sleep.assert_called_once_with(30.0)
    # Após reset, contadores voltam a zero
    assert limiter.request_count == 0 and limiter.token_count == 0


# ---------------------------------------------------------------------------
# CostTracker.reset_daily_stats --------------------------------------------
# ---------------------------------------------------------------------------

def test_cost_tracker_reset_daily_stats():
    pricing = {EmbeddingProvider.OPENAI: {"input": 0.01}}
    tracker = CostTracker(pricing, daily_budget=5.0)
    tracker.track_usage(EmbeddingProvider.OPENAI, 100, 1.0)
    assert tracker.current_spend > 0

    tracker.reset_daily_stats()
    assert tracker.current_spend == 0.0
    stats = tracker.get_usage_stats()
    assert stats["total_cost"] == 0.0 and stats["total_tokens"] == 0 