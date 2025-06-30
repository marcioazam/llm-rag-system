import math
from src.embeddings.api_embedding_service import (
    EmbeddingResponse,
    EmbeddingProvider,
    EmbeddingConfig,
    APIEmbeddingService,
)


# ---------------------------------------------------------------------------
# EmbeddingResponse ----------------------------------------------------------
# ---------------------------------------------------------------------------

def test_embedding_response_empty_properties():
    """Verifica propriedades quando não há embeddings."""
    resp = EmbeddingResponse(embeddings=[], model="test-model")
    assert resp.num_embeddings == 0
    assert resp.embedding_dimension == 0
    assert resp.total_tokens == 0
    # Índices inválidos devem retornar None
    assert resp.get_embedding(0) is None
    assert resp.get_embedding(-1) is None


def test_embedding_response_basic_properties():
    """Propriedades derivadas devem refletir dados fornecidos."""
    data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    usage = {"total_tokens": 12}
    resp = EmbeddingResponse(embeddings=data, model="m", usage=usage)

    assert resp.num_embeddings == 2
    assert resp.embedding_dimension == 3
    assert resp.total_tokens == 12
    assert resp.get_embedding(1) == [0.4, 0.5, 0.6]


# ---------------------------------------------------------------------------
# APIEmbeddingService utilitários -------------------------------------------
# ---------------------------------------------------------------------------


def _make_service(batch_size: int = 50):
    cfg = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        api_key="dummy-key",
        batch_size=batch_size,
    )
    return APIEmbeddingService(cfg)


def test_api_embedding_service_endpoint_headers_payload():
    """Verifica helpers de endpoint, headers e payload."""
    svc = _make_service()

    # Endpoint contém provedor
    endpoint = svc._get_api_endpoint()
    assert "/openai/embeddings" in endpoint

    # Headers contêm Authorization e Content-Type
    headers = svc._get_headers()
    assert "Authorization" in headers and "Bearer" in headers["Authorization"]
    assert headers["Content-Type"] == "application/json"

    # Payload deve refletir entrada e modelo
    payload = svc._prepare_request_payload(["hello", "world"])
    assert payload["model"] == svc.config.model
    assert payload["input"] == ["hello", "world"]


def test_api_embedding_service_supported_models_and_info():
    svc = _make_service()
    models = svc.get_supported_models()
    assert svc.config.model in models

    info = svc.get_model_info()
    assert info["model"] == svc.config.model
    assert info["dimensions"] == svc.config.dimensions
    assert info["provider"] == svc.config.provider


def test_api_embedding_service_estimate_cost_and_performance():
    svc = _make_service()

    texts = ["token one", "two tokens", "three token words"]
    est_cost = svc.estimate_cost(texts)

    # Custo estimado deve ser proporcional ao número de tokens
    total_tokens = sum(len(t.split()) for t in texts)
    expected_cost = svc.cost_tracker.calculate_cost(svc.config.provider, total_tokens)
    assert math.isclose(est_cost, expected_cost, rel_tol=1e-9)

    # Métricas de performance vazias inicialmente
    stats = svc.get_performance_stats()
    assert stats["total_requests"] == 0
    assert stats["total_tokens"] == 0
    assert stats["average_latency"] == 0.0

    # Atualiza métricas manualmente
    svc._track_performance(latency=0.2, tokens=10)
    svc._track_performance(latency=0.4, tokens=20)
    stats2 = svc.get_performance_stats()
    assert stats2["total_requests"] == 2
    assert stats2["total_tokens"] == 30
    assert math.isclose(stats2["average_latency"], 0.3, rel_tol=1e-3) 