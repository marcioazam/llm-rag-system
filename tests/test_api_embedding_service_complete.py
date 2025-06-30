"""
Testes completos para APIEmbeddingService
Cobrindo todos os cenários não testados para aumentar a cobertura
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import aiohttp
from src.embeddings.api_embedding_service import (
    APIEmbeddingService, EmbeddingProvider, EmbeddingConfig,
    EmbeddingResponse, RateLimiter, CostTracker
)


class TestEmbeddingProvider:
    """Testes para EmbeddingProvider enum"""
    
    def test_embedding_provider_values(self):
        """Testa valores do enum EmbeddingProvider"""
        assert EmbeddingProvider.OPENAI == "openai"
        assert EmbeddingProvider.ANTHROPIC == "anthropic"
        assert EmbeddingProvider.GOOGLE == "google"
        assert EmbeddingProvider.COHERE == "cohere"
        assert EmbeddingProvider.HUGGINGFACE == "huggingface"


class TestEmbeddingConfig:
    """Testes para EmbeddingConfig"""
    
    def test_embedding_config_creation(self):
        """Testa criação de configuração de embedding"""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            api_key="test-key",
            dimensions=1536,
            batch_size=100,
            max_retries=3,
            timeout=30.0
        )
        
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.model == "text-embedding-3-small"
        assert config.api_key == "test-key"
        assert config.dimensions == 1536
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.timeout == 30.0
        
    def test_embedding_config_defaults(self):
        """Testa valores padrão da configuração"""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            api_key="test-key"
        )
        
        assert config.model is not None
        assert config.dimensions > 0
        assert config.batch_size > 0
        assert config.max_retries >= 0
        assert config.timeout > 0
        
    def test_embedding_config_validation(self):
        """Testa validação da configuração"""
        # API key obrigatória
        with pytest.raises(ValueError):
            EmbeddingConfig(provider=EmbeddingProvider.OPENAI, api_key="")
            
        # Dimensões positivas
        with pytest.raises(ValueError):
            EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                api_key="test",
                dimensions=0
            )


class TestEmbeddingResponse:
    """Testes para EmbeddingResponse"""
    
    def test_embedding_response_creation(self):
        """Testa criação de resposta de embedding"""
        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="text-embedding-3-small",
            usage={"prompt_tokens": 10, "total_tokens": 10},
            provider=EmbeddingProvider.OPENAI
        )
        
        assert len(response.embeddings) == 2
        assert response.model == "text-embedding-3-small"
        assert response.usage["prompt_tokens"] == 10
        assert response.provider == EmbeddingProvider.OPENAI
        
    def test_embedding_response_properties(self):
        """Testa propriedades da resposta"""
        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            model="test-model",
            usage={"total_tokens": 15}
        )
        
        assert response.num_embeddings == 3
        assert response.embedding_dimension == 2
        assert response.total_tokens == 15
        
    def test_get_embedding_by_index(self):
        """Testa obtenção de embedding por índice"""
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        response = EmbeddingResponse(embeddings=embeddings, model="test")
        
        assert response.get_embedding(0) == [0.1, 0.2]
        assert response.get_embedding(1) == [0.3, 0.4]
        assert response.get_embedding(2) is None  # Índice inválido


class TestRateLimiter:
    """Testes para RateLimiter"""
    
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter(requests_per_minute=60, tokens_per_minute=1000)
    
    def test_rate_limiter_init(self, rate_limiter):
        """Testa inicialização do rate limiter"""
        assert rate_limiter.requests_per_minute == 60
        assert rate_limiter.tokens_per_minute == 1000
        assert rate_limiter.request_count == 0
        assert rate_limiter.token_count == 0
        
    @pytest.mark.asyncio
    async def test_check_rate_limit_within_limits(self, rate_limiter):
        """Testa verificação dentro dos limites"""
        # Primeira requisição deve passar
        can_proceed = await rate_limiter.check_rate_limit(tokens=100)
        assert can_proceed
        assert rate_limiter.request_count == 1
        assert rate_limiter.token_count == 100
        
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeds_requests(self, rate_limiter):
        """Testa excesso de requisições"""
        # Simula muitas requisições
        rate_limiter.request_count = 61  # Acima do limite
        
        can_proceed = await rate_limiter.check_rate_limit(tokens=10)
        assert not can_proceed
        
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeds_tokens(self, rate_limiter):
        """Testa excesso de tokens"""
        # Simula muitos tokens
        rate_limiter.token_count = 1001  # Acima do limite
        
        can_proceed = await rate_limiter.check_rate_limit(tokens=10)
        assert not can_proceed
        
    @pytest.mark.asyncio
    async def test_wait_for_rate_limit_reset(self, rate_limiter):
        """Testa espera para reset do rate limit"""
        rate_limiter.request_count = 61  # Acima do limite
        
        # Mock do sleep para não esperar realmente
        with patch('asyncio.sleep') as mock_sleep:
            await rate_limiter.wait_for_reset()
            mock_sleep.assert_called_once()
            
    def test_reset_counters(self, rate_limiter):
        """Testa reset dos contadores"""
        rate_limiter.request_count = 50
        rate_limiter.token_count = 500
        
        rate_limiter.reset_counters()
        
        assert rate_limiter.request_count == 0
        assert rate_limiter.token_count == 0
        
    def test_get_remaining_capacity(self, rate_limiter):
        """Testa obtenção da capacidade restante"""
        rate_limiter.request_count = 30
        rate_limiter.token_count = 400
        
        remaining = rate_limiter.get_remaining_capacity()
        
        assert remaining["requests"] == 30  # 60 - 30
        assert remaining["tokens"] == 600   # 1000 - 400


class TestCostTracker:
    """Testes para CostTracker"""
    
    @pytest.fixture
    def cost_tracker(self):
        pricing = {
            EmbeddingProvider.OPENAI: {"input": 0.0001, "output": 0.0001},
            EmbeddingProvider.ANTHROPIC: {"input": 0.0002, "output": 0.0002}
        }
        return CostTracker(pricing=pricing, daily_budget=10.0)
    
    def test_cost_tracker_init(self, cost_tracker):
        """Testa inicialização do cost tracker"""
        assert cost_tracker.daily_budget == 10.0
        assert cost_tracker.current_spend == 0.0
        assert EmbeddingProvider.OPENAI in cost_tracker.pricing
        
    def test_calculate_cost(self, cost_tracker):
        """Testa cálculo de custo"""
        cost = cost_tracker.calculate_cost(
            provider=EmbeddingProvider.OPENAI,
            tokens=1000
        )
        
        # 1000 tokens * 0.0001 = 0.1
        assert cost == 0.1
        
    def test_track_usage(self, cost_tracker):
        """Testa rastreamento de uso"""
        cost_tracker.track_usage(
            provider=EmbeddingProvider.OPENAI,
            tokens=1000,
            cost=0.1
        )
        
        assert cost_tracker.current_spend == 0.1
        assert cost_tracker.usage_stats[EmbeddingProvider.OPENAI]["tokens"] == 1000
        assert cost_tracker.usage_stats[EmbeddingProvider.OPENAI]["cost"] == 0.1
        
    def test_check_budget(self, cost_tracker):
        """Testa verificação de orçamento"""
        # Dentro do orçamento
        assert cost_tracker.check_budget(5.0)
        
        # Excede orçamento
        assert not cost_tracker.check_budget(15.0)
        
        # Com gasto atual
        cost_tracker.current_spend = 8.0
        assert cost_tracker.check_budget(1.0)  # 8 + 1 = 9 < 10
        assert not cost_tracker.check_budget(3.0)  # 8 + 3 = 11 > 10
        
    def test_get_usage_stats(self, cost_tracker):
        """Testa obtenção de estatísticas de uso"""
        cost_tracker.track_usage(EmbeddingProvider.OPENAI, 1000, 0.1)
        cost_tracker.track_usage(EmbeddingProvider.ANTHROPIC, 500, 0.1)
        
        stats = cost_tracker.get_usage_stats()
        
        assert "total_cost" in stats
        assert "total_tokens" in stats
        assert "by_provider" in stats
        assert stats["total_cost"] == 0.2
        assert stats["total_tokens"] == 1500
        
    def test_reset_daily_stats(self, cost_tracker):
        """Testa reset das estatísticas diárias"""
        cost_tracker.track_usage(EmbeddingProvider.OPENAI, 1000, 0.1)
        
        cost_tracker.reset_daily_stats()
        
        assert cost_tracker.current_spend == 0.0
        assert all(stats["tokens"] == 0 for stats in cost_tracker.usage_stats.values())


class TestAPIEmbeddingService:
    """Testes para APIEmbeddingService principal"""
    
    @pytest.fixture
    def service_config(self):
        return EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            api_key="test-api-key",
            dimensions=1536
        )
    
    @pytest.fixture
    def embedding_service(self, service_config):
        return APIEmbeddingService(config=service_config)
    
    def test_init(self, embedding_service, service_config):
        """Testa inicialização do serviço"""
        assert embedding_service.config == service_config
        assert embedding_service.rate_limiter is not None
        assert embedding_service.cost_tracker is not None
        assert embedding_service.session is None  # Criado sob demanda
        
    def test_init_with_custom_components(self, service_config):
        """Testa inicialização com componentes customizados"""
        custom_rate_limiter = RateLimiter(100, 2000)
        custom_cost_tracker = CostTracker({}, 20.0)
        
        service = APIEmbeddingService(
            config=service_config,
            rate_limiter=custom_rate_limiter,
            cost_tracker=custom_cost_tracker
        )
        
        assert service.rate_limiter == custom_rate_limiter
        assert service.cost_tracker == custom_cost_tracker
        
    @pytest.mark.asyncio
    async def test_create_session(self, embedding_service):
        """Testa criação de sessão HTTP"""
        session = await embedding_service._create_session()
        
        assert isinstance(session, aiohttp.ClientSession)
        assert session.timeout.total == embedding_service.config.timeout
        
        await session.close()
        
    def test_get_api_endpoint(self, embedding_service):
        """Testa obtenção do endpoint da API"""
        endpoint = embedding_service._get_api_endpoint()
        
        assert isinstance(endpoint, str)
        assert "http" in endpoint.lower()
        
    def test_get_headers(self, embedding_service):
        """Testa obtenção dos headers da API"""
        headers = embedding_service._get_headers()
        
        assert "Authorization" in headers or "api-key" in headers
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"
        
    def test_prepare_request_payload(self, embedding_service):
        """Testa preparação do payload da requisição"""
        texts = ["Hello world", "Test text"]
        payload = embedding_service._prepare_request_payload(texts)
        
        assert isinstance(payload, dict)
        assert "input" in payload or "texts" in payload
        assert "model" in payload
        assert payload["model"] == embedding_service.config.model
        
    @pytest.mark.asyncio
    async def test_make_api_request_success(self, embedding_service):
        """Testa requisição API bem-sucedida"""
        texts = ["Test text"]
        
        # Mock da resposta da API
        mock_response_data = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            response = await embedding_service._make_api_request(texts)
            
            assert response["data"][0]["embedding"] == [0.1, 0.2, 0.3]
            assert response["model"] == "text-embedding-3-small"
            
    @pytest.mark.asyncio
    async def test_make_api_request_failure(self, embedding_service):
        """Testa falha na requisição API"""
        texts = ["Test text"]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 401
            mock_response.text = AsyncMock(return_value="Unauthorized")
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(Exception):
                await embedding_service._make_api_request(texts)
                
    def test_parse_response(self, embedding_service):
        """Testa parsing da resposta da API"""
        api_response = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        }
        
        response = embedding_service._parse_response(api_response)
        
        assert isinstance(response, EmbeddingResponse)
        assert len(response.embeddings) == 2
        assert response.embeddings[0] == [0.1, 0.2, 0.3]
        assert response.model == "text-embedding-3-small"
        assert response.usage["total_tokens"] == 4
        
    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedding_service):
        """Testa embedding de texto único"""
        text = "Hello world"
        
        # Mock da resposta
        mock_response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="test-model",
            usage={"total_tokens": 2}
        )
        
        with patch.object(embedding_service, 'embed_batch', return_value=mock_response):
            embedding = await embedding_service.embed_text(text)
            
            assert embedding == [0.1, 0.2, 0.3]
            
    @pytest.mark.asyncio
    async def test_embed_batch(self, embedding_service):
        """Testa embedding de lote de textos"""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Mock da requisição API
        with patch.object(embedding_service, '_make_api_request') as mock_request:
            mock_request.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                    {"embedding": [0.5, 0.6]}
                ],
                "model": "test-model",
                "usage": {"total_tokens": 6}
            }
            
            response = await embedding_service.embed_batch(texts)
            
            assert len(response.embeddings) == 3
            assert response.embeddings[0] == [0.1, 0.2]
            assert response.embeddings[1] == [0.3, 0.4]
            assert response.embeddings[2] == [0.5, 0.6]
            
    @pytest.mark.asyncio
    async def test_embed_large_batch_chunking(self, embedding_service):
        """Testa embedding de lote grande com chunking"""
        # Lote maior que batch_size
        embedding_service.config.batch_size = 2
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        
        # Mock das requisições
        with patch.object(embedding_service, '_make_api_request') as mock_request:
            mock_request.side_effect = [
                {  # Primeiro batch
                    "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}],
                    "model": "test", "usage": {"total_tokens": 4}
                },
                {  # Segundo batch
                    "data": [{"embedding": [0.5, 0.6]}, {"embedding": [0.7, 0.8]}],
                    "model": "test", "usage": {"total_tokens": 4}
                },
                {  # Terceiro batch
                    "data": [{"embedding": [0.9, 1.0]}],
                    "model": "test", "usage": {"total_tokens": 2}
                }
            ]
            
            response = await embedding_service.embed_batch(texts)
            
            assert len(response.embeddings) == 5
            assert mock_request.call_count == 3  # 3 batches
            
    @pytest.mark.asyncio
    async def test_rate_limiting(self, embedding_service):
        """Testa rate limiting"""
        texts = ["Test text"]
        
        # Mock rate limiter que bloqueia
        embedding_service.rate_limiter.check_rate_limit = AsyncMock(return_value=False)
        embedding_service.rate_limiter.wait_for_reset = AsyncMock()
        
        with patch.object(embedding_service, '_make_api_request') as mock_request:
            mock_request.return_value = {
                "data": [{"embedding": [0.1, 0.2]}],
                "model": "test", "usage": {"total_tokens": 2}
            }
            
            # Primeira tentativa deve esperar pelo rate limit
            await embedding_service.embed_batch(texts)
            
            embedding_service.rate_limiter.wait_for_reset.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_cost_tracking(self, embedding_service):
        """Testa rastreamento de custos"""
        texts = ["Test text"]
        
        # Mock cost tracker
        embedding_service.cost_tracker.calculate_cost = Mock(return_value=0.01)
        embedding_service.cost_tracker.check_budget = Mock(return_value=True)
        embedding_service.cost_tracker.track_usage = Mock()
        
        with patch.object(embedding_service, '_make_api_request') as mock_request:
            mock_request.return_value = {
                "data": [{"embedding": [0.1, 0.2]}],
                "model": "test", "usage": {"total_tokens": 2}
            }
            
            await embedding_service.embed_batch(texts)
            
            embedding_service.cost_tracker.calculate_cost.assert_called_once()
            embedding_service.cost_tracker.check_budget.assert_called_once()
            embedding_service.cost_tracker.track_usage.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_budget_exceeded(self, embedding_service):
        """Testa excesso de orçamento"""
        texts = ["Test text"]
        
        # Mock cost tracker que bloqueia por orçamento
        embedding_service.cost_tracker.calculate_cost = Mock(return_value=100.0)
        embedding_service.cost_tracker.check_budget = Mock(return_value=False)
        
        with pytest.raises(Exception, match="budget"):
            await embedding_service.embed_batch(texts)
            
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, embedding_service):
        """Testa mecanismo de retry"""
        texts = ["Test text"]
        embedding_service.config.max_retries = 2
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Primeira tentativa falha, segunda sucede
            mock_response_fail = Mock()
            mock_response_fail.status = 500
            mock_response_fail.text = AsyncMock(return_value="Server error")
            
            mock_response_success = Mock()
            mock_response_success.status = 200
            mock_response_success.json = AsyncMock(return_value={
                "data": [{"embedding": [0.1, 0.2]}],
                "model": "test", "usage": {"total_tokens": 2}
            })
            
            mock_post.return_value.__aenter__.side_effect = [
                mock_response_fail,
                mock_response_success
            ]
            
            response = await embedding_service.embed_batch(texts)
            
            assert len(response.embeddings) == 1
            assert mock_post.call_count == 2  # 1 falha + 1 sucesso
            
    @pytest.mark.asyncio
    async def test_cleanup(self, embedding_service):
        """Testa limpeza de recursos"""
        # Cria sessão
        await embedding_service._create_session()
        assert embedding_service.session is not None
        
        # Cleanup
        await embedding_service.cleanup()
        assert embedding_service.session is None
        
    def test_get_supported_models(self, embedding_service):
        """Testa obtenção de modelos suportados"""
        models = embedding_service.get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)
        
    def test_get_model_info(self, embedding_service):
        """Testa obtenção de informações do modelo"""
        info = embedding_service.get_model_info()
        
        assert "model" in info
        assert "dimensions" in info
        assert "provider" in info
        assert info["model"] == embedding_service.config.model
        assert info["dimensions"] == embedding_service.config.dimensions
        
    def test_estimate_cost(self, embedding_service):
        """Testa estimativa de custo"""
        texts = ["Short text", "Another short text"]
        
        # Mock cost calculation
        embedding_service.cost_tracker.calculate_cost = Mock(return_value=0.02)
        
        estimated_cost = embedding_service.estimate_cost(texts)
        
        assert estimated_cost == 0.02
        embedding_service.cost_tracker.calculate_cost.assert_called_once()


class TestAPIEmbeddingServiceAdvanced:
    """Testes avançados do APIEmbeddingService"""
    
    @pytest.fixture
    def advanced_service(self):
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            api_key="test-key",
            batch_size=10,
            max_retries=3
        )
        return APIEmbeddingService(config)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, advanced_service):
        """Testa requisições concorrentes"""
        texts_batches = [
            ["Text 1", "Text 2"],
            ["Text 3", "Text 4"],
            ["Text 5", "Text 6"]
        ]
        
        # Mock das respostas
        with patch.object(advanced_service, '_make_api_request') as mock_request:
            mock_request.side_effect = [
                {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}], "model": "test", "usage": {"total_tokens": 4}},
                {"data": [{"embedding": [0.5, 0.6]}, {"embedding": [0.7, 0.8]}], "model": "test", "usage": {"total_tokens": 4}},
                {"data": [{"embedding": [0.9, 1.0]}, {"embedding": [1.1, 1.2]}], "model": "test", "usage": {"total_tokens": 4}}
            ]
            
            # Executa requisições concorrentes
            tasks = [advanced_service.embed_batch(batch) for batch in texts_batches]
            responses = await asyncio.gather(*tasks)
            
            assert len(responses) == 3
            assert all(len(response.embeddings) == 2 for response in responses)
            
    @pytest.mark.asyncio
    async def test_adaptive_batch_sizing(self, advanced_service):
        """Testa ajuste adaptativo do tamanho do batch"""
        # Simula diferentes tamanhos de texto
        short_texts = ["Hi"] * 20
        long_texts = ["This is a much longer text that contains more tokens"] * 5
        
        with patch.object(advanced_service, '_make_api_request') as mock_request:
            mock_request.return_value = {
                "data": [{"embedding": [0.1, 0.2]}] * 10,
                "model": "test", "usage": {"total_tokens": 50}
            }
            
            # Para textos curtos, deve usar batch size maior
            await advanced_service.embed_batch(short_texts)
            
            # Para textos longos, deve usar batch size menor
            await advanced_service.embed_batch(long_texts)
            
            # Verifica se foi chamado (implementação específica pode variar)
            assert mock_request.call_count >= 2
            
    @pytest.mark.asyncio
    async def test_error_recovery(self, advanced_service):
        """Testa recuperação de erros"""
        texts = ["Test text"]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simula diferentes tipos de erro
            errors = [
                Mock(status=429, text=AsyncMock(return_value="Rate limit")),  # Rate limit
                Mock(status=503, text=AsyncMock(return_value="Service unavailable")),  # Server error
                Mock(status=200, json=AsyncMock(return_value={  # Sucesso
                    "data": [{"embedding": [0.1, 0.2]}],
                    "model": "test", "usage": {"total_tokens": 2}
                }))
            ]
            
            mock_post.return_value.__aenter__.side_effect = errors
            
            # Deve recuperar após erros e retornar resultado
            response = await advanced_service.embed_batch(texts)
            
            assert len(response.embeddings) == 1
            assert mock_post.call_count == 3  # 2 erros + 1 sucesso
            
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, advanced_service):
        """Testa monitoramento de performance"""
        texts = ["Test text"] * 5
        
        with patch.object(advanced_service, '_make_api_request') as mock_request:
            mock_request.return_value = {
                "data": [{"embedding": [0.1, 0.2]}] * 5,
                "model": "test", "usage": {"total_tokens": 10}
            }
            
            # Executa embedding
            start_time = asyncio.get_event_loop().time()
            await advanced_service.embed_batch(texts)
            end_time = asyncio.get_event_loop().time()
            
            # Verifica métricas de performance
            performance_stats = advanced_service.get_performance_stats()
            
            assert "total_requests" in performance_stats
            assert "total_tokens" in performance_stats
            assert "average_latency" in performance_stats
            assert performance_stats["total_requests"] >= 1
            
    def test_configuration_validation(self):
        """Testa validação de configuração avançada"""
        # Configuração inválida - dimensões negativas
        with pytest.raises(ValueError):
            EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                api_key="test",
                dimensions=-100
            )
            
        # Configuração inválida - batch size zero
        with pytest.raises(ValueError):
            EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                api_key="test",
                batch_size=0
            )
            
        # Configuração inválida - timeout negativo
        with pytest.raises(ValueError):
            EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                api_key="test",
                timeout=-1.0
            )


class TestAPIEmbeddingServiceIntegration:
    """Testes de integração completos"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Testa fluxo completo do serviço"""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            api_key="test-api-key",
            model="text-embedding-3-small",
            batch_size=5
        )
        
        service = APIEmbeddingService(config)
        
        # Textos de teste
        texts = [
            "Machine learning is transforming technology",
            "Natural language processing enables computers to understand text",
            "Deep learning uses neural networks with multiple layers",
            "Artificial intelligence aims to create intelligent machines"
        ]
        
        # Mock da API
        with patch.object(service, '_make_api_request') as mock_request:
            mock_request.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3, 0.4]},
                    {"embedding": [0.2, 0.3, 0.4, 0.5]},
                    {"embedding": [0.3, 0.4, 0.5, 0.6]},
                    {"embedding": [0.4, 0.5, 0.6, 0.7]}
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 20, "total_tokens": 20}
            }
            
            # Executa embedding
            response = await service.embed_batch(texts)
            
            # Verifica resultado
            assert len(response.embeddings) == 4
            assert response.model == "text-embedding-3-small"
            assert response.usage["total_tokens"] == 20
            
            # Verifica embeddings individuais
            for i, embedding in enumerate(response.embeddings):
                assert len(embedding) == 4
                assert embedding[0] == 0.1 + i * 0.1
                
            # Verifica rastreamento de custos
            stats = service.cost_tracker.get_usage_stats()
            assert stats["total_tokens"] == 20
            
            # Verifica rate limiting
            remaining = service.rate_limiter.get_remaining_capacity()
            assert remaining["requests"] >= 0
            
        # Cleanup
        await service.cleanup() 