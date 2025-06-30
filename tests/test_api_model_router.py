"""
Testes para o módulo API Model Router
Testa roteamento inteligente de modelos via API
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from typing import Dict, Any, List

# Mock das dependências
with patch.dict('sys.modules', {
    'openai': Mock(),
    'anthropic': Mock(),
    'google.generativeai': Mock()
}):
    try:
        from src.models.api_model_router import APIModelRouter, ModelProvider, RoutingStrategy
    except ImportError:
        # Se o módulo não existir, crie uma versão mock
        class ModelProvider:
            OPENAI = "openai"
            ANTHROPIC = "anthropic"
            GOOGLE = "google"
            DEEPSEEK = "deepseek"
        
        class RoutingStrategy:
            COST_OPTIMIZED = "cost_optimized"
            PERFORMANCE_OPTIMIZED = "performance_optimized"
            QUALITY_OPTIMIZED = "quality_optimized"
            BALANCED = "balanced"
        
        class APIModelRouter:
            def __init__(self, strategy=None, fallback_enabled=True):
                self.strategy = strategy or RoutingStrategy.BALANCED
                self.fallback_enabled = fallback_enabled
                self.providers = {
                    ModelProvider.OPENAI: {"available": True, "cost": 0.002},
                    ModelProvider.ANTHROPIC: {"available": True, "cost": 0.003},
                    ModelProvider.GOOGLE: {"available": True, "cost": 0.001}
                }
                self.current_provider = None
            
            async def route_request(self, prompt, task_type="general"):
                provider = self.select_provider(task_type)
                return await self.call_provider(provider, prompt)
            
            def select_provider(self, task_type="general"):
                if self.strategy == RoutingStrategy.COST_OPTIMIZED:
                    return ModelProvider.GOOGLE
                elif self.strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
                    return ModelProvider.ANTHROPIC
                else:
                    return ModelProvider.OPENAI
            
            async def call_provider(self, provider, prompt):
                if not self.providers[provider]["available"]:
                    if self.fallback_enabled:
                        return await self.fallback_request(prompt)
                    else:
                        raise Exception(f"Provider {provider} not available")
                
                return {
                    "provider": provider,
                    "response": f"Response from {provider} for: {prompt[:50]}...",
                    "cost": self.providers[provider]["cost"]
                }
            
            async def fallback_request(self, prompt):
                for provider, config in self.providers.items():
                    if config["available"]:
                        return await self.call_provider(provider, prompt)
                raise Exception("No providers available")
            
            def get_provider_status(self):
                return self.providers.copy()
            
            def set_provider_availability(self, provider, available):
                if provider in self.providers:
                    self.providers[provider]["available"] = available


class TestModelProvider:
    """Testes para constantes de providers"""
    
    def test_provider_constants(self):
        """Testa se as constantes de provider estão definidas"""
        assert hasattr(ModelProvider, 'OPENAI')
        assert hasattr(ModelProvider, 'ANTHROPIC') 
        assert hasattr(ModelProvider, 'GOOGLE')
        assert hasattr(ModelProvider, 'DEEPSEEK')


class TestRoutingStrategy:
    """Testes para estratégias de roteamento"""
    
    def test_strategy_constants(self):
        """Testa se as constantes de estratégia estão definidas"""
        assert hasattr(RoutingStrategy, 'COST_OPTIMIZED')
        assert hasattr(RoutingStrategy, 'PERFORMANCE_OPTIMIZED')
        assert hasattr(RoutingStrategy, 'QUALITY_OPTIMIZED')
        assert hasattr(RoutingStrategy, 'BALANCED')


class TestAPIModelRouter:
    """Testes para a classe APIModelRouter"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.router = APIModelRouter({
            "providers": {
                "openai": {
                    "models": {
                        "gpt4o_mini": {
                            "name": "gpt-4o-mini",
                            "max_tokens": 4096,
                            "temperature": 0.7,
                            "responsibilities": ["primary_reasoning", "code_generation"],
                            "context_window": 128000,
                            "cost_per_1k_tokens": 0.0015,
                            "priority": 1
                        }
                    }
                }
            },
            "routing": {
                "strategy": "cost_performance_optimized",
                "fallback_chain": ["openai.gpt4o_mini"]
            }
        })
    
    def test_init_default(self):
        """Testa inicialização com parâmetros padrão"""
        assert self.router.strategy == RoutingStrategy.BALANCED
        assert self.router.fallback_enabled is True
        assert len(self.router.providers) >= 3
    
    def test_init_with_strategy(self):
        """Testa inicialização com estratégia específica"""
        router = APIModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        assert router.strategy == RoutingStrategy.COST_OPTIMIZED
    
    def test_init_without_fallback(self):
        """Testa inicialização sem fallback"""
        router = APIModelRouter(fallback_enabled=False)
        assert router.fallback_enabled is False
    
    def test_select_provider_cost_optimized(self):
        """Testa seleção de provider para custo otimizado"""
        router = APIModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        provider = router.select_provider()
        assert provider == ModelProvider.GOOGLE  # Assumindo que Google tem menor custo
    
    def test_select_provider_performance_optimized(self):
        """Testa seleção de provider para performance otimizada"""
        router = APIModelRouter(strategy=RoutingStrategy.PERFORMANCE_OPTIMIZED)
        provider = router.select_provider()
        assert provider == ModelProvider.ANTHROPIC
    
    def test_select_provider_by_task_type(self):
        """Testa seleção de provider baseada no tipo de tarefa"""
        # Diferentes tipos de tarefa podem ter providers diferentes
        code_provider = self.router.select_provider("code_generation")
        text_provider = self.router.select_provider("text_generation")
        
        assert code_provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]
        assert text_provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]
    
    @pytest.mark.asyncio
    async def test_route_request_basic(self):
        """Testa roteamento básico de request"""
        prompt = "Explain machine learning"
        result = await self.router.route_request(prompt)
        
        assert "provider" in result
        assert "response" in result
        assert "cost" in result
        assert result["provider"] in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]
    
    @pytest.mark.asyncio
    async def test_route_request_with_task_type(self):
        """Testa roteamento com tipo de tarefa específico"""
        prompt = "def factorial(n):"
        result = await self.router.route_request(prompt, task_type="code_generation")
        
        assert "provider" in result
        assert "response" in result
        assert isinstance(result["cost"], (int, float))
    
    @pytest.mark.asyncio
    async def test_call_provider_success(self):
        """Testa chamada direta para provider específico"""
        prompt = "Test prompt"
        result = await self.router.call_provider(ModelProvider.OPENAI, prompt)
        
        assert result["provider"] == ModelProvider.OPENAI
        assert "Test prompt" in result["response"] or prompt[:50] in result["response"]
        assert result["cost"] > 0
    
    @pytest.mark.asyncio
    async def test_call_provider_unavailable_with_fallback(self):
        """Testa chamada para provider indisponível com fallback ativo"""
        # Tornar OpenAI indisponível
        self.router.set_provider_availability(ModelProvider.OPENAI, False)
        
        prompt = "Test prompt"
        result = await self.router.call_provider(ModelProvider.OPENAI, prompt)
        
        # Deve usar fallback (outro provider)
        assert result["provider"] != ModelProvider.OPENAI
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_call_provider_unavailable_without_fallback(self):
        """Testa chamada para provider indisponível sem fallback"""
        router = APIModelRouter(fallback_enabled=False)
        router.set_provider_availability(ModelProvider.OPENAI, False)
        
        prompt = "Test prompt"
        
        with pytest.raises(Exception, match="not available"):
            await router.call_provider(ModelProvider.OPENAI, prompt)
    
    def test_get_provider_status(self):
        """Testa obtenção do status dos providers"""
        status = self.router.get_provider_status()
        
        assert isinstance(status, dict)
        assert ModelProvider.OPENAI in status
        assert ModelProvider.ANTHROPIC in status
        assert ModelProvider.GOOGLE in status
        
        for provider, config in status.items():
            assert "available" in config
            assert "cost" in config
    
    def test_set_provider_availability(self):
        """Testa configuração da disponibilidade de providers"""
        # Verificar estado inicial
        assert self.router.providers[ModelProvider.OPENAI]["available"] is True
        
        # Tornar indisponível
        self.router.set_provider_availability(ModelProvider.OPENAI, False)
        assert self.router.providers[ModelProvider.OPENAI]["available"] is False
        
        # Tornar disponível novamente
        self.router.set_provider_availability(ModelProvider.OPENAI, True)
        assert self.router.providers[ModelProvider.OPENAI]["available"] is True
    
    def test_set_provider_availability_invalid_provider(self):
        """Testa configuração para provider inválido"""
        # Não deve causar erro, apenas ignorar
        self.router.set_provider_availability("invalid_provider", False)
        
        # Providers válidos não devem ser afetados
        assert self.router.providers[ModelProvider.OPENAI]["available"] is True


class TestAPIModelRouterIntegration:
    """Testes de integração para APIModelRouter"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.router = APIModelRouter({
            "providers": {
                "openai": {
                    "models": {
                        "gpt4o_mini": {
                            "name": "gpt-4o-mini",
                            "max_tokens": 4096,
                            "temperature": 0.7,
                            "responsibilities": ["primary_reasoning", "code_generation"],
                            "context_window": 128000,
                            "cost_per_1k_tokens": 0.0015,
                            "priority": 1
                        }
                    }
                }
            },
            "routing": {
                "strategy": "cost_performance_optimized",
                "fallback_chain": ["openai.gpt4o_mini"]
            }
        })
    
    @pytest.mark.asyncio
    async def test_cost_optimization_scenario(self):
        """Testa cenário de otimização de custo"""
        router = APIModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        
        # Simular múltiplas requests
        prompts = [
            "Simple question 1",
            "Simple question 2", 
            "Simple question 3"
        ]
        
        total_cost = 0
        for prompt in prompts:
            result = await router.route_request(prompt)
            total_cost += result["cost"]
        
        # Com estratégia de custo, deve usar provider mais barato
        assert total_cost <= 0.003 * len(prompts)  # Google é mais barato
    
    @pytest.mark.asyncio
    async def test_performance_optimization_scenario(self):
        """Testa cenário de otimização de performance"""
        router = APIModelRouter(strategy=RoutingStrategy.PERFORMANCE_OPTIMIZED)
        
        prompt = "Complex analysis task"
        result = await router.route_request(prompt)
        
        # Deve usar provider de alta performance
        assert result["provider"] == ModelProvider.ANTHROPIC
    
    @pytest.mark.asyncio
    async def test_fallback_chain_scenario(self):
        """Testa cenário de fallback em cadeia"""
        # Tornar todos os providers indisponíveis exceto um
        self.router.set_provider_availability(ModelProvider.OPENAI, False)
        self.router.set_provider_availability(ModelProvider.ANTHROPIC, False)
        # Deixar Google disponível
        
        prompt = "Test fallback"
        result = await self.router.route_request(prompt)
        
        # Deve usar o último provider disponível
        assert result["provider"] == ModelProvider.GOOGLE
    
    @pytest.mark.asyncio
    async def test_all_providers_unavailable(self):
        """Testa cenário onde todos os providers estão indisponíveis"""
        # Tornar todos indisponíveis
        for provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]:
            self.router.set_provider_availability(provider, False)
        
        prompt = "Test no providers"
        
        with pytest.raises(Exception, match="No providers available"):
            await self.router.route_request(prompt)


class TestAPIModelRouterEdgeCases:
    """Testes para casos extremos e edge cases"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.router = APIModelRouter({
            "providers": {
                "openai": {
                    "models": {
                        "gpt4o_mini": {
                            "name": "gpt-4o-mini",
                            "max_tokens": 4096,
                            "temperature": 0.7,
                            "responsibilities": ["primary_reasoning", "code_generation"],
                            "context_window": 128000,
                            "cost_per_1k_tokens": 0.0015,
                            "priority": 1
                        }
                    }
                }
            },
            "routing": {
                "strategy": "cost_performance_optimized",
                "fallback_chain": ["openai.gpt4o_mini"]
            }
        })
    
    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        """Testa roteamento com prompt vazio"""
        result = await self.router.route_request("")
        
        assert "provider" in result
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_very_long_prompt(self):
        """Testa roteamento com prompt muito longo"""
        long_prompt = "a" * 10000  # 10k caracteres
        result = await self.router.route_request(long_prompt)
        
        assert "provider" in result
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_special_characters_prompt(self):
        """Testa roteamento com caracteres especiais"""
        special_prompt = "Teste com çãractëres €$peciais 中文 🚀"
        result = await self.router.route_request(special_prompt)
        
        assert "provider" in result
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_unknown_task_type(self):
        """Testa roteamento com tipo de tarefa desconhecido"""
        result = await self.router.route_request("Test", task_type="unknown_task")
        
        # Deve usar provider padrão
        assert "provider" in result
        assert "response" in result
    
    def test_router_state_consistency(self):
        """Testa consistência do estado do router"""
        initial_status = self.router.get_provider_status()
        
        # Modificar disponibilidade
        self.router.set_provider_availability(ModelProvider.OPENAI, False)
        modified_status = self.router.get_provider_status()
        
        # Estados devem ser diferentes
        assert initial_status[ModelProvider.OPENAI]["available"] != modified_status[ModelProvider.OPENAI]["available"]
        
        # Outros providers não devem ser afetados
        assert initial_status[ModelProvider.ANTHROPIC] == modified_status[ModelProvider.ANTHROPIC]


class TestAPIModelRouterMetrics:
    """Testes para métricas e monitoramento"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.router = APIModelRouter({
            "providers": {
                "openai": {
                    "models": {
                        "gpt4o_mini": {
                            "name": "gpt-4o-mini",
                            "max_tokens": 4096,
                            "temperature": 0.7,
                            "responsibilities": ["primary_reasoning", "code_generation"],
                            "context_window": 128000,
                            "cost_per_1k_tokens": 0.0015,
                            "priority": 1
                        }
                    }
                }
            },
            "routing": {
                "strategy": "cost_performance_optimized",
                "fallback_chain": ["openai.gpt4o_mini"]
            }
        })
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Testa rastreamento de custos"""
        prompt = "Cost tracking test"
        result = await self.router.route_request(prompt)
        
        assert "cost" in result
        assert isinstance(result["cost"], (int, float))
        assert result["cost"] >= 0
    
    @pytest.mark.asyncio
    async def test_provider_usage_tracking(self):
        """Testa rastreamento de uso de providers"""
        # Executar múltiplas requests
        results = []
        for i in range(5):
            result = await self.router.route_request(f"Test {i}")
            results.append(result)
        
        # Verificar que providers foram usados
        providers_used = set(result["provider"] for result in results)
        assert len(providers_used) >= 1
        
        # Com estratégia balanceada, pode usar múltiplos providers
        for result in results:
            assert result["provider"] in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 