"""
Testes completos para o API Model Router.
Cobertura atual: 19% -> Meta: 80%
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.models.api_model_router import (
    APIModelRouter, TaskType, ModelResponse, ModelConfig
)


class TestAPIModelRouter:
    """Testes para o API Model Router."""

    @pytest.fixture
    def basic_config(self):
        """Configuração básica para testes."""
        return {
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "models": {
                        "gpt4o_mini": {
                            "name": "gpt-4o-mini",
                            "max_tokens": 4096,
                            "temperature": 0.1,
                            "responsibilities": ["primary_reasoning", "quick_queries"],
                            "context_window": 128000,
                            "cost_per_1k_tokens": 0.00015,
                            "priority": 1
                        },
                        "gpt4o": {
                            "name": "gpt-4o",
                            "max_tokens": 4096,
                            "temperature": 0.0,
                            "responsibilities": ["code_generation", "debugging", "architecture_design"],
                            "context_window": 128000,
                            "cost_per_1k_tokens": 0.005,
                            "priority": 2
                        }
                    }
                },
                "anthropic": {
                    "api_key": "test-key",
                    "models": {
                        "claude_3_5_sonnet": {
                            "name": "claude-3-5-sonnet-20241022",
                            "max_tokens": 8192,
                            "temperature": 0.0,
                            "responsibilities": ["document_analysis", "content_creation", "technical_writing"],
                            "context_window": 200000,
                            "cost_per_1k_tokens": 0.003,
                            "priority": 1
                        }
                    }
                }
            },
            "routing": {
                "strategy": "cost_performance_optimized",
                "fallback_chain": ["openai.gpt4o_mini", "anthropic.claude_3_5_sonnet"]
            }
        }

    @pytest.fixture
    def router(self, basic_config):
        """Criar instância do router."""
        return APIModelRouter(basic_config)

    def test_init(self, router, basic_config):
        """Testar inicialização do router."""
        assert router.config == basic_config
        assert len(router.available_models) == 3  # 2 OpenAI + 1 Anthropic
        assert "openai.gpt4o_mini" in router.available_models
        assert "openai.gpt4o" in router.available_models
        assert "anthropic.claude_3_5_sonnet" in router.available_models
        
        # Verificar estatísticas iniciais
        assert router.stats["total_requests"] == 0
        assert router.stats["total_cost"] == 0.0

    def test_load_available_models(self, router):
        """Testar carregamento de modelos disponíveis."""
        models = router.available_models
        
        # Verificar modelo OpenAI
        openai_model = models["openai.gpt4o_mini"]
        assert openai_model.name == "gpt-4o-mini"
        assert openai_model.max_tokens == 4096
        assert "primary_reasoning" in openai_model.responsibilities
        
        # Verificar modelo Anthropic
        anthropic_model = models["anthropic.claude_3_5_sonnet"]
        assert anthropic_model.name == "claude-3-5-sonnet-20241022"
        assert "document_analysis" in anthropic_model.responsibilities

    def test_detect_task_type_code_generation(self, router):
        """Testar detecção de tarefa de geração de código."""
        query = "Preciso implementar uma função para calcular fibonacci"
        task_type = router.detect_task_type(query)
        assert task_type == TaskType.CODE_GENERATION

    def test_detect_task_type_debugging(self, router):
        """Testar detecção de tarefa de debugging."""
        query = "Preciso fazer debug e corrigir um bug no sistema"
        task_type = router.detect_task_type(query)
        assert task_type == TaskType.DEBUGGING

    def test_detect_task_type_document_analysis(self, router):
        """Testar detecção de análise de documento."""
        query = "Preciso analisar este documento e extrair informações importantes"
        task_type = router.detect_task_type(query)
        assert task_type == TaskType.DOCUMENT_ANALYSIS

    def test_detect_task_type_fallback(self, router):
        """Testar fallback para tarefa geral."""
        query = "Uma pergunta genérica sobre qualquer coisa"
        task_type = router.detect_task_type(query)
        assert task_type == TaskType.GENERAL_EXPLANATION

    def test_select_best_model_code_generation(self, router):
        """Testar seleção de modelo para geração de código."""
        model = router.select_best_model(TaskType.CODE_GENERATION)
        assert model == "openai.gpt4o"  # Melhor para code_generation

    def test_select_best_model_quick_queries(self, router):
        """Testar seleção de modelo para queries rápidas."""
        model = router.select_best_model(TaskType.QUICK_QUERIES)
        assert model == "openai.gpt4o_mini"  # Mais barato para queries simples

    def test_select_best_model_document_analysis(self, router):
        """Testar seleção de modelo para análise de documento."""
        model = router.select_best_model(TaskType.DOCUMENT_ANALYSIS)
        assert model == "anthropic.claude_3_5_sonnet"  # Especializado em análise

    def test_select_best_model_context_window_limit(self, router):
        """Testar seleção considerando limite de contexto."""
        # Simular contexto muito grande que excede alguns modelos
        large_context_length = 150000
        model = router.select_best_model(TaskType.DOCUMENT_ANALYSIS, large_context_length)
        assert model == "anthropic.claude_3_5_sonnet"  # Tem maior context_window

    def test_select_best_model_no_suitable_fallback(self, router):
        """Testar fallback quando nenhum modelo é adequado."""
        # Modificar temporariamente os modelos para não ter responsabilidades adequadas
        original_models = router.available_models.copy()
        
        # Limpar responsabilidades
        for model in router.available_models.values():
            model.responsibilities = []
        
        try:
            model = router.select_best_model(TaskType.CODE_GENERATION)
            assert model == "openai.gpt4o_mini"  # Primeiro do fallback_chain
        finally:
            router.available_models = original_models

    @patch('requests.post')
    def test_generate_response_success(self, mock_post, router):
        """Testar geração de resposta bem-sucedida."""
        # Mock da resposta da API OpenAI
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Resposta do modelo"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "model": "gpt-4o-mini"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        query = "Pergunta simples"
        response = router.generate_response(query)

        assert isinstance(response, ModelResponse)
        assert response.content == "Resposta do modelo"
        assert response.provider == "openai"
        assert response.usage["total_tokens"] == 30
        assert response.finish_reason == "stop"

    @patch('requests.post')
    def test_generate_response_with_context(self, mock_post, router):
        """Testar geração de resposta com contexto."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Resposta"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 50},
            "model": "gpt-4o-mini"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        query = "Pergunta"
        context = "Contexto relevante"
        response = router.generate_response(query, context=context)

        assert response.content == "Resposta"
        # Verificar que a requisição incluiu o contexto
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        messages = request_data['messages']
        
        # Deve ter mensagem com contexto
        assert any("contexto" in msg['content'].lower() for msg in messages)

    @patch('requests.post')
    def test_generate_response_with_system_prompt(self, mock_post, router):
        """Testar geração de resposta com system prompt."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Resposta"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 50},
            "model": "gpt-4o-mini"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        query = "Pergunta"
        system_prompt = "Você é um assistente especializado"
        response = router.generate_response(query, system_prompt=system_prompt)

        # Verificar que system prompt foi incluído
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        messages = request_data['messages']
        
        # Primeiro message deve ser system
        assert messages[0]['role'] == 'system'
        assert system_prompt in messages[0]['content']

    def test_generate_response_force_model(self, router):
        """Testar forçar uso de modelo específico."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Resposta"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 50},
                "model": "gpt-4o"
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            query = "Pergunta simples"
            response = router.generate_response(
                query, 
                force_model="openai.gpt4o"
            )

            assert response.model == "gpt-4o"

    @patch('requests.post')
    def test_generate_response_api_error(self, mock_post, router):
        """Testar tratamento de erro da API."""
        # Simular erro HTTP
        mock_post.side_effect = Exception("API Error")

        query = "Pergunta"
        
        # Deve levantar exceção ou retornar erro
        with pytest.raises(Exception):
            router.generate_response(query)

    def test_update_stats(self, router):
        """Testar atualização de estatísticas."""
        initial_requests = router.stats["total_requests"]
        initial_cost = router.stats["total_cost"]

        router._update_stats("openai", "gpt-4o-mini", TaskType.QUICK_QUERIES, 0.01, 1.5)

        assert router.stats["total_requests"] == initial_requests + 1
        assert router.stats["total_cost"] == initial_cost + 0.01
        assert "openai" in router.stats["provider_usage"]
        assert router.stats["provider_usage"]["openai"] == 1
        assert "gpt-4o-mini" in router.stats["model_usage"]

    def test_get_available_models(self, router):
        """Testar obtenção de modelos disponíveis."""
        models = router.get_available_models()
        
        assert isinstance(models, dict)
        assert "models" in models
        assert "openai.gpt4o_mini" in models["models"]
        assert "name" in models["models"]["openai.gpt4o_mini"]
        assert "responsibilities" in models["models"]["openai.gpt4o_mini"]

    def test_get_stats(self, router):
        """Testar obtenção de estatísticas."""
        stats = router.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "total_cost" in stats
        assert "provider_usage" in stats

    def test_cost_optimized_strategy(self, basic_config):
        """Testar estratégia otimizada por custo."""
        basic_config["routing"]["strategy"] = "cost_optimized"
        router = APIModelRouter(basic_config)
        
        # Para quick_queries, deve escolher o mais barato
        model = router.select_best_model(TaskType.QUICK_QUERIES)
        assert model == "openai.gpt4o_mini"  # Mais barato

    def test_performance_optimized_strategy(self, basic_config):
        """Testar estratégia otimizada por performance."""
        basic_config["routing"]["strategy"] = "performance_optimized"
        router = APIModelRouter(basic_config)
        
        # Para code_generation, deve escolher o de maior prioridade
        model = router.select_best_model(TaskType.CODE_GENERATION)
        assert model == "openai.gpt4o"  # Priority 2 vs gpt4o_mini priority 1

    @patch('requests.post')
    def test_anthropic_api_call(self, mock_post, router):
        """Testar chamada para API da Anthropic."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "content": [{"text": "Resposta da Anthropic"}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Forçar uso do modelo Anthropic
        query = "Analisar documento"
        response = router.generate_response(
            query, 
            force_model="anthropic.claude_3_5_sonnet"
        )

        assert response.content == "Resposta da Anthropic"
        assert response.provider == "anthropic"

    @pytest.mark.performance
    def test_response_time_tracking(self, router):
        """Testar rastreamento de tempo de resposta."""
        with patch('requests.post') as mock_post:
            # Simular delay na API
            def slow_response(*args, **kwargs):
                time.sleep(0.1)  # 100ms delay
                mock_resp = Mock()
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content": "Resposta"}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 50},
                    "model": "gpt-4o-mini"
                }
                mock_resp.raise_for_status.return_value = None
                return mock_resp
            
            mock_post.side_effect = slow_response

            response = router.generate_response("Pergunta de teste")
            
            # Deve rastrear tempo de processamento
            assert response.processing_time > 0.05  # Pelo menos 50ms
            assert response.processing_time < 1.0   # Menos que 1s

    def test_context_length_calculation(self, router):
        """Testar cálculo de comprimento de contexto."""
        long_context = "palavra " * 1000  # ~7000 caracteres
        
        # Deve selecionar modelo adequado para contexto longo
        model = router.select_best_model(
            TaskType.DOCUMENT_ANALYSIS, 
            context_length=len(long_context)
        )
        
        # Deve escolher modelo com context_window adequado
        assert model in router.available_models
        selected_config = router.available_models[model]
        assert selected_config.context_window > len(long_context)

    def test_empty_config_handling(self):
        """Testar tratamento de configuração vazia."""
        empty_config = {"providers": {}, "routing": {}}
        router = APIModelRouter(empty_config)
        
        assert len(router.available_models) == 0
        
        # Deve retornar None quando não há modelos
        model = router.select_best_model(TaskType.GENERAL_EXPLANATION)
        assert model is None

    def test_multiple_responsibilities_model(self, router):
        """Testar modelo com múltiplas responsabilidades."""
        # gpt4o_mini tem ["primary_reasoning", "quick_queries"]
        
        # Deve ser selecionado para ambas as responsabilidades
        model1 = router.select_best_model(TaskType.GENERAL_EXPLANATION)  # primary_reasoning
        model2 = router.select_best_model(TaskType.QUICK_QUERIES)        # quick_queries
        
        assert model1 == "openai.gpt4o_mini"
        assert model2 == "openai.gpt4o_mini" 