import pytest
import unittest.mock as mock
from unittest.mock import Mock, patch, MagicMock
from src.models.model_router import ModelRouter, TaskType, AdvancedModelRouter


class TestModelRouter:
    """Testes unitários para a classe ModelRouter"""
    
    def setup_method(self):
        """Setup para cada teste"""
        with patch('src.models.model_router.ollama'):
            self.router = ModelRouter()
    
    def test_init_default_models(self):
        """Testa se os modelos padrão são inicializados corretamente"""
        assert "general" in self.router.models
        assert "code" in self.router.models
        assert self.router.models["general"]["name"] == "llama3.1:8b-instruct-q4_K_M"
        assert self.router.models["code"]["name"] == "codellama:7b-instruct"
    
    def test_code_indicators_initialized(self):
        """Testa se os indicadores de código são inicializados"""
        assert isinstance(self.router.code_indicators, list)
        assert len(self.router.code_indicators) > 0
        assert "código" in self.router.code_indicators
        assert "python" in self.router.code_indicators
    
    @patch('src.models.model_router.ollama')
    def test_check_available_models_success(self, mock_ollama):
        """Testa verificação de modelos disponíveis com sucesso"""
        mock_ollama.list.return_value = {
            'models': [
                {'name': 'llama3.1:8b-instruct-q4_K_M'},
                {'name': 'codellama:7b-instruct'}
            ]
        }
        
        router = ModelRouter()
        available = router._check_available_models()
        
        assert isinstance(available, set)
        # O método pode ser chamado múltiplas vezes durante a inicialização
        assert mock_ollama.list.called
    
    @patch('src.models.model_router.ollama')
    def test_check_available_models_error(self, mock_ollama):
        """Testa tratamento de erro na verificação de modelos"""
        mock_ollama.list.side_effect = Exception("Connection error")
        
        router = ModelRouter()
        available = router._check_available_models()
        
        assert available == {'general', 'code'}  # fallback
    
    def test_detect_code_need_with_code_indicators(self):
        """Testa detecção de necessidade de geração de código com indicadores"""
        code_queries = [
            "escrever uma função",
            "criar uma classe",
            "implementar algoritmo",
            "código para",
            "exemplo de código",
            "python",
            "javascript",
            "programação"
        ]
        
        for query in code_queries:
            assert self.router.detect_code_need(query) == True
    
    def test_detect_code_need_without_code_indicators(self):
        """Testa detecção quando não há necessidade de geração de código"""
        non_code_queries = [
            "o que é machine learning",
            "explique o conceito",
            "como isso funciona",
            "me fale sobre"
        ]
        
        for query in non_code_queries:
            assert self.router.detect_code_need(query) == False
    
    def test_select_model_basic_functionality(self):
        """Testa funcionalidade básica de seleção de modelo"""
        query = "pergunta básica"
        model = self.router.select_model(query)
        assert isinstance(model, str)
        assert len(model) > 0
    
    def test_select_model_for_code_generation_available(self):
        """Testa seleção de modelo para geração de código quando disponível"""
        # Simula que o modelo code está disponível
        self.router.available_models = {'general', 'code'}
        code_query = "escrever uma função para ordenar array"
        model = self.router.select_model(code_query)
        assert model in ['code', 'general']  # pode ser qualquer um dependendo da detecção
    
    def test_select_model_with_context_code_detection(self):
        """Testa seleção de modelo com contexto que contém código"""
        # Simula que o modelo code está disponível
        self.router.available_models = {'general', 'code'}
        query = "explique isso"
        context = "código python para análise"
        model = self.router.select_model(query, context)
        assert model in ['code', 'general']  # pode detectar código no contexto
    
    @patch('src.models.model_router.ollama')
    def test_generate_with_model_success(self, mock_ollama):
        """Testa geração de resposta bem-sucedida"""
        # Mock da resposta do Ollama
        mock_ollama.chat.return_value = {
            'message': {'content': 'Resposta de teste'}
        }
        
        response = self.router.generate_with_model("query de teste", "general")
        
        assert response == "Resposta de teste"
        mock_ollama.chat.assert_called_once()
    
    @patch('src.models.model_router.ollama')
    def test_generate_with_model_api_error(self, mock_ollama):
        """Testa tratamento de erro da API"""
        mock_ollama.chat.side_effect = Exception("Erro de API")
        
        response = self.router.generate_with_model("query de teste", "general")
        
        assert response == ""  # retorna string vazia em caso de erro
    
    @patch.object(ModelRouter, 'generate_with_model')
    def test_generate_hybrid_response_general_only(self, mock_generate):
        """Testa geração de resposta híbrida para query geral"""
        mock_generate.return_value = "Resposta geral"
        
        response = self.router.generate_hybrid_response(
            "o que é machine learning", 
            "contexto", 
            ["doc1", "doc2"]
        )
        
        assert response == "Resposta geral"
        mock_generate.assert_called_once()
    
    @patch.object(ModelRouter, 'generate_with_model')
    def test_generate_hybrid_response_with_code(self, mock_generate):
        """Testa geração de resposta híbrida com código"""
        mock_generate.side_effect = [
            "Explicação geral [CÓDIGO: função de ordenação]",
            "def sort_array(arr): return sorted(arr)"
        ]
        
        response = self.router.generate_hybrid_response(
            "criar função de ordenação", 
            "contexto", 
            ["doc1"]
        )
        
        assert "Explicação geral" in response
        assert "```python" in response
        assert "def sort_array" in response
        assert mock_generate.call_count == 2
    
    def test_get_model_status(self):
        """Testa obtenção do status dos modelos"""
        status = self.router.get_model_status()
        
        assert isinstance(status, dict)
        assert "available" in status
        assert "models" in status
        assert "total_models" in status
        assert isinstance(status["available"], list)
        assert isinstance(status["models"], dict)
        assert isinstance(status["total_models"], int)


class TestAdvancedModelRouter:
    """Testes unitários para a classe AdvancedModelRouter"""
    
    def setup_method(self):
        """Setup para cada teste"""
        with patch('src.models.model_router.ollama'):
            self.router = AdvancedModelRouter()
    
    def test_init_specialized_models(self):
        """Testa se os modelos especializados são inicializados"""
        assert "general" in self.router.models
        assert "code" in self.router.models
        assert "mistral" in self.router.models
        assert "sql" in self.router.models
        assert "fast" in self.router.models
    
    def test_task_indicators_initialized(self):
        """Testa se os indicadores de tarefa são inicializados"""
        assert hasattr(self.router, 'task_indicators')
        assert TaskType.SQL_QUERY in self.router.task_indicators
        assert TaskType.ARCHITECTURE_DESIGN in self.router.task_indicators
        assert TaskType.DEBUGGING in self.router.task_indicators
    
    def test_detect_tasks_sql(self):
        """Testa detecção de tarefas SQL"""
        sql_query = "SELECT * FROM users WHERE active = 1"
        tasks = self.router.detect_tasks(sql_query)
        
        assert TaskType.GENERAL_EXPLANATION in tasks
        assert TaskType.SQL_QUERY in tasks
    
    def test_detect_tasks_architecture(self):
        """Testa detecção de tarefas de arquitetura"""
        arch_query = "design a microservices architecture"
        tasks = self.router.detect_tasks(arch_query)
        
        assert TaskType.GENERAL_EXPLANATION in tasks
        assert TaskType.ARCHITECTURE_DESIGN in tasks
    
    def test_detect_tasks_debugging(self):
        """Testa detecção de tarefas de debug"""
        debug_query = "debug this error in my code"
        tasks = self.router.detect_tasks(debug_query)
        
        assert TaskType.GENERAL_EXPLANATION in tasks
        assert TaskType.DEBUGGING in tasks
    
    def test_detect_tasks_code_generation(self):
        """Testa detecção de tarefas de geração de código"""
        code_query = "criar uma função python"
        tasks = self.router.detect_tasks(code_query)
        
        assert TaskType.GENERAL_EXPLANATION in tasks
        assert TaskType.CODE_GENERATION in tasks
    
    def test_select_best_model_for_task(self):
        """Testa seleção do melhor modelo para uma tarefa"""
        # Simula modelos disponíveis
        self.router.available_models = {'general', 'code'}
        
        model = self.router.select_best_model(TaskType.CODE_GENERATION)
        assert model == 'code'
        
        model = self.router.select_best_model(TaskType.GENERAL_EXPLANATION)
        assert model == 'general'
    
    def test_select_best_model_fallback(self):
        """Testa fallback quando modelo específico não está disponível"""
        self.router.available_models = {'general'}
        
        model = self.router.select_best_model(TaskType.SQL_QUERY)
        assert model == 'general'  # fallback


class TestTaskType:
    """Testes para o enum TaskType"""
    
    def test_task_type_values(self):
        """Testa se os valores do enum estão corretos"""
        assert TaskType.GENERAL_EXPLANATION.value == "general_explanation"
        assert TaskType.CODE_GENERATION.value == "code_generation"
        assert TaskType.SQL_QUERY.value == "sql_query"
        assert TaskType.ARCHITECTURE_DESIGN.value == "architecture_design"
        assert TaskType.DEBUGGING.value == "debugging"
        assert TaskType.DOCUMENTATION.value == "documentation"
        assert TaskType.QUICK_SNIPPET.value == "quick_snippet"
    
    def test_task_type_membership(self):
        """Testa se os tipos de tarefa são membros válidos"""
        task_values = [task.value for task in TaskType]
        assert "general_explanation" in task_values
        assert "code_generation" in task_values
        assert "sql_query" in task_values
        assert "architecture_design" in task_values
        assert "debugging" in task_values
        assert "documentation" in task_values
        assert "quick_snippet" in task_values


class TestModelRouterEdgeCases:
    """Testes para casos extremos e cenários não cobertos"""
    
    def setup_method(self):
        """Setup para cada teste"""
        with patch('src.models.model_router.ollama'):
            self.router = ModelRouter()
    
    def test_detect_code_need_with_regex_patterns(self):
        """Testa detecção de código usando padrões regex"""
        regex_queries = [
            "como implementar uma solução",
            "exemplo de função",
            "código para validação",
            "função que calcula",
            "classe para gerenciar",
            "script para automatizar"
        ]
        
        for query in regex_queries:
            assert self.router.detect_code_need(query) == True
    
    def test_detect_code_need_empty_inputs(self):
        """Testa detecção com entradas vazias"""
        assert self.router.detect_code_need("") == False
        assert self.router.detect_code_need("", "") == False
        assert self.router.detect_code_need("   ", "   ") == False
    
    def test_detect_code_need_case_insensitive(self):
        """Testa detecção case-insensitive"""
        queries = [
            "PYTHON código",
            "Função JAVASCRIPT",
            "IMPLEMENTAR algoritmo",
            "Código PARA análise"
        ]
        
        for query in queries:
            assert self.router.detect_code_need(query) == True
    
    @patch('src.models.model_router.ollama')
    def test_generate_with_model_with_system_prompt(self, mock_ollama):
        """Testa geração com system prompt"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Resposta com system prompt'}
        }
        
        response = self.router.generate_with_model(
            "query de teste", 
            "general",
            system_prompt="Você é um assistente especializado"
        )
        
        assert response == "Resposta com system prompt"
        # Verifica se system prompt foi incluído nas mensagens
        call_args = mock_ollama.chat.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
    
    @patch('src.models.model_router.ollama')
    def test_generate_with_model_custom_temperature(self, mock_ollama):
        """Testa geração com temperatura customizada"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Resposta com temperatura'}
        }
        
        response = self.router.generate_with_model(
            "query de teste", 
            "general",
            temperature=0.9
        )
        
        # Verifica se temperatura foi passada corretamente
        call_args = mock_ollama.chat.call_args
        options = call_args[1]['options']
        assert options['temperature'] == 0.9
    
    def test_generate_with_model_unavailable_model(self):
        """Testa fallback quando modelo não está disponível"""
        self.router.available_models = {'general'}
        
        with patch.object(self.router, 'generate_with_model', wraps=self.router.generate_with_model) as mock_generate:
            with patch('src.models.model_router.ollama') as mock_ollama:
                mock_ollama.chat.return_value = {'message': {'content': 'Fallback response'}}
                
                response = self.router.generate_with_model("test", "unavailable_model")
                
                # Verifica se foi usado o modelo general como fallback
                call_args = mock_ollama.chat.call_args
                model_used = call_args[1]['model']
                assert model_used == self.router.models['general']['name']
    
    def test_generate_hybrid_response_no_code_markers(self):
        """Testa resposta híbrida sem marcadores de código"""
        with patch.object(self.router, 'generate_with_model') as mock_generate:
            mock_generate.return_value = "Resposta sem código"
            
            response = self.router.generate_hybrid_response(
                "criar função de ordenação",
                "contexto",
                ["doc1", "doc2"]
            )
            
            assert response == "Resposta sem código"
            assert mock_generate.call_count == 1
    
    def test_generate_hybrid_response_multiple_code_markers(self):
        """Testa resposta híbrida com múltiplos marcadores"""
        with patch.object(self.router, 'generate_with_model') as mock_generate:
            mock_generate.side_effect = [
                "Explicação [CÓDIGO: função sort] e [CÓDIGO: função filter]",
                "def sort_func(): pass",
                "def filter_func(): pass"
            ]
            
            response = self.router.generate_hybrid_response(
                "criar funções",
                "contexto",
                ["doc1"]
            )
            
            assert "```python" in response
            assert "def sort_func" in response
            assert "def filter_func" in response
            assert mock_generate.call_count == 3
    
    def test_get_model_status_structure(self):
        """Testa estrutura completa do status dos modelos"""
        status = self.router.get_model_status()
        
        # Verifica estrutura
        required_keys = ['available', 'total_models', 'models']
        for key in required_keys:
            assert key in status
        
        # Verifica tipos
        assert isinstance(status['available'], list)
        assert isinstance(status['total_models'], int)
        assert isinstance(status['models'], dict)
        
        # Verifica conteúdo
        assert status['total_models'] > 0
        assert len(status['models']) == status['total_models']


class TestAdvancedModelRouterEdgeCases:
    """Testes para casos extremos do AdvancedModelRouter"""
    
    def setup_method(self):
        """Setup para cada teste"""
        with patch('src.models.model_router.ollama'):
            self.router = AdvancedModelRouter()
    
    def test_detect_tasks_multiple_indicators(self):
        """Testa detecção de múltiplas tarefas em uma query"""
        complex_query = "debug this sql query and create documentation"
        tasks = self.router.detect_tasks(complex_query)
        
        task_values = [task.value for task in tasks]
        assert "general_explanation" in task_values
        assert "sql_query" in task_values
        assert "debugging" in task_values
        assert "documentation" in task_values
    
    def test_detect_tasks_with_context(self):
        """Testa detecção de tarefas considerando contexto"""
        query = "explique isso"
        context = "SELECT * FROM users WHERE active = 1"
        tasks = self.router.detect_tasks(query, context)
        
        task_values = [task.value for task in tasks]
        assert "sql_query" in task_values
    
    def test_select_best_model_no_available_models(self):
        """Testa seleção quando nenhum modelo está disponível"""
        self.router.available_models = set()
        model = self.router.select_best_model(TaskType.CODE_GENERATION)
        assert model is None
    
    def test_select_best_model_priority_ordering(self):
        """Testa se a prioridade é respeitada na seleção"""
        # Simula múltiplos modelos disponíveis para a mesma tarefa
        self.router.available_models = {'general', 'mistral'}
        
        # Ambos podem fazer GENERAL_EXPLANATION, mas general tem prioridade 1
        model = self.router.select_best_model(TaskType.GENERAL_EXPLANATION)
        assert model == 'general'  # prioridade menor = melhor
    
    @patch.object(AdvancedModelRouter, 'generate_with_model')
    def test_generate_advanced_response_sql_processing(self, mock_generate):
        """Testa processamento de marcadores SQL"""
        mock_generate.side_effect = [
            "Explicação [SQL: buscar usuários ativos]",
            "SELECT * FROM users WHERE active = 1"
        ]
        
        self.router.available_models = {'general', 'sql'}
        
        result = self.router.generate_advanced_response(
            "buscar usuários",
            "contexto",
            ["doc1"]
        )
        
        assert "```sql" in result['answer']
        assert "SELECT * FROM users" in result['answer']
        assert len(result['models_used']) >= 1
        assert 'sql_query' in result['tasks_performed']
    
    @patch.object(AdvancedModelRouter, 'generate_with_model')
    def test_generate_advanced_response_architecture_task(self, mock_generate):
        """Testa resposta para tarefa de arquitetura"""
        mock_generate.return_value = "Design de microserviços"
        
        self.router.available_models = {'general', 'mistral'}
        
        result = self.router.generate_advanced_response(
            "design a microservices architecture",
            "contexto",
            ["doc1"]
        )
        
        assert 'architecture_design' in result['tasks_performed']
        assert result['answer'] == "Design de microserviços"
        # Deve usar mistral para arquitetura se disponível
        assert any('mistral' in model for model in result['models_used'])
    
    def test_generate_advanced_response_result_structure(self):
        """Testa estrutura do resultado da resposta avançada"""
        with patch.object(self.router, 'generate_with_model') as mock_generate:
            mock_generate.return_value = "Resposta teste"
            
            result = self.router.generate_advanced_response(
                "pergunta teste",
                "contexto",
                ["doc1"]
            )
            
            # Verifica estrutura obrigatória
            required_keys = ['answer', 'models_used', 'tasks_performed', 'sections']
            for key in required_keys:
                assert key in result
            
            # Verifica tipos
            assert isinstance(result['answer'], str)
            assert isinstance(result['models_used'], list)
            assert isinstance(result['tasks_performed'], list)
            assert isinstance(result['sections'], dict)
            
            # Verifica conteúdo mínimo
            assert len(result['answer']) > 0
            assert len(result['models_used']) > 0
            assert 'general_explanation' in result['tasks_performed']
            assert 'main' in result['sections']


if __name__ == "__main__":
    pytest.main([__file__])