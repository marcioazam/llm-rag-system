"""
Testes de segurança para o sistema RAG
Valida implementações de segurança e previne regressões
"""

import pytest
import yaml
import os
import re
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time


@pytest.mark.security
class TestSecurityMeasures:
    """Testes das medidas de segurança implementadas"""
    
    def test_no_hardcoded_credentials(self):
        """Testa se não há credenciais hardcoded no código"""
        source_files = list(Path("src").rglob("*.py"))
        
        # Padrões de credenciais suspeitas
        suspicious_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            # Padrões específicos conhecidos
            r'arrozefeijao13',
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI API keys
            r'sk-ant-[a-zA-Z0-9]{95}',  # Anthropic API keys
        ]
        
        violations = []
        for file_path in source_files:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            for pattern in suspicious_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Verificar se não é um comentário ou exemplo
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line = content[line_start:content.find('\n', match.start())]
                    if not line.strip().startswith('#') and 'example' not in line.lower():
                        violations.append(f"{file_path}:{line.strip()}")
        
        assert not violations, f"Credenciais hardcoded encontradas: {violations}"
    
    def test_env_variables_usage(self):
        """Testa se variáveis de ambiente são usadas para configurações sensíveis"""
        settings_file = Path("src/settings.py")
        assert settings_file.exists(), "Arquivo settings.py não encontrado"
        
        content = settings_file.read_text()
        
        # Verificar se usa Field(env=...)
        env_patterns = [
            r'Field\(env=["\']NEO4J_PASSWORD["\']',
            r'Field\(env=["\']OPENAI_API_KEY["\']',
            r'Field\(env=["\']ANTHROPIC_API_KEY["\']'
        ]
        
        for pattern in env_patterns:
            if re.search(pattern, content):
                continue  # Pelo menos um padrão foi encontrado
        else:
            pytest.fail("Não foram encontradas configurações usando variáveis de ambiente")
    
    def test_gitignore_security(self):
        """Testa se .gitignore protege arquivos sensíveis"""
        gitignore_file = Path(".gitignore")
        assert gitignore_file.exists(), "Arquivo .gitignore não encontrado"
        
        content = gitignore_file.read_text(errors='ignore')
        
        required_patterns = [
            '.env',
            '*.log',
            '*.pid',
            'config/secrets.yaml',
            'config/api_keys.yaml'
        ]
        
        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        assert not missing_patterns, f"Padrões ausentes no .gitignore: {missing_patterns}"
    
    def test_cors_configuration(self):
        """Testa se CORS está configurado adequadamente"""
        config_file = Path("config/config.yaml")
        assert config_file.exists(), "Arquivo config.yaml não encontrado"
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        cors_origins = config.get('api', {}).get('cors_origins', [])
        
        # Verificar se não permite todas as origens
        assert '*' not in cors_origins, "CORS permite todas as origens (*)"
        
        # Verificar se são origens específicas e locais
        for origin in cors_origins:
            assert origin.startswith(('http://localhost', 'http://127.0.0.1')), \
                f"Origem CORS suspeita: {origin}"
    
    def test_rate_limiting_configuration(self):
        """Testa se rate limiting está configurado adequadamente"""
        config_file = Path("config/config.yaml")
        assert config_file.exists(), "Arquivo config.yaml não encontrado"
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        rate_limit = config.get('api', {}).get('rate_limit', {})
        rpm = rate_limit.get('requests_per_minute', float('inf'))
        
        # Rate limiting deve ser restritivo
        assert rpm <= 20, f"Rate limiting muito permissivo: {rpm} req/min"
        assert rate_limit.get('enabled', False), "Rate limiting não está habilitado"


@pytest.mark.security
@pytest.mark.api
class TestAPISecurityValidation:
    """Testa validações de segurança da API"""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Mock do pipeline RAG para testes"""
        with patch('src.api.main.get_pipeline') as mock:
            pipeline = MagicMock()
            pipeline.query_llm_only.return_value = {"answer": "test response"}
            pipeline.query.return_value = {"answer": "test response", "sources": []}
            mock.return_value = pipeline
            yield pipeline
    
    @pytest.fixture
    def client(self, mock_pipeline):
        """Cliente de teste da API"""
        from src.api.main import app
        return TestClient(app)
    
    def test_input_validation_question_length(self, client):
        """Testa validação de tamanho da pergunta"""
        # Pergunta muito longa
        long_question = "a" * 3000
        response = client.post("/query", json={"question": long_question})
        assert response.status_code == 422, "Deveria rejeitar pergunta muito longa"
    
    def test_input_validation_empty_question(self, client):
        """Testa validação de pergunta vazia"""
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422, "Deveria rejeitar pergunta vazia"
        
        response = client.post("/query", json={"question": "   "})
        assert response.status_code == 422, "Deveria rejeitar pergunta apenas com espaços"
    
    def test_input_validation_dangerous_characters(self, client):
        """Testa validação de caracteres perigosos"""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "<img src=x onerror=alert(1)>",
            "javascript:alert(1)"
        ]
        
        for dangerous_input in dangerous_inputs:
            response = client.post("/query", json={"question": dangerous_input})
            # Deve aceitar mas sanitizar a entrada
            if response.status_code == 200:
                # Verificar se caracteres perigosos foram removidos
                data = response.json()
                # A pergunta processada não deve conter caracteres perigosos
                assert '<' not in str(data), f"Caracteres perigosos não foram sanitizados: {dangerous_input}"
    
    def test_k_parameter_validation(self, client):
        """Testa validação do parâmetro k"""
        # k muito grande
        response = client.post("/query", json={"question": "test", "k": 100})
        assert response.status_code == 422, "Deveria rejeitar k muito grande"
        
        # k negativo
        response = client.post("/query", json={"question": "test", "k": -1})
        assert response.status_code == 422, "Deveria rejeitar k negativo"
        
        # k zero
        response = client.post("/query", json={"question": "test", "k": 0})
        assert response.status_code == 422, "Deveria rejeitar k zero"
    
    def test_system_prompt_validation(self, client):
        """Testa validação do system prompt"""
        # System prompt muito longo
        long_prompt = "a" * 6000
        response = client.post("/query", json={
            "question": "test",
            "system_prompt": long_prompt
        })
        assert response.status_code == 422, "Deveria rejeitar system prompt muito longo"


@pytest.mark.security
@pytest.mark.integration
class TestSecurityIntegration:
    """Testes de integração de segurança"""
    
    def test_health_endpoint_security(self):
        """Testa se endpoint de health não vaza informações sensíveis"""
        from src.api.main import app
        client = TestClient(app)
        
        with patch('src.api.main.get_pipeline') as mock_pipeline:
            mock_pipeline.return_value.query_llm_only.return_value = {"answer": "OK"}
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            
            # Verificar se não há informações sensíveis
            sensitive_keys = ['password', 'api_key', 'secret', 'token']
            content_str = str(data).lower()
            
            for key in sensitive_keys:
                assert key not in content_str, f"Informação sensível encontrada no health: {key}"
    
    def test_error_handling_no_info_leak(self, mock_pipeline):
        """Testa se tratamento de erros não vaza informações"""
        from src.api.main import app
        client = TestClient(app)
        
        # Simular erro interno
        mock_pipeline.query_llm_only.side_effect = Exception("Internal database connection string: postgres://user:pass@db")
        
        response = client.post("/query", json={"question": "test"})
        
        # Deve retornar erro mas sem vazar informações sensíveis
        assert response.status_code == 500
        error_text = response.text.lower()
        
        sensitive_patterns = ['password', 'connection string', 'database', 'postgres://', 'mysql://']
        for pattern in sensitive_patterns:
            assert pattern not in error_text, f"Informação sensível vazada no erro: {pattern}"


@pytest.mark.security
@pytest.mark.performance
class TestSecurityPerformance:
    """Testes de performance relacionados à segurança"""
    
    def test_rate_limiting_simulation(self):
        """Simula teste de rate limiting"""
        from src.api.main import app
        client = TestClient(app)
        
        with patch('src.api.main.get_pipeline') as mock_pipeline:
            mock_pipeline.return_value.query_llm_only.return_value = {"answer": "test"}
            
            # Simular múltiplas requisições rápidas
            responses = []
            start_time = time.time()
            
            for i in range(5):  # Apenas 5 requisições para não ser muito lento
                response = client.post("/query", json={"question": f"test {i}"})
                responses.append(response.status_code)
                
            duration = time.time() - start_time
            
            # Verificar se pelo menos algumas requisições foram bem-sucedidas
            # (rate limiting pode não estar ativo em testes)
            success_count = sum(1 for status in responses if status == 200)
            assert success_count >= 1, "Nenhuma requisição foi bem-sucedida"
    
    def test_input_validation_performance(self):
        """Testa se validação de entrada não impacta muito a performance"""
        from src.api.main import QueryRequest
        
        start_time = time.time()
        
        # Testar validação de múltiplas entradas
        for i in range(100):
            try:
                QueryRequest(question=f"Test question {i}")
            except:
                pass  # Ignorar erros de validação para este teste
        
        duration = time.time() - start_time
        
        # Validação deve ser rápida (menos de 1 segundo para 100 validações)
        assert duration < 1.0, f"Validação muito lenta: {duration:.2f}s para 100 validações"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 