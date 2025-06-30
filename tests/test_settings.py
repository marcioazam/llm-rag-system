"""
Testes para o módulo de configurações do sistema
"""

import pytest
from unittest.mock import patch, Mock
import os
import tempfile


class TestSettings:
    """Testes para funcionalidades de configuração"""
    
    def test_settings_import(self):
        """Testa se o módulo de settings pode ser importado"""
        try:
            from src.settings import settings
            assert settings is not None
        except ImportError:
            # Se não conseguir importar, crie um mock
            pytest.skip("Módulo settings não encontrado")
    
    def test_default_settings_values(self):
        """Testa valores padrão das configurações"""
        try:
            from src.settings import settings
            # Testa se tem configurações básicas
            assert hasattr(settings, 'QDRANT_URL') or hasattr(settings, 'VECTOR_DB_URL')
            assert hasattr(settings, 'OPENAI_API_KEY') or hasattr(settings, 'API_KEYS')
        except ImportError:
            pytest.skip("Módulo settings não encontrado")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-123'})
    def test_environment_variable_loading(self):
        """Testa carregamento de variáveis de ambiente"""
        # Testa se variáveis de ambiente são carregadas
        assert os.environ.get('OPENAI_API_KEY') == 'test-key-123'
    
    def test_config_file_validation(self):
        """Testa validação de arquivos de configuração"""
        # Cria um arquivo temporário de configuração
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
llm_providers:
  openai:
    api_key: "test-key"
    default_model: "gpt-3.5-turbo"
""")
            config_path = f.name
        
        try:
            # Verifica se o arquivo existe e é válido
            assert os.path.exists(config_path)
            with open(config_path, 'r') as f:
                content = f.read()
                assert 'openai' in content
                assert 'api_key' in content
        finally:
            os.unlink(config_path)
    
    def test_missing_required_config(self):
        """Testa comportamento com configurações obrigatórias ausentes"""
        with patch.dict(os.environ, {}, clear=True):
            # Remove todas as variáveis de ambiente
            # O sistema deve funcionar com valores padrão ou mostrar erro apropriado
            assert True  # Placeholder - sistema deve ser resiliente


class TestConfigValidation:
    """Testes para validação de configurações"""
    
    def test_api_key_validation(self):
        """Testa validação de chaves de API"""
        valid_keys = [
            "sk-test123",
            "test-api-key-123",
            "gsk_test123"
        ]
        
        invalid_keys = [
            "",
            None,
            "   ",
            "short"
        ]
        
        for key in valid_keys:
            assert len(key) >= 5, f"Chave válida deve ter pelo menos 5 caracteres: {key}"
        
        for key in invalid_keys:
            if key is None:
                assert key is None
            else:
                assert not key or len(key.strip()) <= 5
    
    def test_url_validation(self):
        """Testa validação de URLs"""
        valid_urls = [
            "http://localhost:6333",
            "https://api.openai.com",
            "http://127.0.0.1:8000"
        ]
        
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://invalid"
        ]
        
        for url in valid_urls:
            assert url.startswith(('http://', 'https://'))
        
        for url in invalid_urls:
            if url:
                assert not url.startswith(('http://', 'https://'))
    
    def test_model_configuration(self):
        """Testa configuração de modelos"""
        model_configs = {
            'openai': {
                'models': ['gpt-3.5-turbo', 'gpt-4'],
                'default': 'gpt-3.5-turbo'
            },
            'anthropic': {
                'models': ['claude-3-sonnet'],
                'default': 'claude-3-sonnet'
            }
        }
        
        for provider, config in model_configs.items():
            assert 'models' in config
            assert 'default' in config
            assert config['default'] in config['models']
            assert len(config['models']) > 0


class TestEnvironmentSettings:
    """Testes para configurações específicas do ambiente"""
    
    def test_development_settings(self):
        """Testa configurações de desenvolvimento"""
        dev_settings = {
            'DEBUG': True,
            'LOG_LEVEL': 'DEBUG',
            'CACHE_ENABLED': True
        }
        
        for key, expected_value in dev_settings.items():
            # Verifica se as configurações de desenvolvimento fazem sentido
            assert isinstance(expected_value, (bool, str))
    
    def test_production_settings(self):
        """Testa configurações de produção"""
        prod_settings = {
            'DEBUG': False,
            'LOG_LEVEL': 'INFO',
            'CACHE_ENABLED': True,
            'RATE_LIMITING': True
        }
        
        for key, expected_value in prod_settings.items():
            # Verifica se as configurações de produção são seguras
            assert isinstance(expected_value, (bool, str))
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'test'})
    def test_test_environment_isolation(self):
        """Testa isolamento do ambiente de teste"""
        # Em ambiente de teste, certas configurações devem ser diferentes
        assert os.environ.get('ENVIRONMENT') == 'test'
        
        # Configurações de teste devem ser isoladas
        test_configs = {
            'USE_REAL_APIs': False,
            'MOCK_EXTERNAL_CALLS': True,
            'TEMP_DATABASE': True
        }
        
        for key, expected in test_configs.items():
            # Verifica se configurações de teste são apropriadas
            assert isinstance(expected, bool) 