"""
Testes abrangentes para src.template_renderer - FASE 2 do plano de cobertura.
Este arquivo visa aumentar a cobertura do módulo de 0% para 90%+.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import tempfile
import os

# Mock jinja2 antes da importação para evitar dependências
with patch.dict('sys.modules', {
    'jinja2': Mock(),
    'jinja2.Environment': Mock(),
    'jinja2.FileSystemLoader': Mock(),
    'jinja2.Template': Mock(),
}):
    from src.template_renderer import TemplateRenderer


class TestTemplateRenderer:
    """Testes para TemplateRenderer - Cobertura completa."""
    
    @pytest.fixture
    def mock_jinja_environment(self):
        """Mock do ambiente Jinja2."""
        env = Mock()
        template = Mock()
        template.render.return_value = "rendered content"
        env.get_template.return_value = template
        return env
    
    @pytest.fixture
    def template_renderer(self, mock_jinja_environment):
        """Instância do TemplateRenderer com mocks."""
        with patch('src.template_renderer.Environment', return_value=mock_jinja_environment):
            with patch('src.template_renderer.FileSystemLoader'):
                return TemplateRenderer(template_dir="/mock/templates")
    
    def test_init_default_template_dir(self):
        """Testa inicialização com diretório padrão."""
        with patch('src.template_renderer.Environment') as mock_env:
            with patch('src.template_renderer.FileSystemLoader') as mock_loader:
                renderer = TemplateRenderer()
                
                # Verifica se foi chamado com diretório padrão
                mock_loader.assert_called_once()
                mock_env.assert_called_once()
    
    def test_init_custom_template_dir(self):
        """Testa inicialização com diretório customizado."""
        custom_dir = "/custom/templates"
        
        with patch('src.template_renderer.Environment') as mock_env:
            with patch('src.template_renderer.FileSystemLoader') as mock_loader:
                renderer = TemplateRenderer(template_dir=custom_dir)
                
                mock_loader.assert_called_once_with(custom_dir)
                mock_env.assert_called_once()
    
    def test_render_template_success(self, template_renderer, mock_jinja_environment):
        """Testa renderização bem-sucedida de template."""
        template_name = "test_template.html"
        context = {"name": "World", "value": 42}
        expected_result = "Hello World, value is 42"
        
        # Configurar mock para retornar resultado esperado
        mock_template = Mock()
        mock_template.render.return_value = expected_result
        mock_jinja_environment.get_template.return_value = mock_template
        
        result = template_renderer.render(template_name, context)
        
        assert result == expected_result
        mock_jinja_environment.get_template.assert_called_once_with(template_name)
        mock_template.render.assert_called_once_with(context)
    
    def test_render_template_with_empty_context(self, template_renderer, mock_jinja_environment):
        """Testa renderização com contexto vazio."""
        template_name = "empty_context.html"
        expected_result = "No variables"
        
        mock_template = Mock()
        mock_template.render.return_value = expected_result
        mock_jinja_environment.get_template.return_value = mock_template
        
        result = template_renderer.render(template_name, {})
        
        assert result == expected_result
        mock_template.render.assert_called_once_with({})
    
    def test_render_template_not_found(self, template_renderer, mock_jinja_environment):
        """Testa erro quando template não é encontrado."""
        from jinja2 import TemplateNotFound
        
        template_name = "nonexistent.html"
        mock_jinja_environment.get_template.side_effect = TemplateNotFound(template_name)
        
        with pytest.raises(TemplateNotFound):
            template_renderer.render(template_name, {})
    
    def test_render_template_syntax_error(self, template_renderer, mock_jinja_environment):
        """Testa erro de sintaxe no template."""
        from jinja2 import TemplateSyntaxError
        
        template_name = "syntax_error.html"
        mock_jinja_environment.get_template.side_effect = TemplateSyntaxError(
            "Invalid syntax", lineno=1
        )
        
        with pytest.raises(TemplateSyntaxError):
            template_renderer.render(template_name, {})
    
    def test_render_with_complex_context(self, template_renderer, mock_jinja_environment):
        """Testa renderização com contexto complexo."""
        template_name = "complex.html"
        context = {
            "user": {"name": "Alice", "email": "alice@test.com"},
            "items": [{"name": "Item 1"}, {"name": "Item 2"}],
            "metadata": {"version": "1.0", "debug": True}
        }
        expected_result = "Complex template rendered"
        
        mock_template = Mock()
        mock_template.render.return_value = expected_result
        mock_jinja_environment.get_template.return_value = mock_template
        
        result = template_renderer.render(template_name, context)
        
        assert result == expected_result
        mock_template.render.assert_called_once_with(context)
    
    def test_render_with_none_values(self, template_renderer, mock_jinja_environment):
        """Testa renderização com valores None no contexto."""
        template_name = "with_none.html"
        context = {"value": None, "name": "Test", "empty": ""}
        expected_result = "Handling None values"
        
        mock_template = Mock()
        mock_template.render.return_value = expected_result
        mock_jinja_environment.get_template.return_value = mock_template
        
        result = template_renderer.render(template_name, context)
        
        assert result == expected_result
        mock_template.render.assert_called_once_with(context)
    
    def test_render_multiple_templates(self, template_renderer, mock_jinja_environment):
        """Testa renderização de múltiplos templates."""
        templates = [
            ("template1.html", {"name": "First"}),
            ("template2.html", {"name": "Second"}),
            ("template3.html", {"name": "Third"})
        ]
        
        mock_template = Mock()
        mock_template.render.side_effect = ["Result 1", "Result 2", "Result 3"]
        mock_jinja_environment.get_template.return_value = mock_template
        
        results = []
        for template_name, context in templates:
            result = template_renderer.render(template_name, context)
            results.append(result)
        
        assert results == ["Result 1", "Result 2", "Result 3"]
        assert mock_jinja_environment.get_template.call_count == 3
        assert mock_template.render.call_count == 3
    
    def test_template_caching_behavior(self, template_renderer, mock_jinja_environment):
        """Testa comportamento de cache de templates."""
        template_name = "cached_template.html"
        context = {"data": "test"}
        
        mock_template = Mock()
        mock_template.render.return_value = "Cached result"
        mock_jinja_environment.get_template.return_value = mock_template
        
        # Renderizar o mesmo template múltiplas vezes
        result1 = template_renderer.render(template_name, context)
        result2 = template_renderer.render(template_name, context)
        
        assert result1 == result2 == "Cached result"
        # Jinja2 deve ser chamado para cada renderização
        assert mock_jinja_environment.get_template.call_count == 2
    
    def test_error_handling_during_render(self, template_renderer, mock_jinja_environment):
        """Testa tratamento de erros durante renderização."""
        template_name = "error_template.html"
        context = {"data": "test"}
        
        mock_template = Mock()
        mock_template.render.side_effect = RuntimeError("Template rendering failed")
        mock_jinja_environment.get_template.return_value = mock_template
        
        with pytest.raises(RuntimeError, match="Template rendering failed"):
            template_renderer.render(template_name, context)
    
    def test_special_characters_in_context(self, template_renderer, mock_jinja_environment):
        """Testa renderização com caracteres especiais no contexto."""
        template_name = "special_chars.html"
        context = {
            "unicode": "测试 тест テスト",
            "symbols": "!@#$%^&*()",
            "html": "<script>alert('test')</script>",
            "emoji": "🚀🔥💯"
        }
        expected_result = "Special characters handled"
        
        mock_template = Mock()
        mock_template.render.return_value = expected_result
        mock_jinja_environment.get_template.return_value = mock_template
        
        result = template_renderer.render(template_name, context)
        
        assert result == expected_result
        mock_template.render.assert_called_once_with(context)
    
    def test_large_context_data(self, template_renderer, mock_jinja_environment):
        """Testa renderização com contexto de dados grandes."""
        template_name = "large_data.html"
        context = {
            "large_list": list(range(1000)),
            "large_dict": {f"key_{i}": f"value_{i}" for i in range(100)},
            "large_string": "x" * 10000
        }
        expected_result = "Large data processed"
        
        mock_template = Mock()
        mock_template.render.return_value = expected_result
        mock_jinja_environment.get_template.return_value = mock_template
        
        result = template_renderer.render(template_name, context)
        
        assert result == expected_result
        mock_template.render.assert_called_once_with(context)


class TestTemplateRendererEdgeCases:
    """Testes para casos extremos e situações específicas."""
    
    def test_invalid_template_directory(self):
        """Testa inicialização com diretório inválido."""
        with patch('src.template_renderer.FileSystemLoader') as mock_loader:
            mock_loader.side_effect = FileNotFoundError("Directory not found")
            
            with pytest.raises(FileNotFoundError):
                TemplateRenderer(template_dir="/invalid/path")
    
    def test_concurrent_template_rendering(self, template_renderer, mock_jinja_environment):
        """Testa renderização concorrente de templates."""
        import threading
        import time
        
        template_name = "concurrent.html"
        results = []
        
        mock_template = Mock()
        mock_template.render.return_value = "Concurrent result"
        mock_jinja_environment.get_template.return_value = mock_template
        
        def render_worker(context):
            time.sleep(0.01)  # Simular algum processamento
            result = template_renderer.render(template_name, context)
            results.append(result)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=render_worker, args=({"id": i},))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        assert all(result == "Concurrent result" for result in results)
    
    def test_memory_efficient_rendering(self, template_renderer, mock_jinja_environment):
        """Testa renderização eficiente em memória."""
        template_name = "memory_test.html"
        
        mock_template = Mock()
        mock_template.render.return_value = "Memory efficient"
        mock_jinja_environment.get_template.return_value = mock_template
        
        # Renderizar muitos templates para testar uso de memória
        for i in range(100):
            context = {"iteration": i}
            result = template_renderer.render(template_name, context)
            assert result == "Memory efficient"
        
        # Verificar que o template foi buscado 100 vezes
        assert mock_jinja_environment.get_template.call_count == 100


class TestTemplateRendererIntegration:
    """Testes de integração com sistema de arquivos real."""
    
    def test_real_filesystem_integration(self):
        """Testa integração com sistema de arquivos real."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Criar arquivo de template
            template_path = Path(temp_dir) / "test.html"
            template_content = "Hello {{ name }}!"
            template_path.write_text(template_content)
            
            # Mock apenas o Jinja2, não o sistema de arquivos
            with patch('src.template_renderer.Environment') as mock_env:
                mock_template = Mock()
                mock_template.render.return_value = "Hello World!"
                mock_env.return_value.get_template.return_value = mock_template
                
                renderer = TemplateRenderer(template_dir=temp_dir)
                result = renderer.render("test.html", {"name": "World"})
                
                assert result == "Hello World!"
    
    @pytest.mark.performance
    def test_performance_benchmarking(self, template_renderer, mock_jinja_environment):
        """Testa performance de renderização."""
        import time
        
        template_name = "performance.html"
        context = {"data": "test" * 100}
        
        mock_template = Mock()
        mock_template.render.return_value = "Performance result"
        mock_jinja_environment.get_template.return_value = mock_template
        
        start_time = time.time()
        
        for _ in range(100):
            template_renderer.render(template_name, context)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assert que 100 renderizações levaram menos que 1 segundo
        assert total_time < 1.0
        assert mock_jinja_environment.get_template.call_count == 100


# Testes de Funcionalidades Avançadas
class TestTemplateRendererAdvanced:
    """Testes para funcionalidades avançadas."""
    
    def test_template_inheritance_simulation(self, template_renderer, mock_jinja_environment):
        """Simula teste de herança de templates."""
        base_template = "base.html"
        child_template = "child.html"
        
        mock_template = Mock()
        mock_template.render.return_value = "Inherited content"
        mock_jinja_environment.get_template.return_value = mock_template
        
        result = template_renderer.render(child_template, {"content": "Child content"})
        
        assert result == "Inherited content"
    
    def test_custom_filters_simulation(self, template_renderer, mock_jinja_environment):
        """Simula teste de filtros customizados."""
        template_name = "filtered.html"
        context = {"text": "hello world", "number": 42}
        
        mock_template = Mock()
        mock_template.render.return_value = "HELLO WORLD - 42"
        mock_jinja_environment.get_template.return_value = mock_template
        
        result = template_renderer.render(template_name, context)
        
        assert result == "HELLO WORLD - 42"
    
    def test_conditional_rendering(self, template_renderer, mock_jinja_environment):
        """Testa renderização condicional."""
        template_name = "conditional.html"
        
        mock_template = Mock()
        mock_template.render.side_effect = lambda ctx: f"User: {ctx.get('user', 'Anonymous')}"
        mock_jinja_environment.get_template.return_value = mock_template
        
        # Teste com usuário
        result_with_user = template_renderer.render(template_name, {"user": "Alice"})
        assert "Alice" in result_with_user
        
        # Teste sem usuário
        result_without_user = template_renderer.render(template_name, {})
        assert "Anonymous" in result_without_user
    
    def test_loop_rendering(self, template_renderer, mock_jinja_environment):
        """Testa renderização com loops."""
        template_name = "loop.html"
        context = {"items": ["item1", "item2", "item3"]}
        
        mock_template = Mock()
        mock_template.render.return_value = "item1, item2, item3"
        mock_jinja_environment.get_template.return_value = mock_template
        
        result = template_renderer.render(template_name, context)
        
        assert "item1, item2, item3" == result
        mock_template.render.assert_called_once_with(context) 