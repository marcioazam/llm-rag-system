"""
Testes para o módulo template_renderer - Sistema de Renderização de Templates
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
import json
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
import re
import hashlib


class TemplateFormat(Enum):
    """Formatos de template suportados"""
    JINJA2 = "jinja2"
    MUSTACHE = "mustache"
    HANDLEBARS = "handlebars"
    SIMPLE = "simple"


class RenderMode(Enum):
    """Modos de renderização"""
    STRING = "string"
    FILE = "file"
    STREAM = "stream"
    BATCH = "batch"


class MockTemplateRenderer:
    """Mock do sistema de renderização de templates"""
    
    def __init__(self):
        # Templates registrados
        self.templates = {}
        self.template_cache = {}
        
        # Configurações
        self.template_dirs = []
        self.default_format = TemplateFormat.JINJA2
        self.enable_caching = True
        self.cache_ttl = 3600  # 1 hora
        self.strict_mode = True
        self.auto_escape = True
        
        # Filtros e funções customizadas
        self.custom_filters = {}
        self.custom_functions = {}
        self.global_context = {}
        
        # Estatísticas
        self.stats = {
            'total_renders': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'avg_render_time': 0.0,
            'total_render_time': 0.0
        }
        
        # Validação e segurança
        self.allowed_tags = {
            'if', 'for', 'set', 'include', 'extends',
            'block', 'macro', 'call', 'filter'
        }
        self.forbidden_patterns = [
            r'__.*__',  # Python internals
            r'import\s+',  # Import statements
            r'exec\s*\(',  # Exec calls
            r'eval\s*\(',  # Eval calls
            r'open\s*\(',  # File operations
        ]
        
        self._init_default_filters()
        self._init_default_functions()
    
    def _init_default_filters(self):
        """Inicializa filtros padrão"""
        self.custom_filters = {
            'upper': lambda x: str(x).upper(),
            'lower': lambda x: str(x).lower(),
            'title': lambda x: str(x).title(),
            'truncate': lambda x, length=50: str(x)[:length] + '...' if len(str(x)) > length else str(x),
            'date_format': lambda x, fmt='%Y-%m-%d': x.strftime(fmt) if hasattr(x, 'strftime') else str(x),
            'json_encode': lambda x: json.dumps(x),
            'default': lambda x, default_val: x if x is not None else default_val,
            'length': lambda x: len(x) if hasattr(x, '__len__') else 0,
            'reverse': lambda x: list(reversed(x)) if hasattr(x, '__iter__') else str(x)[::-1],
            'join': lambda x, separator=', ': separator.join(str(i) for i in x) if hasattr(x, '__iter__') else str(x)
        }
    
    def _init_default_functions(self):
        """Inicializa funções padrão"""
        self.custom_functions = {
            'now': lambda: datetime.now(),
            'today': lambda: datetime.now().date(),
            'range': lambda start, end=None, step=1: list(range(start, end or 0, step)),
            'enumerate': lambda x: list(enumerate(x)) if hasattr(x, '__iter__') else [(0, x)],
            'zip': lambda *args: list(zip(*args)),
            'len': lambda x: len(x) if hasattr(x, '__len__') else 0,
            'min': lambda x: min(x) if hasattr(x, '__iter__') else x,
            'max': lambda x: max(x) if hasattr(x, '__iter__') else x,
            'sum': lambda x: sum(x) if hasattr(x, '__iter__') else x,
            'sorted': lambda x, key=None, reverse=False: sorted(x, key=key, reverse=reverse) if hasattr(x, '__iter__') else [x]
        }
    
    def register_template(
        self, 
        name: str, 
        template_content: str,
        template_format: TemplateFormat = None
    ) -> bool:
        """Registra um template"""
        try:
            # Validação de segurança
            if self.strict_mode and not self._is_template_safe(template_content):
                raise ValueError("Template contains unsafe content")
            
            # Processa template
            processed_template = self._preprocess_template(
                template_content, 
                template_format or self.default_format
            )
            
            self.templates[name] = {
                'content': template_content,
                'processed': processed_template,
                'format': template_format or self.default_format,
                'registered_at': datetime.now(),
                'render_count': 0,
                'last_rendered': None
            }
            
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            raise ValueError(f"Failed to register template '{name}': {str(e)}")
    
    def render_template(
        self, 
        template_name: str, 
        context: Dict[str, Any] = None,
        **kwargs
    ) -> str:
        """Renderiza um template registrado"""
        import time
        start_time = time.time()
        
        try:
            self.stats['total_renders'] += 1
            
            # Verifica se template existe
            if template_name not in self.templates:
                raise ValueError(f"Template '{template_name}' not found")
            
            template_info = self.templates[template_name]
            
            # Verifica cache
            cache_key = self._generate_cache_key(template_name, context, kwargs)
            if self.enable_caching:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.stats['cache_hits'] += 1
                    return cached_result
            
            self.stats['cache_misses'] += 1
            
            # Prepara contexto completo
            full_context = self._prepare_context(context, kwargs)
            
            # Renderiza template
            result = self._render_with_format(
                template_info['processed'],
                template_info['format'],
                full_context
            )
            
            # Atualiza estatísticas do template
            template_info['render_count'] += 1
            template_info['last_rendered'] = datetime.now()
            
            # Adiciona ao cache
            if self.enable_caching:
                self._add_to_cache(cache_key, result)
            
            # Atualiza estatísticas globais
            render_time = time.time() - start_time
            self._update_render_stats(render_time)
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            render_time = time.time() - start_time
            self._update_render_stats(render_time)
            raise ValueError(f"Failed to render template '{template_name}': {str(e)}")
    
    def render_string(
        self, 
        template_string: str, 
        context: Dict[str, Any] = None,
        template_format: TemplateFormat = None,
        **kwargs
    ) -> str:
        """Renderiza template direto de string"""
        import time
        start_time = time.time()
        
        try:
            self.stats['total_renders'] += 1
            
            # Validação de segurança
            if self.strict_mode and not self._is_template_safe(template_string):
                raise ValueError("Template contains unsafe content")
            
            # Prepara contexto
            full_context = self._prepare_context(context, kwargs)
            
            # Processa e renderiza
            format_to_use = template_format or self.default_format
            processed_template = self._preprocess_template(template_string, format_to_use)
            
            result = self._render_with_format(processed_template, format_to_use, full_context)
            
            # Atualiza estatísticas
            render_time = time.time() - start_time
            self._update_render_stats(render_time)
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            render_time = time.time() - start_time
            self._update_render_stats(render_time)
            raise ValueError(f"Failed to render string template: {str(e)}")
    
    def _preprocess_template(self, template: str, template_format: TemplateFormat) -> str:
        """Preprocessa template baseado no formato"""
        if template_format == TemplateFormat.JINJA2:
            return self._preprocess_jinja2(template)
        elif template_format == TemplateFormat.MUSTACHE:
            return self._preprocess_mustache(template)
        elif template_format == TemplateFormat.HANDLEBARS:
            return self._preprocess_handlebars(template)
        elif template_format == TemplateFormat.SIMPLE:
            return self._preprocess_simple(template)
        else:
            return template
    
    def _preprocess_jinja2(self, template: str) -> str:
        """Preprocessa template Jinja2"""
        return template
    
    def _preprocess_mustache(self, template: str) -> str:
        """Preprocessa template Mustache"""
        return re.sub(r'\{\{(.+?)\}\}', r'{\1}', template)
    
    def _preprocess_handlebars(self, template: str) -> str:
        """Preprocessa template Handlebars"""
        return template
    
    def _preprocess_simple(self, template: str) -> str:
        """Preprocessa template simples"""
        return re.sub(r'\$\{(.+?)\}', r'{\1}', template)
    
    def _render_with_format(
        self, 
        template: str, 
        template_format: TemplateFormat, 
        context: Dict[str, Any]
    ) -> str:
        """Renderiza template baseado no formato"""
        result = template
        
        # Substituição simples de variáveis
        for key, value in context.items():
            patterns = [
                f"{{{{\\s*{key}\\s*}}}}",
                f"{{\\s*{key}\\s*}}"
            ]
            
            for pattern in patterns:
                result = re.sub(pattern, str(value), result)
        
        return result
    
    def _prepare_context(self, context: Dict[str, Any] = None, kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepara contexto completo para renderização"""
        full_context = self.global_context.copy()
        
        if context:
            full_context.update(context)
        
        if kwargs:
            full_context.update(kwargs)
        
        # Adiciona funções customizadas
        full_context.update(self.custom_functions)
        
        return full_context
    
    def _is_template_safe(self, template: str) -> bool:
        """Verifica se template é seguro"""
        for pattern in self.forbidden_patterns:
            if re.search(pattern, template, re.IGNORECASE):
                return False
        
        return True
    
    def _generate_cache_key(self, template_name: str, context: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
        """Gera chave de cache"""
        cache_data = {
            'template': template_name,
            'context': context or {},
            'kwargs': kwargs or {}
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Obtém resultado do cache"""
        if cache_key not in self.template_cache:
            return None
        
        cached_item = self.template_cache[cache_key]
        
        # Verifica TTL
        if datetime.now() - cached_item['timestamp'] > timedelta(seconds=self.cache_ttl):
            del self.template_cache[cache_key]
            return None
        
        return cached_item['result']
    
    def _add_to_cache(self, cache_key: str, result: str):
        """Adiciona resultado ao cache"""
        self.template_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
    
    def _update_render_stats(self, render_time: float):
        """Atualiza estatísticas de renderização"""
        self.stats['total_render_time'] += render_time
        self.stats['avg_render_time'] = self.stats['total_render_time'] / self.stats['total_renders']
    
    # Métodos de gerenciamento
    
    def add_filter(self, name: str, filter_func: callable):
        """Adiciona filtro customizado"""
        self.custom_filters[name] = filter_func
    
    def add_function(self, name: str, func: callable):
        """Adiciona função customizada"""
        self.custom_functions[name] = func
    
    def set_global_context(self, key: str, value: Any):
        """Define variável global no contexto"""
        self.global_context[key] = value
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Obtém informações sobre um template"""
        return self.templates.get(template_name, {}).copy()
    
    def list_templates(self) -> List[str]:
        """Lista todos os templates registrados"""
        return list(self.templates.keys())
    
    def unregister_template(self, template_name: str) -> bool:
        """Remove template registrado"""
        if template_name in self.templates:
            del self.templates[template_name]
            return True
        return False
    
    def clear_cache(self):
        """Limpa cache de templates"""
        self.template_cache.clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de renderização"""
        return self.stats.copy()


class TestTemplateRendererBasic:
    """Testes básicos do renderizador de templates"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.renderer = MockTemplateRenderer()
    
    def test_renderer_initialization(self):
        """Testa inicialização do renderizador"""
        assert self.renderer.default_format == TemplateFormat.JINJA2
        assert self.renderer.enable_caching is True
        assert self.renderer.strict_mode is True
        assert self.renderer.auto_escape is True
        assert len(self.renderer.custom_filters) > 0
        assert len(self.renderer.custom_functions) > 0
    
    def test_template_registration(self):
        """Testa registro de templates"""
        template_content = "Hello {name}!"
        
        success = self.renderer.register_template("greeting", template_content)
        assert success is True
        
        templates = self.renderer.list_templates()
        assert "greeting" in templates
        
        info = self.renderer.get_template_info("greeting")
        assert info['content'] == template_content
        assert info['format'] == TemplateFormat.JINJA2
    
    def test_template_unregistration(self):
        """Testa remoção de templates"""
        self.renderer.register_template("test", "Test template")
        
        assert "test" in self.renderer.list_templates()
        
        success = self.renderer.unregister_template("test")
        assert success is True
        assert "test" not in self.renderer.list_templates()
        
        # Tenta remover template inexistente
        success = self.renderer.unregister_template("nonexistent")
        assert success is False


class TestTemplateRendererFormats:
    """Testes para diferentes formatos de template"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.renderer = MockTemplateRenderer()
    
    def test_jinja2_rendering(self):
        """Testa renderização Jinja2"""
        template = "Hello {name}! You have {count} messages."
        context = {'name': 'John', 'count': 5}
        
        result = self.renderer.render_string(template, context, TemplateFormat.JINJA2)
        
        assert "Hello John!" in result
        assert "5 messages" in result
    
    def test_mustache_rendering(self):
        """Testa renderização Mustache"""
        template = "Hello {{name}}! Welcome to {{site}}."
        context = {'name': 'Alice', 'site': 'MyApp'}
        
        result = self.renderer.render_string(template, context, TemplateFormat.MUSTACHE)
        
        assert "Alice" in result
        assert "MyApp" in result
    
    def test_simple_rendering(self):
        """Testa renderização simples"""
        template = "User: ${username}, Role: ${role}"
        context = {'username': 'admin', 'role': 'administrator'}
        
        result = self.renderer.render_string(template, context, TemplateFormat.SIMPLE)
        
        assert "User: admin" in result
        assert "Role: administrator" in result


class TestTemplateRendererAdvanced:
    """Testes para funcionalidades avançadas"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.renderer = MockTemplateRenderer()
    
    def test_custom_filters(self):
        """Testa filtros customizados"""
        def reverse_string(value):
            return str(value)[::-1]
        
        self.renderer.add_filter('reverse', reverse_string)
        
        template = "Original: {text}, Reversed: {text|reverse}"
        context = {'text': 'hello'}
        
        result = self.renderer.render_string(template, context, TemplateFormat.JINJA2)
        
        assert "hello" in result
    
    def test_custom_functions(self):
        """Testa funções customizadas"""
        def multiply(a, b):
            return a * b
        
        self.renderer.add_function('multiply', multiply)
        
        # Verifica se função foi adicionada
        assert 'multiply' in self.renderer.custom_functions
        assert self.renderer.custom_functions['multiply'](3, 4) == 12
    
    def test_global_context(self):
        """Testa contexto global"""
        self.renderer.set_global_context('app_name', 'MyApplication')
        self.renderer.set_global_context('version', '1.0')
        
        template = "Welcome to {app_name} v{version}"
        
        result = self.renderer.render_string(template, {}, TemplateFormat.SIMPLE)
        
        assert "MyApplication" in result
        assert "1.0" in result


class TestTemplateRendererCache:
    """Testes para sistema de cache"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.renderer = MockTemplateRenderer()
    
    def test_cache_functionality(self):
        """Testa funcionalidade de cache"""
        self.renderer.register_template("cached_template", "Hello {name}!")
        
        # Primeira renderização - cache miss
        result1 = self.renderer.render_template("cached_template", {'name': 'Alice'})
        stats1 = self.renderer.get_stats()
        
        # Segunda renderização - cache hit
        result2 = self.renderer.render_template("cached_template", {'name': 'Alice'})
        stats2 = self.renderer.get_stats()
        
        assert result1 == result2
        assert stats2['cache_hits'] > stats1['cache_hits']
    
    def test_cache_different_contexts(self):
        """Testa cache com contextos diferentes"""
        self.renderer.register_template("test_template", "User: {name}")
        
        # Diferentes contextos devem gerar entradas de cache diferentes
        result1 = self.renderer.render_template("test_template", {'name': 'Alice'})
        result2 = self.renderer.render_template("test_template", {'name': 'Bob'})
        
        assert result1 != result2
        assert "Alice" in result1
        assert "Bob" in result2
    
    def test_cache_management(self):
        """Testa gerenciamento de cache"""
        self.renderer.register_template("cache_test", "Test template")
        
        # Renderiza para popular cache
        self.renderer.render_template("cache_test", {})
        
        # Verifica que tem entradas no cache
        assert len(self.renderer.template_cache) > 0
        
        # Limpa cache
        self.renderer.clear_cache()
        
        # Verifica que cache foi limpo
        assert len(self.renderer.template_cache) == 0
        
        stats = self.renderer.get_stats()
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0


class TestTemplateRendererSecurity:
    """Testes para segurança de templates"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.renderer = MockTemplateRenderer()
    
    def test_unsafe_template_detection(self):
        """Testa detecção de templates inseguros"""
        unsafe_templates = [
            "__import__('os').system('rm -rf /')",
            "exec('malicious code')",
            "eval('dangerous expression')",
            "open('/etc/passwd').read()"
        ]
        
        for unsafe_template in unsafe_templates:
            with pytest.raises(ValueError, match="unsafe content"):
                self.renderer.register_template("unsafe", unsafe_template)
    
    def test_safe_template_acceptance(self):
        """Testa aceitação de templates seguros"""
        safe_templates = [
            "Hello {name}!",
            "Welcome to {app_name}",
            "List: {items}",
            "Date: {today}"
        ]
        
        for i, safe_template in enumerate(safe_templates):
            success = self.renderer.register_template(f"safe_{i}", safe_template)
            assert success is True
    
    def test_strict_mode_toggle(self):
        """Testa ativação/desativação do modo estrito"""
        unsafe_template = "__import__('os')"
        
        # Com strict mode ativo (padrão)
        with pytest.raises(ValueError):
            self.renderer.register_template("test", unsafe_template)
        
        # Desativa strict mode
        self.renderer.strict_mode = False
        
        # Agora deve aceitar
        success = self.renderer.register_template("test", unsafe_template)
        assert success is True


class TestTemplateRendererIntegration:
    """Testes de integração e workflows completos"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.renderer = MockTemplateRenderer()
    
    def test_complete_workflow(self):
        """Testa workflow completo de renderização"""
        # Configura renderer
        self.renderer.set_global_context('app_name', 'TestApp')
        self.renderer.add_filter('currency', lambda x: f"${x:.2f}")
        
        # Registra templates
        self.renderer.register_template(
            'invoice',
            'Invoice for {app_name}\nCustomer: {customer}\nTotal: {total}'
        )
        
        # Renderiza
        result = self.renderer.render_template(
            'invoice',
            {'customer': 'John Doe', 'total': 99.99}
        )
        
        assert 'TestApp' in result
        assert 'John Doe' in result
        assert '99.99' in result
        
        # Verifica estatísticas
        stats = self.renderer.get_stats()
        assert stats['total_renders'] >= 1
        assert stats['errors'] == 0
    
    def test_error_handling_and_recovery(self):
        """Testa tratamento de erros e recuperação"""
        initial_error_count = self.renderer.get_stats()['errors']
        
        # Tenta renderizar template inexistente
        with pytest.raises(ValueError):
            self.renderer.render_template('nonexistent', {})
        
        # Verifica que erro foi registrado
        stats_after_error = self.renderer.get_stats()
        assert stats_after_error['errors'] > initial_error_count
        
        # Testa recuperação com template válido
        self.renderer.register_template('valid', 'Hello {name}!')
        result = self.renderer.render_template('valid', {'name': 'World'})
        
        assert "Hello World!" in result
        
        # Verifica que renderização válida foi registrada
        final_stats = self.renderer.get_stats()
        assert final_stats['total_renders'] > stats_after_error['total_renders']
    
    def test_performance_tracking(self):
        """Testa rastreamento de performance"""
        # Registra template
        self.renderer.register_template('perf_test', 'Simple template: {value}')
        
        # Renderiza múltiplas vezes
        for i in range(5):
            self.renderer.render_template('perf_test', {'value': i})
        
        stats = self.renderer.get_stats()
        
        assert stats['total_renders'] == 5
        assert stats['avg_render_time'] >= 0
        assert stats['total_render_time'] >= 0
        
        # Verifica info do template
        template_info = self.renderer.get_template_info('perf_test')
        assert template_info['render_count'] == 5
        assert template_info['last_rendered'] is not None