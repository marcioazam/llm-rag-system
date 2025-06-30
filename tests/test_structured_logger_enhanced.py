"""
Testes para o módulo structured_logger - Sistema de Logging Estruturado
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import threading
from contextlib import contextmanager


class LogLevel(Enum):
    """Níveis de logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MockStructuredLogger:
    """Mock do sistema de logging estruturado"""
    
    def __init__(self, service_name: str = "test_service", version: str = "1.0.0"):
        self.service_name = service_name
        self.version = version
        self.correlation_id = None
        self.context = {}
        self.logs = []
        self.min_level = LogLevel.INFO
        self.enabled = True
        self.formatters = {}
        self.filters = []
        self.buffers = {}
        self.performance_tracking = {}
        
        # Configurações
        self.max_buffer_size = 1000
        self.flush_interval = 30
        self.correlation_header = "X-Correlation-ID"
        self.include_stack_trace = True
        self.redact_sensitive = True
        
        # Estatísticas
        self.stats = {
            'total_logs': 0,
            'logs_by_level': {level.value: 0 for level in LogLevel},
            'errors_count': 0,
            'performance_events': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
    
    def log(self, level: LogLevel, message: str, **context) -> str:
        """Log principal com contexto"""
        if not self.enabled or not self._should_log(level):
            return ""
        
        with self._lock:
            log_entry = self._create_log_entry(level, message, context)
            self.logs.append(log_entry)
            self._update_stats(level)
            
            # Processa filtros
            if self._passes_filters(log_entry):
                formatted_log = self._format_log(log_entry)
                self._output_log(formatted_log)
                return formatted_log
            
            return ""
    
    def debug(self, message: str, **context) -> str:
        """Log de debug"""
        return self.log(LogLevel.DEBUG, message, **context)
    
    def info(self, message: str, **context) -> str:
        """Log de informação"""
        return self.log(LogLevel.INFO, message, **context)
    
    def warning(self, message: str, **context) -> str:
        """Log de warning"""
        return self.log(LogLevel.WARNING, message, **context)
    
    def error(self, message: str, **context) -> str:
        """Log de erro"""
        return self.log(LogLevel.ERROR, message, **context)
    
    def critical(self, message: str, **context) -> str:
        """Log crítico"""
        return self.log(LogLevel.CRITICAL, message, **context)
    
    def _create_log_entry(self, level: LogLevel, message: str, context: Dict) -> Dict[str, Any]:
        """Cria entrada de log estruturada"""
        timestamp = datetime.now().isoformat() + "Z"
        
        log_entry = {
            'timestamp': timestamp,
            'level': level.value,
            'message': message,
            'service': self.service_name,
            'version': self.version,
            'correlation_id': self.correlation_id or self._generate_correlation_id(),
            'context': {**self.context, **context},
            'thread_id': threading.current_thread().ident,
            'process_id': 12345  # Mock PID
        }
        
        # Adiciona stack trace para erros se habilitado
        if level in [LogLevel.ERROR, LogLevel.CRITICAL] and self.include_stack_trace:
            log_entry['stack_trace'] = self._get_mock_stack_trace()
        
        # Redação de dados sensíveis
        if self.redact_sensitive:
            log_entry = self._redact_sensitive_data(log_entry)
        
        return log_entry
    
    def _should_log(self, level: LogLevel) -> bool:
        """Verifica se deve registrar o log baseado no nível"""
        level_order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        return level_order.index(level) >= level_order.index(self.min_level)
    
    def _passes_filters(self, log_entry: Dict) -> bool:
        """Verifica se log passa pelos filtros"""
        for filter_func in self.filters:
            if not filter_func(log_entry):
                return False
        return True
    
    def _format_log(self, log_entry: Dict) -> str:
        """Formata log para output"""
        formatter = self.formatters.get('default', self._default_formatter)
        return formatter(log_entry)
    
    def _default_formatter(self, log_entry: Dict) -> str:
        """Formatador padrão JSON"""
        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))
    
    def _output_log(self, formatted_log: str):
        """Output do log (mock - normalmente seria para arquivo/console)"""
        # Em produção iria para arquivo, console, ou sistema externo
        pass
    
    def _update_stats(self, level: LogLevel):
        """Atualiza estatísticas"""
        self.stats['total_logs'] += 1
        self.stats['logs_by_level'][level.value] += 1
        
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.stats['errors_count'] += 1
    
    def _generate_correlation_id(self) -> str:
        """Gera ID de correlação"""
        return str(uuid.uuid4())
    
    def _get_mock_stack_trace(self) -> str:
        """Mock de stack trace"""
        return "mock_stack_trace_here"
    
    def _redact_sensitive_data(self, log_entry: Dict) -> Dict:
        """Remove dados sensíveis"""
        sensitive_keys = ['password', 'token', 'api_key', 'secret']
        
        def redact_dict(d):
            if isinstance(d, dict):
                return {k: '[REDACTED]' if k.lower() in sensitive_keys else redact_dict(v) 
                       for k, v in d.items()}
            elif isinstance(d, list):
                return [redact_dict(item) for item in d]
            return d
        
        return redact_dict(log_entry)
    
    def set_correlation_id(self, correlation_id: str):
        """Define ID de correlação"""
        self.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Obtém ID de correlação atual"""
        return self.correlation_id
    
    def add_context(self, key: str, value: Any):
        """Adiciona contexto persistente"""
        self.context[key] = value
    
    def remove_context(self, key: str):
        """Remove contexto"""
        self.context.pop(key, None)
    
    def clear_context(self):
        """Limpa todo o contexto"""
        self.context.clear()
    
    @contextmanager
    def context_manager(self, **context):
        """Context manager para contexto temporário"""
        old_context = self.context.copy()
        self.context.update(context)
        try:
            yield
        finally:
            self.context = old_context
    
    def set_level(self, level: LogLevel):
        """Define nível mínimo de logging"""
        self.min_level = level
    
    def add_filter(self, filter_func):
        """Adiciona filtro de log"""
        self.filters.append(filter_func)
    
    def set_formatter(self, name: str, formatter_func):
        """Define formatador customizado"""
        self.formatters[name] = formatter_func
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de logging"""
        return self.stats.copy()
    
    def clear_logs(self):
        """Limpa logs armazenados (para testes)"""
        with self._lock:
            self.logs.clear()
    
    def get_logs(self) -> List[Dict]:
        """Retorna logs registrados (para testes)"""
        with self._lock:
            return self.logs.copy()
    
    def flush_buffers(self):
        """Força flush dos buffers"""
        # Simula flush de buffers pendentes
        for buffer_name, buffer in self.buffers.items():
            buffer.clear()
    
    def start_performance_tracking(self, event_name: str) -> str:
        """Inicia rastreamento de performance"""
        tracking_id = str(uuid.uuid4())
        self.performance_tracking[tracking_id] = {
            'event_name': event_name,
            'start_time': time.time(),
            'end_time': None
        }
        return tracking_id
    
    def end_performance_tracking(self, tracking_id: str) -> Optional[float]:
        """Finaliza rastreamento de performance"""
        if tracking_id in self.performance_tracking:
            event = self.performance_tracking[tracking_id]
            event['end_time'] = time.time()
            duration = event['end_time'] - event['start_time']
            
            self.stats['performance_events'] += 1
            
            # Log do evento de performance
            self.info(
                f"Performance event completed",
                event_name=event['event_name'],
                duration_ms=duration * 1000,
                tracking_id=tracking_id
            )
            
            return duration
        return None


class TestStructuredLoggerBasic:
    """Testes básicos do logger estruturado"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.logger = MockStructuredLogger("test_service", "1.0.0")
    
    def test_logger_initialization(self):
        """Testa inicialização do logger"""
        assert self.logger.service_name == "test_service"
        assert self.logger.version == "1.0.0"
        assert self.logger.enabled is True
        assert self.logger.min_level == LogLevel.INFO
        assert len(self.logger.context) == 0
        assert self.logger.correlation_id is None
    
    def test_basic_logging_methods(self):
        """Testa métodos básicos de logging"""
        # Debug (não deve aparecer com nível INFO)
        debug_result = self.logger.debug("Debug message")
        assert debug_result == ""
        
        # Info
        info_result = self.logger.info("Info message")
        assert info_result != ""
        assert "Info message" in info_result
        
        # Warning
        warning_result = self.logger.warning("Warning message")
        assert warning_result != ""
        assert "WARNING" in warning_result
        
        # Error
        error_result = self.logger.error("Error message")
        assert error_result != ""
        assert "ERROR" in error_result
        
        # Critical
        critical_result = self.logger.critical("Critical message")
        assert critical_result != ""
        assert "CRITICAL" in critical_result
    
    def test_log_level_filtering(self):
        """Testa filtragem por nível de log"""
        # Nível DEBUG - deve logar tudo
        self.logger.set_level(LogLevel.DEBUG)
        
        debug_log = self.logger.debug("Debug test")
        info_log = self.logger.info("Info test")
        
        assert debug_log != ""
        assert info_log != ""
        
        # Nível ERROR - só errors e critical
        self.logger.set_level(LogLevel.ERROR)
        self.logger.clear_logs()
        
        self.logger.info("Should not appear")
        self.logger.warning("Should not appear")
        self.logger.error("Should appear")
        
        logs = self.logger.get_logs()
        assert len(logs) == 1
        assert logs[0]['level'] == 'ERROR'
    
    def test_log_structure(self):
        """Testa estrutura do log gerado"""
        log_result = self.logger.info("Test message", user_id=123, action="test")
        
        # Parse JSON
        log_data = json.loads(log_result)
        
        # Verifica campos obrigatórios
        required_fields = ['timestamp', 'level', 'message', 'service', 'version', 'correlation_id']
        for field in required_fields:
            assert field in log_data
        
        assert log_data['level'] == 'INFO'
        assert log_data['message'] == 'Test message'
        assert log_data['service'] == 'test_service'
        assert log_data['context']['user_id'] == 123
        assert log_data['context']['action'] == 'test'
    
    def test_correlation_id_management(self):
        """Testa gerenciamento de correlation ID"""
        # Correlation ID automático
        log1 = self.logger.info("Message 1")
        log1_data = json.loads(log1)
        correlation_1 = log1_data['correlation_id']
        
        assert correlation_1 is not None
        assert len(correlation_1) > 0
        
        # Correlation ID manual
        custom_id = "custom-correlation-123"
        self.logger.set_correlation_id(custom_id)
        
        log2 = self.logger.info("Message 2")
        log2_data = json.loads(log2)
        
        assert log2_data['correlation_id'] == custom_id
        assert self.logger.get_correlation_id() == custom_id


class TestStructuredLoggerContext:
    """Testes para gerenciamento de contexto"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.logger = MockStructuredLogger()
    
    def test_persistent_context(self):
        """Testa contexto persistente"""
        self.logger.add_context("user_id", 123)
        self.logger.add_context("session_id", "abc123")
        
        log_result = self.logger.info("Test message")
        log_data = json.loads(log_result)
        
        assert log_data['context']['user_id'] == 123
        assert log_data['context']['session_id'] == "abc123"
        
        # Segundo log deve manter contexto
        log_result2 = self.logger.info("Second message")
        log_data2 = json.loads(log_result2)
        
        assert log_data2['context']['user_id'] == 123
        assert log_data2['context']['session_id'] == "abc123"
    
    def test_context_override(self):
        """Testa override de contexto"""
        self.logger.add_context("user_id", 123)
        
        # Override no log específico
        log_result = self.logger.info("Test message", user_id=456, extra="data")
        log_data = json.loads(log_result)
        
        assert log_data['context']['user_id'] == 456  # Override
        assert log_data['context']['extra'] == "data"
        
        # Próximo log volta ao contexto original
        log_result2 = self.logger.info("Second message")
        log_data2 = json.loads(log_result2)
        
        assert log_data2['context']['user_id'] == 123  # Original
        assert 'extra' not in log_data2['context']
    
    def test_context_manager(self):
        """Testa context manager temporário"""
        self.logger.add_context("user_id", 123)
        
        with self.logger.context_manager(operation="test_op", temp_data="temp"):
            log_result = self.logger.info("Inside context manager")
            log_data = json.loads(log_result)
            
            assert log_data['context']['user_id'] == 123
            assert log_data['context']['operation'] == "test_op"
            assert log_data['context']['temp_data'] == "temp"
        
        # Fora do context manager - contexto temporário removido
        log_result2 = self.logger.info("Outside context manager")
        log_data2 = json.loads(log_result2)
        
        assert log_data2['context']['user_id'] == 123
        assert 'operation' not in log_data2['context']
        assert 'temp_data' not in log_data2['context']
    
    def test_context_clearing(self):
        """Testa limpeza de contexto"""
        self.logger.add_context("user_id", 123)
        self.logger.add_context("session_id", "abc")
        
        # Remove contexto específico
        self.logger.remove_context("user_id")
        
        log_result = self.logger.info("After removal")
        log_data = json.loads(log_result)
        
        assert 'user_id' not in log_data['context']
        assert log_data['context']['session_id'] == "abc"
        
        # Limpa todo contexto
        self.logger.clear_context()
        
        log_result2 = self.logger.info("After clear")
        log_data2 = json.loads(log_result2)
        
        assert len(log_data2['context']) == 0


class TestStructuredLoggerSecurity:
    """Testes para funcionalidades de segurança"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.logger = MockStructuredLogger()
        self.logger.redact_sensitive = True
    
    def test_sensitive_data_redaction(self):
        """Testa redação de dados sensíveis"""
        log_result = self.logger.info(
            "User login",
            user_id=123,
            password="secret123",
            api_key="key_abc123",
            token="bearer_token",
            secret="my_secret"
        )
        
        log_data = json.loads(log_result)
        context = log_data['context']
        
        assert context['user_id'] == 123  # Não sensível
        assert context['password'] == '[REDACTED]'
        assert context['api_key'] == '[REDACTED]'
        assert context['token'] == '[REDACTED]'
        assert context['secret'] == '[REDACTED]'
    
    def test_nested_sensitive_data_redaction(self):
        """Testa redação em dados aninhados"""
        nested_data = {
            'user': {
                'id': 123,
                'password': 'secret',
                'profile': {
                    'name': 'John',
                    'api_key': 'key123'
                }
            },
            'tokens': ['token1', 'token2']
        }
        
        log_result = self.logger.info("Complex data", data=nested_data)
        log_data = json.loads(log_result)
        
        user_data = log_data['context']['data']['user']
        assert user_data['id'] == 123
        assert user_data['password'] == '[REDACTED]'
        assert user_data['profile']['name'] == 'John'
        assert user_data['profile']['api_key'] == '[REDACTED]'
    
    def test_redaction_disabled(self):
        """Testa comportamento com redação desabilitada"""
        self.logger.redact_sensitive = False
        
        log_result = self.logger.info("Test", password="secret123")
        log_data = json.loads(log_result)
        
        assert log_data['context']['password'] == "secret123"
    
    def test_stack_trace_inclusion(self):
        """Testa inclusão de stack trace em erros"""
        self.logger.include_stack_trace = True
        
        error_log = self.logger.error("Error occurred", error_type="ValueError")
        log_data = json.loads(error_log)
        
        assert 'stack_trace' in log_data
        assert log_data['stack_trace'] == "mock_stack_trace_here"
        
        # Info não deve ter stack trace
        info_log = self.logger.info("Info message")
        info_data = json.loads(info_log)
        
        assert 'stack_trace' not in info_data


class TestStructuredLoggerFilters:
    """Testes para sistema de filtros"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.logger = MockStructuredLogger()
        self.logger.set_level(LogLevel.DEBUG)  # Aceita todos os níveis
    
    def test_simple_filter(self):
        """Testa filtro simples"""
        def user_filter(log_entry):
            return log_entry['context'].get('user_id') == 123
        
        self.logger.add_filter(user_filter)
        
        # Log que passa no filtro
        log1 = self.logger.info("Message 1", user_id=123)
        assert log1 != ""
        
        # Log que não passa no filtro
        log2 = self.logger.info("Message 2", user_id=456)
        assert log2 == ""
        
        # Log sem user_id
        log3 = self.logger.info("Message 3")
        assert log3 == ""
    
    def test_multiple_filters(self):
        """Testa múltiplos filtros"""
        def user_filter(log_entry):
            return log_entry['context'].get('user_id') == 123
        
        def level_filter(log_entry):
            return log_entry['level'] in ['INFO', 'ERROR']
        
        self.logger.add_filter(user_filter)
        self.logger.add_filter(level_filter)
        
        # Passa em ambos os filtros
        log1 = self.logger.info("Message 1", user_id=123)
        assert log1 != ""
        
        # Falha no filtro de usuário
        log2 = self.logger.info("Message 2", user_id=456)
        assert log2 == ""
        
        # Falha no filtro de nível
        log3 = self.logger.debug("Debug message", user_id=123)
        assert log3 == ""
    
    def test_performance_filter(self):
        """Testa filtro baseado em performance"""
        def slow_operation_filter(log_entry):
            duration = log_entry['context'].get('duration_ms', 0)
            return duration > 100  # Só loga operações lentas
        
        self.logger.add_filter(slow_operation_filter)
        
        # Operação rápida - não deve logar
        fast_log = self.logger.info("Fast operation", duration_ms=50)
        assert fast_log == ""
        
        # Operação lenta - deve logar
        slow_log = self.logger.info("Slow operation", duration_ms=200)
        assert slow_log != ""


class TestStructuredLoggerPerformance:
    """Testes para funcionalidades de performance"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.logger = MockStructuredLogger()
    
    def test_performance_tracking(self):
        """Testa rastreamento de performance"""
        # Inicia tracking
        tracking_id = self.logger.start_performance_tracking("test_operation")
        assert tracking_id is not None
        assert len(tracking_id) > 0
        
        # Simula operação
        time.sleep(0.1)
        
        # Finaliza tracking
        duration = self.logger.end_performance_tracking(tracking_id)
        assert duration is not None
        assert duration >= 0.1
        
        # Verifica se log de performance foi criado
        logs = self.logger.get_logs()
        performance_logs = [log for log in logs if 'duration_ms' in log['context']]
        assert len(performance_logs) == 1
        
        perf_log = performance_logs[0]
        assert perf_log['context']['event_name'] == "test_operation"
        assert perf_log['context']['tracking_id'] == tracking_id
        assert perf_log['context']['duration_ms'] >= 100  # 0.1s = 100ms
    
    def test_invalid_tracking_id(self):
        """Testa ID de tracking inválido"""
        invalid_id = "invalid-tracking-id"
        duration = self.logger.end_performance_tracking(invalid_id)
        
        assert duration is None
    
    def test_statistics_collection(self):
        """Testa coleta de estatísticas"""
        initial_stats = self.logger.get_stats()
        assert initial_stats['total_logs'] == 0
        assert initial_stats['errors_count'] == 0
        
        # Gera alguns logs
        self.logger.info("Info 1")
        self.logger.info("Info 2")
        self.logger.warning("Warning 1")
        self.logger.error("Error 1")
        self.logger.critical("Critical 1")
        
        stats = self.logger.get_stats()
        assert stats['total_logs'] == 5
        assert stats['logs_by_level']['INFO'] == 2
        assert stats['logs_by_level']['WARNING'] == 1
        assert stats['logs_by_level']['ERROR'] == 1
        assert stats['logs_by_level']['CRITICAL'] == 1
        assert stats['errors_count'] == 2  # ERROR + CRITICAL
    
    def test_buffer_management(self):
        """Testa gerenciamento de buffers"""
        # Adiciona dados aos buffers
        self.logger.buffers['test_buffer'] = ['log1', 'log2', 'log3']
        
        assert len(self.logger.buffers['test_buffer']) == 3
        
        # Flush buffers
        self.logger.flush_buffers()
        
        assert len(self.logger.buffers['test_buffer']) == 0


class TestStructuredLoggerIntegration:
    """Testes de integração"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.logger = MockStructuredLogger("integration_service", "2.0.0")
    
    def test_complete_logging_workflow(self):
        """Testa workflow completo de logging"""
        # Setup
        self.logger.set_level(LogLevel.DEBUG)
        self.logger.set_correlation_id("workflow-123")
        self.logger.add_context("user_id", 456)
        self.logger.add_context("session_id", "session_abc")
        
        # Filtro personalizado
        def important_events_filter(log_entry):
            return log_entry['context'].get('important', False)
        
        self.logger.add_filter(important_events_filter)
        
        # Formatador personalizado
        def custom_formatter(log_entry):
            return f"[{log_entry['level']}] {log_entry['message']} | {log_entry['correlation_id']}"
        
        self.logger.set_formatter('custom', custom_formatter)
        
        # Logs que não passam no filtro
        unimportant_log = self.logger.info("Unimportant event")
        assert unimportant_log == ""
        
        # Log importante
        important_log = self.logger.info("Important event occurred", important=True, event_type="user_action")
        assert important_log != ""
        
        # Verifica estrutura do log
        log_data = json.loads(important_log)
        assert log_data['correlation_id'] == "workflow-123"
        assert log_data['context']['user_id'] == 456
        assert log_data['context']['session_id'] == "session_abc"
        assert log_data['context']['important'] is True
        assert log_data['context']['event_type'] == "user_action"
    
    def test_error_handling_workflow(self):
        """Testa workflow de tratamento de erros"""
        self.logger.include_stack_trace = True
        
        try:
            # Simula operação que pode falhar
            with self.logger.context_manager(operation="risky_operation", attempt=1):
                # Tracking de performance
                tracking_id = self.logger.start_performance_tracking("risky_operation")
                
                # Simula erro
                error_context = {
                    'error_type': 'ValueError',
                    'error_code': 'VAL_001',
                    'user_input': 'invalid_data',
                    'password': 'secret123'  # Será redacted
                }
                
                self.logger.error("Operation failed", **error_context)
                
                # Finaliza tracking
                duration = self.logger.end_performance_tracking(tracking_id)
                
        except Exception:
            pass  # Ignora exceção para o teste
        
        # Verifica logs gerados
        logs = self.logger.get_logs()
        assert len(logs) >= 2  # Error log + performance log
        
        # Encontra log de erro
        error_logs = [log for log in logs if log['level'] == 'ERROR']
        assert len(error_logs) == 1
        
        error_log = error_logs[0]
        assert 'stack_trace' in error_log
        assert error_log['context']['error_type'] == 'ValueError'
        assert error_log['context']['password'] == '[REDACTED]'
        assert error_log['context']['operation'] == 'risky_operation'
    
    def test_concurrent_logging(self):
        """Testa logging concorrente (simulado)"""
        import threading
        import time
        
        def worker_thread(worker_id):
            """Função executada por thread worker"""
            for i in range(5):
                self.logger.info(
                    f"Worker {worker_id} message {i}",
                    worker_id=worker_id,
                    message_num=i,
                    thread_id=threading.current_thread().ident
                )
                time.sleep(0.01)  # Pequena pausa
        
        # Executa múltiplas threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker_thread, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Aguarda conclusão
        for thread in threads:
            thread.join()
        
        # Verifica logs
        logs = self.logger.get_logs()
        assert len(logs) == 15  # 3 workers * 5 messages each
        
        # Verifica se todos os workers logaram
        worker_ids = set(log['context']['worker_id'] for log in logs)
        assert worker_ids == {0, 1, 2}
        
        # Verifica integridade dos dados
        for log in logs:
            assert 'worker_id' in log['context']
            assert 'message_num' in log['context']
            assert 'thread_id' in log['context']