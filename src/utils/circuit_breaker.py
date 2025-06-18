# -*- coding: utf-8 -*-
"""
Circuit Breaker Pattern para APIs.
Implementa proteção contra falhas de serviços externos.
"""

import time
import logging
from typing import Any, Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Estados do circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, calls rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit Breaker Pattern implementation.
    
    Protege contra falhas de serviços externos, evitando cascading failures.
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        """
        Inicializa o circuit breaker.
        
        Args:
            failure_threshold: Número de falhas consecutivas para abrir o circuito
            recovery_timeout: Tempo em segundos para tentar recuperação
            expected_exception: Tipo de exceção que conta como falha
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # Estado interno
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
        # Estatísticas
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
    def can_execute(self) -> bool:
        """
        Verifica se o circuit breaker permite execução.
        
        Returns:
            True se pode executar, False se circuito está aberto
        """
        self.total_calls += 1
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Verificar se deve tentar recuperação
            if self._should_attempt_recovery():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker: Tentando recuperação (HALF_OPEN)")
                return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Em half-open, permite apenas uma tentativa
            return True
        
        return False
    
    def record_success(self):
        """Registra uma operação bem-sucedida."""
        self.successful_calls += 1
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Recuperação bem-sucedida
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker: Recuperado (CLOSED)")
        
    def record_failure(self):
        """Registra uma falha."""
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Falha durante recuperação - voltar para OPEN
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker: Falha na recuperação (OPEN)")
        
        elif self.failure_count >= self.failure_threshold:
            # Muitas falhas - abrir circuito
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker: Circuito aberto após {self.failure_count} falhas")
    
    def _should_attempt_recovery(self) -> bool:
        """Verifica se deve tentar recuperação."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do circuit breaker."""
        success_rate = 0.0
        if self.total_calls > 0:
            success_rate = self.successful_calls / self.total_calls
        
        return {
            'state': self.state.value,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': success_rate,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time
        }
    
    def reset(self):
        """Reseta o circuit breaker para estado inicial."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        logger.info("Circuit breaker: Reset para estado CLOSED")


def circuit_breaker_decorator(failure_threshold: int = 5,
                             recovery_timeout: int = 60,
                             expected_exception: type = Exception):
    """
    Decorator para aplicar circuit breaker a funções.
    
    Args:
        failure_threshold: Número de falhas para abrir o circuito
        recovery_timeout: Tempo de recuperação em segundos
        expected_exception: Tipo de exceção que conta como falha
    """
    def decorator(func: Callable) -> Callable:
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception
        )
        
        async def async_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except expected_exception as e:
                breaker.record_failure()
                raise e
        
        def sync_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except expected_exception as e:
                breaker.record_failure()
                raise e
        
        # Detectar se a função é async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator 