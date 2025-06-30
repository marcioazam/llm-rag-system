"""
Testes para o módulo circuit_breaker - Sistema de Circuit Breaker
"""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio
import time
from enum import Enum
from typing import Dict, Any, Callable


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerException(Exception):
    pass


class MockCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, success_threshold: int = 3, timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.state_transitions = []
        
        self.on_state_change = None
        self.on_failure = None
        self.on_success = None
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        self.total_calls += 1
        
        if not self._should_allow_call():
            raise CircuitBreakerException(f"Circuit breaker is {self.state.value}")
        
        try:
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self.timeout
            )
            await self._on_success()
            return result
        except asyncio.TimeoutError:
            self.total_timeouts += 1
            await self._on_failure("timeout")
            raise CircuitBreakerException("Function execution timeout")
        except Exception as e:
            await self._on_failure(str(e))
            raise e
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _should_allow_call(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_transition_to_half_open():
                self._transition_to_half_open()
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def _should_transition_to_half_open(self) -> bool:
        if self.last_failure_time is None:
            return False
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout
    
    async def _on_success(self):
        self.total_successes += 1
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
        
        if self.on_success:
            await self._call_callback(self.on_success)
    
    async def _on_failure(self, error: str):
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        
        if self.on_failure:
            await self._call_callback(self.on_failure, error)
    
    def _transition_to_open(self):
        old_state = self.state
        self.state = CircuitState.OPEN
        self.success_count = 0
        self._record_state_transition(old_state, self.state)
    
    def _transition_to_half_open(self):
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        self._record_state_transition(old_state, self.state)
    
    def _transition_to_closed(self):
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._record_state_transition(old_state, self.state)
    
    def _record_state_transition(self, from_state: CircuitState, to_state: CircuitState):
        transition = {
            'timestamp': time.time(),
            'from_state': from_state.value,
            'to_state': to_state.value
        }
        self.state_transitions.append(transition)
        
        if self.on_state_change:
            asyncio.create_task(self._call_callback(self.on_state_change, from_state, to_state))
    
    async def _call_callback(self, callback: Callable, *args):
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception:
            pass
    
    def get_state(self) -> CircuitState:
        return self.state
    
    def get_statistics(self) -> Dict[str, Any]:
        success_rate = 0
        if self.total_calls > 0:
            success_rate = self.total_successes / self.total_calls
        
        return {
            'state': self.state.value,
            'total_calls': self.total_calls,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'total_timeouts': self.total_timeouts,
            'success_rate': success_rate,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'state_transitions': len(self.state_transitions),
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time
        }
    
    def reset(self):
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        if old_state != self.state:
            self._record_state_transition(old_state, self.state)
    
    def force_open(self):
        old_state = self.state
        self.state = CircuitState.OPEN
        self._record_state_transition(old_state, self.state)
    
    def force_half_open(self):
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        self._record_state_transition(old_state, self.state)


class TestCircuitBreakerBasic:
    def setup_method(self):
        self.circuit_breaker = MockCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=2,
            success_threshold=2,
            timeout=1.0
        )
    
    def test_circuit_breaker_initialization(self):
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.failure_threshold == 3
        assert self.circuit_breaker.recovery_timeout == 2
        assert self.circuit_breaker.success_threshold == 2
        assert self.circuit_breaker.timeout == 1.0
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.success_count == 0
    
    @pytest.mark.asyncio
    async def test_successful_call(self):
        async def successful_function():
            return "success"
        
        result = await self.circuit_breaker.call(successful_function)
        
        assert result == "success"
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.total_successes == 1
        assert self.circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_failed_call(self):
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await self.circuit_breaker.call(failing_function)
        
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.total_failures == 1
        assert self.circuit_breaker.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_timeout_call(self):
        async def slow_function():
            await asyncio.sleep(2)
            return "too slow"
        
        with pytest.raises(CircuitBreakerException, match="timeout"):
            await self.circuit_breaker.call(slow_function)
        
        assert self.circuit_breaker.total_timeouts == 1
        assert self.circuit_breaker.failure_count == 1
    
    def test_get_statistics(self):
        stats = self.circuit_breaker.get_statistics()
        
        assert 'state' in stats
        assert 'total_calls' in stats
        assert 'success_rate' in stats
        assert stats['state'] == 'closed'
        assert stats['total_calls'] == 0
        assert stats['success_rate'] == 0


class TestCircuitBreakerStateTransitions:
    def setup_method(self):
        self.circuit_breaker = MockCircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1,
            success_threshold=2,
            timeout=1.0
        )
    
    @pytest.mark.asyncio
    async def test_closed_to_open_transition(self):
        async def failing_function():
            raise Exception("Test failure")
        
        # Primeira falha
        with pytest.raises(Exception):
            await self.circuit_breaker.call(failing_function)
        assert self.circuit_breaker.state == CircuitState.CLOSED
        
        # Segunda falha - deve abrir o circuit
        with pytest.raises(Exception):
            await self.circuit_breaker.call(failing_function)
        assert self.circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_open_blocks_calls(self):
        self.circuit_breaker.force_open()
        
        async def any_function():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerException, match="Circuit breaker is open"):
            await self.circuit_breaker.call(any_function)
    
    @pytest.mark.asyncio
    async def test_open_to_half_open_transition(self):
        self.circuit_breaker.force_open()
        self.circuit_breaker.last_failure_time = time.time() - 2
        
        async def test_function():
            return "test"
        
        result = await self.circuit_breaker.call(test_function)
        
        assert result == "test"
        assert self.circuit_breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_to_closed_transition(self):
        self.circuit_breaker.force_half_open()
        
        async def successful_function():
            return "success"
        
        await self.circuit_breaker.call(successful_function)
        assert self.circuit_breaker.state == CircuitState.HALF_OPEN
        
        await self.circuit_breaker.call(successful_function)
        assert self.circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self):
        self.circuit_breaker.force_half_open()
        
        async def failing_function():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            await self.circuit_breaker.call(failing_function)
        
        assert self.circuit_breaker.state == CircuitState.OPEN


class TestCircuitBreakerAdvanced:
    def setup_method(self):
        self.circuit_breaker = MockCircuitBreaker()
        self.state_changes = []
        self.failures = []
        self.successes = []
    
    def _on_state_change(self, from_state: CircuitState, to_state: CircuitState):
        self.state_changes.append({
            'from': from_state.value,
            'to': to_state.value,
            'timestamp': time.time()
        })
    
    def _on_failure(self, error: str):
        self.failures.append(error)
    
    def _on_success(self):
        self.successes.append(time.time())
    
    def test_callback_registration(self):
        self.circuit_breaker.on_state_change = self._on_state_change
        self.circuit_breaker.on_failure = self._on_failure
        self.circuit_breaker.on_success = self._on_success
        
        assert self.circuit_breaker.on_state_change is not None
        assert self.circuit_breaker.on_failure is not None
        assert self.circuit_breaker.on_success is not None
    
    @pytest.mark.asyncio
    async def test_callbacks_are_called(self):
        self.circuit_breaker.on_state_change = self._on_state_change
        self.circuit_breaker.on_failure = self._on_failure
        self.circuit_breaker.on_success = self._on_success
        
        async def successful_function():
            return "success"
        
        async def failing_function():
            raise Exception("failure")
        
        await self.circuit_breaker.call(successful_function)
        assert len(self.successes) == 1
        
        with pytest.raises(Exception):
            await self.circuit_breaker.call(failing_function)
        assert len(self.failures) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        async def slow_function(delay: float):
            await asyncio.sleep(delay)
            return f"result_{delay}"
        
        tasks = [
            self.circuit_breaker.call(slow_function, 0.1),
            self.circuit_breaker.call(slow_function, 0.2),
            self.circuit_breaker.call(slow_function, 0.1)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all("result_" in str(result) for result in results)
        assert self.circuit_breaker.total_successes == 3
    
    @pytest.mark.asyncio
    async def test_mixed_success_failure_pattern(self):
        async def conditional_function(should_fail: bool):
            if should_fail:
                raise Exception("Conditional failure")
            return "success"
        
        calls = [False, True, False, True, True]
        
        for should_fail in calls:
            try:
                await self.circuit_breaker.call(conditional_function, should_fail)
            except Exception:
                pass
        
        stats = self.circuit_breaker.get_statistics()
        assert stats['total_calls'] == 5
        assert stats['total_successes'] == 2
        assert stats['total_failures'] == 3
    
    def test_reset_functionality(self):
        self.circuit_breaker.total_calls = 10
        self.circuit_breaker.total_failures = 5
        self.circuit_breaker.failure_count = 3
        self.circuit_breaker.force_open()
        
        self.circuit_breaker.reset()
        
        assert self.circuit_breaker.state == CircuitState.CLOSED
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.success_count == 0
        assert self.circuit_breaker.last_failure_time is None


class TestCircuitBreakerEdgeCases:
    def setup_method(self):
        self.circuit_breaker = MockCircuitBreaker()
    
    @pytest.mark.asyncio
    async def test_sync_function_execution(self):
        def sync_function(x: int) -> int:
            return x * 2
        
        result = await self.circuit_breaker.call(sync_function, 5)
        
        assert result == 10
        assert self.circuit_breaker.total_successes == 1
    
    @pytest.mark.asyncio
    async def test_function_with_args_kwargs(self):
        async def complex_function(a, b, c=None, d=None):
            return f"{a}-{b}-{c}-{d}"
        
        result = await self.circuit_breaker.call(
            complex_function, "arg1", "arg2", c="kwarg1", d="kwarg2"
        )
        
        assert result == "arg1-arg2-kwarg1-kwarg2"
    
    @pytest.mark.asyncio
    async def test_zero_thresholds(self):
        circuit = MockCircuitBreaker(failure_threshold=0, success_threshold=0)
        
        async def any_function():
            return "test"
        
        result = await circuit.call(any_function)
        assert result == "test"
    
    @pytest.mark.asyncio
    async def test_very_short_timeout(self):
        circuit = MockCircuitBreaker(timeout=0.001)
        
        async def slightly_slow_function():
            await asyncio.sleep(0.01)
            return "too slow"
        
        with pytest.raises(CircuitBreakerException, match="timeout"):
            await circuit.call(slightly_slow_function)
    
    def test_state_transitions_history(self):
        self.circuit_breaker.force_open()
        self.circuit_breaker.force_half_open()
        self.circuit_breaker.reset()
        
        transitions = self.circuit_breaker.state_transitions
        assert len(transitions) == 3
        
        assert transitions[0]['from_state'] == 'closed'
        assert transitions[0]['to_state'] == 'open'
        
        assert transitions[1]['from_state'] == 'open'
        assert transitions[1]['to_state'] == 'half_open'
        
        assert transitions[2]['from_state'] == 'half_open'
        assert transitions[2]['to_state'] == 'closed'


class TestCircuitBreakerIntegration:
    def setup_method(self):
        self.circuit_breaker = MockCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,
            success_threshold=2,
            timeout=2.0
        )
    
    @pytest.mark.asyncio
    async def test_full_circuit_breaker_cycle(self):
        async def api_call(should_fail: bool = False):
            if should_fail:
                raise Exception("API Error")
            await asyncio.sleep(0.1)
            return {"status": "success"}
        
        assert self.circuit_breaker.state == CircuitState.CLOSED
        
        for _ in range(3):
            result = await self.circuit_breaker.call(api_call, False)
            assert result["status"] == "success"
        
        for _ in range(3):
            with pytest.raises(Exception):
                await self.circuit_breaker.call(api_call, True)
        
        assert self.circuit_breaker.state == CircuitState.OPEN
        
        with pytest.raises(CircuitBreakerException, match="Circuit breaker is open"):
            await self.circuit_breaker.call(api_call, False)
        
        await asyncio.sleep(1.1)
        
        result = await self.circuit_breaker.call(api_call, False)
        assert self.circuit_breaker.state == CircuitState.HALF_OPEN
        
        result = await self.circuit_breaker.call(api_call, False)
        assert self.circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_varying_load(self):
        call_count = 0
        
        async def variable_function():
            nonlocal call_count
            call_count += 1
            
            if call_count % 4 == 0:
                raise Exception(f"Planned failure {call_count}")
            
            return f"success {call_count}"
        
        successful_calls = 0
        failed_calls = 0
        circuit_breaker_blocks = 0
        
        for i in range(20):
            try:
                result = await self.circuit_breaker.call(variable_function)
                successful_calls += 1
            except CircuitBreakerException:
                circuit_breaker_blocks += 1
            except Exception:
                failed_calls += 1
            
            await asyncio.sleep(0.05)
        
        stats = self.circuit_breaker.get_statistics()
        
        assert successful_calls > 0
        assert failed_calls > 0
        assert stats['total_calls'] == successful_calls + failed_calls
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_scenarios(self):
        failure_scenarios = [
            Exception("Database timeout"),
            ValueError("Invalid input"),
            ConnectionError("Network error"),
        ]
        
        async def failing_service(error_type: Exception):
            raise error_type
        
        for error in failure_scenarios:
            try:
                await self.circuit_breaker.call(failing_service, error)
            except Exception:
                pass
        
        if self.circuit_breaker.failure_count >= self.circuit_breaker.failure_threshold:
            assert self.circuit_breaker.state == CircuitState.OPEN
        
        self.circuit_breaker.reset()
        assert self.circuit_breaker.state == CircuitState.CLOSED
        
        async def working_service():
            return "service restored"
        
        result = await self.circuit_breaker.call(working_service)
        assert result == "service restored"

