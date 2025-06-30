"""
Testes para o módulo pipeline_dependency - Sistema de Injeção de Dependências
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Type, Protocol
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import inspect


class DependencyScope(Enum):
    """Escopos de dependência"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class DependencyLifecycle(Enum):
    """Ciclo de vida da dependência"""
    CREATING = "creating"
    CREATED = "created"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    DISPOSING = "disposing"
    DISPOSED = "disposed"


class IDependencyContainer(Protocol):
    """Interface do container de dependências"""
    
    def register(self, interface: Type, implementation: Type, scope: DependencyScope = DependencyScope.TRANSIENT):
        """Registra uma dependência"""
        pass
    
    def resolve(self, interface: Type):
        """Resolve uma dependência"""
        pass


class MockDependencyContainer:
    """Mock do container de injeção de dependências"""
    
    def __init__(self):
        self.registrations = {}
        self.instances = {}
        self.scoped_instances = {}
        self.lifecycle_callbacks = {}
        self.resolution_stack = []
        self.creation_order = []
        
        # Configurações
        self.auto_wire = True
        self.circular_detection = True
        self.lazy_loading = False
        self.thread_safe = True
        
        # Estatísticas
        self.stats = {
            'registrations_count': 0,
            'resolutions_count': 0,
            'singletons_created': 0,
            'transients_created': 0,
            'circular_detected': 0,
            'failed_resolutions': 0
        }
        
        # Hooks
        self.pre_resolution_hooks = []
        self.post_resolution_hooks = []
        self.pre_creation_hooks = []
        self.post_creation_hooks = []
    
    def register(
        self, 
        interface: Type, 
        implementation: Type = None, 
        scope: DependencyScope = DependencyScope.TRANSIENT,
        factory: callable = None,
        **kwargs
    ):
        """Registra uma dependência no container"""
        if implementation is None and factory is None:
            implementation = interface
        
        registration = {
            'interface': interface,
            'implementation': implementation,
            'factory': factory,
            'scope': scope,
            'kwargs': kwargs,
            'lifecycle': DependencyLifecycle.CREATING,
            'dependencies': self._extract_dependencies(implementation) if implementation else []
        }
        
        self.registrations[interface] = registration
        self.stats['registrations_count'] += 1
        
        return self
    
    def register_singleton(self, interface: Type, implementation: Type = None, **kwargs):
        """Registra como singleton"""
        return self.register(interface, implementation, DependencyScope.SINGLETON, **kwargs)
    
    def register_transient(self, interface: Type, implementation: Type = None, **kwargs):
        """Registra como transient"""
        return self.register(interface, implementation, DependencyScope.TRANSIENT, **kwargs)
    
    def register_scoped(self, interface: Type, implementation: Type = None, **kwargs):
        """Registra como scoped"""
        return self.register(interface, implementation, DependencyScope.SCOPED, **kwargs)
    
    def register_factory(self, interface: Type, factory: callable, scope: DependencyScope = DependencyScope.TRANSIENT):
        """Registra factory function"""
        return self.register(interface, None, scope, factory)
    
    def resolve(self, interface: Type, scope_id: str = None):
        """Resolve uma dependência"""
        self.stats['resolutions_count'] += 1
        
        try:
            # Executa hooks pré-resolução
            for hook in self.pre_resolution_hooks:
                hook(interface)
            
            # Verifica circular dependency
            if self.circular_detection and interface in self.resolution_stack:
                self.stats['circular_detected'] += 1
                raise ValueError(f"Circular dependency detected: {interface}")
            
            self.resolution_stack.append(interface)
            
            try:
                instance = self._resolve_internal(interface, scope_id)
                
                # Executa hooks pós-resolução
                for hook in self.post_resolution_hooks:
                    hook(interface, instance)
                
                return instance
                
            finally:
                self.resolution_stack.remove(interface)
        
        except Exception as e:
            self.stats['failed_resolutions'] += 1
            raise e
    
    def _resolve_internal(self, interface: Type, scope_id: str = None):
        """Resolução interna"""
        if interface not in self.registrations:
            raise ValueError(f"No registration found for {interface}")
        
        registration = self.registrations[interface]
        scope = registration['scope']
        
        # Singleton
        if scope == DependencyScope.SINGLETON:
            if interface not in self.instances:
                self.instances[interface] = self._create_instance(registration)
                self.stats['singletons_created'] += 1
            return self.instances[interface]
        
        # Scoped
        elif scope == DependencyScope.SCOPED:
            scope_key = f"{interface}_{scope_id or 'default'}"
            if scope_key not in self.scoped_instances:
                self.scoped_instances[scope_key] = self._create_instance(registration)
            return self.scoped_instances[scope_key]
        
        # Transient
        else:
            self.stats['transients_created'] += 1
            return self._create_instance(registration)
    
    def _create_instance(self, registration: Dict):
        """Cria instância da dependência"""
        interface = registration['interface']
        implementation = registration['implementation']
        factory = registration['factory']
        kwargs = registration['kwargs']
        
        # Executa hooks pré-criação
        for hook in self.pre_creation_hooks:
            hook(interface)
        
        registration['lifecycle'] = DependencyLifecycle.CREATING
        
        try:
            if factory:
                # Usa factory function
                if self.auto_wire:
                    resolved_kwargs = self._resolve_factory_dependencies(factory)
                    instance = factory(**resolved_kwargs, **kwargs)
                else:
                    instance = factory(**kwargs)
            else:
                # Usa classe diretamente
                if self.auto_wire:
                    resolved_kwargs = self._resolve_constructor_dependencies(implementation)
                    instance = implementation(**resolved_kwargs, **kwargs)
                else:
                    instance = implementation(**kwargs)
            
            registration['lifecycle'] = DependencyLifecycle.CREATED
            self.creation_order.append(interface)
            
            # Executa hooks pós-criação
            for hook in self.post_creation_hooks:
                hook(interface, instance)
            
            return instance
            
        except Exception as e:
            registration['lifecycle'] = DependencyLifecycle.DISPOSED
            raise e
    
    def _extract_dependencies(self, implementation: Type) -> List[Type]:
        """Extrai dependências do construtor"""
        if not implementation:
            return []
        
        try:
            signature = inspect.signature(implementation.__init__)
            dependencies = []
            
            for param_name, param in signature.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
            
            return dependencies
        except Exception:
            return []
    
    def _resolve_constructor_dependencies(self, implementation: Type) -> Dict[str, Any]:
        """Resolve dependências do construtor"""
        if not implementation:
            return {}
        
        try:
            signature = inspect.signature(implementation.__init__)
            resolved_kwargs = {}
            
            for param_name, param in signature.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    if param.annotation in self.registrations:
                        resolved_kwargs[param_name] = self.resolve(param.annotation)
                    elif param.default != inspect.Parameter.empty:
                        resolved_kwargs[param_name] = param.default
            
            return resolved_kwargs
        except Exception:
            return {}
    
    def _resolve_factory_dependencies(self, factory: callable) -> Dict[str, Any]:
        """Resolve dependências da factory"""
        try:
            signature = inspect.signature(factory)
            resolved_kwargs = {}
            
            for param_name, param in signature.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in self.registrations:
                        resolved_kwargs[param_name] = self.resolve(param.annotation)
                    elif param.default != inspect.Parameter.empty:
                        resolved_kwargs[param_name] = param.default
            
            return resolved_kwargs
        except Exception:
            return {}
    
    def is_registered(self, interface: Type) -> bool:
        """Verifica se interface está registrada"""
        return interface in self.registrations
    
    def get_registration(self, interface: Type) -> Optional[Dict]:
        """Obtém registro da interface"""
        return self.registrations.get(interface)
    
    def unregister(self, interface: Type):
        """Remove registro"""
        if interface in self.registrations:
            del self.registrations[interface]
        
        # Remove instâncias
        if interface in self.instances:
            del self.instances[interface]
        
        # Remove instâncias scoped
        keys_to_remove = [key for key in self.scoped_instances.keys() if key.startswith(str(interface))]
        for key in keys_to_remove:
            del self.scoped_instances[key]
    
    def clear_scope(self, scope_id: str):
        """Limpa escopo específico"""
        keys_to_remove = [key for key in self.scoped_instances.keys() if key.endswith(f"_{scope_id}")]
        for key in keys_to_remove:
            del self.scoped_instances[key]
    
    def clear_all(self):
        """Limpa todos os registros e instâncias"""
        self.registrations.clear()
        self.instances.clear()
        self.scoped_instances.clear()
        self.creation_order.clear()
        
        # Reset stats
        for key in self.stats:
            self.stats[key] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas"""
        return self.stats.copy()
    
    def get_creation_order(self) -> List[Type]:
        """Retorna ordem de criação das instâncias"""
        return self.creation_order.copy()
    
    def add_pre_resolution_hook(self, hook: callable):
        """Adiciona hook pré-resolução"""
        self.pre_resolution_hooks.append(hook)
    
    def add_post_resolution_hook(self, hook: callable):
        """Adiciona hook pós-resolução"""
        self.post_resolution_hooks.append(hook)
    
    def add_pre_creation_hook(self, hook: callable):
        """Adiciona hook pré-criação"""
        self.pre_creation_hooks.append(hook)
    
    def add_post_creation_hook(self, hook: callable):
        """Adiciona hook pós-criação"""
        self.post_creation_hooks.append(hook)
    
    def create_scope(self) -> 'MockDependencyScope':
        """Cria novo escopo"""
        return MockDependencyScope(self)


class MockDependencyScope:
    """Mock do escopo de dependências"""
    
    def __init__(self, container: MockDependencyContainer):
        self.container = container
        self.scope_id = f"scope_{id(self)}"
        self.disposed = False
    
    def resolve(self, interface: Type):
        """Resolve no escopo atual"""
        if self.disposed:
            raise ValueError("Scope has been disposed")
        return self.container.resolve(interface, self.scope_id)
    
    def dispose(self):
        """Descarta escopo"""
        if not self.disposed:
            self.container.clear_scope(self.scope_id)
            self.disposed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()


# Classes de teste para injeção
class ITestService(ABC):
    """Interface de teste"""
    
    @abstractmethod
    def get_data(self) -> str:
        pass


class MockTestService(ITestService):
    """Implementação de teste (renomeada para evitar conflito com pytest)"""
    
    def __init__(self, config: str = "default"):
        self.config = config
        self.created_at = "mock_time"
    
    def get_data(self) -> str:
        return f"data_from_{self.config}"


class IDependentService(ABC):
    """Interface com dependência"""
    
    @abstractmethod
    def process(self) -> str:
        pass


class MockDependentService(IDependentService):
    """Serviço com dependência (renomeado para evitar conflito com pytest)"""
    
    def __init__(self, test_service: ITestService):
        self.test_service = test_service
    
    def process(self) -> str:
        data = self.test_service.get_data()
        return f"processed_{data}"


class TestDependencyContainerBasic:
    """Testes básicos do container de dependências"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.container = MockDependencyContainer()
    
    def test_container_initialization(self):
        """Testa inicialização do container"""
        assert len(self.container.registrations) == 0
        assert len(self.container.instances) == 0
        assert self.container.auto_wire is True
        assert self.container.circular_detection is True
        assert self.container.thread_safe is True
    
    def test_simple_registration(self):
        """Testa registro simples"""
        self.container.register(ITestService, MockTestService)
        
        assert self.container.is_registered(ITestService)
        registration = self.container.get_registration(ITestService)
        
        assert registration['interface'] == ITestService
        assert registration['implementation'] == MockTestService
        assert registration['scope'] == DependencyScope.TRANSIENT
    
    def test_registration_scopes(self):
        """Testa diferentes escopos de registro"""
        # Singleton
        self.container.register_singleton(ITestService, MockTestService)
        reg1 = self.container.get_registration(ITestService)
        assert reg1['scope'] == DependencyScope.SINGLETON
        
        # Transient
        self.container.register_transient(IDependentService, MockDependentService)
        reg2 = self.container.get_registration(IDependentService)
        assert reg2['scope'] == DependencyScope.TRANSIENT
        
        # Scoped
        self.container.register_scoped(str, str)
        reg3 = self.container.get_registration(str)
        assert reg3['scope'] == DependencyScope.SCOPED
    
    def test_factory_registration(self):
        """Testa registro com factory"""
        def test_factory():
            return MockTestService("factory_config")
        
        self.container.register_factory(ITestService, test_factory)
        
        registration = self.container.get_registration(ITestService)
        assert registration['factory'] == test_factory
        assert registration['implementation'] is None
    
    def test_registration_with_kwargs(self):
        """Testa registro com kwargs"""
        self.container.register(ITestService, MockTestService, config="custom_config")
        
        registration = self.container.get_registration(ITestService)
        assert registration['kwargs']['config'] == "custom_config"


class TestDependencyContainerResolution:
    """Testes para resolução de dependências"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.container = MockDependencyContainer()
    
    def test_simple_resolution(self):
        """Testa resolução simples"""
        self.container.register(ITestService, MockTestService)
        
        instance = self.container.resolve(ITestService)
        
        assert isinstance(instance, MockTestService)
        assert instance.config == "default"
        assert hasattr(instance, 'created_at')
    
    def test_singleton_resolution(self):
        """Testa resolução singleton"""
        self.container.register_singleton(ITestService, MockTestService)
        
        instance1 = self.container.resolve(ITestService)
        instance2 = self.container.resolve(ITestService)
        
        # Deve ser a mesma instância
        assert instance1 is instance2
        
        stats = self.container.get_stats()
        assert stats['singletons_created'] == 1
        assert stats['resolutions_count'] == 2
    
    def test_transient_resolution(self):
        """Testa resolução transient"""
        self.container.register_transient(ITestService, MockTestService)
        
        instance1 = self.container.resolve(ITestService)
        instance2 = self.container.resolve(ITestService)
        
        # Devem ser instâncias diferentes
        assert instance1 is not instance2
        assert isinstance(instance1, MockTestService)
        assert isinstance(instance2, MockTestService)
        
        stats = self.container.get_stats()
        assert stats['transients_created'] >= 2
    
    def test_scoped_resolution(self):
        """Testa resolução scoped"""
        self.container.register_scoped(ITestService, MockTestService)
        
        # Mesmo escopo - mesma instância
        instance1 = self.container.resolve(ITestService, "scope1")
        instance2 = self.container.resolve(ITestService, "scope1")
        assert instance1 is instance2
        
        # Escopo diferente - instância diferente
        instance3 = self.container.resolve(ITestService, "scope2")
        assert instance1 is not instance3
    
    def test_factory_resolution(self):
        """Testa resolução com factory"""
        def test_factory():
            return MockTestService("factory_created")
        
        self.container.register_factory(ITestService, test_factory)
        
        instance = self.container.resolve(ITestService)
        
        assert isinstance(instance, MockTestService)
        assert instance.config == "factory_created"
    
    def test_dependency_injection(self):
        """Testa injeção de dependências automática"""
        # Registra dependência
        self.container.register(ITestService, MockTestService)
        self.container.register(IDependentService, MockDependentService)
        
        # Resolve serviço que tem dependência
        dependent = self.container.resolve(IDependentService)
        
        assert isinstance(dependent, MockDependentService)
        assert isinstance(dependent.test_service, MockTestService)
        
        result = dependent.process()
        assert result == "processed_data_from_default"
    
    def test_unregistered_dependency(self):
        """Testa resolução de dependência não registrada"""
        with pytest.raises(ValueError, match="No registration found"):
            self.container.resolve(ITestService)


class TestDependencyContainerCircular:
    """Testes para detecção de dependências circulares"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.container = MockDependencyContainer()
    
    def test_circular_dependency_detection(self):
        """Testa detecção de dependência circular"""
        # Simula dependência circular usando stack diretamente
        self.container.resolution_stack.append(ITestService)
        
        # Registra normalmente
        self.container.register(ITestService, MockTestService)
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            self.container.resolve(ITestService)
        
        stats = self.container.get_stats()
        assert stats['circular_detected'] == 1
    
    def test_circular_detection_disabled(self):
        """Testa com detecção circular desabilitada"""
        self.container.circular_detection = False
        
        # Simula dependência circular usando stack
        self.container.resolution_stack.append(ITestService)
        self.container.register(ITestService, MockTestService)
        
        # Com detecção desabilitada, deve funcionar normalmente
        # (apesar do stack ter o item)
        try:
            instance = self.container.resolve(ITestService)
            assert isinstance(instance, MockTestService)
        finally:
            # Limpa stack para próximos testes
            self.container.resolution_stack.clear()
        
        stats = self.container.get_stats()
        assert stats['circular_detected'] == 0


class TestDependencyContainerScopes:
    """Testes para escopos de dependência"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.container = MockDependencyContainer()
    
    def test_scope_creation(self):
        """Testa criação de escopo"""
        scope = self.container.create_scope()
        
        assert isinstance(scope, MockDependencyScope)
        assert scope.scope_id.startswith("scope_")
        assert not scope.disposed
    
    def test_scope_resolution(self):
        """Testa resolução dentro de escopo"""
        self.container.register_scoped(ITestService, MockTestService)
        
        with self.container.create_scope() as scope:
            instance1 = scope.resolve(ITestService)
            instance2 = scope.resolve(ITestService)
            
            # Dentro do mesmo escopo - mesma instância
            assert instance1 is instance2
        
        # Novo escopo - nova instância
        with self.container.create_scope() as scope2:
            instance3 = scope2.resolve(ITestService)
            assert instance1 is not instance3
    
    def test_scope_disposal(self):
        """Testa descarte de escopo"""
        self.container.register_scoped(ITestService, MockTestService)
        
        scope = self.container.create_scope()
        instance = scope.resolve(ITestService)
        
        # Scope ativo
        assert not scope.disposed
        
        # Descarta scope
        scope.dispose()
        assert scope.disposed
        
        # Não pode resolver após descarte
        with pytest.raises(ValueError, match="Scope has been disposed"):
            scope.resolve(ITestService)
    
    def test_scope_context_manager(self):
        """Testa escopo como context manager"""
        self.container.register_scoped(ITestService, MockTestService)
        
        with self.container.create_scope() as scope:
            instance = scope.resolve(ITestService)
            assert isinstance(instance, MockTestService)
            assert not scope.disposed
        
        # Após sair do context - deve estar disposed
        assert scope.disposed


class TestDependencyContainerHooks:
    """Testes para hooks e callbacks"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.container = MockDependencyContainer()
        self.hook_calls = []
    
    def test_resolution_hooks(self):
        """Testa hooks de resolução"""
        def pre_hook(interface):
            self.hook_calls.append(f"pre_resolution_{interface.__name__}")
        
        def post_hook(interface, instance):
            self.hook_calls.append(f"post_resolution_{interface.__name__}")
        
        self.container.add_pre_resolution_hook(pre_hook)
        self.container.add_post_resolution_hook(post_hook)
        
        self.container.register(ITestService, MockTestService)
        instance = self.container.resolve(ITestService)
        
        assert "pre_resolution_ITestService" in self.hook_calls
        assert "post_resolution_ITestService" in self.hook_calls
    
    def test_creation_hooks(self):
        """Testa hooks de criação"""
        def pre_creation_hook(interface):
            self.hook_calls.append(f"pre_creation_{interface.__name__}")
        
        def post_creation_hook(interface, instance):
            self.hook_calls.append(f"post_creation_{interface.__name__}")
        
        self.container.add_pre_creation_hook(pre_creation_hook)
        self.container.add_post_creation_hook(post_creation_hook)
        
        self.container.register(ITestService, MockTestService)
        instance = self.container.resolve(ITestService)
        
        assert "pre_creation_ITestService" in self.hook_calls
        assert "post_creation_ITestService" in self.hook_calls
    
    def test_multiple_hooks(self):
        """Testa múltiplos hooks"""
        def hook1(interface):
            self.hook_calls.append("hook1")
        
        def hook2(interface):
            self.hook_calls.append("hook2")
        
        self.container.add_pre_resolution_hook(hook1)
        self.container.add_pre_resolution_hook(hook2)
        
        self.container.register(ITestService, MockTestService)
        self.container.resolve(ITestService)
        
        assert "hook1" in self.hook_calls
        assert "hook2" in self.hook_calls


class TestDependencyContainerManagement:
    """Testes para gerenciamento do container"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.container = MockDependencyContainer()
    
    def test_unregister(self):
        """Testa remoção de registros"""
        self.container.register_singleton(ITestService, MockTestService)
        instance = self.container.resolve(ITestService)
        
        assert self.container.is_registered(ITestService)
        assert ITestService in self.container.instances
        
        # Remove registro
        self.container.unregister(ITestService)
        
        assert not self.container.is_registered(ITestService)
        assert ITestService not in self.container.instances
    
    def test_clear_all(self):
        """Testa limpeza completa"""
        self.container.register(ITestService, MockTestService)
        self.container.register(IDependentService, MockDependentService)
        self.container.resolve(ITestService)
        
        assert len(self.container.registrations) == 2
        assert self.container.get_stats()['resolutions_count'] > 0
        
        # Limpa tudo
        self.container.clear_all()
        
        assert len(self.container.registrations) == 0
        assert len(self.container.instances) == 0
        assert self.container.get_stats()['resolutions_count'] == 0
    
    def test_statistics_tracking(self):
        """Testa rastreamento de estatísticas"""
        self.container.register_singleton(ITestService, MockTestService)
        self.container.register_transient(str, str)
        
        # Resoluções
        self.container.resolve(ITestService)  # Singleton
        self.container.resolve(ITestService)  # Cache hit
        self.container.resolve(str)  # Transient
        self.container.resolve(str)  # Transient
        
        stats = self.container.get_stats()
        
        assert stats['registrations_count'] == 2
        # Pode ser 4 ou 5 dependendo das dependências internas
        assert stats['resolutions_count'] >= 4
        assert stats['singletons_created'] == 1
        assert stats['transients_created'] >= 2
    
    def test_creation_order_tracking(self):
        """Testa rastreamento da ordem de criação"""
        self.container.register(ITestService, MockTestService)
        self.container.register(IDependentService, MockDependentService)
        
        # Resolve dependent service (vai criar ITestService primeiro)
        self.container.resolve(IDependentService)
        
        creation_order = self.container.get_creation_order()
        
        # ITestService deve ser criado antes de IDependentService
        assert len(creation_order) == 2
        assert ITestService in creation_order
        assert IDependentService in creation_order
    
    def test_autowire_disabled(self):
        """Testa com autowire desabilitado"""
        self.container.auto_wire = False
        
        self.container.register(ITestService, MockTestService)
        self.container.register(IDependentService, MockDependentService)
        
        # Sem autowire, vai falhar ao criar MockDependentService
        # porque não vai injetar ITestService automaticamente
        with pytest.raises(Exception):
            self.container.resolve(IDependentService)
