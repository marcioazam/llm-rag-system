"""
FASE 3 - Redis Enterprise: Configuração Avançada para Produção
Sistema de configuração e otimização do Redis para ambientes enterprise
"""

import asyncio
import logging
import json
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import redis.asyncio as aioredis
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RedisClusterNode:
    """Configuração de nó do cluster Redis"""
    host: str
    port: int
    role: str  # "master", "slave", "sentinel"
    memory_mb: int
    max_connections: int
    enabled: bool = True


@dataclass
class RedisPerformanceProfile:
    """Perfil de performance do Redis"""
    memory_usage_mb: float
    connected_clients: int
    ops_per_second: float
    hit_rate: float
    evicted_keys: int
    expired_keys: int
    network_io_mb: float
    cpu_usage_percent: float
    replication_lag_ms: float


@dataclass
class RedisOptimizationRule:
    """Regra de otimização do Redis"""
    name: str
    condition: str
    config_changes: Dict[str, Any]
    priority: int
    description: str


class RedisEnterpriseManager:
    """Gerenciador avançado do Redis Enterprise"""
    
    def __init__(self,
                 redis_configs: Optional[Dict] = None,
                 cluster_nodes: Optional[List[RedisClusterNode]] = None,
                 auto_optimization: bool = True,
                 monitoring_interval_seconds: int = 30):
        
        self.redis_configs = redis_configs or {}
        self.cluster_nodes = cluster_nodes or []
        self.auto_optimization = auto_optimization
        self.monitoring_interval = monitoring_interval_seconds
        
        # Conexões Redis
        self.redis_pools = {}
        self.cluster_client = None
        self.sentinel_client = None
        
        # Monitoramento
        self.performance_history = []
        self.current_performance = None
        self.optimization_rules = []
        
        # Estado
        self.is_monitoring = False
        self.last_optimization = None
        
        # Configurações enterprise
        self._load_enterprise_configs()
        self._setup_optimization_rules()
        
        logger.info("RedisEnterpriseManager inicializado")
    
    def _load_enterprise_configs(self):
        """Carrega configurações enterprise padrão"""
        self.enterprise_configs = {
            # Configurações de Memória
            "memory": {
                "maxmemory-policy": "allkeys-lru",
                "maxmemory-samples": 10,
                "hash-max-ziplist-entries": 512,
                "hash-max-ziplist-value": 64,
                "list-max-ziplist-size": -2,
                "list-compress-depth": 0,
                "set-max-intset-entries": 512,
                "zset-max-ziplist-entries": 128,
                "zset-max-ziplist-value": 64
            },
            
            # Configurações de Persistência
            "persistence": {
                "save": "3600 1 300 100 60 10000",  # Smart save intervals
                "rdbcompression": "yes",
                "rdbchecksum": "yes", 
                "rdb-save-incremental-fsync": "yes",
                "aof-rewrite-incremental-fsync": "yes",
                "aof-use-rdb-preamble": "yes"
            },
            
            # Configurações de Rede
            "network": {
                "tcp-keepalive": 300,
                "tcp-backlog": 2048,
                "timeout": 0,
                "client-output-buffer-limit": "normal 0 0 0",
                "client-output-buffer-limit-replica": "256mb 64mb 60",
                "client-output-buffer-limit-pubsub": "32mb 8mb 60"
            },
            
            # Configurações de Performance
            "performance": {
                "hz": 10,
                "dynamic-hz": "yes",
                "lazyfree-lazy-eviction": "yes",
                "lazyfree-lazy-expire": "yes",
                "lazyfree-lazy-server-del": "yes",
                "replica-lazy-flush": "yes",
                "io-threads": 4,
                "io-threads-do-reads": "yes"
            },
            
            # Configurações de Cluster
            "cluster": {
                "cluster-enabled": "yes",
                "cluster-config-file": "nodes.conf",
                "cluster-node-timeout": 15000,
                "cluster-announce-ip": "",
                "cluster-announce-port": 0,
                "cluster-announce-bus-port": 0,
                "cluster-require-full-coverage": "no"
            },
            
            # Configurações de Segurança
            "security": {
                "protected-mode": "yes",
                "requirepass": "",  # Definido via env
                "masterauth": "",   # Definido via env
                "acl-log-max-len": 128,
                "rename-command-flushdb": "FLUSHDB_DISABLED",
                "rename-command-flushall": "FLUSHALL_DISABLED",
                "rename-command-debug": "DEBUG_DISABLED"
            }
        }
    
    def _setup_optimization_rules(self):
        """Configura regras de otimização automática"""
        self.optimization_rules = [
            RedisOptimizationRule(
                name="High Memory Usage",
                condition="memory_usage_mb > 0.8 * max_memory_mb",
                config_changes={
                    "maxmemory-policy": "allkeys-lru",
                    "maxmemory-samples": 5,
                    "lazyfree-lazy-eviction": "yes"
                },
                priority=1,
                description="Otimizar uso de memória quando próximo do limite"
            ),
            
            RedisOptimizationRule(
                name="High CPU Usage",
                condition="cpu_usage_percent > 80",
                config_changes={
                    "hz": 5,
                    "dynamic-hz": "no",
                    "io-threads": 2
                },
                priority=2,
                description="Reduzir uso de CPU em alta carga"
            ),
            
            RedisOptimizationRule(
                name="Low Hit Rate",
                condition="hit_rate < 0.7",
                config_changes={
                    "maxmemory-policy": "volatile-lru",
                    "maxmemory-samples": 10
                },
                priority=3,
                description="Melhorar hit rate com política mais agressiva"
            ),
            
            RedisOptimizationRule(
                name="High Network IO",
                condition="network_io_mb > 100",
                config_changes={
                    "client-output-buffer-limit": "normal 32mb 8mb 60",
                    "tcp-keepalive": 60
                },
                priority=4,
                description="Otimizar rede em alta transferência"
            ),
            
            RedisOptimizationRule(
                name="Many Connected Clients",
                condition="connected_clients > 1000",
                config_changes={
                    "timeout": 300,
                    "tcp-keepalive": 30,
                    "maxclients": 2000
                },
                priority=5,
                description="Otimizar para muitas conexões simultâneas"
            )
        ]
    
    async def initialize_cluster(self) -> bool:
        """Inicializa cluster Redis"""
        try:
            logger.info("🚀 Inicializando cluster Redis Enterprise")
            
            # Configurar nós do cluster
            if not self.cluster_nodes:
                await self._setup_default_cluster()
            
            # Criar pools de conexão
            await self._create_connection_pools()
            
            # Aplicar configurações enterprise
            await self._apply_enterprise_configs()
            
            # Verificar saúde do cluster
            cluster_health = await self._check_cluster_health()
            
            if cluster_health["status"] == "healthy":
                logger.info("✅ Cluster Redis inicializado com sucesso")
                
                # Iniciar monitoramento se habilitado
                if self.auto_optimization:
                    asyncio.create_task(self._start_monitoring())
                
                return True
            else:
                logger.error(f"❌ Cluster Redis com problemas: {cluster_health}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao inicializar cluster Redis: {e}")
            return False
    
    async def _setup_default_cluster(self):
        """Configura cluster padrão baseado nas variáveis de ambiente"""
        # Configuração baseada em ambiente
        redis_hosts = os.getenv('REDIS_CLUSTER_HOSTS', 'localhost:7000,localhost:7001,localhost:7002').split(',')
        
        for i, host_port in enumerate(redis_hosts):
            host, port = host_port.split(':')
            
            node = RedisClusterNode(
                host=host.strip(),
                port=int(port.strip()),
                role="master" if i < len(redis_hosts) // 2 else "slave",
                memory_mb=int(os.getenv('REDIS_NODE_MEMORY_MB', '2048')),
                max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', '1000'))
            )
            
            self.cluster_nodes.append(node)
        
        logger.info(f"Configurado cluster padrão com {len(self.cluster_nodes)} nós")
    
    async def _create_connection_pools(self):
        """Cria pools de conexão para os nós"""
        try:
            for node in self.cluster_nodes:
                if not node.enabled:
                    continue
                
                pool_key = f"{node.host}:{node.port}"
                
                # Configurações de conexão otimizadas
                pool = aioredis.ConnectionPool(
                    host=node.host,
                    port=node.port,
                    password=os.getenv('REDIS_PASSWORD'),
                    max_connections=node.max_connections,
                    retry_on_timeout=True,
                    retry_on_error=[ConnectionError, TimeoutError],
                    health_check_interval=30,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    socket_keepalive_options={}
                )
                
                self.redis_pools[pool_key] = aioredis.Redis(connection_pool=pool)
                
                # Testar conexão
                await self.redis_pools[pool_key].ping()
                
                logger.debug(f"✅ Pool criado para {pool_key}")
            
            # Configurar cliente de cluster se há múltiplos nós
            if len(self.cluster_nodes) > 1:
                startup_nodes = [
                    {"host": node.host, "port": node.port}
                    for node in self.cluster_nodes
                    if node.enabled and node.role == "master"
                ]
                
                self.cluster_client = aioredis.RedisCluster(
                    startup_nodes=startup_nodes,
                    password=os.getenv('REDIS_PASSWORD'),
                    decode_responses=True,
                    skip_full_coverage_check=True,
                    max_connections_per_node=100
                )
                
                await self.cluster_client.ping()
                logger.info("✅ Cliente de cluster configurado")
            
        except Exception as e:
            logger.error(f"Erro ao criar pools de conexão: {e}")
            raise
    
    async def _apply_enterprise_configs(self):
        """Aplica configurações enterprise em todos os nós"""
        try:
            for pool_key, redis_client in self.redis_pools.items():
                logger.info(f"Aplicando configurações enterprise em {pool_key}")
                
                # Aplicar cada categoria de configuração
                for category, configs in self.enterprise_configs.items():
                    for config_key, config_value in configs.items():
                        try:
                            # Configurações que precisam de tratamento especial
                            if config_key.startswith("rename-command"):
                                command, new_name = config_key.split('-', 2)[2], config_value
                                await redis_client.config_set(f"rename-command {command}", new_name)
                            
                            elif config_key == "save":
                                # Configurar save points
                                await redis_client.config_set("save", config_value)
                            
                            elif config_key.startswith("client-output-buffer-limit"):
                                buffer_type = config_key.split('-')[-1]
                                await redis_client.config_set(f"client-output-buffer-limit {buffer_type}", config_value)
                            
                            else:
                                # Configuração padrão
                                await redis_client.config_set(config_key, config_value)
                            
                            logger.debug(f"✅ {config_key} = {config_value}")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Não foi possível aplicar {config_key}: {e}")
                
                # Rewrite AOF se habilitado
                try:
                    await redis_client.bgrewriteaof()
                    logger.debug(f"✅ AOF rewrite iniciado em {pool_key}")
                except:
                    pass  # AOF pode não estar habilitado
            
            logger.info("🔧 Configurações enterprise aplicadas em todos os nós")
            
        except Exception as e:
            logger.error(f"Erro ao aplicar configurações enterprise: {e}")
    
    async def _check_cluster_health(self) -> Dict[str, Any]:
        """Verifica saúde do cluster"""
        health_report = {
            "status": "healthy",
            "nodes": {},
            "cluster_info": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Verificar cada nó
            for node in self.cluster_nodes:
                if not node.enabled:
                    continue
                
                pool_key = f"{node.host}:{node.port}"
                node_health = await self._check_node_health(pool_key)
                health_report["nodes"][pool_key] = node_health
                
                if node_health["status"] != "healthy":
                    health_report["status"] = "degraded"
                    if node_health["status"] == "failed":
                        health_report["errors"].append(f"Nó {pool_key} falhou")
                    else:
                        health_report["warnings"].append(f"Nó {pool_key} com problemas")
            
            # Verificar cluster se disponível
            if self.cluster_client:
                try:
                    cluster_info = await self.cluster_client.cluster_info()
                    health_report["cluster_info"] = cluster_info
                    
                    if cluster_info.get("cluster_state") != "ok":
                        health_report["status"] = "degraded"
                        health_report["warnings"].append("Estado do cluster não está OK")
                        
                except Exception as e:
                    health_report["errors"].append(f"Erro ao verificar cluster: {e}")
                    health_report["status"] = "degraded"
            
            return health_report
            
        except Exception as e:
            logger.error(f"Erro na verificação de saúde: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "nodes": {},
                "cluster_info": {},
                "warnings": [],
                "errors": [str(e)]
            }
    
    async def _check_node_health(self, pool_key: str) -> Dict[str, Any]:
        """Verifica saúde de um nó específico"""
        try:
            redis_client = self.redis_pools.get(pool_key)
            if not redis_client:
                return {"status": "failed", "error": "Cliente não encontrado"}
            
            # Teste básico de conectividade
            ping_result = await redis_client.ping()
            if not ping_result:
                return {"status": "failed", "error": "Ping falhou"}
            
            # Obter informações do servidor
            info = await redis_client.info()
            
            # Verificar métricas críticas
            node_health = {
                "status": "healthy",
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                "hit_rate": self._calculate_hit_rate(info),
                "warnings": []
            }
            
            # Verificar condições de warning
            if node_health["connected_clients"] > 800:
                node_health["warnings"].append("Alto número de clientes conectados")
            
            if node_health["used_memory_mb"] > 1500:  # 1.5GB
                node_health["warnings"].append("Alto uso de memória")
            
            if node_health["hit_rate"] < 0.8:
                node_health["warnings"].append("Hit rate baixo")
            
            if node_health["warnings"]:
                node_health["status"] = "warning"
            
            return node_health
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calcula hit rate do Redis"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return hits / max(total, 1)
    
    async def _start_monitoring(self):
        """Inicia monitoramento contínuo"""
        self.is_monitoring = True
        logger.info("📊 Iniciando monitoramento Redis Enterprise")
        
        while self.is_monitoring:
            try:
                # Coletar métricas de performance
                performance = await self._collect_performance_metrics()
                
                if performance:
                    self.current_performance = performance
                    self.performance_history.append(performance)
                    
                    # Manter apenas últimas 1000 medições
                    if len(self.performance_history) > 1000:
                        self.performance_history.pop(0)
                    
                    # Verificar se otimização é necessária
                    if await self._should_optimize(performance):
                        await self._apply_automatic_optimization(performance)
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                await asyncio.sleep(60)  # Retry em 1 minuto
    
    async def _collect_performance_metrics(self) -> Optional[RedisPerformanceProfile]:
        """Coleta métricas de performance de todos os nós"""
        try:
            total_memory = 0
            total_clients = 0
            total_ops = 0
            total_hit_rate = 0
            total_evicted = 0
            total_expired = 0
            total_network = 0
            total_cpu = 0
            max_replication_lag = 0
            
            node_count = 0
            
            for pool_key, redis_client in self.redis_pools.items():
                try:
                    info = await redis_client.info()
                    
                    total_memory += info.get("used_memory", 0)
                    total_clients += info.get("connected_clients", 0)
                    total_ops += info.get("instantaneous_ops_per_sec", 0)
                    total_hit_rate += self._calculate_hit_rate(info)
                    total_evicted += info.get("evicted_keys", 0)
                    total_expired += info.get("expired_keys", 0)
                    total_network += info.get("instantaneous_input_kbps", 0) + info.get("instantaneous_output_kbps", 0)
                    total_cpu += info.get("used_cpu_sys", 0) + info.get("used_cpu_user", 0)
                    
                    # Replication lag (se for slave)
                    if info.get("role") == "slave":
                        lag = info.get("master_last_io_seconds_ago", 0)
                        max_replication_lag = max(max_replication_lag, lag)
                    
                    node_count += 1
                    
                except Exception as e:
                    logger.warning(f"Erro ao coletar métricas de {pool_key}: {e}")
            
            if node_count == 0:
                return None
            
            # Calcular médias
            profile = RedisPerformanceProfile(
                memory_usage_mb=total_memory / 1024 / 1024,
                connected_clients=total_clients,
                ops_per_second=total_ops,
                hit_rate=total_hit_rate / node_count,
                evicted_keys=total_evicted,
                expired_keys=total_expired,
                network_io_mb=total_network / 1024,  # kbps to MB
                cpu_usage_percent=min(100, (total_cpu / node_count) * 100),
                replication_lag_ms=max_replication_lag * 1000
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Erro ao coletar métricas de performance: {e}")
            return None
    
    async def _should_optimize(self, performance: RedisPerformanceProfile) -> bool:
        """Verifica se otimização automática é necessária"""
        if not self.auto_optimization:
            return False
        
        # Cooldown de otimização (mínimo 5 minutos)
        if self.last_optimization:
            time_since_last = datetime.now() - self.last_optimization
            if time_since_last < timedelta(minutes=5):
                return False
        
        # Critérios para otimização
        return (
            performance.memory_usage_mb > 1500 or      # > 1.5GB
            performance.cpu_usage_percent > 80 or      # > 80% CPU
            performance.hit_rate < 0.7 or             # < 70% hit rate
            performance.connected_clients > 1000 or   # > 1000 clientes
            performance.replication_lag_ms > 5000     # > 5s lag
        )
    
    async def _apply_automatic_optimization(self, performance: RedisPerformanceProfile):
        """Aplica otimizações automáticas baseadas nas métricas"""
        try:
            logger.info("🔧 Aplicando otimizações automáticas")
            
            applied_optimizations = []
            
            # Avaliar cada regra de otimização
            for rule in sorted(self.optimization_rules, key=lambda r: r.priority):
                
                # Construir contexto para avaliação
                context = {
                    "memory_usage_mb": performance.memory_usage_mb,
                    "max_memory_mb": 2048,  # Configuração padrão
                    "cpu_usage_percent": performance.cpu_usage_percent,
                    "hit_rate": performance.hit_rate,
                    "connected_clients": performance.connected_clients,
                    "network_io_mb": performance.network_io_mb,
                    "replication_lag_ms": performance.replication_lag_ms
                }
                
                try:
                    # Avaliar condição da regra
                    if eval(rule.condition, {"__builtins__": {}}, context):
                        
                        # Aplicar configurações da regra
                        success = await self._apply_config_changes(rule.config_changes)
                        
                        if success:
                            applied_optimizations.append(rule.name)
                            logger.info(f"✅ Aplicada otimização: {rule.name}")
                        else:
                            logger.warning(f"❌ Falha ao aplicar: {rule.name}")
                
                except Exception as e:
                    logger.warning(f"Erro ao avaliar regra {rule.name}: {e}")
            
            if applied_optimizations:
                self.last_optimization = datetime.now()
                logger.info(f"🎯 Otimizações aplicadas: {', '.join(applied_optimizations)}")
            
        except Exception as e:
            logger.error(f"Erro na otimização automática: {e}")
    
    async def _apply_config_changes(self, config_changes: Dict[str, Any]) -> bool:
        """Aplica mudanças de configuração em todos os nós"""
        try:
            success_count = 0
            
            for pool_key, redis_client in self.redis_pools.items():
                try:
                    for config_key, config_value in config_changes.items():
                        await redis_client.config_set(config_key, config_value)
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Erro ao aplicar config em {pool_key}: {e}")
            
            # Considerar sucesso se aplicado na maioria dos nós
            return success_count >= len(self.redis_pools) // 2
            
        except Exception as e:
            logger.error(f"Erro ao aplicar mudanças de configuração: {e}")
            return False
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Retorna status completo do cluster"""
        try:
            health = await self._check_cluster_health()
            
            status = {
                "cluster_health": health,
                "current_performance": asdict(self.current_performance) if self.current_performance else {},
                "node_count": len([n for n in self.cluster_nodes if n.enabled]),
                "total_memory_mb": sum(n.memory_mb for n in self.cluster_nodes if n.enabled),
                "monitoring_active": self.is_monitoring,
                "auto_optimization": self.auto_optimization,
                "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
                "performance_trend": self._analyze_performance_trend(),
                "configuration_summary": self._get_config_summary()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Erro ao obter status do cluster: {e}")
            return {"error": str(e)}
    
    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """Analisa tendência de performance"""
        if len(self.performance_history) < 10:
            return {"status": "insufficient_data"}
        
        recent = self.performance_history[-10:]
        older = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else []
        
        if not older:
            return {"status": "insufficient_data"}
        
        # Calcular médias
        recent_hit_rate = sum(p.hit_rate for p in recent) / len(recent)
        older_hit_rate = sum(p.hit_rate for p in older) / len(older)
        
        recent_ops = sum(p.ops_per_second for p in recent) / len(recent)
        older_ops = sum(p.ops_per_second for p in older) / len(older)
        
        recent_memory = sum(p.memory_usage_mb for p in recent) / len(recent)
        older_memory = sum(p.memory_usage_mb for p in older) / len(older)
        
        return {
            "status": "analyzed",
            "hit_rate_trend": "up" if recent_hit_rate > older_hit_rate else "down",
            "hit_rate_change": recent_hit_rate - older_hit_rate,
            "ops_trend": "up" if recent_ops > older_ops else "down", 
            "ops_change": recent_ops - older_ops,
            "memory_trend": "up" if recent_memory > older_memory else "down",
            "memory_change_mb": recent_memory - older_memory
        }
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Retorna resumo das configurações atuais"""
        return {
            "enterprise_configs_applied": True,
            "cluster_enabled": len(self.cluster_nodes) > 1,
            "auto_optimization_enabled": self.auto_optimization,
            "monitoring_interval_seconds": self.monitoring_interval,
            "optimization_rules_count": len(self.optimization_rules)
        }
    
    async def export_cluster_config(self, output_file: str = "redis_cluster_config.yaml") -> str:
        """Exporta configuração do cluster para arquivo"""
        try:
            config_export = {
                "cluster_nodes": [asdict(node) for node in self.cluster_nodes],
                "enterprise_configs": self.enterprise_configs,
                "optimization_rules": [
                    {
                        "name": rule.name,
                        "condition": rule.condition,
                        "config_changes": rule.config_changes,
                        "priority": rule.priority,
                        "description": rule.description
                    }
                    for rule in self.optimization_rules
                ],
                "monitoring_settings": {
                    "auto_optimization": self.auto_optimization,
                    "monitoring_interval_seconds": self.monitoring_interval
                },
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(output_file, 'w') as f:
                yaml.dump(config_export, f, default_flow_style=False)
            
            logger.info(f"📄 Configuração exportada para: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Erro ao exportar configuração: {e}")
            return ""
    
    async def cleanup(self):
        """Limpeza de recursos"""
        try:
            # Parar monitoramento
            self.is_monitoring = False
            
            # Fechar conexões
            for pool_key, redis_client in self.redis_pools.items():
                try:
                    await redis_client.close()
                    logger.debug(f"✅ Pool {pool_key} fechado")
                except:
                    pass
            
            if self.cluster_client:
                try:
                    await self.cluster_client.close()
                    logger.debug("✅ Cliente de cluster fechado")
                except:
                    pass
            
            if self.sentinel_client:
                try:
                    await self.sentinel_client.close()
                    logger.debug("✅ Cliente sentinel fechado")
                except:
                    pass
            
            self.redis_pools.clear()
            
            logger.info("🧹 Cleanup Redis Enterprise concluído")
            
        except Exception as e:
            logger.warning(f"Erro no cleanup Redis: {e}")


# Utilitários
def create_production_cluster_config() -> List[RedisClusterNode]:
    """Cria configuração de cluster para produção"""
    return [
        RedisClusterNode("redis-master-1", 7000, "master", 4096, 2000),
        RedisClusterNode("redis-master-2", 7001, "master", 4096, 2000),
        RedisClusterNode("redis-master-3", 7002, "master", 4096, 2000),
        RedisClusterNode("redis-slave-1", 7003, "slave", 4096, 1000),
        RedisClusterNode("redis-slave-2", 7004, "slave", 4096, 1000),
        RedisClusterNode("redis-slave-3", 7005, "slave", 4096, 1000),
    ]


def get_redis_enterprise_best_practices() -> Dict[str, str]:
    """Retorna melhores práticas para Redis Enterprise"""
    return {
        "memory_management": "Use maxmemory-policy allkeys-lru para caches, volatile-lru para dados com TTL",
        "persistence": "Configure AOF + RDB para máxima durabilidade em produção",
        "clustering": "Use pelo menos 3 masters para alta disponibilidade",
        "monitoring": "Monitore memory usage, hit rate, connected clients e replication lag",
        "security": "Sempre configure requirepass e protected-mode em produção",
        "network": "Configure tcp-keepalive e ajuste client-output-buffer-limit",
        "performance": "Use io-threads e lazyfree para melhor performance",
        "backup": "Configure save intervals baseado na criticidade dos dados"
    } 