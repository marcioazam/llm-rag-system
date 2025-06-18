"""
Performance Tuner para Sistema RAG
Implementa otimizações específicas para Qdrant, Neo4j e APIs
Baseado em:
- https://qdrant.tech/articles/vector-search-resource-optimization/
- https://neo4j.com/docs/operations-manual/current/performance/
- https://www.arsturn.com/blog/optimizing-slow-performance-in-ollama
"""

import os
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import yaml
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import GraphDatabase
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    """Recursos do sistema disponíveis"""
    cpu_count: int
    memory_gb: float
    available_memory_gb: float
    disk_type: str  # 'ssd' ou 'hdd'
    gpu_available: bool
    gpu_memory_gb: float = 0.0

@dataclass
class OptimizationProfile:
    """Perfil de otimização para diferentes cenários"""
    name: str
    description: str
    qdrant_config: Dict[str, Any]
    neo4j_config: Dict[str, Any]
    api_config: Dict[str, Any]

class QdrantOptimizer:
    """
    Otimizador específico para Qdrant
    Baseado em: https://qdrant.tech/articles/vector-search-resource-optimization/
    """
    
    def __init__(self, client: QdrantClient):
        self.client = client
        self.optimization_applied = False
    
    def analyze_current_config(self, collection_name: str) -> Dict[str, Any]:
        """Analisa configuração atual da collection"""
        try:
            info = self.client.get_collection(collection_name)
            
            return {
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'segments_count': info.segments_count,
                'config': info.config.dict(),
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Erro ao analisar collection: {e}")
            return {}
    
    async def apply_optimizations(
        self, 
        collection_name: str,
        resources: SystemResources
    ) -> Dict[str, Any]:
        """
        Aplica otimizações baseadas nos recursos disponíveis
        """
        logger.info(f"Aplicando otimizações Qdrant para collection: {collection_name}")
        
        optimizations = {}
        
        # 1. Scalar Quantization (75% memory reduction)
        if resources.memory_gb < 16:
            optimizations['quantization'] = await self._apply_scalar_quantization(collection_name)
        
        # 2. Segment optimization (equal to CPU cores)
        optimizations['segments'] = await self._optimize_segments(collection_name, resources.cpu_count)
        
        # 3. HNSW parameters optimization
        optimizations['hnsw'] = await self._optimize_hnsw_parameters(collection_name, resources)
        
        # 4. On-disk storage for large datasets
        if resources.available_memory_gb < 4:
            optimizations['on_disk'] = await self._enable_on_disk_storage(collection_name)
        
        self.optimization_applied = True
        return optimizations
    
    async def _apply_scalar_quantization(self, collection_name: str) -> Dict[str, Any]:
        """
        Aplica Scalar Quantization para reduzir uso de memória em 75%
        """
        try:
            # Configurar quantização
            self.client.update_collection(
                collection_name=collection_name,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )
            )
            
            logger.info("Scalar quantization aplicada com sucesso")
            return {
                'status': 'applied',
                'memory_reduction': '75%',
                'type': 'INT8',
                'quantile': 0.99
            }
            
        except Exception as e:
            logger.error(f"Erro ao aplicar quantization: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _optimize_segments(self, collection_name: str, cpu_count: int) -> Dict[str, Any]:
        """
        Otimiza número de segmentos baseado em CPU cores
        """
        try:
            # Configurar segmentos igual ao número de CPUs
            optimal_segments = min(cpu_count, 8)  # Cap em 8 segmentos
            
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    default_segment_number=optimal_segments,
                    max_segment_size=100000,  # 100k vectors por segmento
                    memmap_threshold=50000,
                    indexing_threshold=20000,
                    flush_interval_sec=5
                )
            )
            
            logger.info(f"Segmentos otimizados: {optimal_segments}")
            return {
                'status': 'optimized',
                'segments': optimal_segments,
                'based_on': f'{cpu_count} CPU cores'
            }
            
        except Exception as e:
            logger.error(f"Erro ao otimizar segmentos: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _optimize_hnsw_parameters(
        self, 
        collection_name: str,
        resources: SystemResources
    ) -> Dict[str, Any]:
        """
        Otimiza parâmetros HNSW baseado em recursos
        """
        try:
            # Determinar parâmetros baseado em recursos
            if resources.memory_gb >= 32:
                # High performance
                m = 32
                ef_construct = 256
                ef = 128
            elif resources.memory_gb >= 16:
                # Balanced
                m = 16
                ef_construct = 128
                ef = 64
            else:
                # Memory constrained
                m = 8
                ef_construct = 64
                ef = 32
            
            # Aplicar configuração
            self.client.update_collection(
                collection_name=collection_name,
                hnsw_config=models.HnswConfigDiff(
                    m=m,
                    ef_construct=ef_construct,
                    ef=ef,
                    full_scan_threshold=10000,
                    max_indexing_threads=min(resources.cpu_count, 8),
                    on_disk=resources.memory_gb < 8
                )
            )
            
            logger.info(f"HNSW otimizado: m={m}, ef_construct={ef_construct}")
            return {
                'status': 'optimized',
                'm': m,
                'ef_construct': ef_construct,
                'ef': ef,
                'profile': 'high' if m == 32 else 'balanced' if m == 16 else 'memory_constrained'
            }
            
        except Exception as e:
            logger.error(f"Erro ao otimizar HNSW: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _enable_on_disk_storage(self, collection_name: str) -> Dict[str, Any]:
        """
        Habilita storage em disco para economizar RAM
        """
        try:
            self.client.update_collection(
                collection_name=collection_name,
                vectors_config={
                    "": models.VectorParamsDiff(
                        on_disk=True
                    )
                }
            )
            
            logger.info("On-disk storage habilitado")
            return {
                'status': 'enabled',
                'reason': 'low_memory',
                'expected_impact': 'slower_search_but_lower_memory'
            }
            
        except Exception as e:
            logger.error(f"Erro ao habilitar on-disk storage: {e}")
            return {'status': 'failed', 'error': str(e)}

class Neo4jOptimizer:
    """
    Otimizador específico para Neo4j
    Baseado em: https://neo4j.com/docs/operations-manual/current/performance/
    """
    
    def __init__(self, uri: str, auth: tuple):
        self.uri = uri
        self.auth = auth
        self.driver = None
    
    async def connect(self):
        """Conecta ao Neo4j"""
        self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
    
    async def apply_optimizations(self, resources: SystemResources) -> Dict[str, Any]:
        """
        Aplica otimizações baseadas em recursos
        """
        logger.info("Aplicando otimizações Neo4j")
        
        optimizations = {}
        
        # 1. Memory settings (Heap e Page Cache)
        optimizations['memory'] = self._calculate_memory_settings(resources)
        
        # 2. Index optimization
        optimizations['indexes'] = await self._optimize_indexes()
        
        # 3. Query optimization
        optimizations['queries'] = await self._analyze_slow_queries()
        
        # 4. Connection pool
        optimizations['connection_pool'] = self._optimize_connection_pool(resources)
        
        return optimizations
    
    def _calculate_memory_settings(self, resources: SystemResources) -> Dict[str, Any]:
        """
        Calcula configurações ótimas de memória
        Regra: Heap = 8GB (ou 25% da RAM), Page Cache = 1.5x database size
        """
        total_memory_gb = resources.memory_gb
        
        # Heap size (máximo 31GB para compressed pointers)
        heap_gb = min(
            max(8, total_memory_gb * 0.25),  # 25% da RAM ou 8GB
            31  # Máximo para compressed OOPs
        )
        
        # Page cache (restante da memória menos overhead do OS)
        os_overhead_gb = 2  # Reserve 2GB para OS
        page_cache_gb = total_memory_gb - heap_gb - os_overhead_gb
        
        # Garantir mínimos
        page_cache_gb = max(page_cache_gb, 4)
        
        config = {
            'dbms.memory.heap.initial_size': f'{int(heap_gb)}g',
            'dbms.memory.heap.max_size': f'{int(heap_gb)}g',
            'dbms.memory.pagecache.size': f'{int(page_cache_gb)}g',
            'dbms.memory.off_heap.max_size': '2g'
        }
        
        logger.info(f"Configurações de memória Neo4j: Heap={heap_gb}GB, PageCache={page_cache_gb}GB")
        
        return {
            'heap_gb': heap_gb,
            'page_cache_gb': page_cache_gb,
            'config': config,
            'recommendation': 'Add these settings to neo4j.conf'
        }
    
    async def _optimize_indexes(self) -> Dict[str, Any]:
        """
        Cria índices compostos para queries multi-property
        """
        index_commands = [
            # Índices para code entities
            "CREATE INDEX code_entity_type_name IF NOT EXISTS FOR (n:CodeEntity) ON (n.type, n.name)",
            "CREATE INDEX code_entity_file_path IF NOT EXISTS FOR (n:CodeEntity) ON (n.file_path)",
            
            # Índices para relationships
            "CREATE INDEX rel_type_timestamp IF NOT EXISTS FOR ()-[r:CALLS]-() ON (r.timestamp)",
            "CREATE INDEX rel_imports_weight IF NOT EXISTS FOR ()-[r:IMPORTS]-() ON (r.weight)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX entity_content_search IF NOT EXISTS FOR (n:CodeEntity) ON EACH [n.content, n.name]"
        ]
        
        created_indexes = []
        
        with self.driver.session() as session:
            for command in index_commands:
                try:
                    session.run(command)
                    created_indexes.append(command.split()[2])  # Nome do índice
                except Exception as e:
                    logger.warning(f"Erro ao criar índice: {e}")
        
        return {
            'created': created_indexes,
            'total': len(created_indexes),
            'recommendation': 'Monitor index usage with SHOW INDEXES'
        }
    
    async def _analyze_slow_queries(self) -> Dict[str, Any]:
        """
        Analisa queries lentas e sugere otimizações
        """
        # Query para encontrar queries lentas
        slow_query_analysis = """
        CALL dbms.listQueries() 
        YIELD query, elapsedTimeMillis, allocatedBytes
        WHERE elapsedTimeMillis > 1000
        RETURN query, elapsedTimeMillis, allocatedBytes
        ORDER BY elapsedTimeMillis DESC
        LIMIT 10
        """
        
        slow_queries = []
        
        try:
            with self.driver.session() as session:
                result = session.run(slow_query_analysis)
                for record in result:
                    slow_queries.append({
                        'query': record['query'][:100] + '...',
                        'elapsed_ms': record['elapsedTimeMillis'],
                        'memory_mb': record['allocatedBytes'] / (1024 * 1024)
                    })
        except:
            # Query monitoring pode não estar habilitado
            pass
        
        return {
            'slow_queries': slow_queries,
            'recommendations': [
                'Use EXPLAIN to analyze query plans',
                'Add indexes for frequently filtered properties',
                'Limit result sets with LIMIT clause',
                'Use parameters instead of literals in queries'
            ]
        }
    
    def _optimize_connection_pool(self, resources: SystemResources) -> Dict[str, Any]:
        """
        Otimiza pool de conexões baseado em recursos
        """
        # Calcular tamanho ótimo do pool
        pool_size = min(
            resources.cpu_count * 2,  # 2x CPU cores
            50  # Máximo razoável
        )
        
        config = {
            'max_connection_pool_size': pool_size,
            'connection_acquisition_timeout': 60,
            'max_connection_lifetime': 3600,  # 1 hora
            'connection_timeout': 30
        }
        
        return {
            'pool_size': pool_size,
            'config': config,
            'based_on': f'{resources.cpu_count} CPU cores'
        }
    
    async def close(self):
        """Fecha conexão"""
        if self.driver:
            self.driver.close()

class APIOptimizer:
    """
    Otimizador para chamadas de API (OpenAI, Anthropic, etc.)
    Foca em concorrência, batching e caching
    """
    
    def __init__(self):
        self.optimizations = {
            'openai': {
                'max_concurrent_requests': 50,
                'batch_size': 20,
                'retry_strategy': 'exponential_backoff',
                'timeout': 30
            },
            'anthropic': {
                'max_concurrent_requests': 30,
                'batch_size': 10,
                'retry_strategy': 'exponential_backoff',
                'timeout': 60
            },
            'local_llm': {
                'num_threads': 8,
                'batch_size': 1,
                'model_cache_size': 4,
                'context_size': 4096
            }
        }
    
    def optimize_for_provider(
        self, 
        provider: str,
        resources: SystemResources
    ) -> Dict[str, Any]:
        """
        Retorna configurações otimizadas para provider específico
        """
        base_config = self.optimizations.get(provider, {})
        
        # Ajustar baseado em recursos
        if provider in ['openai', 'anthropic']:
            # APIs externas - otimizar concorrência
            if resources.cpu_count >= 8:
                base_config['max_concurrent_requests'] = min(
                    base_config.get('max_concurrent_requests', 20) * 1.5,
                    100
                )
            
            # Batch size baseado em memória
            if resources.memory_gb >= 16:
                base_config['batch_size'] = min(
                    base_config.get('batch_size', 10) * 2,
                    50
                )
        
        elif provider == 'local_llm':
            # LLMs locais - otimizar threads e cache
            base_config['num_threads'] = min(resources.cpu_count, 16)
            
            # Model cache baseado em memória
            if resources.memory_gb >= 32:
                base_config['model_cache_size'] = 8
            elif resources.memory_gb >= 16:
                base_config['model_cache_size'] = 4
            else:
                base_config['model_cache_size'] = 2
        
        return base_config
    
    def get_batching_strategy(self, provider: str) -> Dict[str, Any]:
        """
        Retorna estratégia de batching para provider
        """
        strategies = {
            'openai': {
                'embeddings': {
                    'max_batch_size': 100,
                    'optimal_batch_size': 50,
                    'wait_time_ms': 100
                },
                'completions': {
                    'max_batch_size': 20,
                    'optimal_batch_size': 10,
                    'wait_time_ms': 200
                }
            },
            'anthropic': {
                'completions': {
                    'max_batch_size': 10,
                    'optimal_batch_size': 5,
                    'wait_time_ms': 300
                }
            }
        }
        
        return strategies.get(provider, {})

class PerformanceTuner:
    """
    Sistema principal de performance tuning
    Coordena otimizações entre Qdrant, Neo4j e APIs
    """
    
    def __init__(self, config_path: str = "config/performance_config.yaml"):
        self.config = self._load_config(config_path)
        self.system_resources = self._detect_system_resources()
        
        # Otimizadores
        self.qdrant_optimizer = None
        self.neo4j_optimizer = None
        self.api_optimizer = APIOptimizer()
        
        # Perfis de otimização
        self.profiles = self._create_optimization_profiles()
        
        logger.info(f"Performance Tuner inicializado. Recursos: {self.system_resources}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Configuração padrão"""
        return {
            'auto_tune': True,
            'profile': 'balanced',
            'qdrant': {
                'enabled': True,
                'collection': 'hybrid_rag_collection'
            },
            'neo4j': {
                'enabled': True,
                'uri': 'bolt://localhost:7687',
                'auth': ('neo4j', 'password')
            },
            'api': {
                'enabled': True,
                'providers': ['openai', 'anthropic']
            }
        }
    
    def _detect_system_resources(self) -> SystemResources:
        """Detecta recursos do sistema"""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        # Detectar tipo de disco (simplificado)
        disk_type = 'ssd'  # Assumir SSD por padrão
        
        # Detectar GPU (simplificado)
        gpu_available = False
        gpu_memory_gb = 0.0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        
        return SystemResources(
            cpu_count=cpu_count,
            memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            disk_type=disk_type,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb
        )
    
    def _create_optimization_profiles(self) -> Dict[str, OptimizationProfile]:
        """Cria perfis de otimização predefinidos"""
        return {
            'development': OptimizationProfile(
                name='development',
                description='Otimizado para desenvolvimento local',
                qdrant_config={
                    'segments': 2,
                    'hnsw_m': 8,
                    'hnsw_ef_construct': 64,
                    'quantization': False
                },
                neo4j_config={
                    'heap_gb': 2,
                    'page_cache_gb': 2,
                    'pool_size': 10
                },
                api_config={
                    'max_concurrent': 5,
                    'batch_size': 5
                }
            ),
            'balanced': OptimizationProfile(
                name='balanced',
                description='Balanceado entre performance e recursos',
                qdrant_config={
                    'segments': 4,
                    'hnsw_m': 16,
                    'hnsw_ef_construct': 128,
                    'quantization': True
                },
                neo4j_config={
                    'heap_gb': 8,
                    'page_cache_gb': 16,
                    'pool_size': 20
                },
                api_config={
                    'max_concurrent': 20,
                    'batch_size': 10
                }
            ),
            'production': OptimizationProfile(
                name='production',
                description='Máxima performance para produção',
                qdrant_config={
                    'segments': 8,
                    'hnsw_m': 32,
                    'hnsw_ef_construct': 256,
                    'quantization': False
                },
                neo4j_config={
                    'heap_gb': 16,
                    'page_cache_gb': 32,
                    'pool_size': 50
                },
                api_config={
                    'max_concurrent': 50,
                    'batch_size': 20
                }
            )
        }
    
    async def auto_tune(self) -> Dict[str, Any]:
        """
        Executa auto-tuning baseado nos recursos detectados
        """
        logger.info("Iniciando auto-tuning do sistema")
        
        results = {
            'resources': self.system_resources.__dict__,
            'selected_profile': self._select_profile(),
            'optimizations': {}
        }
        
        # Otimizar Qdrant
        if self.config['qdrant']['enabled']:
            results['optimizations']['qdrant'] = await self._tune_qdrant()
        
        # Otimizar Neo4j
        if self.config['neo4j']['enabled']:
            results['optimizations']['neo4j'] = await self._tune_neo4j()
        
        # Otimizar APIs
        if self.config['api']['enabled']:
            results['optimizations']['api'] = self._tune_apis()
        
        # Salvar configurações otimizadas
        await self._save_optimized_config(results)
        
        return results
    
    def _select_profile(self) -> str:
        """Seleciona perfil baseado em recursos"""
        if self.system_resources.memory_gb < 8:
            return 'development'
        elif self.system_resources.memory_gb < 32:
            return 'balanced'
        else:
            return 'production'
    
    async def _tune_qdrant(self) -> Dict[str, Any]:
        """Otimiza Qdrant"""
        if not self.qdrant_optimizer:
            client = QdrantClient(":memory:")  # Ou URL real
            self.qdrant_optimizer = QdrantOptimizer(client)
        
        return await self.qdrant_optimizer.apply_optimizations(
            self.config['qdrant']['collection'],
            self.system_resources
        )
    
    async def _tune_neo4j(self) -> Dict[str, Any]:
        """Otimiza Neo4j"""
        if not self.neo4j_optimizer:
            self.neo4j_optimizer = Neo4jOptimizer(
                self.config['neo4j']['uri'],
                self.config['neo4j']['auth']
            )
            await self.neo4j_optimizer.connect()
        
        results = await self.neo4j_optimizer.apply_optimizations(self.system_resources)
        
        await self.neo4j_optimizer.close()
        
        return results
    
    def _tune_apis(self) -> Dict[str, Any]:
        """Otimiza configurações de API"""
        results = {}
        
        for provider in self.config['api']['providers']:
            results[provider] = {
                'config': self.api_optimizer.optimize_for_provider(
                    provider,
                    self.system_resources
                ),
                'batching': self.api_optimizer.get_batching_strategy(provider)
            }
        
        return results
    
    async def _save_optimized_config(self, results: Dict[str, Any]):
        """Salva configurações otimizadas"""
        output_path = "config/optimized_performance_config.yaml"
        
        optimized_config = {
            'generated_at': asyncio.get_event_loop().time(),
            'system_resources': results['resources'],
            'profile': results['selected_profile'],
            'optimizations': results['optimizations']
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
        
        logger.info(f"Configurações otimizadas salvas em: {output_path}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Retorna resumo das otimizações aplicadas"""
        profile = self.profiles.get(self._select_profile())
        
        return {
            'system_profile': self._select_profile(),
            'resources': {
                'cpu_cores': self.system_resources.cpu_count,
                'memory_gb': self.system_resources.memory_gb,
                'gpu_available': self.system_resources.gpu_available
            },
            'recommendations': self._get_recommendations(),
            'profile_config': profile.__dict__ if profile else {}
        }
    
    def _get_recommendations(self) -> List[str]:
        """Gera recomendações baseadas nos recursos"""
        recommendations = []
        
        # Memória
        if self.system_resources.memory_gb < 16:
            recommendations.append(
                "Consider upgrading RAM to 16GB+ for better performance"
            )
            recommendations.append(
                "Enable Qdrant scalar quantization to reduce memory usage by 75%"
            )
        
        # CPU
        if self.system_resources.cpu_count < 8:
            recommendations.append(
                "More CPU cores would improve concurrent request handling"
            )
        
        # GPU
        if not self.system_resources.gpu_available:
            recommendations.append(
                "GPU acceleration not available - using CPU for embeddings"
            )
        
        # Disco
        recommendations.append(
            "Ensure Qdrant data is stored on SSD for optimal performance"
        )
        
        return recommendations

# Factory function
async def create_performance_tuner(
    config_path: Optional[str] = None
) -> PerformanceTuner:
    """Cria e executa performance tuner"""
    if config_path is None:
        config_path = "config/performance_config.yaml"
    
    tuner = PerformanceTuner(config_path)
    
    # Auto-tune se configurado
    if tuner.config.get('auto_tune', True):
        await tuner.auto_tune()
    
    return tuner 