"""
Script de Setup: Configuração Rápida do Sistema RAG Avançado
Configura todos os componentes e dependências necessárias
"""

import os
import sys
import subprocess
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRAGSetup:
    """Setup automatizado para Sistema RAG Avançado"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        self.metrics_dir = self.project_root / "metrics"
        
    def run(self):
        """Executa setup completo"""
        logger.info("🚀 Iniciando setup do Sistema RAG Avançado...")
        
        # 1. Verificar Python version
        self.check_python_version()
        
        # 2. Instalar dependências
        self.install_dependencies()
        
        # 3. Configurar serviços externos
        self.setup_external_services()
        
        # 4. Criar estrutura de diretórios
        self.create_directories()
        
        # 5. Gerar arquivos de configuração
        self.generate_configs()
        
        # 6. Verificar instalação
        self.verify_installation()
        
        logger.info("✅ Setup concluído com sucesso!")
        self.print_next_steps()
    
    def check_python_version(self):
        """Verifica versão do Python"""
        logger.info("Verificando versão do Python...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("Python 3.8+ é necessário!")
            sys.exit(1)
        
        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    def install_dependencies(self):
        """Instala dependências do projeto"""
        logger.info("Instalando dependências...")
        
        # Dependências adicionais para features avançadas
        additional_deps = [
            "tree-sitter==0.20.4",
            "tree-sitter-languages==1.8.0",
            "python-louvain==0.16",
            "networkx==3.1",
            "ragatouille==0.0.7",
            "colbert-ai==0.2.0",
            "redis==5.0.1",
            "aioredis==2.0.1",
            "prometheus-client==0.19.0",
            "psutil==5.9.8",
            "sentence-transformers==2.2.2",
            "transformers==4.36.0",
            "torch==2.1.0"
        ]
        
        # Adicionar ao requirements.txt
        requirements_path = self.project_root / "requirements.txt"
        existing_reqs = set()
        
        if requirements_path.exists():
            with open(requirements_path, 'r') as f:
                existing_reqs = set(line.strip() for line in f if line.strip())
        
        with open(requirements_path, 'a') as f:
            for dep in additional_deps:
                if dep not in existing_reqs:
                    f.write(f"\n{dep}")
        
        # Instalar
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        logger.info("✓ Dependências instaladas")
    
    def setup_external_services(self):
        """Configura serviços externos"""
        logger.info("Configurando serviços externos...")
        
        # Docker Compose para serviços
        docker_compose = """version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    
  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password123
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=2G
    volumes:
      - neo4j_data:/data
      
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
      
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

volumes:
  redis_data:
  neo4j_data:
  qdrant_data:
  prometheus_data:
"""
        
        # Salvar docker-compose.yml
        with open(self.project_root / "docker-compose.yml", 'w') as f:
            f.write(docker_compose)
        
        # Prometheus config
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag_system'
    static_configs:
      - targets: ['localhost:8000']
"""
        
        self.config_dir.mkdir(exist_ok=True)
        with open(self.config_dir / "prometheus.yml", 'w') as f:
            f.write(prometheus_config)
        
        logger.info("✓ Configuração de serviços criada")
        logger.info("  Execute 'docker-compose up -d' para iniciar os serviços")
    
    def create_directories(self):
        """Cria estrutura de diretórios"""
        logger.info("Criando estrutura de diretórios...")
        
        directories = [
            self.config_dir,
            self.logs_dir,
            self.metrics_dir,
            self.project_root / "cache",
            self.project_root / "data" / "graphs",
            self.project_root / "data" / "indexes" / "advanced"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("✓ Diretórios criados")
    
    def generate_configs(self):
        """Gera arquivos de configuração"""
        logger.info("Gerando arquivos de configuração...")
        
        # Configuração principal do sistema avançado
        advanced_config = {
            'system': {
                'name': 'Advanced RAG System',
                'version': '2.0',
                'environment': 'development'
            },
            'advanced_features': {
                'language_aware_chunking': {
                    'enabled': True,
                    'target_chunk_size': 500,
                    'languages': ['python', 'javascript', 'typescript', 'csharp', 'java'],
                    'preserve_context': True,
                    'tree_sitter_timeout': 30
                },
                'graphrag': {
                    'enabled': True,
                    'neo4j_uri': 'bolt://localhost:7687',
                    'neo4j_auth': ['neo4j', 'password123'],
                    'louvain_resolution': 1.0,
                    'max_hops': 3,
                    'community_min_size': 3,
                    'enable_multi_hop': True
                },
                'reranking': {
                    'enabled': True,
                    'colbert_model': 'colbert-ir/colbertv2.0',
                    'cross_encoder_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                    'improvement_target': 0.20,
                    'use_gpu': False,
                    'batch_size': 16
                },
                'multi_layer_cache': {
                    'enabled': True,
                    'semantic_threshold': 0.95,
                    'enable_redis': True,
                    'redis_url': 'redis://localhost:6379',
                    'cache_ttl': 3600,
                    'max_cache_size': 10000,
                    'prefix_min_length': 10
                },
                'monitoring': {
                    'enabled': True,
                    'sample_interval': 60,
                    'enable_adaptive_routing': True,
                    'prometheus_port': 8000,
                    'alert_thresholds': {
                        'latency_p95': 5.0,
                        'error_rate': 0.05,
                        'memory_usage_mb': 8192
                    }
                },
                'performance_tuning': {
                    'auto_tune': True,
                    'profile': 'auto',
                    'qdrant_collection': 'advanced_rag',
                    'enable_quantization': True,
                    'optimize_on_startup': True
                }
            },
            'api_providers': {
                'openai': {
                    'enabled': True,
                    'models': ['gpt-4o', 'gpt-4o-mini', 'text-embedding-3-small']
                },
                'anthropic': {
                    'enabled': True,
                    'models': ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307']
                }
            }
        }
        
        # Salvar configuração
        config_path = self.config_dir / "advanced_rag_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(advanced_config, f, default_flow_style=False)
        
        # Configuração de monitoring
        monitoring_config = {
            'monitoring': {
                'sample_interval': 60,
                'metrics_retention': 7,
                'export_format': 'prometheus',
                'alert_channels': ['log', 'console'],
                'performance_tracking': {
                    'track_embeddings': True,
                    'track_retrieval': True,
                    'track_reranking': True,
                    'track_generation': True,
                    'track_cache': True
                }
            },
            'optimization': {
                'enable_adaptive_routing': True,
                'optimization_interval': 300,
                'min_samples_for_optimization': 100,
                'strategies': {
                    'simple_threshold': 0.5,
                    'complex_threshold': 2.0,
                    'cache_benefit_threshold': 0.7
                }
            }
        }
        
        monitoring_path = self.config_dir / "monitoring_config.yaml"
        with open(monitoring_path, 'w') as f:
            yaml.dump(monitoring_config, f, default_flow_style=False)
        
        # Performance config
        performance_config = {
            'auto_tune': True,
            'profile': 'auto',
            'qdrant': {
                'enabled': True,
                'collection': 'advanced_rag',
                'url': 'http://localhost:6333'
            },
            'neo4j': {
                'enabled': True,
                'uri': 'bolt://localhost:7687',
                'auth': ['neo4j', 'password123']
            },
            'api': {
                'enabled': True,
                'providers': ['openai', 'anthropic']
            }
        }
        
        perf_path = self.config_dir / "performance_config.yaml"
        with open(perf_path, 'w') as f:
            yaml.dump(performance_config, f, default_flow_style=False)
        
        logger.info("✓ Configurações geradas")
    
    def verify_installation(self):
        """Verifica se tudo foi instalado corretamente"""
        logger.info("Verificando instalação...")
        
        # Verificar imports críticos
        critical_imports = [
            "tree_sitter",
            "tree_sitter_languages",
            "community",
            "networkx",
            "redis",
            "prometheus_client",
            "transformers",
            "sentence_transformers"
        ]
        
        failed_imports = []
        for module in critical_imports:
            try:
                __import__(module)
            except ImportError:
                failed_imports.append(module)
        
        if failed_imports:
            logger.error(f"Falha ao importar: {', '.join(failed_imports)}")
            logger.error("Execute: pip install -r requirements.txt")
            sys.exit(1)
        
        logger.info("✓ Todos os módulos importados com sucesso")
    
    def print_next_steps(self):
        """Imprime próximos passos"""
        print("\n" + "="*60)
        print("🎉 SETUP CONCLUÍDO COM SUCESSO!")
        print("="*60)
        print("\n📋 PRÓXIMOS PASSOS:\n")
        print("1. Iniciar serviços externos:")
        print("   $ docker-compose up -d")
        print()
        print("2. Configurar variáveis de ambiente:")
        print("   $ cp config/env_example.txt .env")
        print("   $ # Editar .env com suas API keys")
        print()
        print("3. Testar o sistema:")
        print("   $ python examples/advanced_rag_example.py")
        print()
        print("4. Monitorar métricas:")
        print("   - Prometheus: http://localhost:9090")
        print("   - Neo4j Browser: http://localhost:7474")
        print("   - Qdrant Dashboard: http://localhost:6333/dashboard")
        print()
        print("5. Ver documentação completa:")
        print("   $ cat ADVANCED_RAG_IMPLEMENTATION_GUIDE.md")
        print("\n" + "="*60)

def main():
    """Executa setup"""
    setup = AdvancedRAGSetup()
    
    try:
        setup.run()
    except Exception as e:
        logger.error(f"Erro durante setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 