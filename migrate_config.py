#!/usr/bin/env python3
"""
Script para migrar config.yaml para suportar multi-modelo
Preserva configurações existentes e adiciona novas
"""

import yaml
import shutil
from datetime import datetime
from pathlib import Path
import sys

def backup_config(config_path):
    """Cria backup da configuração atual"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.parent / f"{config_path.stem}.backup.{timestamp}{config_path.suffix}"
    shutil.copy2(config_path, backup_path)
    print(f"✓ Backup criado: {backup_path}")
    return backup_path

def load_config(config_path):
    """Carrega configuração existente"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def migrate_config(old_config):
    """Migra configuração antiga para novo formato"""
    # Preserva configurações existentes
    new_config = old_config.copy()
    
    # Adiciona configurações multi-modelo ao LLM
    if 'llm' in new_config:
        # Preserva configurações existentes
        llm_config = new_config['llm']
        
        # Adiciona novas configurações
        llm_config['routing_mode'] = 'advanced'
        llm_config['models'] = {
            'general': {
                'name': llm_config.get('model', 'llama3.1:8b-instruct-q4_K_M'),
                'temperature': llm_config.get('temperature', 0.7),
                'max_tokens': llm_config.get('max_tokens', 2048),
                'tasks': ['general_explanation', 'documentation'],
                'priority': 1
            },
            'code': {
                'name': 'codellama:7b-instruct',
                'temperature': 0.3,
                'max_tokens': 4096,
                'tasks': ['code_generation', 'debugging'],
                'priority': 1
            },
            'mistral': {
                'name': 'mistral:7b-instruct-q4_0',
                'temperature': 0.8,
                'max_tokens': 2048,
                'tasks': ['architecture_design', 'system_analysis'],
                'priority': 2,
                'optional': True
            },
            'sql': {
                'name': 'sqlcoder:7b-q4_0',
                'temperature': 0.1,
                'max_tokens': 1024,
                'tasks': ['sql_query'],
                'priority': 1,
                'optional': True
            },
            'fast': {
                'name': 'phi:2.7b',
                'temperature': 0.5,
                'max_tokens': 512,
                'tasks': ['quick_snippet', 'validation'],
                'priority': 3,
                'optional': True
            }
        }
    
    # Adiciona configurações de performance
    new_config['performance'] = {
        'max_concurrent_models': 2,
        'cpu_threads': 6,
        'use_gpu': False,
        'model_timeout': 300,
        'cache_models': True,
        'model_cache_ttl': 3600
    }
    
    # Atualiza embeddings para CPU (MX150 tem pouca VRAM)
    if 'embeddings' in new_config:
        new_config['embeddings']['device'] = 'cpu'
        new_config['embeddings']['cache_embeddings'] = True
        new_config['embeddings']['normalize_embeddings'] = True
    
    # Adiciona metadata fields ao vectordb
    if 'vectordb' in new_config:
        new_config['vectordb']['metadata_fields'] = [
            'source', 'doc_type', 'language', 'complexity'
        ]
    
    # Adiciona adaptive chunking
    if 'chunking' in new_config:
        new_config['chunking']['adaptive_chunking'] = {
            'code': {
                'chunk_size': 1024,
                'chunk_overlap': 100
            },
            'documentation': {
                'chunk_size': 512,
                'chunk_overlap': 50
            },
            'sql': {
                'chunk_size': 256,
                'chunk_overlap': 25
            }
        }
    
    # Adiciona context-aware retrieval
    if 'retrieval' in new_config:
        new_config['retrieval']['context_aware'] = {
            'enabled': True,
            'boost_recent': True,
            'boost_factor': 1.2
        }
    
    # Adiciona configurações multi-modelo ao RAG
    if 'rag' in new_config:
        new_config['rag']['multi_model'] = {
            'enabled': True,
            'task_detection': {
                'enabled': True,
                'confidence_threshold': 0.7
            },
            'model_selection': {
                'strategy': 'best_available',
                'fallback_chain': ['general', 'code', 'fast']
            },
            'response_fusion': {
                'enabled': True,
                'method': 'sequential'
            }
        }
    
    # Adiciona novas seções
    new_config['logging'] = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/rag_system.log',
        'max_file_size': '10MB',
        'backup_count': 3,
        'model_logging': {
            'log_prompts': False,
            'log_responses': False,
            'log_performance': True
        }
    }
    
    new_config['api'] = {
        'host': '0.0.0.0',
        'port': 8000,
        'reload': True,
        'workers': 1,
        'cors_origins': ['*'],
        'rate_limit': {
            'enabled': True,
            'requests_per_minute': 60
        }
    }
    
    new_config['monitoring'] = {
        'enabled': True,
        'check_interval': 60,
        'alerts': {
            'memory_threshold': 85,
            'cpu_threshold': 90,
            'model_load_time_threshold': 30
        }
    }
    
    new_config['debug'] = {
        'verbose': False,
        'show_model_selection': True,
        'show_retrieval_scores': False,
        'show_chunk_details': False,
        'benchmark_mode': False
    }
    
    return new_config

def save_config(config, config_path):
    """Salva nova configuração"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    print(f"✓ Configuração atualizada: {config_path}")

def main():
    """Função principal"""
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        print(f"❌ Arquivo não encontrado: {config_path}")
        sys.exit(1)
    
    print("🔄 Migrando configuração para suportar multi-modelo...")
    
    # 1. Backup
    backup_path = backup_config(config_path)
    
    try:
        # 2. Carregar config atual
        old_config = load_config(config_path)
        
        # 3. Migrar
        new_config = migrate_config(old_config)
        
        # 4. Salvar
        save_config(new_config, config_path)
        
        print("\n✅ Migração concluída com sucesso!")
        print("\nNovas funcionalidades adicionadas:")
        print("  • Suporte multi-modelo (5 modelos)")
        print("  • Roteamento inteligente de tarefas")
        print("  • Configurações de performance otimizadas")
        print("  • Chunking adaptativo por tipo de conteúdo")
        print("  • Monitoramento de sistema")
        print("  • Logging aprimorado")
        
        print(f"\n💡 Dica: Compare com o backup em {backup_path}")
        
    except Exception as e:
        print(f"\n❌ Erro durante migração: {e}")
        print(f"Restaurando backup...")
        shutil.copy2(backup_path, config_path)
        print("✓ Backup restaurado")
        sys.exit(1)

if __name__ == "__main__":
    main()
