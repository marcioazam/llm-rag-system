#!/usr/bin/env python3
"""
Script de Integração - Funcionalidades Avançadas de RAG
Integra Multi-Head RAG, Adaptive Router e MemoRAG com o pipeline existente
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_advanced_features_config() -> Dict[str, Any]:
    """Cria configuração para as funcionalidades avançadas"""
    
    config = {
        "advanced_features": {
            "multi_head_rag": {
                "enabled": True,
                "num_heads": 5,
                "attention_dim": 768,
                "voting_strategy": "weighted_majority",
                "head_weights": {
                    "factual": 1.2,
                    "conceptual": 1.0,
                    "procedural": 1.1,
                    "contextual": 0.9,
                    "temporal": 0.8
                }
            },
            
            "adaptive_router": {
                "enabled": True,
                "optimization_objective": "balanced",
                "complexity_classifier": {
                    "model": "heuristic",  # ou "ml_based" quando treinar modelo
                    "confidence_threshold": 0.7
                },
                "routing_config": {
                    "simple": {
                        "strategies": ["direct"],
                        "k": 3,
                        "max_time": 2.0
                    },
                    "single_hop": {
                        "strategies": ["standard", "multi_query"],
                        "k": 5,
                        "max_time": 5.0
                    },
                    "multi_hop": {
                        "strategies": ["graph_enhanced", "multi_head"],
                        "k": 8,
                        "max_time": 10.0
                    },
                    "complex": {
                        "strategies": ["hybrid", "multi_head", "corrective"],
                        "k": 10,
                        "max_time": 15.0
                    }
                }
            },
            
            "memo_rag": {
                "enabled": True,
                "max_memory_tokens": 2000000,
                "compression_threshold": 10000,
                "segment_size": 1000,
                "clue_guided_retrieval": True,
                "memory_hierarchy": {
                    "hot_threshold_hours": 24,
                    "warm_threshold_hours": 168,  # 1 semana
                    "compression_levels": {
                        "hot": 0,  # Sem compressão
                        "warm": 3,  # Compressão média
                        "cold": 6   # Compressão máxima
                    }
                },
                "persistence": {
                    "enabled": True,
                    "path": "storage/memo_rag_memory.pkl",
                    "auto_save_interval": 3600  # 1 hora
                }
            }
        }
    }
    
    return config


def generate_integration_code() -> str:
    """Gera código de integração para o pipeline"""
    
    integration_code = '''# Adicionar ao src/rag_pipeline_advanced.py

# Importar novos componentes
from src.retrieval.multi_head_rag import create_multi_head_retriever
from src.retrieval.adaptive_rag_router import create_adaptive_router
from src.retrieval.memo_rag import create_memo_rag

class AdvancedRAGPipeline(APIRAGPipeline):
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        
        # Carregar configurações avançadas
        self.advanced_features_config = self.config.get("advanced_features", {})
        
        # Inicializar Multi-Head RAG
        if self.advanced_features_config.get("multi_head_rag", {}).get("enabled", False):
            self._init_multi_head_rag()
        
        # Inicializar MemoRAG
        if self.advanced_features_config.get("memo_rag", {}).get("enabled", False):
            self._init_memo_rag()
        
        # Inicializar Adaptive Router (deve ser último)
        if self.advanced_features_config.get("adaptive_router", {}).get("enabled", False):
            self._init_adaptive_router()
        
        logger.info("Funcionalidades avançadas de RAG inicializadas")
    
    def _init_multi_head_rag(self):
        """Inicializa Multi-Head RAG"""
        try:
            config = self.advanced_features_config["multi_head_rag"]
            self.multi_head_retriever = create_multi_head_retriever(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store,
                config=config
            )
            logger.info("Multi-Head RAG inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar Multi-Head RAG: {e}")
            self.multi_head_retriever = None
    
    def _init_memo_rag(self):
        """Inicializa MemoRAG"""
        try:
            config = self.advanced_features_config["memo_rag"]
            self.memo_rag = create_memo_rag(
                embedding_service=self.embedding_service,
                llm_service=self.llm_service,
                config=config
            )
            logger.info("MemoRAG inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar MemoRAG: {e}")
            self.memo_rag = None
    
    def _init_adaptive_router(self):
        """Inicializa Adaptive Router com todos os componentes"""
        try:
            # Coletar todos os componentes disponíveis
            rag_components = {
                "simple_retriever": self.retriever,
                "standard_rag": self,
                "multi_query_rag": self.multi_query_rag if hasattr(self, 'multi_query_rag') else None,
                "corrective_rag": self.corrective_rag if hasattr(self, 'corrective_rag') else None,
                "enhanced_corrective_rag": self.enhanced_corrective_rag if hasattr(self, 'enhanced_corrective_rag') else None,
                "graph_rag": self.graph_enhancer if hasattr(self, 'graph_enhancer') else None,
                "multi_head_rag": self.multi_head_retriever if hasattr(self, 'multi_head_retriever') else None,
                "memo_rag": self.memo_rag if hasattr(self, 'memo_rag') else None
            }
            
            # Remover componentes None
            rag_components = {k: v for k, v in rag_components.items() if v is not None}
            
            config = self.advanced_features_config["adaptive_router"]
            self.adaptive_router = create_adaptive_router(
                rag_components=rag_components,
                optimization=config.get("optimization_objective", "balanced")
            )
            
            logger.info(f"Adaptive Router inicializado com {len(rag_components)} componentes")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar Adaptive Router: {e}")
            self.adaptive_router = None
    
    async def query_advanced(self, question: str, config: Optional[Dict] = None, **kwargs) -> Dict:
        """Query avançada com roteamento adaptativo"""
        
        # Se Adaptive Router está habilitado, usar por padrão
        if hasattr(self, 'adaptive_router') and self.adaptive_router:
            use_adaptive = kwargs.get("use_adaptive_routing", True)
            
            if use_adaptive:
                logger.info("Usando Adaptive Router para processar query")
                result = await self.adaptive_router.route_query(question, **kwargs)
                
                # Adicionar metadados do pipeline
                result["pipeline_metadata"] = {
                    "method": "adaptive_routing",
                    "features_enabled": {
                        "multi_head": hasattr(self, 'multi_head_retriever') and self.multi_head_retriever is not None,
                        "memo_rag": hasattr(self, 'memo_rag') and self.memo_rag is not None,
                        "adaptive_router": True
                    }
                }
                
                return result
        
        # Fallback para método específico se solicitado
        method = kwargs.get("method")
        
        if method == "multi_head" and hasattr(self, 'multi_head_retriever'):
            return await self._query_multi_head(question, **kwargs)
        
        elif method == "memo_rag" and hasattr(self, 'memo_rag'):
            return await self._query_memo_rag(question, **kwargs)
        
        # Fallback para query padrão
        return await super().query_advanced(question, config, **kwargs)
    
    async def _query_multi_head(self, question: str, **kwargs) -> Dict:
        """Query usando Multi-Head RAG"""
        k = kwargs.get("k", 10)
        
        documents, metadata = await self.multi_head_retriever.retrieve_multi_head(question, k=k)
        
        # Gerar resposta com documentos multi-head
        context = self._prepare_context(documents)
        answer = await self._generate_response(question, context)
        
        return {
            "answer": answer,
            "sources": documents,
            "method": "multi_head",
            "multi_head_metadata": metadata
        }
    
    async def _query_memo_rag(self, question: str, **kwargs) -> Dict:
        """Query usando MemoRAG"""
        return await self.memo_rag.query_with_memory(question, **kwargs)
    
    async def add_to_memory(self, document: str, metadata: Optional[Dict] = None, **kwargs) -> Dict:
        """Adiciona documento à memória global do MemoRAG"""
        if not hasattr(self, 'memo_rag') or not self.memo_rag:
            return {"error": "MemoRAG não está habilitado"}
        
        return await self.memo_rag.add_document(document, metadata, **kwargs)
    
    def get_advanced_stats(self) -> Dict:
        """Estatísticas incluindo funcionalidades avançadas"""
        stats = super().get_advanced_stats()
        
        # Adicionar stats das novas funcionalidades
        if hasattr(self, 'multi_head_retriever') and self.multi_head_retriever:
            stats["multi_head_rag"] = self.multi_head_retriever.get_stats()
        
        if hasattr(self, 'adaptive_router') and self.adaptive_router:
            stats["adaptive_router"] = self.adaptive_router.get_routing_stats()
        
        if hasattr(self, 'memo_rag') and self.memo_rag:
            stats["memo_rag"] = self.memo_rag.get_stats()
        
        return stats
'''
    
    return integration_code


async def test_integration():
    """Testa a integração das funcionalidades"""
    
    print("\n🧪 Testando integração...")
    
    try:
        # Tentar importar o pipeline
        from src.rag_pipeline_advanced import AdvancedRAGPipeline
        
        # Criar instância com config
        config_path = Path("config/config.yaml")
        
        if config_path.exists():
            pipeline = AdvancedRAGPipeline(str(config_path))
            print("✅ Pipeline carregado com sucesso")
            
            # Verificar funcionalidades
            features = {
                "multi_head": hasattr(pipeline, 'multi_head_retriever'),
                "adaptive_router": hasattr(pipeline, 'adaptive_router'),
                "memo_rag": hasattr(pipeline, 'memo_rag')
            }
            
            print("\n📊 Funcionalidades disponíveis:")
            for feature, available in features.items():
                status = "✅" if available else "❌"
                print(f"   {status} {feature}")
            
            return True
        else:
            print("⚠️ Arquivo de configuração não encontrado")
            return False
            
    except ImportError as e:
        print(f"❌ Erro ao importar pipeline: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro ao testar integração: {e}")
        return False


def main():
    """Função principal do script de integração"""
    
    print("🚀 === INTEGRAÇÃO DE FUNCIONALIDADES AVANÇADAS RAG === 🚀")
    print("=" * 60)
    
    # Passo 1: Criar configuração
    print("\n📋 PASSO 1: Criando configuração avançada...")
    
    config = create_advanced_features_config()
    config_path = Path("config/advanced_rag_features.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Configuração salva: {config_path}")
    
    # Passo 2: Gerar código de integração
    print("\n📝 PASSO 2: Gerando código de integração...")
    
    integration_code = generate_integration_code()
    integration_file = Path("integration_advanced_features.py")
    
    with open(integration_file, 'w', encoding='utf-8') as f:
        f.write(integration_code)
    
    print(f"✅ Código de integração salvo: {integration_file}")
    print("   ⚠️ IMPORTANTE: Adicione este código ao src/rag_pipeline_advanced.py")
    
    # Passo 3: Atualizar config principal
    print("\n⚙️ PASSO 3: Atualizando configuração principal...")
    
    main_config_path = Path("config/config.yaml")
    
    if main_config_path.exists():
        with open(main_config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f) or {}
        
        # Adicionar referência às funcionalidades avançadas
        main_config["advanced_features_config"] = str(config_path)
        
        with open(main_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(main_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Configuração principal atualizada")
    else:
        print("⚠️ Configuração principal não encontrada - criar manualmente")
    
    # Passo 4: Instruções finais
    print("\n📋 PASSO 4: Instruções para completar a integração:")
    print("\n1. Adicione o código de integração ao arquivo:")
    print("   src/rag_pipeline_advanced.py")
    print("\n2. Instale dependências adicionais (se necessário):")
    print("   pip install torch numpy")
    print("\n3. Configure as funcionalidades em:")
    print(f"   {config_path}")
    print("\n4. Teste a integração:")
    print("   python demo_advanced_rag_features.py")
    
    # Passo 5: Criar exemplo de uso
    print("\n💡 PASSO 5: Exemplo de uso:")
    
    usage_example = '''
# Uso das funcionalidades avançadas

from src.rag_pipeline_advanced import AdvancedRAGPipeline

# Criar pipeline
pipeline = AdvancedRAGPipeline("config/config.yaml")

# 1. Usar Adaptive Router (automático)
result = await pipeline.query_advanced(
    "Compare REST vs GraphQL para microserviços",
    use_adaptive_routing=True
)

# 2. Forçar Multi-Head RAG
result = await pipeline.query_advanced(
    "Explique machine learning",
    method="multi_head"
)

# 3. Adicionar à memória global
await pipeline.add_to_memory(
    document="Conteúdo técnico extenso...",
    metadata={"source": "manual", "importance": 0.9}
)

# 4. Query com MemoRAG
result = await pipeline.query_advanced(
    "Informações sobre o manual técnico",
    method="memo_rag"
)

# 5. Obter estatísticas
stats = pipeline.get_advanced_stats()
print(f"Router stats: {stats['adaptive_router']}")
print(f"MemoRAG memory: {stats['memo_rag']['memory_stats']}")
'''
    
    print(usage_example)
    
    # Testar se possível
    print("\n🧪 Tentando testar integração...")
    asyncio.run(test_integration())
    
    print("\n✅ === SCRIPT DE INTEGRAÇÃO CONCLUÍDO === ✅")
    print("\n🎯 Próximos passos:")
    print("1. Integre o código gerado ao pipeline")
    print("2. Execute o demo para validar")
    print("3. Ajuste configurações conforme necessário")
    print("4. Monitore performance em produção")


if __name__ == "__main__":
    main() 