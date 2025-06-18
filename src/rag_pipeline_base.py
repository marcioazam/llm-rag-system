"""
Classe base para RAG Pipeline - Criada para a Fase 1
Interface mínima para manter compatibilidade
"""

import logging
from typing import Dict, List, Optional, Any
import asyncio

logger = logging.getLogger(__name__)


class BaseRAGPipeline:
    """
    Classe base para RAG Pipeline - Interface mínima
    
    Esta classe fornece a interface básica necessária para o AdvancedRAGPipeline
    sem dependências complexas.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa pipeline base
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config_path = config_path
        self.config = self._load_config(config_path) if config_path else {}
        
        # Métricas básicas
        self.base_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0,
            "total_documents_processed": 0
        }
        
        logger.info("BaseRAGPipeline inicializado")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configuração básica"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Erro ao carregar config {config_path}: {e}")
            return {}
    
    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Query básica - deve ser implementada pelas subclasses
        
        Args:
            question: Pergunta do usuário
            **kwargs: Parâmetros adicionais
            
        Returns:
            Dict com resposta
        """
        self.base_metrics["total_queries"] += 1
        
        try:
            # Implementação básica de fallback
            result = {
                "answer": f"Desculpe, não foi possível processar a pergunta: {question}",
                "confidence": 0.0,
                "sources": [],
                "model_used": "base_fallback",
                "cost": 0.0
            }
            
            self.base_metrics["successful_queries"] += 1
            return result
            
        except Exception as e:
            self.base_metrics["failed_queries"] += 1
            logger.error(f"Erro na query base: {e}")
            raise
    
    async def add_documents(self, documents: List[str], **kwargs) -> bool:
        """
        Adiciona documentos - implementação básica
        
        Args:
            documents: Lista de documentos
            **kwargs: Parâmetros adicionais
            
        Returns:
            bool: Success status
        """
        try:
            self.base_metrics["total_documents_processed"] += len(documents)
            logger.info(f"Documentos processados (base): {len(documents)}")
            return True
        except Exception as e:
            logger.error(f"Erro ao adicionar documentos: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas básicas
        
        Returns:
            Dict com métricas
        """
        success_rate = 0.0
        if self.base_metrics["total_queries"] > 0:
            success_rate = (
                self.base_metrics["successful_queries"] / 
                self.base_metrics["total_queries"]
            )
        
        return {
            "base_metrics": {
                **self.base_metrics,
                "success_rate": success_rate
            },
            "config_loaded": bool(self.config),
            "pipeline_type": "base"
        }
    
    async def _call_llm_api(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Método básico para chamada de LLM - deve ser implementado pelas subclasses
        
        Args:
            prompt: Prompt para o modelo
            system_prompt: System prompt opcional
            
        Returns:
            Dict com resposta do modelo
        """
        # Implementação de fallback básica
        return {
            "answer": "Resposta não disponível - APIs não configuradas",
            "model_used": "fallback",
            "cost": 0.0
        }


# Alias para compatibilidade
APIRAGPipeline = BaseRAGPipeline 