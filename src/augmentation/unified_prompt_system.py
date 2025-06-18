from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .dynamic_prompt_system import DynamicPromptSystem
from ..prompt_selector import select_prompt, classify_query

logger = logging.getLogger(__name__)


@dataclass
class PromptGenerationResult:
    """Resultado da geração de prompt com metadados."""
    final_prompt: str
    template_id: str
    task_type: str
    prompt_source: str  # "template" ou "dynamic"
    confidence: float
    metadata: Dict[str, Any]


class UnifiedPromptSystem:
    """
    Sistema unificado que combina:
    1. Prompt Selector - Seleção inteligente baseada em heurísticas
    2. Dynamic Prompt System - Geração dinâmica com contexto e templates
    
    Fluxo:
    1. Classifica a query (bugfix, code, review, etc.)
    2. Seleciona template apropriado baseado na classificação
    3. Gera prompt dinâmico combinando template + contexto + query
    4. Aplica otimizações baseadas no tipo de tarefa
    """

    def __init__(self):
        self.dynamic_system = DynamicPromptSystem()
        self.prompt_cache = {}  # Cache para templates processados
        
        # Configurações por tipo de tarefa
        self.task_configs = {
            "bugfix": {
                "reasoning_required": True,
                "max_context_chunks": 3,
                "temperature": 0.3,  # Mais conservador
                "system_enhancement": "Foque em identificar a causa raiz e fornecer soluções testáveis."
            },
            "review": {
                "reasoning_required": True,
                "max_context_chunks": 5,
                "temperature": 0.4,
                "system_enhancement": "Analise criticamente seguindo melhores práticas de code review."
            },
            "perf": {
                "reasoning_required": True,
                "max_context_chunks": 4,
                "temperature": 0.3,
                "system_enhancement": "Identifique gargalos e sugira otimizações mensuráveis."
            },
            "arch": {
                "reasoning_required": True,
                "max_context_chunks": 6,
                "temperature": 0.5,
                "system_enhancement": "Considere trade-offs, escalabilidade e manutenibilidade."
            },
            "test": {
                "reasoning_required": False,
                "max_context_chunks": 3,
                "temperature": 0.4,
                "system_enhancement": "Gere testes abrangentes seguindo pirâmide de testes."
            },
            "testgen": {
                "reasoning_required": False,
                "max_context_chunks": 2,
                "temperature": 0.3,
                "system_enhancement": "Crie casos de teste que cubram cenários edge case."
            },
            "data_viz": {
                "reasoning_required": False,
                "max_context_chunks": 3,
                "temperature": 0.6,
                "system_enhancement": "Sugira visualizações que revelem insights nos dados."
            },
            "ci": {
                "reasoning_required": True,
                "max_context_chunks": 4,
                "temperature": 0.3,
                "system_enhancement": "Diagnostique problemas de CI/CD e sugira correções."
            },
            "geral": {
                "reasoning_required": False,
                "max_context_chunks": 5,
                "temperature": 0.5,
                "system_enhancement": "Forneça resposta clara e bem estruturada."
            }
        }

    async def generate_optimal_prompt(
        self,
        query: str,
        context_chunks: List[str],
        language: str = "Português",
        depth: str = "quick",
        force_template: Optional[str] = None
    ) -> PromptGenerationResult:
        """
        Gera prompt otimizado combinando seleção inteligente com geração dinâmica.
        
        Args:
            query: Query do usuário
            context_chunks: Chunks de contexto relevantes
            language: Idioma da resposta
            depth: "quick" ou "deep" para complexidade
            force_template: Forçar uso de template específico
            
        Returns:
            PromptGenerationResult com prompt final e metadados
        """
        # 1. Classificar query e selecionar template
        task_type = classify_query(query)
        task_config = self.task_configs.get(task_type, self.task_configs["geral"])
        
        logger.info(f"🎯 Query classificada como: {task_type}")
        
        # 2. Selecionar template apropriado
        try:
            template_id, template_text = select_prompt(query, depth=depth)
            prompt_source = "template"
            confidence = 0.9  # Alta confiança em templates selecionados
            
            logger.info(f"📋 Template selecionado: {template_id}")
            
        except Exception as e:
            logger.warning(f"Erro ao selecionar template: {e}. Usando sistema dinâmico.")
            template_text = None
            template_id = "dynamic_fallback"
            prompt_source = "dynamic"
            confidence = 0.7

        # 3. Preparar contexto otimizado baseado no tipo de tarefa
        max_chunks = task_config["max_context_chunks"]
        optimized_chunks = context_chunks[:max_chunks] if context_chunks else []
        
        # 4. Gerar prompt base usando sistema dinâmico
        if template_text:
            # Usar template como base e enriquecer
            final_prompt = self._merge_template_with_dynamic(
                template_text, query, optimized_chunks, task_type, language
            )
        else:
            # Fallback para sistema dinâmico puro
            final_prompt = self.dynamic_system.generate_prompt(
                query, optimized_chunks, task_type, language
            )

        # 5. Aplicar enhancements específicos da tarefa
        final_prompt = self._apply_task_enhancements(
            final_prompt, task_type, task_config, query
        )

        # 6. Criar resultado com metadados
        result = PromptGenerationResult(
            final_prompt=final_prompt,
            template_id=template_id,
            task_type=task_type,
            prompt_source=prompt_source,
            confidence=confidence,
            metadata={
                "language": language,
                "depth": depth,
                "context_chunks_used": len(optimized_chunks),
                "max_chunks_allowed": max_chunks,
                "temperature_suggestion": task_config["temperature"],
                "reasoning_required": task_config["reasoning_required"],
                "system_enhancement": task_config["system_enhancement"]
            }
        )

        logger.info(f"✅ Prompt gerado: {len(final_prompt)} chars, confiança: {confidence:.2f}")
        return result

    def _merge_template_with_dynamic(
        self,
        template_text: str,
        query: str,
        context_chunks: List[str],
        task_type: str,
        language: str
    ) -> str:
        """Combina template selecionado com sistema dinâmico."""
        
        # Gerar prompt dinâmico para contexto e estrutura
        dynamic_prompt = self.dynamic_system.generate_prompt(
            query, context_chunks, task_type, language
        )
        
        # Extrair partes do prompt dinâmico
        dynamic_parts = dynamic_prompt.split("\n\n")
        system_prompt = dynamic_parts[0] if dynamic_parts else ""
        context_block = ""
        
        # Encontrar bloco de contexto
        for i, part in enumerate(dynamic_parts):
            if "Contexto:" in part and i + 1 < len(dynamic_parts):
                context_block = f"{part}\n\n{dynamic_parts[i + 1]}"
                break
        
        # Combinar template com contexto dinâmico
        merged_parts = []
        
        # 1. Sistema prompt melhorado
        if system_prompt:
            merged_parts.append(system_prompt)
        
        # 2. Template como estrutura principal
        merged_parts.append("# Template de Resposta:")
        merged_parts.append(template_text)
        
        # 3. Contexto dinâmico
        if context_block:
            merged_parts.append("---")
            merged_parts.append(context_block)
        
        # 4. Query específica
        merged_parts.append("---")
        merged_parts.append(f"Query específica: {query}")
        
        return "\n\n".join(merged_parts)

    def _apply_task_enhancements(
        self,
        prompt: str,
        task_type: str,
        task_config: Dict[str, Any],
        query: str
    ) -> str:
        """Aplica enhancements específicos por tipo de tarefa."""
        
        enhanced_parts = [prompt]
        
        # Enhancement específico da tarefa
        system_enhancement = task_config.get("system_enhancement", "")
        if system_enhancement:
            enhanced_parts.append(f"\n🎯 Instrução específica: {system_enhancement}")
        
        # Adicionar raciocínio se necessário
        if task_config.get("reasoning_required", False):
            if self.dynamic_system._needs_reasoning(query):
                enhanced_parts.append("\n🧠 Pense passo a passo antes de responder:")
                enhanced_parts.append("1. Analise o problema")
                enhanced_parts.append("2. Identifique as causas")
                enhanced_parts.append("3. Proponha soluções")
                enhanced_parts.append("4. Justifique sua resposta")
        
        # Enhancement por tipo específico
        if task_type == "bugfix":
            enhanced_parts.append("\n🐛 Para debugging, inclua:")
            enhanced_parts.append("- Diagnóstico da causa raiz")
            enhanced_parts.append("- Reprodução do problema")
            enhanced_parts.append("- Solução com código corrigido")
            enhanced_parts.append("- Prevenção futura")
            
        elif task_type == "review":
            enhanced_parts.append("\n👀 Para code review, verifique:")
            enhanced_parts.append("- Legibilidade e manutenibilidade")
            enhanced_parts.append("- Performance e otimizações")
            enhanced_parts.append("- Segurança e vulnerabilidades")
            enhanced_parts.append("- Testes e cobertura")
            
        elif task_type == "perf":
            enhanced_parts.append("\n⚡ Para otimização, analise:")
            enhanced_parts.append("- Gargalos identificados")
            enhanced_parts.append("- Métricas de performance")
            enhanced_parts.append("- Soluções com impacto mensurado")
            enhanced_parts.append("- Trade-offs das otimizações")
            
        elif task_type == "arch":
            enhanced_parts.append("\n🏗️ Para arquitetura, considere:")
            enhanced_parts.append("- Requisitos funcionais e não-funcionais")
            enhanced_parts.append("- Padrões arquiteturais apropriados")
            enhanced_parts.append("- Escalabilidade e manutenibilidade")
            enhanced_parts.append("- Decisões e trade-offs")
            
        elif task_type in ["test", "testgen"]:
            enhanced_parts.append("\n🧪 Para testes, inclua:")
            enhanced_parts.append("- Casos de teste positivos e negativos")
            enhanced_parts.append("- Edge cases e boundary conditions")
            enhanced_parts.append("- Mocks e fixtures necessários")
            enhanced_parts.append("- Estratégia de cobertura")
        
        return "\n".join(enhanced_parts)

    def get_task_suggestions(self, query: str) -> Dict[str, Any]:
        """Retorna sugestões de configuração baseadas na query."""
        task_type = classify_query(query)
        task_config = self.task_configs.get(task_type, self.task_configs["geral"])
        
        return {
            "task_type": task_type,
            "suggested_temperature": task_config["temperature"],
            "reasoning_recommended": task_config["reasoning_required"],
            "max_context_chunks": task_config["max_context_chunks"],
            "system_enhancement": task_config["system_enhancement"]
        }

    def clear_cache(self):
        """Limpa cache de prompts."""
        self.prompt_cache.clear()
        logger.info("Cache de prompts limpo")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema."""
        return {
            "total_task_types": len(self.task_configs),
            "cache_size": len(self.prompt_cache),
            "available_enhancements": list(self.task_configs.keys())
        }
