#!/usr/bin/env python3
"""
Hybrid Model Router - Sistema que pode usar múltiplas APIs colaborativas
Permite tanto respostas simples (1 API) quanto respostas híbridas (múltiplas APIs)
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .api_model_router import APIModelRouter, TaskType, ModelResponse

logger = logging.getLogger(__name__)

class ResponseMode(Enum):
    """Modos de resposta do sistema"""
    SINGLE_API = "single_api"          # Uma única API (atual)
    HYBRID_COLLABORATIVE = "hybrid"    # Múltiplas APIs colaborativas
    HYBRID_COMPETITIVE = "competitive" # Múltiplas APIs competitivas (escolhe melhor)

@dataclass
class HybridStep:
    """Representa um passo no processo híbrido"""
    step_name: str
    provider: str
    model: str
    purpose: str
    input_query: str
    output: str
    cost: float
    processing_time: float
    success: bool

@dataclass
class HybridResponse:
    """Resposta de um processo híbrido"""
    final_answer: str
    mode: ResponseMode
    steps: List[HybridStep]
    total_cost: float
    total_time: float
    providers_used: List[str]
    success: bool
    error: Optional[str] = None

class HybridModelRouter(APIModelRouter):
    """
    Router híbrido que pode usar uma ou múltiplas APIs por resposta
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configurações específicas do sistema híbrido
        self.hybrid_config = config.get("hybrid", {
            "enabled": True,
            "default_mode": ResponseMode.SINGLE_API,
            "cost_threshold": 0.01,  # Limite de custo para modo híbrido
            "time_threshold": 10.0,  # Limite de tempo para modo híbrido
            "quality_threshold": 0.8  # Threshold de qualidade para usar híbrido
        })
        
        # Workflows pré-definidos para diferentes tipos de tarefa
        self.hybrid_workflows = {
            TaskType.CODE_GENERATION: [
                {"step": "analysis", "provider": "anthropic", "model": "claude_3_5_sonnet", "purpose": "Analisar requisitos"},
                {"step": "generation", "provider": "openai", "model": "gpt4o_mini", "purpose": "Gerar código"},
                {"step": "optimization", "provider": "deepseek", "model": "deepseek_coder", "purpose": "Otimizar código"},
                {"step": "documentation", "provider": "anthropic", "model": "claude_3_5_sonnet", "purpose": "Documentar solução"}
            ],
            TaskType.DOCUMENT_ANALYSIS: [
                {"step": "extraction", "provider": "anthropic", "model": "claude_3_5_sonnet", "purpose": "Extrair informações"},
                {"step": "analysis", "provider": "openai", "model": "gpt4o", "purpose": "Análise profunda"},
                {"step": "synthesis", "provider": "google", "model": "gemini_1_5_pro", "purpose": "Sintetizar resultados"}
            ],
            TaskType.COMPLEX_ANALYSIS: [
                {"step": "decomposition", "provider": "openai", "model": "gpt4o", "purpose": "Quebrar problema"},
                {"step": "research", "provider": "google", "model": "gemini_1_5_pro", "purpose": "Pesquisar contexto"},
                {"step": "reasoning", "provider": "deepseek", "model": "deepseek_chat", "purpose": "Raciocínio lógico"},
                {"step": "conclusion", "provider": "anthropic", "model": "claude_3_5_sonnet", "purpose": "Conclusão final"}
            ]
        }

    def generate_response(self, 
                         query: str, 
                         context: str = "",
                         system_prompt: Optional[str] = None,
                         task_type: Optional[TaskType] = None,
                         force_model: Optional[str] = None,
                         response_mode: Optional[ResponseMode] = None) -> Any:
        """
        Gera resposta usando modo simples ou híbrido
        """
        
        # Detectar tipo de tarefa se não fornecido
        if task_type is None:
            task_type = self.detect_task_type(query, context)
        
        # Determinar modo de resposta
        if response_mode is None:
            response_mode = self._determine_response_mode(query, task_type, force_model)
        
        # Executar baseado no modo
        if response_mode == ResponseMode.SINGLE_API:
            return self._generate_single_response(query, context, system_prompt, task_type, force_model)
        elif response_mode == ResponseMode.HYBRID_COLLABORATIVE:
            return self._generate_hybrid_collaborative(query, context, system_prompt, task_type)
        elif response_mode == ResponseMode.HYBRID_COMPETITIVE:
            return self._generate_hybrid_competitive(query, context, system_prompt, task_type)
        else:
            raise ValueError(f"Modo de resposta não suportado: {response_mode}")

    def _determine_response_mode(self, query: str, task_type: TaskType, force_model: Optional[str]) -> ResponseMode:
        """Determina automaticamente o melhor modo de resposta"""
        
        # Se modelo específico foi forçado, usar modo simples
        if force_model:
            return ResponseMode.SINGLE_API
        
        # Se híbrido está desabilitado, usar modo simples
        if not self.hybrid_config.get("enabled", True):
            return ResponseMode.SINGLE_API
        
        # Critérios para usar modo híbrido
        complexity_indicators = [
            len(query) > 200,  # Query longa
            "e também" in query.lower(),  # Múltiplas solicitações
            "primeiro" in query.lower() and "depois" in query.lower(),  # Etapas sequenciais
            task_type in [TaskType.CODE_GENERATION, TaskType.COMPLEX_ANALYSIS, TaskType.DOCUMENT_ANALYSIS],
            "otimizar" in query.lower() and "código" in query.lower(),  # Otimização de código
            "analisar" in query.lower() and ("implementar" in query.lower() or "criar" in query.lower())
        ]
        
        complexity_score = sum(complexity_indicators) / len(complexity_indicators)
        
        if complexity_score > self.hybrid_config.get("quality_threshold", 0.8):
            return ResponseMode.HYBRID_COLLABORATIVE
        else:
            return ResponseMode.SINGLE_API

    def _generate_single_response(self, query: str, context: str, system_prompt: Optional[str], 
                                task_type: TaskType, force_model: Optional[str]) -> ModelResponse:
        """Gera resposta usando uma única API (modo atual)"""
        return super().generate_response(query, context, system_prompt, task_type, force_model)

    def _generate_hybrid_collaborative(self, query: str, context: str, system_prompt: Optional[str], 
                                     task_type: TaskType) -> HybridResponse:
        """Gera resposta usando múltiplas APIs colaborativas"""
        
        start_time = time.time()
        steps = []
        total_cost = 0.0
        providers_used = []
        
        # Obter workflow para o tipo de tarefa
        workflow = self.hybrid_workflows.get(task_type, self.hybrid_workflows[TaskType.COMPLEX_ANALYSIS])
        
        try:
            current_context = context
            
            for i, step_config in enumerate(workflow):
                step_start_time = time.time()
                
                # Preparar query para esta etapa
                if i == 0:
                    step_query = query
                else:
                    # Usar resultado da etapa anterior como contexto
                    previous_output = steps[-1].output if steps else ""
                    step_query = f"{step_config['purpose']}: {query}"
                    current_context = f"{context}\n\nEtapa anterior: {previous_output}"
                
                # Executar etapa
                try:
                    model_key = f"{step_config['provider']}.{step_config['model']}"
                    response = super().generate_response(
                        query=step_query,
                        context=current_context,
                        system_prompt=system_prompt,
                        force_model=model_key
                    )
                    
                    step = HybridStep(
                        step_name=step_config['step'],
                        provider=step_config['provider'],
                        model=step_config['model'],
                        purpose=step_config['purpose'],
                        input_query=step_query,
                        output=response.content,
                        cost=response.cost,
                        processing_time=time.time() - step_start_time,
                        success=True
                    )
                    
                    steps.append(step)
                    total_cost += response.cost
                    providers_used.append(step_config['provider'])
                    
                    logger.info(f"Etapa {step_config['step']} concluída: {step_config['provider']}")
                    
                except Exception as e:
                    logger.error(f"Erro na etapa {step_config['step']}: {e}")
                    
                    # Tentar fallback para OpenAI
                    try:
                        fallback_response = super().generate_response(
                            query=step_query,
                            context=current_context,
                            system_prompt=system_prompt,
                            force_model="openai.gpt4o_mini"
                        )
                        
                        step = HybridStep(
                            step_name=step_config['step'],
                            provider="openai",
                            model="gpt4o_mini",
                            purpose=f"{step_config['purpose']} (fallback)",
                            input_query=step_query,
                            output=fallback_response.content,
                            cost=fallback_response.cost,
                            processing_time=time.time() - step_start_time,
                            success=True
                        )
                        
                        steps.append(step)
                        total_cost += fallback_response.cost
                        providers_used.append("openai")
                        
                    except Exception as fallback_error:
                        step = HybridStep(
                            step_name=step_config['step'],
                            provider=step_config['provider'],
                            model=step_config['model'],
                            purpose=step_config['purpose'],
                            input_query=step_query,
                            output=f"Erro: {str(fallback_error)}",
                            cost=0.0,
                            processing_time=time.time() - step_start_time,
                            success=False
                        )
                        steps.append(step)
            
            # Consolidar resposta final
            if steps and steps[-1].success:
                final_answer = steps[-1].output
            else:
                # Tentar síntese manual se última etapa falhou
                successful_outputs = [step.output for step in steps if step.success]
                final_answer = "\n\n".join(successful_outputs) if successful_outputs else "Erro no processamento híbrido"
            
            total_time = time.time() - start_time
            
            return HybridResponse(
                final_answer=final_answer,
                mode=ResponseMode.HYBRID_COLLABORATIVE,
                steps=steps,
                total_cost=total_cost,
                total_time=total_time,
                providers_used=list(set(providers_used)),
                success=any(step.success for step in steps)
            )
            
        except Exception as e:
            logger.error(f"Erro no processo híbrido: {e}")
            
            return HybridResponse(
                final_answer=f"Erro no processo híbrido: {str(e)}",
                mode=ResponseMode.HYBRID_COLLABORATIVE,
                steps=steps,
                total_cost=total_cost,
                total_time=time.time() - start_time,
                providers_used=list(set(providers_used)),
                success=False,
                error=str(e)
            )

    def _generate_hybrid_competitive(self, query: str, context: str, system_prompt: Optional[str], 
                                   task_type: TaskType) -> HybridResponse:
        """Gera múltiplas respostas e escolhe a melhor"""
        
        start_time = time.time()
        steps = []
        total_cost = 0.0
        providers_used = []
        
        # Modelos para competição baseado no tipo de tarefa
        competing_models = self._get_competing_models(task_type)
        
        try:
            responses = []
            
            # Gerar respostas de múltiplos modelos
            for model_key in competing_models:
                try:
                    response = super().generate_response(
                        query=query,
                        context=context,
                        system_prompt=system_prompt,
                        force_model=model_key
                    )
                    
                    provider = model_key.split('.')[0]
                    model = model_key.split('.')[1]
                    
                    step = HybridStep(
                        step_name="competitive_generation",
                        provider=provider,
                        model=model,
                        purpose="Gerar resposta competitiva",
                        input_query=query,
                        output=response.content,
                        cost=response.cost,
                        processing_time=response.processing_time,
                        success=True
                    )
                    
                    steps.append(step)
                    responses.append(response)
                    total_cost += response.cost
                    providers_used.append(provider)
                    
                except Exception as e:
                    logger.error(f"Erro com modelo {model_key}: {e}")
            
            # Escolher melhor resposta (por agora, a mais longa)
            if responses:
                best_response = max(responses, key=lambda r: len(r.content))
                final_answer = best_response.content
            else:
                final_answer = "Erro: Nenhuma resposta gerada"
            
            total_time = time.time() - start_time
            
            return HybridResponse(
                final_answer=final_answer,
                mode=ResponseMode.HYBRID_COMPETITIVE,
                steps=steps,
                total_cost=total_cost,
                total_time=total_time,
                providers_used=list(set(providers_used)),
                success=len(responses) > 0
            )
            
        except Exception as e:
            return HybridResponse(
                final_answer=f"Erro no modo competitivo: {str(e)}",
                mode=ResponseMode.HYBRID_COMPETITIVE,
                steps=steps,
                total_cost=total_cost,
                total_time=time.time() - start_time,
                providers_used=list(set(providers_used)),
                success=False,
                error=str(e)
            )

    def _get_competing_models(self, task_type: TaskType) -> List[str]:
        """Retorna modelos para competição baseado no tipo de tarefa"""
        
        if task_type == TaskType.CODE_GENERATION:
            return ["openai.gpt4o_mini", "deepseek.deepseek_coder"]
        elif task_type == TaskType.DOCUMENT_ANALYSIS:
            return ["anthropic.claude_3_5_sonnet", "openai.gpt4o"]
        elif task_type == TaskType.QUICK_QUERIES:
            return ["google.gemini_1_5_flash", "openai.gpt35_turbo"]
        else:
            return ["openai.gpt4o", "anthropic.claude_3_5_sonnet"]

    def get_hybrid_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas específicas do modo híbrido"""
        base_stats = self.get_stats()
        
        # Adicionar estatísticas híbridas
        hybrid_stats = {
            **base_stats,
            "hybrid_enabled": self.hybrid_config.get("enabled", True),
            "hybrid_workflows": len(self.hybrid_workflows),
            "supported_modes": [mode.value for mode in ResponseMode]
        }
        
        return hybrid_stats 