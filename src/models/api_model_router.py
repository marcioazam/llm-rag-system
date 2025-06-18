"""
Roteador de Modelos via API externa.
Substitui completamente modelos locais como Ollama.
Suporta OpenAI, Anthropic, Google e Groq.
"""

import os
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

import requests

logger = logging.getLogger(__name__)


class TaskType(Enum):
    GENERAL_EXPLANATION = "general_explanation"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    DOCUMENT_ANALYSIS = "document_analysis"
    CONTENT_CREATION = "content_creation"
    RESEARCH_SYNTHESIS = "research_synthesis"
    TECHNICAL_WRITING = "technical_writing"
    QUICK_QUERIES = "quick_queries"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    SIMPLE_EXPLANATIONS = "simple_explanations"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"


@dataclass
class ModelResponse:
    """Response do modelo"""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    cost: float = 0.0
    processing_time: float = 0.0
    finish_reason: str = "stop"


@dataclass
class ModelConfig:
    """Configuração do modelo"""
    name: str
    max_tokens: int
    temperature: float
    responsibilities: List[str]
    context_window: int
    cost_per_1k_tokens: float
    priority: int


class APIModelRouter:
    """
    Roteador inteligente para modelos via API.
    Seleciona o melhor modelo baseado na tarefa e otimiza custo/performance.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers_config = config.get("providers", {})
        self.routing_config = config.get("routing", {})
        
        # Carregar modelos disponíveis
        self.available_models = self._load_available_models()
        
        # Mapeamento de tarefas para responsabilidades
        self.task_responsibility_mapping = {
            TaskType.GENERAL_EXPLANATION: "primary_reasoning",
            TaskType.CODE_GENERATION: "code_generation",
            TaskType.DEBUGGING: "debugging",
            TaskType.DOCUMENT_ANALYSIS: "document_analysis",
            TaskType.CONTENT_CREATION: "content_creation",
            TaskType.RESEARCH_SYNTHESIS: "research_synthesis",
            TaskType.TECHNICAL_WRITING: "technical_writing",
            TaskType.QUICK_QUERIES: "quick_queries",
            TaskType.SUMMARIZATION: "summarization",
            TaskType.TRANSLATION: "translation",
            TaskType.SIMPLE_EXPLANATIONS: "simple_explanations",
            TaskType.ARCHITECTURE_DESIGN: "architecture_design",
            TaskType.CODE_REVIEW: "code_review",
            TaskType.REFACTORING: "refactoring"
        }
        
        # Estatísticas
        self.stats = {
            "total_requests": 0,
            "total_cost": 0.0,
            "provider_usage": {},
            "model_usage": {},
            "task_distribution": {},
            "average_response_time": 0.0,
            "errors": 0
        }
        
        logger.info("APIModelRouter inicializado com {} modelos".format(len(self.available_models)))

    def _load_available_models(self) -> Dict[str, ModelConfig]:
        """Carrega modelos disponíveis dos provedores configurados"""
        available = {}
        
        for provider_name, provider_config in self.providers_config.items():
            models = provider_config.get("models", {})
            
            for model_key, model_config in models.items():
                full_key = f"{provider_name}.{model_key}"
                available[full_key] = ModelConfig(
                    name=model_config["name"],
                    max_tokens=model_config["max_tokens"],
                    temperature=model_config["temperature"],
                    responsibilities=model_config["responsibilities"],
                    context_window=model_config["context_window"],
                    cost_per_1k_tokens=model_config["cost_per_1k_tokens"],
                    priority=model_config["priority"]
                )
        
        return available

    def detect_task_type(self, query: str, context: str = "") -> TaskType:
        """Detecta o tipo de tarefa baseado na query"""
        combined_text = (query + " " + context).lower()
        
        # Palavras-chave para diferentes tipos de tarefas
        task_keywords = {
            TaskType.CODE_GENERATION: ["código", "codigo", "programação", "implementar", "função", "classe", "script"],
            TaskType.DEBUGGING: ["erro", "bug", "debug", "falha", "problema", "corrigir", "consertar"],
            TaskType.DOCUMENT_ANALYSIS: ["analisar", "análise", "documento", "revisar", "examinar"],
            TaskType.CONTENT_CREATION: ["criar", "escrever", "redigir", "produzir", "gerar conteúdo"],
            TaskType.RESEARCH_SYNTHESIS: ["pesquisa", "síntese", "combinar", "unir informações"],
            TaskType.TECHNICAL_WRITING: ["documentação", "manual", "especificação", "guia técnico"],
            TaskType.QUICK_QUERIES: ["rápido", "simples", "básico", "direto"],
            TaskType.SUMMARIZATION: ["resumir", "resumo", "sumarizar", "sintetizar"],
            TaskType.TRANSLATION: ["traduzir", "tradução", "converter idioma"],
            TaskType.ARCHITECTURE_DESIGN: ["arquitetura", "design", "estrutura", "padrão"],
            TaskType.CODE_REVIEW: ["review", "revisar código", "analisar código"],
            TaskType.REFACTORING: ["refatorar", "melhorar código", "otimizar"]
        }
        
        # Detectar baseado em palavras-chave
        for task_type, keywords in task_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return task_type
        
        # Fallback para tarefa geral
        return TaskType.GENERAL_EXPLANATION

    def select_best_model(self, task_type: TaskType, context_length: int = 0) -> Optional[str]:
        """Seleciona o melhor modelo para a tarefa"""
        
        # Mapear tarefa para responsabilidade
        responsibility = self.task_responsibility_mapping.get(task_type, "primary_reasoning")
        
        # Encontrar modelos que podem lidar com a responsabilidade
        suitable_models = []
        for model_key, model_config in self.available_models.items():
            if responsibility in model_config.responsibilities:
                # Verificar se o modelo pode lidar com o contexto
                if context_length <= model_config.context_window:
                    suitable_models.append((model_key, model_config))
        
        if not suitable_models:
            # Fallback para o modelo padrão
            fallback_chain = self.routing_config.get("fallback_chain", ["openai.gpt4o_mini"])
            for fallback in fallback_chain:
                if fallback in self.available_models:
                    return fallback
            return None
        
        # Aplicar estratégia de seleção
        strategy = self.routing_config.get("strategy", "cost_performance_optimized")
        
        if strategy == "cost_optimized":
            # Selecionar o modelo mais barato
            return min(suitable_models, key=lambda x: x[1].cost_per_1k_tokens)[0]
        elif strategy == "performance_optimized":
            # Selecionar o modelo com maior prioridade
            return min(suitable_models, key=lambda x: x[1].priority)[0]
        else:  # cost_performance_optimized ou balanced
            # Otimizar custo vs performance
            def score(model_tuple):
                _, config = model_tuple
                cost_factor = config.cost_per_1k_tokens * 100
                performance_factor = config.priority
                return cost_factor + performance_factor
            return min(suitable_models, key=score)[0]

    def generate_response(self, 
                         query: str, 
                         context: str = "",
                         system_prompt: Optional[str] = None,
                         task_type: Optional[TaskType] = None,
                         force_model: Optional[str] = None) -> ModelResponse:
        """Gera resposta usando o modelo mais adequado"""
        
        start_time = time.time()
        
        # Detectar tipo de tarefa se não fornecido
        if task_type is None:
            task_type = self.detect_task_type(query, context)
        
        # Selecionar modelo
        if force_model and force_model in self.available_models:
            selected_model = force_model
        else:
            context_length = len(query) + len(context) + len(system_prompt or "")
            selected_model = self.select_best_model(task_type, context_length)
        
        if not selected_model:
            raise ValueError("Nenhum modelo disponível para a tarefa")
        
        # Extrair provedor e modelo
        provider, model_key = selected_model.split(".", 1)
        model_config = self.available_models[selected_model]
        
        # Gerar resposta
        try:
            if provider == "openai":
                response = self._call_openai_api(query, context, system_prompt, model_config)
            elif provider == "anthropic":
                response = self._call_anthropic_api(query, context, system_prompt, model_config)
            elif provider == "google":
                response = self._call_google_api(query, context, system_prompt, model_config)
            elif provider == "deepseek":
                response = self._call_deepseek_api(query, context, system_prompt, model_config)
            else:
                raise ValueError(f"Provedor {provider} não suportado")
            
            # Atualizar estatísticas
            self._update_stats(provider, selected_model, task_type, response.cost, response.processing_time)
            
            return response
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Erro ao gerar resposta: {e}")
            raise

    def _call_openai_api(self, query: str, context: str, system_prompt: Optional[str], model_config: ModelConfig) -> ModelResponse:
        """Chama API da OpenAI"""
        start_time = time.time()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não configurada")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            messages.append({"role": "user", "content": f"Contexto: {context}\n\nPergunta: {query}"})
        else:
            messages.append({"role": "user", "content": query})

        payload = {
            "model": model_config.name,
            "messages": messages,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Calcular custo
        total_tokens = data["usage"]["total_tokens"]
        cost = (total_tokens / 1000) * model_config.cost_per_1k_tokens
        
        processing_time = time.time() - start_time
        
        return ModelResponse(
            content=content,
            model=model_config.name,
            provider="openai",
            usage=data["usage"],
            cost=cost,
            processing_time=processing_time,
            finish_reason=data["choices"][0]["finish_reason"]
        )

    def _call_anthropic_api(self, query: str, context: str, system_prompt: Optional[str], model_config: ModelConfig) -> ModelResponse:
        """Chama API da Anthropic"""
        start_time = time.time()
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY não configurada")

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        # Construir prompt
        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        if context:
            prompt += f"Contexto: {context}\n\n"
        prompt += f"Pergunta: {query}\n\nResposta:"

        payload = {
            "model": model_config.name,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["content"][0]["text"]
        
        # Estimar custo (Anthropic não retorna usage sempre)
        estimated_tokens = len(prompt.split()) + len(content.split())
        cost = (estimated_tokens / 1000) * model_config.cost_per_1k_tokens
        
        processing_time = time.time() - start_time
        
        return ModelResponse(
            content=content,
            model=model_config.name,
            provider="anthropic",
            usage={"total_tokens": estimated_tokens},
            cost=cost,
            processing_time=processing_time
        )

    def _update_stats(self, provider: str, model: str, task_type: TaskType, cost: float, processing_time: float):
        """Atualiza estatísticas de uso"""
        self.stats["total_requests"] += 1
        self.stats["total_cost"] += cost
        
        if provider not in self.stats["provider_usage"]:
            self.stats["provider_usage"][provider] = 0
        self.stats["provider_usage"][provider] += 1
        
        if model not in self.stats["model_usage"]:
            self.stats["model_usage"][model] = 0
        self.stats["model_usage"][model] += 1
        
        if task_type.value not in self.stats["task_distribution"]:
            self.stats["task_distribution"][task_type.value] = 0
        self.stats["task_distribution"][task_type.value] += 1
        
        # Calcular média de tempo de resposta
        total_time = self.stats["average_response_time"] * (self.stats["total_requests"] - 1)
        self.stats["average_response_time"] = (total_time + processing_time) / self.stats["total_requests"]

    def get_available_models(self) -> Dict[str, Any]:
        """Retorna modelos disponíveis"""
        return {
            "total": len(self.available_models),
            "models": {k: {
                "name": v.name,
                "provider": k.split(".")[0],
                "responsibilities": v.responsibilities,
                "context_window": v.context_window,
                "cost_per_1k_tokens": v.cost_per_1k_tokens
            } for k, v in self.available_models.items()},
            "providers": list(set(k.split(".")[0] for k in self.available_models.keys()))
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso"""
        return self.stats

    def _call_google_api(self, query: str, context: str, system_prompt: Optional[str], model_config: ModelConfig) -> ModelResponse:
        """Chama API do Google Gemini"""
        start_time = time.time()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY não configurada")

        # Construir prompt combinado
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Contexto: {context}\n\n"
        full_prompt += f"Pergunta: {query}"

        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": model_config.temperature,
                "maxOutputTokens": model_config.max_tokens,
                "topP": 0.8,
                "topK": 10
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_config.name}:generateContent?key={api_key}"
        
        response = requests.post(
            url,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Estimar custo (Google não fornece tokens exatos sempre)
        estimated_tokens = len(full_prompt.split()) + len(content.split())
        cost = (estimated_tokens / 1000) * model_config.cost_per_1k_tokens
        
        processing_time = time.time() - start_time
        
        return ModelResponse(
            content=content,
            model=model_config.name,
            provider="google",
            usage={"total_tokens": estimated_tokens},
            cost=cost,
            processing_time=processing_time,
            finish_reason=data["candidates"][0].get("finishReason", "stop")
        )

    def _call_deepseek_api(self, query: str, context: str, system_prompt: Optional[str], model_config: ModelConfig) -> ModelResponse:
        """Chama API do DeepSeek"""
        start_time = time.time()
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY não configurada")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            messages.append({"role": "user", "content": f"Contexto: {context}\n\nPergunta: {query}"})
        else:
            messages.append({"role": "user", "content": query})

        payload = {
            "model": model_config.name,
            "messages": messages,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "stream": False
        }

        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Calcular custo
        total_tokens = data["usage"]["total_tokens"]
        cost = (total_tokens / 1000) * model_config.cost_per_1k_tokens
        
        processing_time = time.time() - start_time
        
        return ModelResponse(
            content=content,
            model=model_config.name,
            provider="deepseek",
            usage=data["usage"],
            cost=cost,
            processing_time=processing_time,
            finish_reason=data["choices"][0]["finish_reason"]
        ) 