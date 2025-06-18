"""
Enhanced Corrective RAG - Auto-correção avançada com T5 evaluator e decompose-then-recompose.
Baseado nos papers:
- "Corrective Retrieval Augmented Generation" (2024)
- "T5-based Retrieval Evaluator for RAG Systems" (2024)
- "Decompose-then-Recompose: Enhanced Query Processing" (2024)
"""

import logging
import asyncio
import re
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import time
import aiohttp
import os
from datetime import datetime, timedelta

from src.retrieval.hybrid_retriever import HybridRetriever
from src.models.api_model_router import APIModelRouter
from src.vectordb.qdrant_store import QdrantVectorStore
from src.cache.multi_layer_cache import MultiLayerCache
from src.utils.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Níveis de complexidade da query."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    MULTI_ASPECT = "multi_aspect"


class CorrectionStrategy(Enum):
    """Estratégias de correção."""
    QUERY_EXPANSION = "query_expansion"
    QUERY_REFORMULATION = "query_reformulation"
    DECOMPOSITION = "decomposition"
    SEMANTIC_ENHANCEMENT = "semantic_enhancement"
    CONTEXT_INJECTION = "context_injection"


@dataclass
class QueryComponent:
    """Componente de uma query decomposta."""
    text: str
    aspect: str
    importance: float
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Resultado da avaliação T5."""
    relevance_score: float
    confidence: float
    explanation: str
    categories: List[str] = field(default_factory=list)
    semantic_similarity: float = 0.0
    factual_accuracy: float = 0.0
    completeness: float = 0.0


@dataclass
class EnhancedDocumentWithScore:
    """Documento com score enhanced e metadados extras."""
    content: str
    metadata: Dict
    relevance_score: float
    evaluation_result: EvaluationResult
    validation_status: str = "pending"
    correction_applied: bool = False
    source_component: Optional[str] = None
    rerank_score: float = 0.0


class T5RetrievalEvaluator:
    """
    Avaliador de relevância baseado em metodologia T5 com APIs reais.
    Suporta OpenAI GPT-4, Anthropic Claude e HuggingFace T5.
    """
    
    def __init__(self, 
                 model_router=None,
                 cache: Optional[MultiLayerCache] = None,
                 config: Optional[Dict] = None):
        self.model_router = model_router
        self.cache = cache
        self.config = config or {}
        
        # Configurações da API
        self.api_config = {
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': 'https://api.openai.com/v1',
                'model': 'gpt-4o-mini',
                'max_tokens': 500,
                'temperature': 0.1
            },
            'anthropic': {
                'api_key': os.getenv('ANTHROPIC_API_KEY'),
                'base_url': 'https://api.anthropic.com/v1',
                'model': 'claude-3-haiku-20240307',
                'max_tokens': 500,
                'temperature': 0.1
            },
            'huggingface': {
                'api_key': os.getenv('HUGGINGFACE_API_KEY'),
                'base_url': 'https://api-inference.huggingface.co/models',
                'model': 'google/flan-t5-large',
                'max_tokens': 500
            }
        }
        
        # Provider prioritário (fallback chain)
        self.provider_chain = ['openai', 'anthropic', 'huggingface']
        
        # Circuit breakers para cada provider
        self.circuit_breakers = {
            provider: CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=60,
                expected_exception=Exception
            ) for provider in self.provider_chain
        }
        
        # Métricas
        self.evaluation_stats = {
            'total_evaluations': 0,
            'api_calls_by_provider': {p: 0 for p in self.provider_chain},
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'provider_success_rate': {p: 1.0 for p in self.provider_chain}
        }
    
    async def evaluate_relevance(self, 
                                query: str, 
                                document: str, 
                                context: Optional[str] = None) -> EvaluationResult:
        """
        Avalia relevância usando APIs reais com fallback chain.
        """
        start_time = time.time()
        
        # Verificar cache primeiro
        cache_key = f"t5_eval:{hash(query + document)}"
        if self.cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.evaluation_stats['cache_hits'] += 1
                logger.debug(f"Cache hit para avaliação T5: {cache_key}")
                return EvaluationResult(**cached_result)
        
        # Tentar cada provider na chain
        for provider in self.provider_chain:
            try:
                # Verificar circuit breaker
                if not self.circuit_breakers[provider].can_execute():
                    logger.warning(f"Circuit breaker aberto para {provider}, pulando...")
                    continue
                
                # Fazer avaliação
                result = await self._evaluate_with_provider(
                    provider, query, document, context
                )
                
                if result:
                    # Sucesso - atualizar métricas
                    self.circuit_breakers[provider].record_success()
                    self.evaluation_stats['api_calls_by_provider'][provider] += 1
                    self.evaluation_stats['total_evaluations'] += 1
                    
                    # Cache do resultado
                    if self.cache:
                        await self.cache.set(
                            cache_key, 
                            result.__dict__, 
                            ttl=3600  # 1 hora
                        )
                    
                    # Atualizar tempo médio de resposta
                    response_time = time.time() - start_time
                    self._update_response_time(response_time)
                    
                    logger.info(f"✅ Avaliação T5 realizada com {provider} em {response_time:.2f}s")
                    return result
                    
            except Exception as e:
                # Falha - registrar no circuit breaker
                self.circuit_breakers[provider].record_failure()
                self._update_success_rate(provider, False)
                logger.warning(f"Erro ao avaliar com {provider}: {e}")
                continue
        
        # Se todos os providers falharam, usar fallback local
        logger.warning("Todos os providers falharam, usando avaliação fallback")
        return self._fallback_evaluation(query, document)
    
    async def _evaluate_with_provider(self, 
                                     provider: str, 
                                     query: str, 
                                     document: str, 
                                     context: Optional[str] = None) -> Optional[EvaluationResult]:
        """
        Avalia com um provider específico.
        """
        config = self.api_config[provider]
        
        if provider == 'openai':
            return await self._evaluate_openai(query, document, context, config)
        elif provider == 'anthropic':
            return await self._evaluate_anthropic(query, document, context, config)
        elif provider == 'huggingface':
            return await self._evaluate_huggingface(query, document, context, config)
        else:
            raise ValueError(f"Provider não suportado: {provider}")
    
    async def _evaluate_openai(self, 
                              query: str, 
                              document: str, 
                              context: Optional[str], 
                              config: Dict) -> EvaluationResult:
        """
        Avaliação usando OpenAI GPT-4.
        """
        prompt = self._create_evaluation_prompt(query, document, context)
        
        headers = {
            'Authorization': f'Bearer {config["api_key"]}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': config['model'],
            'messages': [
                {
                    'role': 'system',
                    'content': 'Você é um avaliador de relevância especializado. Responda sempre em JSON válido.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': config['max_tokens'],
            'temperature': config['temperature'],
            'response_format': {'type': 'json_object'}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise Exception(f"OpenAI API error: {response.status}")
                
                data = await response.json()
                content = data['choices'][0]['message']['content']
                
                return self._parse_evaluation_response(content, 'openai')
    
    async def _evaluate_anthropic(self, 
                                 query: str, 
                                 document: str, 
                                 context: Optional[str], 
                                 config: Dict) -> EvaluationResult:
        """
        Avaliação usando Anthropic Claude.
        """
        prompt = self._create_evaluation_prompt(query, document, context)
        
        headers = {
            'x-api-key': config['api_key'],
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': config['model'],
            'max_tokens': config['max_tokens'],
            'temperature': config['temperature'],
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Anthropic API error: {response.status}")
                
                data = await response.json()
                content = data['content'][0]['text']
                
                return self._parse_evaluation_response(content, 'anthropic')
    
    async def _evaluate_huggingface(self, 
                                   query: str, 
                                   document: str, 
                                   context: Optional[str], 
                                   config: Dict) -> EvaluationResult:
        """
        Avaliação usando HuggingFace T5.
        """
        # Para T5, usamos um prompt mais direto
        prompt = f"""
        Avaliar relevância do documento para a query.
        Query: {query}
        Documento: {document[:500]}...
        
        Responder em JSON: {{"overall_score": 0.0-1.0, "reasoning": "explicação"}}
        """
        
        headers = {
            'Authorization': f'Bearer {config["api_key"]}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': config['max_tokens'],
                'return_full_text': False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/{config['model']}",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise Exception(f"HuggingFace API error: {response.status}")
                
                data = await response.json()
                
                # HuggingFace retorna formato diferente
                if isinstance(data, list) and len(data) > 0:
                    content = data[0].get('generated_text', '')
                else:
                    content = str(data)
                
                return self._parse_evaluation_response(content, 'huggingface')
    
    def _create_evaluation_prompt(self, 
                                 query: str, 
                                 document: str, 
                                 context: Optional[str] = None) -> str:
        """
        Cria prompt estruturado para avaliação T5.
        """
        context_part = f"\nContexto adicional: {context}" if context else ""
        
        return f"""
        Avalie a relevância do documento para responder à query fornecida. 
        Use a metodologia T5 (Text-to-Text Transfer Transformer) para análise estruturada.

        QUERY: {query}

        DOCUMENTO: {document[:1000]}{"..." if len(document) > 1000 else ""}{context_part}

        INSTRUÇÕES:
        1. Analise a relevância semântica entre query e documento
        2. Avalie a precisão factual do conteúdo
        3. Determine a completude da informação
        4. Calcule o nível de confiança na avaliação
        5. Forneça um score geral (0.0-1.0)

        RESPONDA EM JSON VÁLIDO:
        {{
            "semantic_relevance": 0.0,
            "factual_accuracy": 0.0,
            "completeness": 0.0,
            "confidence": 0.0,
            "overall_score": 0.0,
            "reasoning": "Explicação detalhada da avaliação em 1-2 frases"
        }}
        """
    
    def _parse_evaluation_response(self, 
                                  response: str, 
                                  provider: str) -> EvaluationResult:
        """
        Parse da resposta da API com fallback robusto.
        """
        try:
            # Tentar extrair JSON da resposta
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                return EvaluationResult(
                    relevance_score=float(data.get('semantic_relevance', 0.0)),
                    confidence=float(data.get('confidence', 0.0)),
                    explanation=data.get('reasoning', ''),
                    categories=data.get('categories', []),
                    semantic_similarity=float(data.get('semantic_relevance', 0.0)),
                    factual_accuracy=float(data.get('factual_accuracy', 0.0)),
                    completeness=float(data.get('completeness', 0.0)),
                    overall_score=float(data.get('overall_score', 0.0)),
                    api_provider=provider
                )
        except Exception as e:
            logger.warning(f"Erro ao parsear resposta de {provider}: {e}")
        
        # Fallback com regex
        return self._regex_fallback_parse(response, provider)
    
    def _regex_fallback_parse(self, response: str, provider: str) -> EvaluationResult:
        """
        Parse usando regex como fallback.
        """
        try:
            # Buscar scores numéricos na resposta
            score_patterns = [
                r'overall_score["\']?\s*:\s*([0-9.]+)',
                r'score["\']?\s*:\s*([0-9.]+)',
                r'relevance["\']?\s*:\s*([0-9.]+)',
                r'([0-9.]+)/10',
                r'([0-9.]+)\s*out\s*of\s*10'
            ]
            
            overall_score = 0.5  # Default moderado
            
            for pattern in score_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if score > 1.0:  # Assumir escala 0-10
                        score = score / 10.0
                    overall_score = max(0.0, min(1.0, score))
                    break
            
            return EvaluationResult(
                relevance_score=overall_score,
                confidence=0.7,  # Confiança moderada para fallback
                explanation=f"Parsed from {provider} response using regex fallback",
                semantic_similarity=overall_score,
                factual_accuracy=overall_score,
                completeness=overall_score,
                overall_score=overall_score,
                api_provider=provider
            )
            
        except Exception:
            # Ultimate fallback
            return self._fallback_evaluation("", "")
    
    def _fallback_evaluation(self, query: str, document: str) -> EvaluationResult:
        """
        Avaliação local simples quando APIs falham.
        """
        # Cálculo simples baseado em palavras-chave
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        overlap = len(query_words.intersection(doc_words))
        total_words = len(query_words)
        
        if total_words == 0:
            relevance = 0.0
        else:
            relevance = min(1.0, overlap / total_words)
        
        return EvaluationResult(
            relevance_score=relevance,
            confidence=0.3,  # Baixa confiança para fallback
            explanation="Fallback evaluation based on keyword overlap",
            semantic_similarity=relevance,
            factual_accuracy=relevance,
            completeness=relevance,
            overall_score=relevance,
            api_provider="local_fallback"
        )
    
    def _update_response_time(self, response_time: float):
        """Atualiza tempo médio de resposta."""
        current_avg = self.evaluation_stats['avg_response_time']
        total_evals = self.evaluation_stats['total_evaluations']
        
        if total_evals > 0:
            self.evaluation_stats['avg_response_time'] = (
                (current_avg * (total_evals - 1) + response_time) / total_evals
            )
        else:
            self.evaluation_stats['avg_response_time'] = response_time
    
    def _update_success_rate(self, provider: str, success: bool):
        """Atualiza taxa de sucesso do provider."""
        current_rate = self.evaluation_stats['provider_success_rate'][provider]
        total_calls = self.evaluation_stats['api_calls_by_provider'][provider]
        
        if total_calls > 0:
            new_rate = (current_rate * total_calls + (1.0 if success else 0.0)) / (total_calls + 1)
            self.evaluation_stats['provider_success_rate'][provider] = new_rate


class QueryDecomposer:
    """
    Implementa algoritmo decompose-then-recompose para queries complexas.
    """
    
    def __init__(self, model_router: APIModelRouter):
        self.model_router = model_router
        
    async def analyze_complexity(self, query: str) -> QueryComplexity:
        """Analisa complexidade da query."""
        
        # Análise básica por enquanto
        word_count = len(query.split())
        question_marks = query.count('?')
        and_or_count = len([w for w in query.lower().split() if w in ['and', 'or', 'but', 'also']])
        
        if word_count < 5:
            return QueryComplexity.SIMPLE
        elif word_count < 15 and and_or_count < 2:
            return QueryComplexity.MEDIUM
        elif and_or_count >= 2 or question_marks > 1:
            return QueryComplexity.MULTI_ASPECT
        else:
            return QueryComplexity.COMPLEX
    
    async def decompose_query(self, query: str) -> List[QueryComponent]:
        """
        Decompõe query complexa em componentes menores.
        
        Args:
            query: Query original
            
        Returns:
            Lista de componentes da query
        """
        decomposition_prompt = f"""
        Decompose this complex query into smaller, focused components:
        
        Query: {query}
        
        For each component, identify:
        1. The specific aspect/question
        2. Importance (0.0-1.0)
        3. Dependencies on other components
        4. Search strategy needed
        
        Format as JSON:
        {{
            "components": [
                {{
                    "text": "specific component query",
                    "aspect": "aspect name",
                    "importance": 0.8,
                    "dependencies": ["other_aspect"],
                    "metadata": {{"strategy": "semantic/keyword"}}
                }},
                ...
            ]
        }}
        """
        
        try:
            response = await self.model_router.route_request(
                decomposition_prompt,
                task_type="decomposition"
            )
            
            # Parse JSON response
            result = json.loads(response.get("answer", "{}"))
            components = []
            
            for comp_data in result.get("components", []):
                component = QueryComponent(
                    text=comp_data.get("text", ""),
                    aspect=comp_data.get("aspect", ""),
                    importance=comp_data.get("importance", 0.5),
                    dependencies=comp_data.get("dependencies", []),
                    metadata=comp_data.get("metadata", {})
                )
                components.append(component)
            
            return components
            
        except Exception as e:
            logger.error(f"Erro na decomposição: {e}")
            # Fallback: criar componente único
            return [QueryComponent(
                text=query,
                aspect="main",
                importance=1.0,
                dependencies=[],
                metadata={"strategy": "hybrid"}
            )]
    
    async def recompose_results(self, 
                               original_query: str,
                               component_results: Dict[str, List[EnhancedDocumentWithScore]]) -> List[EnhancedDocumentWithScore]:
        """
        Recompõe resultados dos componentes em resposta final.
        
        Args:
            original_query: Query original
            component_results: Resultados por componente
            
        Returns:
            Lista recomposta e ranqueada de documentos
        """
        # 1. Coletar todos os documentos únicos
        all_docs = {}
        doc_components = defaultdict(list)
        
        for component_name, docs in component_results.items():
            for doc in docs:
                doc_id = hash(doc.content[:200])
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc
                doc_components[doc_id].append(component_name)
        
        # 2. Calcular scores combinados
        recomposed_docs = []
        for doc_id, doc in all_docs.items():
            # Score baseado em múltiplos componentes
            components = doc_components[doc_id]
            
            # Boost para documentos que atendem múltiplos componentes
            multi_component_boost = 1.0 + (len(components) - 1) * 0.2
            
            # Score final combinado
            final_score = doc.relevance_score * multi_component_boost
            
            # Atualizar documento
            doc.rerank_score = final_score
            doc.metadata.update({
                "matching_components": components,
                "multi_component_boost": multi_component_boost
            })
            
            recomposed_docs.append(doc)
        
        # 3. Re-ranquear baseado no context da query original
        reranked_docs = await self._contextual_rerank(original_query, recomposed_docs)
        
        # 4. Ordenar por score final
        reranked_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return reranked_docs
    
    async def _contextual_rerank(self, 
                                original_query: str, 
                                docs: List[EnhancedDocumentWithScore]) -> List[EnhancedDocumentWithScore]:
        """Re-ranqueia documentos considerando context da query original."""
        
        # Para queries simples, manter ordem atual
        if len(docs) <= 3:
            return docs
        
        # Preparar prompt para re-ranking contextual
        doc_summaries = []
        for i, doc in enumerate(docs[:10]):  # Top 10 para re-ranking
            doc_summaries.append(f"{i+1}. {doc.content[:150]}... (score: {doc.rerank_score:.3f})")
        
        rerank_prompt = f"""
        Given the original query and these retrieved documents, reorder them by relevance:
        
        Original Query: {original_query}
        
        Documents:
        {chr(10).join(doc_summaries)}
        
        Respond with just the reordered numbers (e.g., "3,1,5,2,4,...")
        Consider: semantic relevance, completeness, and overall utility for answering the query.
        """
        
        try:
            response = await self.model_router.route_request(
                rerank_prompt,
                task_type="reranking",
                force_model="openai.gpt35_turbo"
            )
            
            # Parse order
            order_str = response.get("answer", "").strip()
            order_nums = [int(x.strip()) - 1 for x in order_str.split(",") if x.strip().isdigit()]
            
            # Aplicar nova ordem
            reranked = []
            used_indices = set()
            
            for idx in order_nums:
                if 0 <= idx < len(docs) and idx not in used_indices:
                    # Boost baseado na nova posição
                    position_boost = 1.0 - (len(reranked) * 0.05)  # Decréscimo gradual
                    docs[idx].rerank_score *= position_boost
                    reranked.append(docs[idx])
                    used_indices.add(idx)
            
            # Adicionar documentos restantes
            for i, doc in enumerate(docs):
                if i not in used_indices:
                    reranked.append(doc)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Erro no re-ranking contextual: {e}")
            return docs


class EnhancedCorrectiveRAG:
    """
    Corrective RAG Enhanced com T5 evaluator e decompose-then-recompose.
    """
    
    def __init__(self, 
                 retriever: Optional[HybridRetriever] = None,
                 relevance_threshold: float = 0.75,
                 max_reformulation_attempts: int = 3,
                 enable_decomposition: bool = True):
        self.retriever = retriever or HybridRetriever()
        self.model_router = APIModelRouter({})
        self.relevance_threshold = relevance_threshold
        self.max_reformulation_attempts = max_reformulation_attempts
        self.enable_decomposition = enable_decomposition
        
        # Componentes enhanced
        self.t5_evaluator = T5RetrievalEvaluator(self.model_router)
        self.query_decomposer = QueryDecomposer(self.model_router)
        
        # Métricas
        self.correction_stats = {
            "total_queries": 0,
            "corrections_applied": 0,
            "decompositions_used": 0,
            "avg_relevance_improvement": 0.0
        }
    
    async def retrieve_and_correct(self, 
                                   query: str, 
                                   k: int = 10,
                                   use_decomposition: Optional[bool] = None) -> Dict:
        """
        Retrieval enhanced com correção automática.
        
        Args:
            query: Query do usuário
            k: Número de documentos finais
            use_decomposition: Forçar uso/não-uso de decomposição
            
        Returns:
            Dict com resultados enhanced
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Enhanced retrieval iniciado para query: {query[:50]}...")
        
        self.correction_stats["total_queries"] += 1
        
        # 1. Análise de complexidade
        complexity = await self.query_decomposer.analyze_complexity(query)
        
        # 2. Decidir estratégia baseada na complexidade
        should_decompose = (use_decomposition if use_decomposition is not None 
                           else (self.enable_decomposition and 
                                complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_ASPECT]))
        
        if should_decompose:
            # Estratégia de decomposição
            results = await self._decompose_and_retrieve(query, k, complexity)
            self.correction_stats["decompositions_used"] += 1
        else:
            # Estratégia tradicional enhanced
            results = await self._traditional_enhanced_retrieve(query, k)
        
        # 3. Calcular métricas finais
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # 4. Atualizar estatísticas
        if results.get("correction_applied", False):
            self.correction_stats["corrections_applied"] += 1
        
        # 5. Adicionar metadados de performance
        results.update({
            "processing_time": processing_time,
            "complexity": complexity.value,
            "strategy_used": "decomposition" if should_decompose else "traditional_enhanced",
            "correction_stats": self.correction_stats.copy()
        })
        
        logger.info(f"Enhanced retrieval concluído em {processing_time:.3f}s")
        
        return results
    
    async def _decompose_and_retrieve(self, 
                                     query: str, 
                                     k: int, 
                                     complexity: QueryComplexity) -> Dict:
        """Estratégia de decomposição para queries complexas."""
        
        # 1. Decompor query
        components = await self.query_decomposer.decompose_query(query)
        
        logger.info(f"Query decomposta em {len(components)} componentes")
        
        # 2. Recuperar para cada componente em paralelo
        component_tasks = []
        for component in components:
            task = self._retrieve_for_component(component, k)
            component_tasks.append(task)
        
        component_results = await asyncio.gather(*component_tasks, return_exceptions=True)
        
        # 3. Processar resultados
        valid_results = {}
        for i, result in enumerate(component_results):
            if not isinstance(result, Exception):
                component_name = components[i].aspect
                valid_results[component_name] = result
        
        # 4. Recompor resultados
        if valid_results:
            final_docs = await self.query_decomposer.recompose_results(query, valid_results)
        else:
            # Fallback para estratégia tradicional
            return await self._traditional_enhanced_retrieve(query, k)
        
        # 5. Preparar resposta
        avg_relevance = np.mean([doc.rerank_score for doc in final_docs]) if final_docs else 0
        
        return {
            "documents": final_docs[:k],
            "original_query": query,
            "decomposition_used": True,
            "num_components": len(components),
            "component_results": {name: len(docs) for name, docs in valid_results.items()},
            "avg_relevance_score": avg_relevance,
            "correction_applied": True,  # Decomposição conta como correção
            "total_evaluated": sum(len(docs) for docs in valid_results.values()),
            "total_relevant": len([doc for doc in final_docs if doc.validation_status == "relevant"])
        }
    
    async def _retrieve_for_component(self, component: QueryComponent, k: int) -> List[EnhancedDocumentWithScore]:
        """Recupera documentos para um componente específico."""
        try:
            # Recuperar documentos brutos
            raw_docs = await self.retriever.retrieve(component.text, limit=k * 2)
            
            # Avaliar com T5
            enhanced_docs = []
            for doc in raw_docs:
                evaluation = await self.t5_evaluator.evaluate_relevance(
                    component.text, 
                    doc.get("content", ""),
                    context={"component_aspect": component.aspect, "importance": component.importance}
                )
                
                enhanced_doc = EnhancedDocumentWithScore(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    relevance_score=evaluation.relevance_score,
                    evaluation_result=evaluation,
                    validation_status="relevant" if evaluation.relevance_score >= self.relevance_threshold else "irrelevant",
                    source_component=component.aspect
                )
                
                enhanced_docs.append(enhanced_doc)
            
            # Filtrar relevantes e ordenar
            relevant_docs = [doc for doc in enhanced_docs if doc.validation_status == "relevant"]
            relevant_docs.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return relevant_docs[:k]
            
        except Exception as e:
            logger.error(f"Erro ao recuperar para componente {component.aspect}: {e}")
            return []
    
    async def _traditional_enhanced_retrieve(self, query: str, k: int) -> Dict:
        """Estratégia tradicional enhanced com T5 evaluator."""
        
        reformulation_count = 0
        original_query = query
        current_query = query
        
        while reformulation_count <= self.max_reformulation_attempts:
            # 1. Recuperar documentos
            raw_docs = await self.retriever.retrieve(current_query, limit=k * 2)
            
            # 2. Avaliar com T5
            enhanced_docs = []
            evaluation_tasks = []
            
            for doc in raw_docs:
                task = self.t5_evaluator.evaluate_relevance(current_query, doc.get("content", ""))
                evaluation_tasks.append(task)
            
            evaluations = await asyncio.gather(*evaluation_tasks)
            
            for doc, evaluation in zip(raw_docs, evaluations):
                enhanced_doc = EnhancedDocumentWithScore(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    relevance_score=evaluation.relevance_score,
                    evaluation_result=evaluation,
                    validation_status="relevant" if evaluation.relevance_score >= self.relevance_threshold else "irrelevant",
                    correction_applied=reformulation_count > 0
                )
                enhanced_docs.append(enhanced_doc)
            
            # 3. Verificar qualidade
            relevant_docs = [doc for doc in enhanced_docs if doc.validation_status == "relevant"]
            
            if relevant_docs:
                avg_relevance = np.mean([doc.relevance_score for doc in relevant_docs])
                
                if avg_relevance >= self.relevance_threshold or reformulation_count >= self.max_reformulation_attempts:
                    # Sucesso ou máximo de tentativas
                    relevant_docs.sort(key=lambda x: x.relevance_score, reverse=True)
                    
                    return {
                        "documents": relevant_docs[:k],
                        "original_query": original_query,
                        "final_query": current_query,
                        "reformulation_count": reformulation_count,
                        "avg_relevance_score": avg_relevance,
                        "correction_applied": reformulation_count > 0,
                        "total_evaluated": len(enhanced_docs),
                        "total_relevant": len(relevant_docs)
                    }
            
            # 4. Reformular se necessário
            if reformulation_count < self.max_reformulation_attempts:
                current_query = await self._enhanced_reformulate_query(original_query, current_query, enhanced_docs)
                reformulation_count += 1
            else:
                break
        
        # Fallback: retornar melhores documentos mesmo com baixa relevância
        enhanced_docs.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return {
            "documents": enhanced_docs[:k],
            "original_query": original_query,
            "final_query": current_query,
            "reformulation_count": reformulation_count,
            "avg_relevance_score": np.mean([doc.relevance_score for doc in enhanced_docs]) if enhanced_docs else 0,
            "correction_applied": reformulation_count > 0,
            "total_evaluated": len(enhanced_docs),
            "total_relevant": len([doc for doc in enhanced_docs if doc.validation_status == "relevant"]),
            "fallback_used": True
        }
    
    async def _enhanced_reformulate_query(self, 
                                         original_query: str, 
                                         current_query: str, 
                                         low_relevance_docs: List[EnhancedDocumentWithScore]) -> str:
        """Reformulação enhanced baseada em análise T5."""
        
        # Analisar feedback do T5 evaluator
        evaluation_feedback = []
        for doc in low_relevance_docs[:3]:
            eval_result = doc.evaluation_result
            feedback = f"Doc relevance: {eval_result.relevance_score:.2f}, "
            feedback += f"Completeness: {eval_result.completeness:.2f}, "
            feedback += f"Explanation: {eval_result.explanation[:100]}..."
            evaluation_feedback.append(feedback)
        
        reformulation_prompt = f"""
        ADVANCED QUERY REFORMULATION
        
        Original Query: {original_query}
        Current Query: {current_query}
        
        T5 Evaluator Feedback:
        {chr(10).join(evaluation_feedback)}
        
        The current query is not retrieving sufficiently relevant documents.
        Based on the T5 evaluator feedback, reformulate the query to:
        
        1. Address completeness gaps identified
        2. Improve semantic alignment
        3. Add specific technical terms if needed
        4. Clarify ambiguous aspects
        5. Expand context where necessary
        
        Consider multiple reformulation strategies:
        - Query expansion with synonyms
        - Technical term addition
        - Context clarification
        - Aspect specification
        
        Reformed Query:
        """
        
        try:
            response = await self.model_router.route_request(
                reformulation_prompt,
                task_type="enhanced_reformulation"
            )
            
            reformulated = response.get("answer", "").strip()
            
            # Validação da reformulação
            if not reformulated or reformulated == current_query:
                # Fallback: adicionar contexto técnico
                reformulated = f"{original_query} technical implementation details examples"
            
            return reformulated
            
        except Exception as e:
            logger.error(f"Erro na reformulação enhanced: {e}")
            return f"{current_query} with practical examples and detailed explanation"
    
    def get_correction_stats(self) -> Dict:
        """Retorna estatísticas de correção."""
        stats = self.correction_stats.copy()
        if stats["total_queries"] > 0:
            stats["correction_rate"] = stats["corrections_applied"] / stats["total_queries"]
            stats["decomposition_rate"] = stats["decompositions_used"] / stats["total_queries"]
        else:
            stats["correction_rate"] = 0.0
            stats["decomposition_rate"] = 0.0
        
        return stats
    
    async def evaluate_system_performance(self, test_queries: List[str]) -> Dict:
        """Avalia performance do sistema com conjunto de queries de teste."""
        
        results = {
            "total_queries": len(test_queries),
            "avg_processing_time": 0.0,
            "avg_relevance_score": 0.0,
            "correction_rate": 0.0,
            "decomposition_rate": 0.0,
            "detailed_results": []
        }
        
        processing_times = []
        relevance_scores = []
        corrections_count = 0
        decompositions_count = 0
        
        for query in test_queries:
            start_time = asyncio.get_event_loop().time()
            
            result = await self.retrieve_and_correct(query, k=5)
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            processing_times.append(processing_time)
            relevance_scores.append(result.get("avg_relevance_score", 0))
            
            if result.get("correction_applied", False):
                corrections_count += 1
            
            if result.get("decomposition_used", False):
                decompositions_count += 1
            
            results["detailed_results"].append({
                "query": query,
                "processing_time": processing_time,
                "relevance_score": result.get("avg_relevance_score", 0),
                "num_results": len(result.get("documents", [])),
                "correction_applied": result.get("correction_applied", False),
                "decomposition_used": result.get("decomposition_used", False)
            })
        
        # Calcular médias
        results["avg_processing_time"] = np.mean(processing_times)
        results["avg_relevance_score"] = np.mean(relevance_scores)
        results["correction_rate"] = corrections_count / len(test_queries)
        results["decomposition_rate"] = decompositions_count / len(test_queries)
        
        return results


# Função de factory para facilitar uso
def create_enhanced_corrective_rag(config: Dict = None) -> EnhancedCorrectiveRAG:
    """
    Factory para criar Enhanced Corrective RAG com configurações.
    
    Args:
        config: Configurações do sistema
        
    Returns:
        Instância configurada do EnhancedCorrectiveRAG
    """
    if config is None:
        config = {}
    
    # Configurações padrão
    default_config = {
        'relevance_threshold': 0.75,
        'max_reformulation_attempts': 3,
        'enable_decomposition': True,
        'cache_evaluations': True,
        'api_providers': ['openai', 'anthropic', 'huggingface']
    }
    
    # Mesclar configurações
    merged_config = {**default_config, **config}
    
    # Criar cache se especificado
    cache = None
    if merged_config.get('cache_evaluations', True):
        cache_config = merged_config.get('cache')
        if cache_config:
            # Se cache_config é uma instância, usar diretamente
            if hasattr(cache_config, 'get') and hasattr(cache_config, 'set'):
                cache = cache_config
            else:
                # Se é configuração, criar nova instância
                from src.cache.multi_layer_cache import create_multi_layer_cache
                cache = create_multi_layer_cache(cache_config)
        else:
            # Configuração padrão do cache
            from src.cache.multi_layer_cache import create_multi_layer_cache
            default_cache_config = {
                'enable_l1': True,
                'enable_l2': True,
                'enable_l3': True,
                'l1_max_size': 1000,
                'redis_host': os.getenv('REDIS_HOST', 'localhost'),
                'redis_port': int(os.getenv('REDIS_PORT', '6379')),
                'redis_db': int(os.getenv('REDIS_DB', '1')),
                'redis_password': os.getenv('REDIS_PASSWORD'),
                'sqlite_path': 'cache/enhanced_rag_evaluations.db',
                'default_ttl': 3600
            }
            cache = create_multi_layer_cache(default_cache_config)
    
    # Criar modelo router se não fornecido
    model_router = merged_config.get('model_router')
    if model_router is None:
        try:
            from src.models.api_model_router import APIModelRouter
            model_router = APIModelRouter()
        except ImportError:
            logger.warning("APIModelRouter não disponível, usando None")
            model_router = None
    
    # Criar retriever se não fornecido
    retriever = merged_config.get('retriever')
    if retriever is None:
        try:
            from src.retrieval.hybrid_retriever import HybridRetriever
            retriever = HybridRetriever()
        except ImportError:
            logger.warning("HybridRetriever não disponível, usando None")
            retriever = None
    
    # Criar instância do Enhanced Corrective RAG
    enhanced_rag = EnhancedCorrectiveRAG(
        retriever=retriever,
        relevance_threshold=merged_config['relevance_threshold'],
        max_reformulation_attempts=merged_config['max_reformulation_attempts'],
        enable_decomposition=merged_config['enable_decomposition']
    )
    
    # Configurar T5 Evaluator com cache
    if enhanced_rag.t5_evaluator:
        enhanced_rag.t5_evaluator.cache = cache
        enhanced_rag.t5_evaluator.model_router = model_router
        
        # Configurar providers da API
        api_providers = merged_config.get('api_providers', ['openai'])
        enhanced_rag.t5_evaluator.provider_chain = api_providers
    
    # Configurar Query Decomposer
    if enhanced_rag.query_decomposer:
        enhanced_rag.query_decomposer.model_router = model_router
    
    logger.info(f"✅ Enhanced Corrective RAG criado com:")
    logger.info(f"   Threshold: {merged_config['relevance_threshold']}")
    logger.info(f"   Decomposition: {merged_config['enable_decomposition']}")
    logger.info(f"   Cache: {'habilitado' if cache else 'desabilitado'}")
    logger.info(f"   API providers: {merged_config.get('api_providers', ['openai'])}")
    
    return enhanced_rag