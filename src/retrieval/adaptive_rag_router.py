"""
Adaptive RAG Router - Sistema de Roteamento Inteligente
Classifica complexidade de queries e roteia dinamicamente entre estratégias
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Níveis de complexidade de query"""
    SIMPLE = "simple"              # Factual, resposta direta
    SINGLE_HOP = "single_hop"      # Requer uma conexão
    MULTI_HOP = "multi_hop"        # Requer múltiplas conexões
    COMPLEX = "complex"            # Análise profunda, síntese
    AMBIGUOUS = "ambiguous"        # Precisa clarificação


class RetrievalStrategy(Enum):
    """Estratégias de retrieval baseadas em complexidade"""
    DIRECT = "direct"              # Busca direta simples
    STANDARD = "standard"          # RAG padrão
    MULTI_QUERY = "multi_query"    # Múltiplas queries
    CORRECTIVE = "corrective"      # Com correção
    GRAPH_ENHANCED = "graph"       # Com grafo de conhecimento
    MULTI_HEAD = "multi_head"      # Multi-head attention
    HYBRID = "hybrid"              # Combinação de métodos


@dataclass
class QueryAnalysis:
    """Resultado da análise de query"""
    query: str
    complexity: QueryComplexity
    confidence: float
    reasoning_type: str  # factual, analytical, comparative, etc.
    key_entities: List[str]
    temporal_aspect: bool
    requires_context: bool
    ambiguity_score: float
    suggested_strategies: List[RetrievalStrategy]
    estimated_hops: int


class QueryComplexityClassifier:
    """Classificador de complexidade de queries"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.classifier = None
        self._initialize_classifier()
        
        # Padrões heurísticos (fallback)
        self.complexity_patterns = {
            QueryComplexity.SIMPLE: [
                "what is", "define", "who is", "when was",
                "qual é", "defina", "quem é", "quando foi"
            ],
            QueryComplexity.SINGLE_HOP: [
                "how does", "why does", "what causes",
                "como funciona", "por que", "o que causa"
            ],
            QueryComplexity.MULTI_HOP: [
                "compare", "contrast", "relationship between",
                "compare", "contraste", "relação entre"
            ],
            QueryComplexity.COMPLEX: [
                "analyze", "evaluate", "implications of",
                "analise", "avalie", "implicações de"
            ]
        }
    
    def _initialize_classifier(self):
        """Inicializa classificador (implementação simplificada)"""
        try:
            # Em produção, usar modelo treinado
            logger.info("Classificador de complexidade inicializado (modo heurístico)")
        except Exception as e:
            logger.warning(f"Usando classificador heurístico: {e}")
    
    async def classify(self, query: str) -> QueryAnalysis:
        """Classifica complexidade da query"""
        
        # Análise básica
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Detectar entidades (simplificado)
        key_entities = self._extract_entities(query)
        
        # Aspectos temporais
        temporal_aspect = any(word in query_lower for word in 
                            ["when", "before", "after", "during", "history", 
                             "quando", "antes", "depois", "durante", "história"])
        
        # Score de ambiguidade
        ambiguity_score = self._calculate_ambiguity(query)
        
        # Classificar complexidade
        complexity, confidence = self._classify_complexity(
            query_lower, word_count, key_entities
        )
        
        # Estimar número de hops
        estimated_hops = self._estimate_hops(complexity, key_entities)
        
        # Sugerir estratégias
        suggested_strategies = self._suggest_strategies(
            complexity, temporal_aspect, ambiguity_score
        )
        
        # Tipo de raciocínio
        reasoning_type = self._identify_reasoning_type(query_lower)
        
        return QueryAnalysis(
            query=query,
            complexity=complexity,
            confidence=confidence,
            reasoning_type=reasoning_type,
            key_entities=key_entities,
            temporal_aspect=temporal_aspect,
            requires_context=complexity != QueryComplexity.SIMPLE,
            ambiguity_score=ambiguity_score,
            suggested_strategies=suggested_strategies,
            estimated_hops=estimated_hops
        )
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extrai entidades principais (simplificado)"""
        # Em produção, usar NER
        words = query.split()
        # Palavras capitalizadas como proxy para entidades
        entities = [w for w in words if w and w[0].isupper()]
        return entities[:5]  # Limitar a 5 entidades
    
    def _calculate_ambiguity(self, query: str) -> float:
        """Calcula score de ambiguidade"""
        ambiguous_terms = ["it", "this", "that", "they", "isso", "este", "aquele"]
        query_lower = query.lower()
        
        ambiguity_count = sum(1 for term in ambiguous_terms if term in query_lower)
        word_count = len(query.split())
        
        return min(ambiguity_count / max(word_count, 1), 1.0)
    
    def _classify_complexity(self, 
                           query_lower: str, 
                           word_count: int,
                           entities: List[str]) -> Tuple[QueryComplexity, float]:
        """Classifica complexidade baseado em heurísticas"""
        
        # Verificar padrões simples primeiro
        for pattern_list in self.complexity_patterns[QueryComplexity.SIMPLE]:
            if pattern_list in query_lower:
                return QueryComplexity.SIMPLE, 0.8
        
        # Queries muito curtas geralmente são simples
        if word_count < 5:
            return QueryComplexity.SIMPLE, 0.7
        
        # Verificar padrões complexos
        if any(pattern in query_lower for pattern in self.complexity_patterns[QueryComplexity.COMPLEX]):
            return QueryComplexity.COMPLEX, 0.75
        
        # Multi-hop se menciona múltiplas entidades
        if len(entities) >= 3:
            return QueryComplexity.MULTI_HOP, 0.7
        
        # Verificar single-hop
        if any(pattern in query_lower for pattern in self.complexity_patterns[QueryComplexity.SINGLE_HOP]):
            return QueryComplexity.SINGLE_HOP, 0.75
        
        # Default para single-hop com confiança média
        return QueryComplexity.SINGLE_HOP, 0.5
    
    def _estimate_hops(self, complexity: QueryComplexity, entities: List[str]) -> int:
        """Estima número de hops necessários"""
        
        base_hops = {
            QueryComplexity.SIMPLE: 0,
            QueryComplexity.SINGLE_HOP: 1,
            QueryComplexity.MULTI_HOP: 2,
            QueryComplexity.COMPLEX: 3,
            QueryComplexity.AMBIGUOUS: 1
        }
        
        # Adicionar hops baseado em entidades
        entity_hops = max(0, len(entities) - 2)
        
        return base_hops.get(complexity, 1) + entity_hops
    
    def _suggest_strategies(self,
                           complexity: QueryComplexity,
                           temporal: bool,
                           ambiguity: float) -> List[RetrievalStrategy]:
        """Sugere estratégias de retrieval baseadas na análise"""
        
        strategies = []
        
        # Mapeamento base
        if complexity == QueryComplexity.SIMPLE:
            strategies = [RetrievalStrategy.DIRECT, RetrievalStrategy.STANDARD]
        elif complexity == QueryComplexity.SINGLE_HOP:
            strategies = [RetrievalStrategy.STANDARD, RetrievalStrategy.MULTI_QUERY]
        elif complexity == QueryComplexity.MULTI_HOP:
            strategies = [RetrievalStrategy.GRAPH_ENHANCED, RetrievalStrategy.MULTI_HEAD]
        elif complexity == QueryComplexity.COMPLEX:
            strategies = [RetrievalStrategy.HYBRID, RetrievalStrategy.MULTI_HEAD, 
                         RetrievalStrategy.CORRECTIVE]
        
        # Ajustes baseados em características
        if temporal and RetrievalStrategy.GRAPH_ENHANCED not in strategies:
            strategies.append(RetrievalStrategy.GRAPH_ENHANCED)
        
        if ambiguity > 0.3 and RetrievalStrategy.CORRECTIVE not in strategies:
            strategies.append(RetrievalStrategy.CORRECTIVE)
        
        return strategies[:3]  # Limitar a 3 estratégias
    
    def _identify_reasoning_type(self, query_lower: str) -> str:
        """Identifica tipo de raciocínio necessário"""
        
        if any(word in query_lower for word in ["compare", "versus", "difference"]):
            return "comparative"
        elif any(word in query_lower for word in ["analyze", "evaluate", "assess"]):
            return "analytical"
        elif any(word in query_lower for word in ["how", "why", "explain"]):
            return "explanatory"
        elif any(word in query_lower for word in ["what", "who", "when", "where"]):
            return "factual"
        else:
            return "general"


class AdaptiveRAGRouter:
    """
    Router adaptativo que seleciona estratégia RAG baseado em complexidade
    """
    
    def __init__(self, 
                 rag_components: Dict[str, Any],
                 optimization_objective: str = "balanced"):
        """
        Args:
            rag_components: Dicionário com componentes RAG disponíveis
            optimization_objective: balanced, speed, accuracy, cost
        """
        
        self.components = rag_components
        self.optimization = optimization_objective
        self.classifier = QueryComplexityClassifier()
        
        # Configurações de roteamento
        self.routing_config = {
            QueryComplexity.SIMPLE: {
                "strategies": [RetrievalStrategy.DIRECT],
                "k": 3,
                "rerank": False,
                "max_time": 2.0
            },
            QueryComplexity.SINGLE_HOP: {
                "strategies": [RetrievalStrategy.STANDARD, RetrievalStrategy.MULTI_QUERY],
                "k": 5,
                "rerank": True,
                "max_time": 5.0
            },
            QueryComplexity.MULTI_HOP: {
                "strategies": [RetrievalStrategy.GRAPH_ENHANCED, RetrievalStrategy.MULTI_HEAD],
                "k": 8,
                "rerank": True,
                "max_time": 10.0
            },
            QueryComplexity.COMPLEX: {
                "strategies": [RetrievalStrategy.HYBRID],
                "k": 10,
                "rerank": True,
                "max_time": 15.0
            },
            QueryComplexity.AMBIGUOUS: {
                "strategies": [RetrievalStrategy.CORRECTIVE, RetrievalStrategy.MULTI_QUERY],
                "k": 7,
                "rerank": True,
                "max_time": 8.0
            }
        }
        
        # Estatísticas
        self.stats = {
            "total_queries": 0,
            "complexity_distribution": defaultdict(int),
            "strategy_usage": defaultdict(int),
            "average_latency": defaultdict(float),
            "success_rate": defaultdict(float)
        }
        
        logger.info(f"AdaptiveRAGRouter inicializado com objetivo: {optimization_objective}")
    
    async def route_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Roteia query para estratégia apropriada
        
        Returns:
            Resultado do processamento com metadados
        """
        
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Analisar query
        analysis = await self.classifier.classify(query)
        self.stats["complexity_distribution"][analysis.complexity] += 1
        
        logger.info(f"Query classificada como {analysis.complexity.value} "
                   f"(confidence: {analysis.confidence:.2f})")
        
        # Obter configuração de roteamento
        routing = self.routing_config[analysis.complexity]
        
        # Ajustar baseado em otimização
        routing = self._adjust_for_optimization(routing, analysis)
        
        # Executar estratégias
        result = await self._execute_strategies(
            query=query,
            analysis=analysis,
            routing=routing,
            **kwargs
        )
        
        # Métricas
        latency = time.time() - start_time
        complexity_key = analysis.complexity.value
        
        # Atualizar latência média
        current_avg = self.stats["average_latency"][complexity_key]
        count = self.stats["complexity_distribution"][analysis.complexity]
        self.stats["average_latency"][complexity_key] = (
            (current_avg * (count - 1) + latency) / count
        )
        
        # Adicionar metadados
        result["routing_metadata"] = {
            "complexity": analysis.complexity.value,
            "confidence": analysis.confidence,
            "reasoning_type": analysis.reasoning_type,
            "strategies_used": [s.value for s in routing["strategies"]],
            "latency": latency,
            "optimization": self.optimization
        }
        
        return result
    
    def _adjust_for_optimization(self, 
                                routing: Dict, 
                                analysis: QueryAnalysis) -> Dict:
        """Ajusta configuração baseado no objetivo de otimização"""
        
        adjusted = routing.copy()
        
        if self.optimization == "speed":
            # Reduzir K e tempo máximo
            adjusted["k"] = max(3, routing["k"] - 2)
            adjusted["max_time"] = routing["max_time"] * 0.7
            adjusted["rerank"] = False
            
        elif self.optimization == "accuracy":
            # Aumentar K e usar mais estratégias
            adjusted["k"] = min(15, routing["k"] + 3)
            adjusted["rerank"] = True
            
            # Adicionar estratégias complementares
            if analysis.complexity != QueryComplexity.SIMPLE:
                if RetrievalStrategy.CORRECTIVE not in adjusted["strategies"]:
                    adjusted["strategies"].append(RetrievalStrategy.CORRECTIVE)
                    
        elif self.optimization == "cost":
            # Minimizar chamadas caras
            adjusted["k"] = max(2, routing["k"] - 3)
            
            # Remover estratégias caras
            expensive_strategies = [RetrievalStrategy.MULTI_HEAD, RetrievalStrategy.HYBRID]
            adjusted["strategies"] = [s for s in adjusted["strategies"] 
                                    if s not in expensive_strategies]
        
        return adjusted
    
    async def _execute_strategies(self,
                                 query: str,
                                 analysis: QueryAnalysis,
                                 routing: Dict,
                                 **kwargs) -> Dict[str, Any]:
        """Executa estratégias selecionadas"""
        
        strategies = routing["strategies"]
        k = routing.get("k", 5)
        rerank = routing.get("rerank", False)
        max_time = routing.get("max_time", 10.0)
        
        # Resultados de cada estratégia
        strategy_results = []
        
        # Executar estratégias (com timeout)
        try:
            async with asyncio.timeout(max_time):
                tasks = []
                
                for strategy in strategies:
                    if strategy == RetrievalStrategy.DIRECT:
                        task = self._execute_direct_retrieval(query, k)
                    elif strategy == RetrievalStrategy.STANDARD:
                        task = self._execute_standard_rag(query, k, analysis)
                    elif strategy == RetrievalStrategy.MULTI_QUERY:
                        task = self._execute_multi_query(query, k, analysis)
                    elif strategy == RetrievalStrategy.CORRECTIVE:
                        task = self._execute_corrective_rag(query, k, analysis)
                    elif strategy == RetrievalStrategy.GRAPH_ENHANCED:
                        task = self._execute_graph_rag(query, k, analysis)
                    elif strategy == RetrievalStrategy.MULTI_HEAD:
                        task = self._execute_multi_head(query, k, analysis)
                    elif strategy == RetrievalStrategy.HYBRID:
                        task = self._execute_hybrid(query, k, analysis)
                    else:
                        continue
                    
                    tasks.append((strategy, task))
                    self.stats["strategy_usage"][strategy] += 1
                
                # Executar em paralelo
                for strategy, task in tasks:
                    try:
                        result = await task
                        strategy_results.append({
                            "strategy": strategy.value,
                            "result": result
                        })
                    except Exception as e:
                        logger.warning(f"Estratégia {strategy.value} falhou: {e}")
        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout ({max_time}s) atingido para query")
        
        # Combinar resultados
        final_result = await self._combine_results(
            strategy_results,
            analysis,
            rerank=rerank
        )
        
        return final_result
    
    async def _execute_direct_retrieval(self, query: str, k: int) -> Dict:
        """Execução direta simples"""
        
        if "simple_retriever" in self.components:
            retriever = self.components["simple_retriever"]
            docs = await retriever.retrieve(query, k)
            return {"documents": docs, "method": "direct"}
        
        return {"documents": [], "method": "direct", "error": "No simple retriever"}
    
    async def _execute_standard_rag(self, query: str, k: int, analysis: QueryAnalysis) -> Dict:
        """RAG padrão"""
        
        if "standard_rag" in self.components:
            rag = self.components["standard_rag"]
            result = await rag.query(query, k=k)
            return {"documents": result.get("sources", []), "answer": result.get("answer", ""), "method": "standard"}
        
        return {"documents": [], "method": "standard", "error": "No standard RAG"}
    
    async def _execute_multi_query(self, query: str, k: int, analysis: QueryAnalysis) -> Dict:
        """Multi-Query RAG"""
        
        if "multi_query_rag" in self.components:
            mq_rag = self.components["multi_query_rag"]
            queries = await mq_rag.generate_queries(query)
            
            all_docs = []
            for q in queries[:3]:  # Limitar queries
                docs = await mq_rag.retrieve(q, k=k//len(queries))
                all_docs.extend(docs)
            
            return {"documents": all_docs, "queries": queries, "method": "multi_query"}
        
        return {"documents": [], "method": "multi_query", "error": "No multi-query RAG"}
    
    async def _execute_corrective_rag(self, query: str, k: int, analysis: QueryAnalysis) -> Dict:
        """Corrective RAG"""
        
        if "corrective_rag" in self.components:
            corrective = self.components["corrective_rag"]
            result = await corrective.retrieve_and_correct(query, k=k)
            return {"documents": result.get("corrected_docs", []), "method": "corrective"}
        
        return {"documents": [], "method": "corrective", "error": "No corrective RAG"}
    
    async def _execute_graph_rag(self, query: str, k: int, analysis: QueryAnalysis) -> Dict:
        """Graph-enhanced RAG"""
        
        if "graph_rag" in self.components:
            graph = self.components["graph_rag"]
            
            # Usar entidades detectadas
            entities = analysis.key_entities
            result = await graph.query_with_entities(query, entities, k=k)
            
            return {"documents": result.get("documents", []), "graph_context": result.get("graph", {}), "method": "graph"}
        
        return {"documents": [], "method": "graph", "error": "No graph RAG"}
    
    async def _execute_multi_head(self, query: str, k: int, analysis: QueryAnalysis) -> Dict:
        """Multi-Head RAG"""
        
        if "multi_head_rag" in self.components:
            multi_head = self.components["multi_head_rag"]
            docs, metadata = await multi_head.retrieve_multi_head(query, k=k)
            
            return {
                "documents": docs,
                "multi_head_metadata": metadata,
                "method": "multi_head"
            }
        
        return {"documents": [], "method": "multi_head", "error": "No multi-head RAG"}
    
    async def _execute_hybrid(self, query: str, k: int, analysis: QueryAnalysis) -> Dict:
        """Estratégia híbrida combinando múltiplos métodos"""
        
        # Executar múltiplas estratégias em paralelo
        sub_strategies = [
            RetrievalStrategy.STANDARD,
            RetrievalStrategy.GRAPH_ENHANCED,
            RetrievalStrategy.MULTI_HEAD
        ]
        
        results = []
        for strategy in sub_strategies:
            if strategy == RetrievalStrategy.STANDARD:
                result = await self._execute_standard_rag(query, k//3, analysis)
            elif strategy == RetrievalStrategy.GRAPH_ENHANCED:
                result = await self._execute_graph_rag(query, k//3, analysis)
            elif strategy == RetrievalStrategy.MULTI_HEAD:
                result = await self._execute_multi_head(query, k//3, analysis)
            else:
                continue
            
            results.append(result)
        
        # Combinar documentos
        all_docs = []
        for r in results:
            all_docs.extend(r.get("documents", []))
        
        # Deduplicar e ranquear
        unique_docs = self._deduplicate_documents(all_docs)
        
        return {
            "documents": unique_docs[:k],
            "method": "hybrid",
            "sub_methods": [r.get("method") for r in results]
        }
    
    def _deduplicate_documents(self, documents: List[Dict]) -> List[Dict]:
        """Remove documentos duplicados mantendo melhor score"""
        
        seen = {}
        for doc in documents:
            content_hash = hash(doc.get("content", ""))
            
            if content_hash not in seen:
                seen[content_hash] = doc
            else:
                # Manter documento com maior score
                if doc.get("score", 0) > seen[content_hash].get("score", 0):
                    seen[content_hash] = doc
        
        # Ordenar por score
        unique_docs = list(seen.values())
        unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return unique_docs
    
    async def _combine_results(self,
                             strategy_results: List[Dict],
                             analysis: QueryAnalysis,
                             rerank: bool = False) -> Dict[str, Any]:
        """Combina resultados de múltiplas estratégias"""
        
        if not strategy_results:
            return {
                "documents": [],
                "answer": "Não foi possível processar a consulta.",
                "strategies_used": [],
                "error": "No results from strategies"
            }
        
        # Se apenas uma estratégia, retornar direto
        if len(strategy_results) == 1:
            result = strategy_results[0]["result"]
            result["strategies_used"] = [strategy_results[0]["strategy"]]
            return result
        
        # Combinar documentos de todas as estratégias
        all_documents = []
        all_answers = []
        metadata = {
            "strategies_used": [],
            "strategy_details": {}
        }
        
        for sr in strategy_results:
            strategy_name = sr["strategy"]
            result = sr["result"]
            
            metadata["strategies_used"].append(strategy_name)
            metadata["strategy_details"][strategy_name] = {
                "doc_count": len(result.get("documents", [])),
                "has_answer": "answer" in result
            }
            
            all_documents.extend(result.get("documents", []))
            
            if "answer" in result:
                all_answers.append(result["answer"])
        
        # Deduplicar documentos
        unique_documents = self._deduplicate_documents(all_documents)
        
        # Reranquear se solicitado
        if rerank and len(unique_documents) > 5:
            # Implementar reranking (simplificado aqui)
            pass
        
        # Combinar respostas (se houver)
        final_answer = ""
        if all_answers:
            if len(all_answers) == 1:
                final_answer = all_answers[0]
            else:
                # Combinar múltiplas respostas
                final_answer = self._combine_answers(all_answers, analysis)
        
        return {
            "documents": unique_documents,
            "answer": final_answer,
            "complexity_analysis": {
                "complexity": analysis.complexity.value,
                "reasoning_type": analysis.reasoning_type,
                "key_entities": analysis.key_entities,
                "estimated_hops": analysis.estimated_hops
            },
            **metadata
        }
    
    def _combine_answers(self, answers: List[str], analysis: QueryAnalysis) -> str:
        """Combina múltiplas respostas em uma resposta coerente"""
        
        if analysis.reasoning_type == "comparative":
            # Para queries comparativas, estruturar comparação
            return f"Comparando as informações:\n\n" + "\n\n".join(
                f"Perspectiva {i+1}: {ans}" for i, ans in enumerate(answers)
            )
        else:
            # Para outros tipos, sintetizar
            return f"Síntese das informações encontradas:\n\n" + " ".join(answers)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de roteamento"""
        
        total = self.stats["total_queries"]
        
        return {
            "total_queries": total,
            "complexity_distribution": {
                k.value: v for k, v in self.stats["complexity_distribution"].items()
            },
            "strategy_usage": dict(self.stats["strategy_usage"]),
            "average_latency_by_complexity": dict(self.stats["average_latency"]),
            "optimization_mode": self.optimization,
            "routing_efficiency": {
                "simple_percentage": self.stats["complexity_distribution"][QueryComplexity.SIMPLE] / max(total, 1),
                "complex_percentage": (
                    self.stats["complexity_distribution"][QueryComplexity.COMPLEX] + 
                    self.stats["complexity_distribution"][QueryComplexity.MULTI_HOP]
                ) / max(total, 1)
            }
        }
    
    async def optimize_routing(self, 
                             performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Otimiza configurações de roteamento baseado em dados de performance
        
        Args:
            performance_data: Lista de {query, complexity, strategy, latency, success}
        """
        
        # Analisar performance por complexidade e estratégia
        performance_by_complexity = defaultdict(lambda: defaultdict(list))
        
        for data in performance_data:
            complexity = data["complexity"]
            strategy = data["strategy"]
            latency = data["latency"]
            success = data["success"]
            
            performance_by_complexity[complexity][strategy].append({
                "latency": latency,
                "success": success
            })
        
        # Calcular métricas médias
        recommendations = {}
        
        for complexity, strategies in performance_by_complexity.items():
            best_strategy = None
            best_score = -1
            
            for strategy, metrics in strategies.items():
                avg_latency = np.mean([m["latency"] for m in metrics])
                success_rate = np.mean([m["success"] for m in metrics])
                
                # Score combinando sucesso e latência
                score = success_rate / (1 + avg_latency / 10)  # Normalizar latência
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
            
            if best_strategy:
                recommendations[complexity] = {
                    "recommended_strategy": best_strategy,
                    "score": best_score
                }
        
        return {
            "optimization_recommendations": recommendations,
            "data_points_analyzed": len(performance_data)
        }


def create_adaptive_router(rag_components: Dict[str, Any],
                         optimization: str = "balanced") -> AdaptiveRAGRouter:
    """Factory para criar Adaptive RAG Router"""
    
    return AdaptiveRAGRouter(
        rag_components=rag_components,
        optimization_objective=optimization
    ) 