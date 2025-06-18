"""
Adaptive Retriever - Ajusta dinamicamente parâmetros de busca baseado na query.
Otimiza o número K de documentos e estratégias de busca.
"""

import logging
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass

from src.retrieval.hybrid_retriever import HybridRetriever


logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Análise detalhada de uma query."""
    query_type: str  # definition, list, comparison, implementation, analysis
    complexity_score: float  # 0-1
    expected_answer_length: str  # short, medium, long
    optimal_k: int
    search_strategy: str  # dense, sparse, hybrid
    confidence: float


class AdaptiveRetriever:
    """
    Implementa retrieval adaptativo que:
    1. Analisa a complexidade e tipo da query
    2. Ajusta dinamicamente o número K
    3. Seleciona estratégia de busca ideal
    4. Otimiza parâmetros por tipo de pergunta
    """
    
    def __init__(self, base_retriever: Optional[HybridRetriever] = None):
        self.retriever = base_retriever or HybridRetriever()
        
        # Padrões para identificar tipos de query
        self.query_patterns = {
            "definition": [
                r"o que é\b", r"defin[ae]\b", r"significa\b", 
                r"conceito de\b", r"explique o que\b"
            ],
            "list": [
                r"list[ae]\b", r"quais são\b", r"enumere\b", 
                r"cite\b", r"exemplos de\b", r"tipos de\b"
            ],
            "comparison": [
                r"diferença entre\b", r"compare\b", r"versus\b", 
                r"vs\b", r"melhor que\b", r"vantagens e desvantagens\b"
            ],
            "implementation": [
                r"como implementar\b", r"como fazer\b", r"código para\b",
                r"exemplo de código\b", r"implementação de\b", r"tutorial\b"
            ],
            "analysis": [
                r"analis[ae]\b", r"avalie\b", r"quando usar\b",
                r"por que\b", r"benefícios\b", r"problemas com\b"
            ]
        }
        
        # Configurações otimizadas por tipo
        self.type_configs = {
            "definition": {
                "base_k": 3,
                "complexity_multiplier": 1.0,
                "strategy": "hybrid"
            },
            "list": {
                "base_k": 8,
                "complexity_multiplier": 1.5,
                "strategy": "hybrid"
            },
            "comparison": {
                "base_k": 6,
                "complexity_multiplier": 1.2,
                "strategy": "hybrid"
            },
            "implementation": {
                "base_k": 5,
                "complexity_multiplier": 1.3,
                "strategy": "sparse"  # Melhor para código
            },
            "analysis": {
                "base_k": 7,
                "complexity_multiplier": 1.4,
                "strategy": "dense"   # Melhor para conceitos
            }
        }
    
    async def retrieve_adaptive(self, query: str) -> Dict:
        """
        Executa busca adaptativa baseada na análise da query.
        
        Args:
            query: Query do usuário
            
        Returns:
            Dict com documentos e análise da busca
        """
        # 1. Analisar query
        analysis = self.analyze_query(query)
        
        logger.info(f"Query analysis: type={analysis.query_type}, "
                   f"complexity={analysis.complexity_score:.2f}, "
                   f"optimal_k={analysis.optimal_k}")
        
        # 2. Executar busca com parâmetros otimizados
        results = await self.retriever.retrieve(
            query=query,
            limit=analysis.optimal_k,
            strategy=analysis.search_strategy
        )
        
        # 3. Pós-processar baseado no tipo
        processed_results = self._post_process_results(results, analysis)
        
        # 4. Preparar resposta
        return {
            "documents": processed_results,
            "query_analysis": {
                "type": analysis.query_type,
                "complexity": analysis.complexity_score,
                "optimal_k": analysis.optimal_k,
                "strategy": analysis.search_strategy,
                "confidence": analysis.confidence
            },
            "retrieval_metadata": {
                "total_retrieved": len(results),
                "after_processing": len(processed_results),
                "expected_answer_length": analysis.expected_answer_length
            }
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analisa a query para determinar tipo e complexidade.
        """
        query_lower = query.lower().strip()
        
        # 1. Identificar tipo de query
        query_type, type_confidence = self._identify_query_type(query_lower)
        
        # 2. Calcular complexidade
        complexity_score = self._calculate_complexity(query_lower)
        
        # 3. Determinar K ótimo
        optimal_k = self._determine_optimal_k(query_type, complexity_score)
        
        # 4. Selecionar estratégia
        search_strategy = self._select_search_strategy(query_type, query_lower)
        
        # 5. Estimar tamanho da resposta
        expected_length = self._estimate_answer_length(query_type, complexity_score)
        
        return QueryAnalysis(
            query_type=query_type,
            complexity_score=complexity_score,
            expected_answer_length=expected_length,
            optimal_k=optimal_k,
            search_strategy=search_strategy,
            confidence=type_confidence
        )
    
    def _identify_query_type(self, query: str) -> Tuple[str, float]:
        """Identifica o tipo da query com confiança."""
        scores = {}
        
        for qtype, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 1
            
            if score > 0:
                scores[qtype] = score / len(patterns)
        
        if scores:
            # Tipo com maior score
            best_type = max(scores.items(), key=lambda x: x[1])
            return best_type[0], best_type[1]
        
        # Default: analysis
        return "analysis", 0.5
    
    def _calculate_complexity(self, query: str) -> float:
        """
        Calcula complexidade da query (0-1).
        
        Fatores:
        - Comprimento
        - Número de cláusulas
        - Termos técnicos
        - Operadores lógicos
        """
        score = 0.0
        
        # Comprimento normalizado (0-50 palavras)
        word_count = len(query.split())
        score += min(word_count / 50, 0.3)
        
        # Cláusulas (e, ou, mas, porém)
        clauses = len(re.findall(r'\b(e|ou|mas|porém|além|também)\b', query))
        score += min(clauses * 0.1, 0.2)
        
        # Termos técnicos
        tech_terms = len(re.findall(
            r'\b(api|algoritmo|arquitetura|framework|biblioteca|'
            r'implementação|otimização|performance|complexidade)\b', 
            query
        ))
        score += min(tech_terms * 0.1, 0.3)
        
        # Pontos de interrogação múltiplos ou sub-perguntas
        questions = query.count('?')
        score += min(questions * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _determine_optimal_k(self, query_type: str, complexity: float) -> int:
        """Determina número ótimo de documentos."""
        config = self.type_configs.get(query_type, self.type_configs["analysis"])
        
        # K base do tipo
        base_k = config["base_k"]
        
        # Ajustar por complexidade
        multiplier = config["complexity_multiplier"]
        complexity_adjustment = int(complexity * multiplier * 3)
        
        optimal_k = base_k + complexity_adjustment
        
        # Limites
        return max(3, min(optimal_k, 15))
    
    def _select_search_strategy(self, query_type: str, query: str) -> str:
        """Seleciona estratégia de busca ideal."""
        # Configuração base por tipo
        base_strategy = self.type_configs.get(query_type, {}).get("strategy", "hybrid")
        
        # Ajustes específicos
        if "código" in query or "function" in query or "classe" in query:
            return "sparse"  # Melhor para termos exatos
        
        if "conceito" in query or "teoria" in query or "princípio" in query:
            return "dense"   # Melhor para semântica
        
        return base_strategy
    
    def _estimate_answer_length(self, query_type: str, complexity: float) -> str:
        """Estima tamanho esperado da resposta."""
        if query_type == "definition":
            return "short" if complexity < 0.3 else "medium"
        
        elif query_type == "list":
            return "medium" if complexity < 0.5 else "long"
        
        elif query_type == "implementation":
            return "long"  # Código geralmente é longo
        
        elif query_type == "comparison":
            return "medium" if complexity < 0.4 else "long"
        
        else:  # analysis
            if complexity < 0.3:
                return "short"
            elif complexity < 0.6:
                return "medium"
            else:
                return "long"
    
    def _post_process_results(self, 
                              results: List[Dict], 
                              analysis: QueryAnalysis) -> List[Dict]:
        """
        Pós-processa resultados baseado no tipo de query.
        """
        processed = results.copy()
        
        # Para listas, priorizar documentos com enumerações
        if analysis.query_type == "list":
            for doc in processed:
                content = doc.get("content", "")
                # Boost score se contiver listas
                list_indicators = len(re.findall(r'(\n\s*[-•*\d]+\.?\s+|\n\s*\d+\))', content))
                if list_indicators > 0:
                    doc["score"] = doc.get("score", 0.5) * 1.2
        
        # Para definições, priorizar respostas concisas
        elif analysis.query_type == "definition":
            for doc in processed:
                content = doc.get("content", "")
                # Boost se começar com definição clara
                if re.match(r'^[^.]+\s+(é|são|significa|refere-se a)\s+', content):
                    doc["score"] = doc.get("score", 0.5) * 1.3
        
        # Para implementações, priorizar código
        elif analysis.query_type == "implementation":
            for doc in processed:
                content = doc.get("content", "")
                # Boost se contiver blocos de código
                code_blocks = len(re.findall(r'```[\s\S]*?```', content))
                if code_blocks > 0:
                    doc["score"] = doc.get("score", 0.5) * (1.1 + code_blocks * 0.1)
        
        # Re-ordenar por novo score
        processed.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return processed 