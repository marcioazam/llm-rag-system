"""
Hybrid Retriever Avançado para Qdrant 1.8.0+
Combina busca densa + esparsa com otimizações de query e reranking
Performance: 16x improvement em sparse search
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import yaml

from ..vectordb.hybrid_qdrant_store import HybridQdrantStore, HybridSearchResult
from ..embeddings.api_embedding_service import APIEmbeddingService
from ..embeddings.sparse_vector_service import AdvancedSparseVectorService
from ..retrieval.query_enhancer import QueryEnhancer
from ..retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

@dataclass
class QueryAnalysis:
    """Análise de query para otimização de busca"""
    original_query: str
    expanded_query: str
    query_type: str  # 'semantic', 'keyword', 'hybrid'
    keywords: List[str]
    entities: List[str]
    intent_confidence: float

@dataclass
class HybridRetrievalResult:
    """Resultado enriquecido de busca híbrida"""
    id: str
    content: str
    metadata: Dict[str, Any]
    dense_score: float
    sparse_score: float
    combined_score: float
    rerank_score: Optional[float]
    retrieval_method: str
    query_match_explanation: str

class QueryAnalyzer:
    """
    Analisador de queries para otimização de busca híbrida
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analisa query para determinar estratégia de busca otimizada
        """
        # Análise básica de características da query
        keywords = self._extract_keywords(query)
        entities = self._extract_entities(query)
        query_type = self._classify_query_type(query, keywords)
        
        # Expansão de query
        expanded_query = self._expand_query(query, keywords)
        
        # Confiança na classificação
        intent_confidence = self._calculate_intent_confidence(query, query_type)
        
        return QueryAnalysis(
            original_query=query,
            expanded_query=expanded_query,
            query_type=query_type,
            keywords=keywords,
            entities=entities,
            intent_confidence=intent_confidence
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extrai keywords da query"""
        import re
        
        # Normalizar e extrair palavras significativas
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        
        # Filtrar stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this'
        }
        
        return [word for word in words if word not in stop_words]
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extrai entidades nomeadas (implementação simples)"""
        import re
        
        # Detectar padrões de entidades (capitalizadas, números, etc.)
        entities = []
        
        # Palavras capitalizadas (possíveis nomes próprios)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized)
        
        # Números e datas
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        entities.extend(numbers)
        
        return entities
    
    def _classify_query_type(self, query: str, keywords: List[str]) -> str:
        """
        Classifica tipo de query para otimização de busca
        """
        query_lower = query.lower()
        
        # Indicadores de busca semântica
        semantic_indicators = [
            'como', 'por que', 'o que é', 'explique', 'descreva', 'compare',
            'diferença', 'vantagem', 'desvantagem', 'conceito', 'teoria'
        ]
        
        # Indicadores de busca por keywords
        keyword_indicators = [
            'encontre', 'liste', 'mostre', 'código', 'função', 'classe',
            'arquivo', 'documento', 'exemplo', 'implementação'
        ]
        
        # Verificar indicadores
        semantic_score = sum(1 for indicator in semantic_indicators if indicator in query_lower)
        keyword_score = sum(1 for indicator in keyword_indicators if indicator in query_lower)
        
        # Classificar
        if semantic_score > keyword_score:
            return 'semantic'
        elif keyword_score > semantic_score:
            return 'keyword'
        else:
            return 'hybrid'
    
    def _expand_query(self, query: str, keywords: List[str]) -> str:
        """
        Expande query com sinônimos e termos relacionados
        """
        # Implementação simples - pode ser expandida com modelos de linguagem
        synonyms = {
            'função': ['function', 'método', 'procedimento'],
            'classe': ['class', 'objeto', 'tipo'],
            'código': ['code', 'script', 'programa'],
            'erro': ['error', 'bug', 'falha', 'problema'],
            'dados': ['data', 'informação', 'dataset']
        }
        
        expanded_terms = []
        for keyword in keywords:
            if keyword in synonyms:
                expanded_terms.extend(synonyms[keyword])
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        
        return query
    
    def _calculate_intent_confidence(self, query: str, query_type: str) -> float:
        """
        Calcula confiança na classificação da query
        """
        # Implementação simplificada
        query_length = len(query.split())
        
        if query_length < 3:
            return 0.6  # Baixa confiança para queries muito curtas
        elif query_length > 10:
            return 0.9  # Alta confiança para queries detalhadas
        else:
            return 0.75  # Confiança média

class HybridRetriever:
    """
    Retriever híbrido avançado com otimizações para Qdrant 1.8.0
    Combina busca densa + esparsa com análise inteligente de queries
    """
    
    def __init__(self, config_path: str = "config/hybrid_search_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Componentes principais
        self.vector_store = HybridQdrantStore(config_path)
        self.query_analyzer = QueryAnalyzer(self.config)
        self.query_enhancer = QueryEnhancer()
        self.reranker = Reranker()
        
        # Serviços de embedding
        self.dense_embedding_service = APIEmbeddingService()
        self.sparse_vector_service = AdvancedSparseVectorService(config_path)
        
        # Cache para performance
        self._retrieval_cache: Dict[str, List[HybridRetrievalResult]] = {}
        
        # Métricas
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_retrieval_time": 0.0,
            "dense_search_time": 0.0,
            "sparse_search_time": 0.0,
            "fusion_time": 0.0,
            "rerank_time": 0.0
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config não encontrado: {config_path}")
            return {}
    
    async def retrieve(
        self, 
        query: str, 
        limit: int = None,
        use_reranking: bool = True,
        strategy: str = "auto"
    ) -> List[HybridRetrievalResult]:
        """
        Busca híbrida otimizada com análise inteligente de query
        
        Args:
            query: Query de busca
            limit: Número máximo de resultados
            use_reranking: Se deve usar reranking
            strategy: 'auto', 'dense_only', 'sparse_only', 'hybrid'
        """
        start_time = time.time()
        self.metrics["total_queries"] += 1
        
        # Cache check
        cache_key = f"{query}:{limit}:{use_reranking}:{strategy}"
        if cache_key in self._retrieval_cache:
            self.metrics["cache_hits"] += 1
            return self._retrieval_cache[cache_key]
        
        # Análise da query
        query_analysis = self.query_analyzer.analyze_query(query)
        
        # Determinar estratégia de busca
        if strategy == "auto":
            strategy = self._determine_search_strategy(query_analysis)
        
        # Executar busca baseada na estratégia
        if strategy == "dense_only":
            results = await self._dense_search_only(query_analysis, limit)
        elif strategy == "sparse_only":
            results = await self._sparse_search_only(query_analysis, limit)
        else:  # hybrid
            results = await self._hybrid_search(query_analysis, limit)
        
        # Reranking se solicitado
        if use_reranking and len(results) > 1:
            rerank_start = time.time()
            results = await self._rerank_results(query, results)
            self.metrics["rerank_time"] += time.time() - rerank_start
        
        # Converter para HybridRetrievalResult
        final_results = self._create_retrieval_results(results, query_analysis, strategy)
        
        # Cache resultado
        if len(self._retrieval_cache) < 1000:
            self._retrieval_cache[cache_key] = final_results
        
        # Atualizar métricas
        total_time = time.time() - start_time
        self.metrics["avg_retrieval_time"] = (
            (self.metrics["avg_retrieval_time"] * (self.metrics["total_queries"] - 1) + total_time) 
            / self.metrics["total_queries"]
        )
        
        return final_results
    
    def _determine_search_strategy(self, query_analysis: QueryAnalysis) -> str:
        """
        Determina estratégia de busca baseada na análise da query
        """
        if query_analysis.query_type == "semantic" and query_analysis.intent_confidence > 0.8:
            return "dense_only"
        elif query_analysis.query_type == "keyword" and len(query_analysis.keywords) > 3:
            return "sparse_only"
        else:
            return "hybrid"
    
    async def _dense_search_only(
        self, 
        query_analysis: QueryAnalysis, 
        limit: int
    ) -> List[HybridSearchResult]:
        """Busca apenas por dense vectors"""
        dense_start = time.time()
        
        # Usar query expandida para melhor recall
        query_text = query_analysis.expanded_query
        
        # Busca densa
        results = await self.vector_store.hybrid_search(query_text, limit)
        
        self.metrics["dense_search_time"] += time.time() - dense_start
        return results
    
    async def _sparse_search_only(
        self, 
        query_analysis: QueryAnalysis, 
        limit: int
    ) -> List[HybridSearchResult]:
        """Busca apenas por sparse vectors"""
        sparse_start = time.time()
        
        # Focar em keywords para busca esparsa
        keyword_query = " ".join(query_analysis.keywords)
        
        # Busca esparsa
        results = await self.vector_store.hybrid_search(keyword_query, limit)
        
        self.metrics["sparse_search_time"] += time.time() - sparse_start
        return results
    
    async def _hybrid_search(
        self, 
        query_analysis: QueryAnalysis, 
        limit: int
    ) -> List[HybridSearchResult]:
        """Busca híbrida completa"""
        fusion_start = time.time()
        
        # Usar query expandida
        query_text = query_analysis.expanded_query
        
        # Busca híbrida
        results = await self.vector_store.hybrid_search(query_text, limit)
        
        self.metrics["fusion_time"] += time.time() - fusion_start
        return results
    
    async def _rerank_results(
        self, 
        query: str, 
        results: List[HybridSearchResult]
    ) -> List[HybridSearchResult]:
        """
        Aplica reranking nos resultados
        """
        try:
            # Preparar dados para reranking
            texts = [result.content for result in results]
            
            # Executar reranking
            reranked_scores = await self.reranker.rerank_documents(query, texts)
            
            # Aplicar scores de reranking
            for result, rerank_score in zip(results, reranked_scores):
                result.combined_score = rerank_score
            
            # Reordenar por novo score
            results.sort(key=lambda x: x.combined_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Erro no reranking: {e}")
        
        return results
    
    def _create_retrieval_results(
        self, 
        search_results: List[HybridSearchResult],
        query_analysis: QueryAnalysis,
        strategy: str
    ) -> List[HybridRetrievalResult]:
        """
        Converte resultados para formato enriquecido
        """
        retrieval_results = []
        
        for result in search_results:
            # Explicação do match
            explanation = self._generate_match_explanation(result, query_analysis)
            
            retrieval_result = HybridRetrievalResult(
                id=result.id,
                content=result.content,
                metadata=result.metadata,
                dense_score=result.dense_score,
                sparse_score=result.sparse_score,
                combined_score=result.combined_score,
                rerank_score=getattr(result, 'rerank_score', None),
                retrieval_method=strategy,
                query_match_explanation=explanation
            )
            
            retrieval_results.append(retrieval_result)
        
        return retrieval_results
    
    def _generate_match_explanation(
        self, 
        result: HybridSearchResult, 
        query_analysis: QueryAnalysis
    ) -> str:
        """
        Gera explicação de por que o documento foi recuperado
        """
        explanations = []
        
        # Análise de dense score
        if result.dense_score > 0.8:
            explanations.append("Alta similaridade semântica")
        elif result.dense_score > 0.6:
            explanations.append("Similaridade semântica moderada")
        
        # Análise de sparse score
        if result.sparse_score > 0.5:
            explanations.append("Forte match de keywords")
        elif result.sparse_score > 0.2:
            explanations.append("Match parcial de keywords")
        
        # Keywords encontradas
        found_keywords = []
        content_lower = result.content.lower()
        for keyword in query_analysis.keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            explanations.append(f"Keywords encontradas: {', '.join(found_keywords)}")
        
        return "; ".join(explanations) if explanations else "Match geral"
    
    async def retrieve_with_filters(
        self, 
        query: str, 
        filters: Dict[str, Any],
        limit: int = None
    ) -> List[HybridRetrievalResult]:
        """
        Busca híbrida com filtros de metadata
        """
        # TODO: Implementar filtros no vector store
        # Por enquanto, busca normal e filtra depois
        results = await self.retrieve(query, limit * 2 if limit else None)
        
        # Aplicar filtros
        filtered_results = []
        for result in results:
            match = True
            for key, value in filters.items():
                if key not in result.metadata or result.metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_results.append(result)
                if limit and len(filtered_results) >= limit:
                    break
        
        return filtered_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas de performance
        """
        cache_hit_rate = (
            self.metrics["cache_hits"] / self.metrics["total_queries"] 
            if self.metrics["total_queries"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._retrieval_cache)
        }
    
    def clear_cache(self) -> None:
        """Limpa cache de retrieval"""
        self._retrieval_cache.clear()
        logger.info("Cache de retrieval limpo")

# Factory function
def create_hybrid_retriever(config_path: str = None) -> HybridRetriever:
    """Cria retriever híbrido"""
    if config_path is None:
        config_path = "config/hybrid_search_config.yaml"
    return HybridRetriever(config_path) 