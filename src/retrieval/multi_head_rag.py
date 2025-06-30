"""
Multi-Head RAG - Sistema de Retrieval com Múltiplas Attention Heads
Captura diferentes aspectos semânticos e consolida resultados via voting
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AttentionHead:
    """Representa uma attention head com foco semântico específico"""
    name: str
    semantic_focus: str
    weight_matrix: Optional[np.ndarray] = None
    temperature: float = 1.0
    top_k: int = 5
    
    def __post_init__(self):
        if self.weight_matrix is None:
            # Inicializar com pesos aleatórios
            self.weight_matrix = np.random.randn(768, 768) * 0.1


class MultiHeadRetriever:
    """
    Multi-Head RAG com diferentes focos semânticos:
    - Factual: Fatos e informações objetivas
    - Conceptual: Conceitos e definições  
    - Procedural: Processos e procedimentos
    - Contextual: Contexto e relações
    - Temporal: Aspectos temporais e sequenciais
    """
    
    def __init__(self,
                 embedding_service,
                 vector_store,
                 num_heads: int = 5,
                 attention_dim: int = 768,
                 voting_strategy: str = "weighted_majority"):
        
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.voting_strategy = voting_strategy
        
        # Definir heads com diferentes focos semânticos
        self.attention_heads = [
            AttentionHead(
                name="factual",
                semantic_focus="facts, data, objective information",
                temperature=0.7,
                top_k=8
            ),
            AttentionHead(
                name="conceptual", 
                semantic_focus="concepts, definitions, theory",
                temperature=0.8,
                top_k=6
            ),
            AttentionHead(
                name="procedural",
                semantic_focus="steps, procedures, how-to",
                temperature=0.9,
                top_k=5
            ),
            AttentionHead(
                name="contextual",
                semantic_focus="context, relationships, connections",
                temperature=1.0,
                top_k=7
            ),
            AttentionHead(
                name="temporal",
                semantic_focus="time, sequence, chronology",
                temperature=0.8,
                top_k=4
            )
        ]
        
        # Inicializar pesos das heads
        self._initialize_attention_weights()
        
        # Estatísticas
        self.stats = {
            "total_queries": 0,
            "head_contributions": defaultdict(int),
            "voting_outcomes": defaultdict(int),
            "average_diversity_score": 0.0
        }
        
        logger.info(f"MultiHeadRetriever inicializado com {self.num_heads} heads")
    
    def _initialize_attention_weights(self):
        """Inicializa pesos de atenção para cada head com bias semântico"""
        
        for head in self.attention_heads:
            if head.name == "factual":
                # Bias para precisão e objetividade
                head.weight_matrix = self._create_factual_weights()
            elif head.name == "conceptual":
                # Bias para abstração e relações conceituais
                head.weight_matrix = self._create_conceptual_weights()
            elif head.name == "procedural":
                # Bias para sequências e ações
                head.weight_matrix = self._create_procedural_weights()
            elif head.name == "contextual":
                # Bias para contexto amplo
                head.weight_matrix = self._create_contextual_weights()
            elif head.name == "temporal":
                # Bias para aspectos temporais
                head.weight_matrix = self._create_temporal_weights()
    
    def _create_factual_weights(self) -> np.ndarray:
        """Cria matriz de pesos para foco factual"""
        # Implementação simplificada - em produção usar técnicas mais sofisticadas
        base_weights = np.eye(self.attention_dim)
        
        # Adicionar bias para tokens relacionados a fatos
        fact_indices = [i for i in range(0, self.attention_dim, 10)]  # Simulação
        for idx in fact_indices:
            base_weights[idx, :] *= 1.2
        
        return base_weights + np.random.randn(self.attention_dim, self.attention_dim) * 0.05
    
    def _create_conceptual_weights(self) -> np.ndarray:
        """Cria matriz de pesos para foco conceitual"""
        weights = np.random.randn(self.attention_dim, self.attention_dim) * 0.1
        
        # Aumentar conexões entre dimensões (conceitos relacionados)
        for i in range(self.attention_dim):
            for j in range(max(0, i-5), min(self.attention_dim, i+5)):
                weights[i, j] += 0.1
        
        return weights
    
    def _create_procedural_weights(self) -> np.ndarray:
        """Cria matriz de pesos para foco procedural"""
        weights = np.zeros((self.attention_dim, self.attention_dim))
        
        # Criar padrão sequencial
        for i in range(self.attention_dim - 1):
            weights[i, i+1] = 0.8  # Forte conexão com próximo
            weights[i, i] = 1.0    # Auto-conexão
        
        return weights + np.random.randn(self.attention_dim, self.attention_dim) * 0.05
    
    def _create_contextual_weights(self) -> np.ndarray:
        """Cria matriz de pesos para foco contextual"""
        # Pesos mais distribuídos para capturar contexto amplo
        weights = np.ones((self.attention_dim, self.attention_dim)) * 0.1
        weights += np.random.randn(self.attention_dim, self.attention_dim) * 0.1
        
        return weights
    
    def _create_temporal_weights(self) -> np.ndarray:
        """Cria matriz de pesos para foco temporal"""
        weights = np.zeros((self.attention_dim, self.attention_dim))
        
        # Padrão para capturar sequências temporais
        for i in range(self.attention_dim):
            # Conexões com vizinhos temporais
            for offset in [-2, -1, 0, 1, 2]:
                j = i + offset * 10  # Saltos temporais
                if 0 <= j < self.attention_dim:
                    weights[i, j] = 1.0 / (abs(offset) + 1)
        
        return weights
    
    async def retrieve_multi_head(self, 
                                 query: str,
                                 k: int = 10,
                                 **kwargs) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Realiza retrieval usando múltiplas attention heads
        
        Returns:
            (documentos, metadados_de_análise)
        """
        
        self.stats["total_queries"] += 1
        
        # Obter embedding da query
        query_embedding = await self._get_query_embedding(query)
        
        # Aplicar cada attention head
        head_results = {}
        head_scores = {}
        
        tasks = []
        for head in self.attention_heads:
            task = self._apply_attention_head(
                query_embedding, 
                head,
                k=head.top_k
            )
            tasks.append((head.name, task))
        
        # Executar heads em paralelo
        for head_name, task in tasks:
            results, scores = await task
            head_results[head_name] = results
            head_scores[head_name] = scores
            self.stats["head_contributions"][head_name] += len(results)
        
        # Consolidar resultados via voting
        final_documents, voting_details = await self._consolidate_results(
            head_results,
            head_scores,
            k=k
        )
        
        # Calcular diversidade
        diversity_score = self._calculate_diversity(head_results)
        self.stats["average_diversity_score"] = (
            (self.stats["average_diversity_score"] * (self.stats["total_queries"] - 1) + diversity_score) /
            self.stats["total_queries"]
        )
        
        # Preparar metadados
        metadata = {
            "retrieval_method": "multi_head",
            "num_heads": len(self.attention_heads),
            "head_contributions": {
                name: len(results) for name, results in head_results.items()
            },
            "voting_strategy": self.voting_strategy,
            "voting_details": voting_details,
            "diversity_score": diversity_score,
            "semantic_coverage": self._analyze_semantic_coverage(head_results)
        }
        
        logger.info(f"Multi-head retrieval: {len(final_documents)} docs, "
                   f"diversity={diversity_score:.3f}")
        
        return final_documents, metadata
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Obtém embedding da query"""
        # Assumindo que embedding_service retorna numpy array
        embedding = await self.embedding_service.aembed_query(query)
        return np.array(embedding)
    
    async def _apply_attention_head(self,
                                   query_embedding: np.ndarray,
                                   head: AttentionHead,
                                   k: int) -> Tuple[List[Dict], Dict[str, float]]:
        """Aplica uma attention head específica"""
        
        # Transformar embedding com attention weights
        attended_embedding = np.dot(query_embedding, head.weight_matrix)
        
        # Normalizar
        attended_embedding = attended_embedding / (np.linalg.norm(attended_embedding) + 1e-8)
        
        # Adicionar bias semântico baseado no foco da head
        semantic_query = f"{query_embedding} {head.semantic_focus}"
        
        # Buscar no vector store com embedding modificado
        results = await self.vector_store.similarity_search_with_score(
            attended_embedding.tolist(),
            k=k
        )
        
        # Processar resultados
        documents = []
        scores = {}
        
        for doc, score in results:
            # Ajustar score com temperatura da head
            adjusted_score = float(score) ** (1.0 / head.temperature)
            
            doc_dict = {
                "content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "attention_head": head.name,
                    "semantic_focus": head.semantic_focus,
                    "raw_score": float(score),
                    "adjusted_score": adjusted_score
                },
                "score": adjusted_score
            }
            
            documents.append(doc_dict)
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            scores[doc_id] = adjusted_score
        
        return documents, scores
    
    async def _consolidate_results(self,
                                  head_results: Dict[str, List[Dict]],
                                  head_scores: Dict[str, Dict[str, float]],
                                  k: int) -> Tuple[List[Dict], Dict]:
        """Consolida resultados de múltiplas heads via voting"""
        
        if self.voting_strategy == "weighted_majority":
            return await self._weighted_majority_voting(head_results, head_scores, k)
        elif self.voting_strategy == "borda_count":
            return await self._borda_count_voting(head_results, head_scores, k)
        elif self.voting_strategy == "coverage_optimization":
            return await self._coverage_optimization_voting(head_results, head_scores, k)
        else:
            # Fallback para união simples
            return await self._union_voting(head_results, head_scores, k)
    
    async def _weighted_majority_voting(self,
                                       head_results: Dict[str, List[Dict]],
                                       head_scores: Dict[str, Dict[str, float]],
                                       k: int) -> Tuple[List[Dict], Dict]:
        """Voting ponderado pela maioria"""
        
        # Agregar votos ponderados
        document_votes = defaultdict(lambda: {
            "score": 0.0,
            "heads": [],
            "content": "",
            "metadata": {}
        })
        
        # Pesos para cada head (podem ser aprendidos)
        head_weights = {
            "factual": 1.2,
            "conceptual": 1.0,
            "procedural": 1.1,
            "contextual": 0.9,
            "temporal": 0.8
        }
        
        # Coletar votos
        for head_name, results in head_results.items():
            weight = head_weights.get(head_name, 1.0)
            
            for doc in results:
                doc_id = hash(doc["content"])
                
                # Voto ponderado
                vote_score = doc["score"] * weight
                
                document_votes[doc_id]["score"] += vote_score
                document_votes[doc_id]["heads"].append(head_name)
                document_votes[doc_id]["content"] = doc["content"]
                document_votes[doc_id]["metadata"].update(doc["metadata"])
        
        # Ordenar por score total
        sorted_docs = sorted(
            document_votes.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Preparar documentos finais
        final_documents = []
        for doc_id, doc_info in sorted_docs[:k]:
            final_doc = {
                "content": doc_info["content"],
                "score": doc_info["score"] / len(doc_info["heads"]),  # Normalizar
                "metadata": {
                    **doc_info["metadata"],
                    "voting_heads": doc_info["heads"],
                    "num_votes": len(doc_info["heads"])
                }
            }
            final_documents.append(final_doc)
        
        voting_details = {
            "strategy": "weighted_majority",
            "head_weights": head_weights,
            "total_candidates": len(document_votes),
            "selected": k
        }
        
        self.stats["voting_outcomes"]["weighted_majority"] += 1
        
        return final_documents, voting_details
    
    async def _borda_count_voting(self,
                                 head_results: Dict[str, List[Dict]],
                                 head_scores: Dict[str, Dict[str, float]],
                                 k: int) -> Tuple[List[Dict], Dict]:
        """Voting usando Borda Count"""
        
        borda_scores = defaultdict(lambda: {"score": 0, "content": "", "metadata": {}})
        
        # Para cada head, atribuir pontos Borda
        for head_name, results in head_results.items():
            num_candidates = len(results)
            
            for rank, doc in enumerate(results):
                doc_id = hash(doc["content"])
                
                # Pontos Borda: N-1 para primeiro, N-2 para segundo, etc.
                borda_points = num_candidates - rank - 1
                
                borda_scores[doc_id]["score"] += borda_points
                borda_scores[doc_id]["content"] = doc["content"]
                borda_scores[doc_id]["metadata"].update(doc["metadata"])
        
        # Ordenar por pontos Borda
        sorted_docs = sorted(
            borda_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Preparar resultado
        final_documents = []
        for doc_id, doc_info in sorted_docs[:k]:
            final_doc = {
                "content": doc_info["content"],
                "score": doc_info["score"],
                "metadata": {
                    **doc_info["metadata"],
                    "borda_score": doc_info["score"]
                }
            }
            final_documents.append(final_doc)
        
        voting_details = {
            "strategy": "borda_count",
            "total_candidates": len(borda_scores),
            "selected": k
        }
        
        return final_documents, voting_details
    
    async def _coverage_optimization_voting(self,
                                          head_results: Dict[str, List[Dict]],
                                          head_scores: Dict[str, Dict[str, float]],
                                          k: int) -> Tuple[List[Dict], Dict]:
        """Voting otimizando cobertura semântica"""
        
        selected_documents = []
        covered_aspects = set()
        
        # Pool de candidatos
        all_candidates = []
        for head_name, results in head_results.items():
            for doc in results:
                doc["_head"] = head_name
                all_candidates.append(doc)
        
        # Ordenar por score
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Selecionar maximizando cobertura
        for candidate in all_candidates:
            if len(selected_documents) >= k:
                break
            
            head_name = candidate["_head"]
            
            # Preferir documentos de heads ainda não representadas
            if head_name not in covered_aspects or len(selected_documents) < k // 2:
                selected_documents.append(candidate)
                covered_aspects.add(head_name)
        
        # Preencher restante se necessário
        while len(selected_documents) < k and all_candidates:
            for candidate in all_candidates:
                if candidate not in selected_documents:
                    selected_documents.append(candidate)
                    if len(selected_documents) >= k:
                        break
        
        voting_details = {
            "strategy": "coverage_optimization",
            "covered_aspects": list(covered_aspects),
            "coverage_ratio": len(covered_aspects) / len(self.attention_heads)
        }
        
        return selected_documents[:k], voting_details
    
    async def _union_voting(self,
                          head_results: Dict[str, List[Dict]],
                          head_scores: Dict[str, Dict[str, float]],
                          k: int) -> Tuple[List[Dict], Dict]:
        """Voting por união simples (fallback)"""
        
        seen = set()
        final_documents = []
        
        # Coletar todos os documentos únicos
        for head_name, results in head_results.items():
            for doc in results:
                doc_id = hash(doc["content"])
                if doc_id not in seen:
                    seen.add(doc_id)
                    final_documents.append(doc)
        
        # Ordenar por score e retornar top-k
        final_documents.sort(key=lambda x: x["score"], reverse=True)
        
        voting_details = {
            "strategy": "union",
            "total_unique": len(final_documents),
            "selected": k
        }
        
        return final_documents[:k], voting_details
    
    def _calculate_diversity(self, head_results: Dict[str, List[Dict]]) -> float:
        """Calcula diversidade dos resultados entre heads"""
        
        if len(head_results) < 2:
            return 0.0
        
        # Conjuntos de documentos por head
        doc_sets = {}
        for head_name, results in head_results.items():
            doc_sets[head_name] = set(hash(doc["content"]) for doc in results)
        
        # Calcular Jaccard distances entre pares
        distances = []
        head_names = list(doc_sets.keys())
        
        for i in range(len(head_names)):
            for j in range(i + 1, len(head_names)):
                set1 = doc_sets[head_names[i]]
                set2 = doc_sets[head_names[j]]
                
                if len(set1) == 0 and len(set2) == 0:
                    jaccard = 0.0
                else:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    jaccard = 1.0 - (intersection / union if union > 0 else 0.0)
                
                distances.append(jaccard)
        
        # Diversidade média
        return np.mean(distances) if distances else 0.0
    
    def _analyze_semantic_coverage(self, head_results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Analisa cobertura semântica por cada head"""
        
        coverage = {}
        total_docs = sum(len(results) for results in head_results.values())
        
        if total_docs == 0:
            return {head: 0.0 for head in head_results.keys()}
        
        for head_name, results in head_results.items():
            coverage[head_name] = len(results) / total_docs
        
        return coverage
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do Multi-Head RAG"""
        
        return {
            "total_queries": self.stats["total_queries"],
            "head_performance": dict(self.stats["head_contributions"]),
            "voting_strategies_used": dict(self.stats["voting_outcomes"]),
            "average_diversity": self.stats["average_diversity_score"],
            "heads_config": [
                {
                    "name": head.name,
                    "semantic_focus": head.semantic_focus,
                    "temperature": head.temperature,
                    "top_k": head.top_k
                }
                for head in self.attention_heads
            ]
        }
    
    async def optimize_head_weights(self, 
                                   feedback_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Otimiza pesos das heads baseado em feedback
        
        Args:
            feedback_data: Lista de {query, selected_docs, relevance_scores}
            
        Returns:
            Novos pesos otimizados
        """
        
        # Implementação simplificada - em produção usar RL ou gradient descent
        head_performance = defaultdict(list)
        
        for feedback in feedback_data:
            query = feedback["query"]
            relevance_scores = feedback["relevance_scores"]
            
            # Analisar quais heads contribuíram para docs relevantes
            for doc_id, relevance in relevance_scores.items():
                # Verificar qual head originou o documento
                # (necessário rastrear origem dos documentos)
                pass
        
        # Por enquanto, retornar pesos atuais
        return {head.name: 1.0 for head in self.attention_heads}


def create_multi_head_retriever(embedding_service,
                               vector_store,
                               config: Optional[Dict] = None) -> MultiHeadRetriever:
    """Factory para criar Multi-Head Retriever"""
    
    default_config = {
        "num_heads": 5,
        "attention_dim": 768,
        "voting_strategy": "weighted_majority"
    }
    
    if config:
        default_config.update(config)
    
    return MultiHeadRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        **default_config
    ) 