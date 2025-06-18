"""
Corrective RAG - Auto-correção de respostas com validação de relevância.
Baseado no paper "Corrective Retrieval Augmented Generation" (2024).
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from src.retrieval.hybrid_retriever import HybridRetriever
from src.models.api_model_router import APIModelRouter


logger = logging.getLogger(__name__)


@dataclass
class DocumentWithScore:
    """Documento com score de relevância."""
    content: str
    metadata: Dict
    relevance_score: float
    validation_status: str = "pending"  # pending, relevant, irrelevant


class CorrectiveRAG:
    """
    Implementa Corrective RAG com:
    1. Avaliação de relevância dos documentos recuperados
    2. Reformulação de query se necessário
    3. Validação cruzada com knowledge graph
    4. Estratégias de fallback
    """
    
    def __init__(self, 
                 retriever: Optional[HybridRetriever] = None,
                 relevance_threshold: float = 0.7,
                 max_reformulation_attempts: int = 2):
        self.retriever = retriever or HybridRetriever()
        self.model_router = APIModelRouter({})
        self.relevance_threshold = relevance_threshold
        self.max_reformulation_attempts = max_reformulation_attempts
        self.reformulation_count = 0
        
    async def retrieve_and_correct(self, 
                                   query: str, 
                                   k: int = 5,
                                   use_graph_validation: bool = True) -> Dict:
        """
        Recupera documentos com correção automática.
        
        Args:
            query: Query do usuário
            k: Número de documentos a recuperar
            use_graph_validation: Se deve validar com knowledge graph
            
        Returns:
            Dict com documentos validados e metadados
        """
        logger.info(f"Iniciando Corrective RAG para query: {query[:50]}...")
        
        # Reset contador
        self.reformulation_count = 0
        
        # 1. Retrieval inicial
        initial_docs = await self._retrieve_documents(query, k)
        
        # 2. Avaliar relevância
        evaluated_docs = await self._evaluate_relevance(query, initial_docs)
        
        # 3. Verificar se precisa reformular
        relevance_scores = [doc.relevance_score for doc in evaluated_docs]
        max_relevance = max(relevance_scores) if relevance_scores else 0
        
        logger.info(f"Relevância máxima: {max_relevance:.3f} (threshold: {self.relevance_threshold})")
        
        # 4. Se baixa relevância, reformular e tentar novamente
        if max_relevance < self.relevance_threshold and self.reformulation_count < self.max_reformulation_attempts:
            logger.info("Baixa relevância detectada. Reformulando query...")
            reformulated_query = await self._reformulate_query(query, evaluated_docs)
            
            # Recursive call com query reformulada
            self.reformulation_count += 1
            return await self.retrieve_and_correct(reformulated_query, k, use_graph_validation)
        
        # 5. Validação com knowledge graph (se disponível)
        if use_graph_validation:
            evaluated_docs = await self._validate_with_graph(evaluated_docs)
        
        # 6. Filtrar e ordenar documentos
        relevant_docs = [doc for doc in evaluated_docs if doc.validation_status == "relevant"]
        relevant_docs.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 7. Preparar resposta
        return {
            "documents": relevant_docs[:k],
            "original_query": query,
            "reformulation_count": self.reformulation_count,
            "avg_relevance_score": np.mean([d.relevance_score for d in relevant_docs]) if relevant_docs else 0,
            "correction_applied": self.reformulation_count > 0,
            "total_evaluated": len(evaluated_docs),
            "total_relevant": len(relevant_docs)
        }
    
    async def _retrieve_documents(self, query: str, k: int) -> List[Dict]:
        """Recupera documentos usando o retriever."""
        try:
            results = await self.retriever.retrieve(query, limit=k * 2)  # Recuperar mais para ter margem
            return results
        except Exception as e:
            logger.error(f"Erro ao recuperar documentos: {e}")
            return []
    
    async def _evaluate_relevance(self, query: str, documents: List[Dict]) -> List[DocumentWithScore]:
        """
        Avalia relevância de cada documento para a query.
        Usa um modelo LLM para scoring de relevância.
        """
        evaluated_docs = []
        
        for doc in documents:
            try:
                # Preparar prompt para avaliação
                evaluation_prompt = f"""
                Avalie a relevância do seguinte documento para a query do usuário.
                Responda APENAS com um score de 0.0 a 1.0 e uma justificativa curta.
                
                Query: {query}
                
                Documento:
                {doc.get('content', '')[:1000]}
                
                Formato de resposta:
                Score: [0.0-1.0]
                Justificativa: [uma linha explicando a relevância]
                """
                
                # Usar modelo para avaliar
                response = await self.model_router.route_request(
                    evaluation_prompt,
                    task_type="evaluation",
                    force_model="openai.gpt35_turbo"  # Modelo rápido para avaliação
                )
                
                # Parse score
                score = self._parse_relevance_score(response.get("answer", ""))
                
                # Criar documento avaliado
                evaluated_doc = DocumentWithScore(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    relevance_score=score,
                    validation_status="relevant" if score >= self.relevance_threshold else "irrelevant"
                )
                
                evaluated_docs.append(evaluated_doc)
                
            except Exception as e:
                logger.error(f"Erro ao avaliar documento: {e}")
                # Em caso de erro, assume relevância média
                evaluated_docs.append(DocumentWithScore(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    relevance_score=0.5,
                    validation_status="pending"
                ))
        
        return evaluated_docs
    
    async def _reformulate_query(self, original_query: str, low_relevance_docs: List[DocumentWithScore]) -> str:
        """
        Reformula a query baseada nos documentos de baixa relevância.
        Tenta entender o que está faltando e criar uma query melhor.
        """
        # Análise dos documentos recuperados
        doc_summaries = []
        for doc in low_relevance_docs[:3]:  # Usar top 3
            doc_summaries.append(f"- {doc.content[:200]}... (relevância: {doc.relevance_score:.2f})")
        
        reformulation_prompt = f"""
        A query original não retornou documentos muito relevantes. 
        Analise a query e os documentos recuperados, então reformule a query para ser mais específica e clara.
        
        Query original: {original_query}
        
        Documentos recuperados (baixa relevância):
        {chr(10).join(doc_summaries)}
        
        Reformule a query para:
        1. Ser mais específica
        2. Usar termos técnicos apropriados
        3. Clarificar a intenção
        4. Adicionar contexto se necessário
        
        Query reformulada:
        """
        
        response = await self.model_router.route_request(
            reformulation_prompt,
            task_type="query_reformulation"
        )
        
        reformulated = response.get("answer", "").strip()
        
        # Se falhar, adicionar contexto genérico
        if not reformulated or reformulated == original_query:
            reformulated = f"{original_query} com exemplos práticos e implementação detalhada"
        
        logger.info(f"Query reformulada: {reformulated}")
        return reformulated
    
    async def _validate_with_graph(self, documents: List[DocumentWithScore]) -> List[DocumentWithScore]:
        """
        Valida documentos usando knowledge graph do Neo4j.
        Verifica se as entidades mencionadas existem e estão relacionadas.
        """
        # TODO: Implementar quando Neo4j estiver configurado
        # Por enquanto, retorna os documentos como estão
        logger.info("Validação com grafo ainda não implementada")
        return documents
    
    def _parse_relevance_score(self, response: str) -> float:
        """Extrai score numérico da resposta do modelo."""
        try:
            # Procurar por "Score:" ou números decimais
            import re
            
            # Tentar encontrar padrão "Score: X.X"
            score_match = re.search(r'Score:\s*([0-9.]+)', response, re.IGNORECASE)
            if score_match:
                return float(score_match.group(1))
            
            # Tentar encontrar qualquer número decimal entre 0 e 1
            decimal_match = re.search(r'(0\.[0-9]+|1\.0)', response)
            if decimal_match:
                return float(decimal_match.group(1))
            
            # Fallback baseado em keywords
            response_lower = response.lower()
            if any(word in response_lower for word in ["muito relevante", "altamente relevante", "extremamente"]):
                return 0.9
            elif any(word in response_lower for word in ["relevante", "relacionado", "útil"]):
                return 0.7
            elif any(word in response_lower for word in ["pouco relevante", "parcialmente"]):
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"Erro ao parsear score: {e}")
            return 0.5  # Score médio como fallback 