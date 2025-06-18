"""
Multi-Query RAG - Geração de múltiplas perspectivas para melhorar recall.
Baseado no paper "MultiQueryRetriever" do LangChain.
"""

import logging
from typing import List, Dict, Set, Optional
import asyncio
from collections import defaultdict

from src.models.api_model_router import APIModelRouter
from src.retrieval.hybrid_retriever import HybridRetriever


logger = logging.getLogger(__name__)


class MultiQueryRAG:
    """
    Implementa Multi-Query RAG que:
    1. Gera múltiplas variações de uma query
    2. Busca documentos para cada variação
    3. Combina e deduplica resultados
    4. Aplica reranking fusion
    """
    
    def __init__(self,
                 retriever: Optional[HybridRetriever] = None,
                 num_variations: int = 3,
                 aggregation_method: str = "weighted_fusion"):
        self.retriever = retriever or HybridRetriever()
        self.model_router = APIModelRouter({})
        self.num_variations = num_variations
        self.aggregation_method = aggregation_method
        
    async def generate_multi_queries(self, original_query: str) -> List[str]:
        """
        Gera múltiplas variações da query original.
        
        Estratégias:
        1. Mais específica (narrow down)
        2. Mais geral (broaden)
        3. Perspectiva relacionada (related angle)
        """
        prompt = f"""
        Você é um especialista em reformulação de perguntas para melhorar busca em sistemas RAG.
        
        Dada a pergunta original abaixo, gere {self.num_variations} variações que capturam diferentes perspectivas.
        As variações devem ajudar a encontrar documentos relevantes que a pergunta original pode perder.
        
        Pergunta original: {original_query}
        
        Gere exatamente {self.num_variations} variações seguindo estas estratégias:
        1. Uma versão mais ESPECÍFICA e DETALHADA
        2. Uma versão mais GERAL e ABRANGENTE
        3. Uma pergunta RELACIONADA de ângulo diferente
        
        Formato de resposta (uma pergunta por linha):
        1. [pergunta específica]
        2. [pergunta geral]
        3. [pergunta relacionada]
        """
        
        try:
            response = await self.model_router.route_request(
                prompt,
                task_type="query_expansion"
            )
            
            # Parse variações
            variations = self._parse_query_variations(response.get("answer", ""))
            
            # Incluir query original
            all_queries = [original_query] + variations
            
            logger.info(f"Geradas {len(variations)} variações para: {original_query}")
            for i, q in enumerate(all_queries):
                logger.debug(f"  Query {i}: {q}")
            
            return all_queries
            
        except Exception as e:
            logger.error(f"Erro ao gerar variações: {e}")
            # Fallback: retornar apenas query original
            return [original_query]
    
    async def retrieve_multi_query(self, 
                                   query: str, 
                                   k: int = 5,
                                   per_query_k: int = 3) -> Dict:
        """
        Executa busca com múltiplas queries e combina resultados.
        
        Args:
            query: Query original
            k: Número final de documentos desejados
            per_query_k: Número de documentos por query individual
            
        Returns:
            Dict com documentos combinados e metadados
        """
        # 1. Gerar variações de query
        queries = await self.generate_multi_queries(query)
        
        # 2. Buscar documentos para cada query em paralelo
        search_tasks = [
            self._search_with_metadata(q, per_query_k, idx)
            for idx, q in enumerate(queries)
        ]
        
        all_results = await asyncio.gather(*search_tasks)
        
        # 3. Combinar resultados
        combined_docs = self._combine_results(all_results, queries)
        
        # 4. Deduplica e rerank
        final_docs = self._deduplicate_and_rerank(combined_docs, k)
        
        # 5. Preparar resposta
        return {
            "documents": final_docs,
            "original_query": query,
            "query_variations": queries[1:],  # Excluir original
            "total_queries": len(queries),
            "aggregation_method": self.aggregation_method,
            "unique_sources": len(set(d.get("metadata", {}).get("source", "") for d in final_docs))
        }
    
    async def _search_with_metadata(self, query: str, k: int, query_idx: int) -> Dict:
        """Busca documentos com metadata sobre qual query foi usada."""
        try:
            results = await self.retriever.retrieve(query, limit=k)
            
            # Adicionar metadata da query
            for doc in results:
                if "metadata" not in doc:
                    doc["metadata"] = {}
                doc["metadata"]["query_idx"] = query_idx
                doc["metadata"]["query_used"] = query
                
            return {
                "query": query,
                "query_idx": query_idx,
                "documents": results
            }
            
        except Exception as e:
            logger.error(f"Erro ao buscar com query '{query}': {e}")
            return {
                "query": query,
                "query_idx": query_idx,
                "documents": []
            }
    
    def _combine_results(self, all_results: List[Dict], queries: List[str]) -> List[Dict]:
        """
        Combina resultados de múltiplas queries aplicando pesos.
        
        Estratégias de peso:
        - Query original: peso 1.0
        - Variações: peso 0.8
        """
        combined = []
        
        for result_set in all_results:
            query_idx = result_set["query_idx"]
            weight = 1.0 if query_idx == 0 else 0.8  # Query original tem peso maior
            
            for doc in result_set["documents"]:
                # Ajustar score com peso
                original_score = doc.get("score", 0.5)
                doc["weighted_score"] = original_score * weight
                doc["original_score"] = original_score
                doc["query_weight"] = weight
                
                combined.append(doc)
        
        return combined
    
    def _deduplicate_and_rerank(self, documents: List[Dict], k: int) -> List[Dict]:
        """
        Remove duplicatas e aplica reranking fusion.
        
        Usa Reciprocal Rank Fusion (RRF) para combinar scores.
        """
        # Agrupar por conteúdo único
        doc_groups = defaultdict(list)
        
        for doc in documents:
            # Usar hash do conteúdo como chave
            content_hash = hash(doc.get("content", "")[:200])  # Primeiros 200 chars
            doc_groups[content_hash].append(doc)
        
        # Aplicar fusion para documentos duplicados
        fused_docs = []
        
        for content_hash, doc_list in doc_groups.items():
            if self.aggregation_method == "weighted_fusion":
                # Weighted fusion: soma ponderada dos scores
                fused_score = sum(d.get("weighted_score", 0) for d in doc_list)
                
                # Usar o documento com maior score individual como base
                best_doc = max(doc_list, key=lambda d: d.get("weighted_score", 0))
                best_doc["fusion_score"] = fused_score
                best_doc["fusion_count"] = len(doc_list)
                best_doc["appeared_in_queries"] = list(set(d["metadata"]["query_idx"] for d in doc_list))
                
                fused_docs.append(best_doc)
                
            elif self.aggregation_method == "rrf":
                # Reciprocal Rank Fusion
                k_rrf = 60  # Constante RRF padrão
                rrf_score = sum(1.0 / (k_rrf + idx + 1) for idx, _ in enumerate(doc_list))
                
                best_doc = doc_list[0]
                best_doc["fusion_score"] = rrf_score
                best_doc["fusion_count"] = len(doc_list)
                
                fused_docs.append(best_doc)
        
        # Ordenar por fusion score
        fused_docs.sort(key=lambda d: d.get("fusion_score", 0), reverse=True)
        
        return fused_docs[:k]
    
    def _parse_query_variations(self, response: str) -> List[str]:
        """Extrai variações de query da resposta do modelo."""
        variations = []
        
        # Tentar extrair linhas numeradas
        lines = response.strip().split('\n')
        
        for line in lines:
            # Remover numeração e espaços
            cleaned = line.strip()
            if cleaned:
                # Remover padrões como "1.", "1)", "1 -"
                import re
                cleaned = re.sub(r'^[\d]+[\.\)\-\s]+', '', cleaned).strip()
                
                if cleaned and len(cleaned) > 10:  # Mínimo 10 caracteres
                    variations.append(cleaned)
        
        # Garantir que temos o número correto de variações
        if len(variations) < self.num_variations:
            # Adicionar variações genéricas se necessário
            base_variations = [
                "com exemplos práticos",
                "explicação detalhada",
                "casos de uso"
            ]
            
            while len(variations) < self.num_variations and base_variations:
                variations.append(f"{response.split('.')[0]} {base_variations.pop(0)}")
        
        return variations[:self.num_variations] 