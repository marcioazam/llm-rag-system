"""
HyDE (Hypothetical Document Embeddings) Enhancer
Implementa geração de documentos hipotéticos para melhorar retrieval
Integrado com APIEmbeddingService e sistema de configuração existente
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import yaml
from pathlib import Path

from ..embeddings.api_embedding_service import APIEmbeddingService
from ..models.api_model_router import APIModelRouter

logger = logging.getLogger(__name__)

@dataclass
class HyDEResult:
    """Resultado do processamento HyDE"""
    original_query: str
    hypothetical_documents: List[str]
    enhanced_embedding: np.ndarray
    confidence_score: float
    generation_time: float

class HyDEEnhancer:
    """
    HyDE (Hypothetical Document Embeddings) Enhancer
    
    Gera documentos hipotéticos baseados na query para melhorar retrieval,
    seguindo a técnica proposta em "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.hyde_config = self.config.get("hyde", self._default_hyde_config())
        
        # Serviços integrados
        try:
            self.embedding_service = APIEmbeddingService(self.config)
            self.model_router = APIModelRouter(self.config)
        except (ImportError, TypeError) as e:
            logger.warning(f"Serviços não disponíveis: {e}, usando fallbacks")
            self.embedding_service = None
            self.model_router = None
        
        # Cache para performance
        self._cache: Dict[str, HyDEResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"HyDE Enhancer inicializado com {self.hyde_config['num_hypothetical_docs']} docs por query")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração do sistema"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config não encontrado: {config_path}, usando padrões")
            return {}
    
    def _default_hyde_config(self) -> Dict:
        """Configuração padrão para HyDE"""
        return {
            "enabled": True,
            "num_hypothetical_docs": 3,
            "max_tokens": 400,
            "temperature": 0.7,
            "model": "gpt-3.5-turbo",
            "cache_enabled": True,
            "embedding_strategy": "mean",  # mean, weighted, max
            "prompt_template": {
                "system": "Você é um especialista em gerar documentos informativos e precisos.",
                "user": """Dada a pergunta abaixo, gere um parágrafo detalhado que responda à pergunta de forma completa e informativa.
O texto deve ser factual, bem estruturado e conter informações relevantes que poderiam estar em um documento real.

Pergunta: {query}

Parágrafo de resposta:"""
            },
            "quality_threshold": 0.3  # Threshold mínimo de qualidade
        }
    
    async def enhance_query(self, query: str) -> HyDEResult:
        """
        Aplica HyDE para melhorar retrieval da query
        
        Args:
            query: Query original do usuário
            
        Returns:
            HyDEResult com embedding melhorado
        """
        import time
        start_time = time.time()
        
        # Verificar cache
        if self.hyde_config["cache_enabled"] and query in self._cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit para query: {query[:50]}...")
            return self._cache[query]
        
        self._cache_misses += 1
        logger.info(f"Processando HyDE para query: {query[:50]}...")
        
        try:
            # 1. Gerar documentos hipotéticos
            hypothetical_docs = await self._generate_hypothetical_documents(query)
            
            if not hypothetical_docs:
                logger.warning("Nenhum documento hipotético gerado, usando embedding original")
                if self.embedding_service:
                    original_embedding = await self.embedding_service.embed_texts([query])
                    enhanced_embedding = original_embedding[0]
                else:
                    enhanced_embedding = np.zeros(384)  # Fallback embedding
                return HyDEResult(
                    original_query=query,
                    hypothetical_documents=[],
                    enhanced_embedding=enhanced_embedding,
                    confidence_score=0.0,
                    generation_time=time.time() - start_time
                )
            
            # 2. Filtrar documentos por qualidade
            quality_docs = await self._filter_by_quality(query, hypothetical_docs)
            
            if not quality_docs:
                logger.warning("Nenhum documento passou no filtro de qualidade")
                quality_docs = hypothetical_docs[:1]  # Usar pelo menos um
            
            # 3. Gerar embeddings dos documentos hipotéticos
            if self.embedding_service:
                doc_embeddings = await self.embedding_service.embed_texts(quality_docs)
            else:
                # Fallback: embeddings simulados
                doc_embeddings = [np.random.normal(0, 1, 384) for _ in quality_docs]
            
            # 4. Combinar embeddings usando estratégia configurada
            enhanced_embedding = self._combine_embeddings(
                doc_embeddings, 
                strategy=self.hyde_config["embedding_strategy"]
            )
            
            # 5. Calcular score de confiança
            confidence = self._calculate_confidence(quality_docs, doc_embeddings)
            
            result = HyDEResult(
                original_query=query,
                hypothetical_documents=quality_docs,
                enhanced_embedding=enhanced_embedding,
                confidence_score=confidence,
                generation_time=time.time() - start_time
            )
            
            # Armazenar em cache
            if self.hyde_config["cache_enabled"]:
                self._cache[query] = result
            
            logger.info(f"HyDE processado: {len(quality_docs)} docs, confiança={confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Erro no processamento HyDE: {e}")
            # Fallback para embedding original
            if self.embedding_service:
                try:
                    original_embedding = await self.embedding_service.embed_texts([query])
                    enhanced_embedding = original_embedding[0]
                except:
                    enhanced_embedding = np.zeros(384)
            else:
                enhanced_embedding = np.zeros(384)
            
            return HyDEResult(
                original_query=query,
                hypothetical_documents=[],
                enhanced_embedding=enhanced_embedding,
                confidence_score=0.0,
                generation_time=time.time() - start_time
            )
    
    async def _generate_hypothetical_documents(self, query: str) -> List[str]:
        """Gera documentos hipotéticos usando LLM"""
        hypothetical_docs = []
        num_docs = self.hyde_config["num_hypothetical_docs"]
        
        if not self.model_router:
            # Fallback: documentos simulados
            logger.info("Model router não disponível, gerando documentos simulados")
            for i in range(num_docs):
                doc = f"Documento hipotético {i+1} para a query: {query}. Este é um exemplo de resposta que contém informações relevantes sobre o tópico solicitado."
                hypothetical_docs.append(doc)
            return hypothetical_docs
        
        # Preparar prompt
        prompt_config = self.hyde_config["prompt_template"]
        user_prompt = prompt_config["user"].format(query=query)
        
        for i in range(num_docs):
            try:
                response = await self.model_router.generate_response(
                    prompt=user_prompt,
                    system_prompt=prompt_config["system"],
                    model=self.hyde_config["model"],
                    max_tokens=self.hyde_config["max_tokens"],
                    temperature=self.hyde_config["temperature"]
                )
                
                if response and response.strip():
                    hypothetical_docs.append(response.strip())
                    logger.debug(f"Documento hipotético {i+1} gerado: {len(response)} chars")
                else:
                    logger.warning(f"Documento hipotético {i+1} vazio")
                    
            except Exception as e:
                logger.error(f"Erro ao gerar documento hipotético {i+1}: {e}")
                continue
        
        return hypothetical_docs
    
    async def _filter_by_quality(self, query: str, documents: List[str]) -> List[str]:
        """
        Filtra documentos por qualidade usando similaridade semântica
        """
        if not documents:
            return []
        
        try:
            if not self.embedding_service:
                # Fallback: retornar todos os documentos
                return documents
            
            # Embedding da query original
            query_embedding = await self.embedding_service.embed_texts([query])
            
            # Embeddings dos documentos
            doc_embeddings = await self.embedding_service.embed_texts(documents)
            
            # Calcular similaridades
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Filtrar por threshold
            threshold = self.hyde_config["quality_threshold"]
            quality_docs = []
            
            for i, (doc, similarity) in enumerate(zip(documents, similarities)):
                if similarity >= threshold:
                    quality_docs.append(doc)
                    logger.debug(f"Documento {i+1} aprovado: similaridade={similarity:.3f}")
                else:
                    logger.debug(f"Documento {i+1} rejeitado: similaridade={similarity:.3f}")
            
            return quality_docs
            
        except Exception as e:
            logger.error(f"Erro no filtro de qualidade: {e}")
            return documents  # Retornar todos se houver erro
    
    def _combine_embeddings(self, embeddings: List[np.ndarray], strategy: str = "mean") -> np.ndarray:
        """
        Combina múltiplos embeddings usando estratégia especificada
        """
        if not embeddings:
            raise ValueError("Lista de embeddings vazia")
        
        embeddings_array = np.array(embeddings)
        
        if strategy == "mean":
            return np.mean(embeddings_array, axis=0)
        elif strategy == "weighted":
            # Peso maior para embeddings com maior norma (mais informação)
            norms = np.linalg.norm(embeddings_array, axis=1)
            weights = norms / np.sum(norms)
            return np.average(embeddings_array, axis=0, weights=weights)
        elif strategy == "max":
            # Elemento-wise maximum
            return np.max(embeddings_array, axis=0)
        else:
            logger.warning(f"Estratégia desconhecida '{strategy}', usando 'mean'")
            return np.mean(embeddings_array, axis=0)
    
    def _calculate_confidence(self, documents: List[str], embeddings: List[np.ndarray]) -> float:
        """
        Calcula score de confiança baseado na consistência dos documentos
        """
        if len(documents) <= 1:
            return 0.5
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calcular similaridades par-a-par
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(sim)
            
            # Confiança baseada na consistência (alta similaridade = alta confiança)
            if similarities:
                avg_similarity = np.mean(similarities)
                confidence = min(max(avg_similarity, 0.0), 1.0)
            else:
                confidence = 0.5
            
            return confidence
            
        except Exception as e:
            logger.error(f"Erro no cálculo de confiança: {e}")
            return 0.5
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance do HyDE"""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "config": self.hyde_config
        }
    
    def clear_cache(self) -> None:
        """Limpa cache do HyDE"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cache HyDE limpo")

# Função de conveniência para criar HyDE enhancer
def create_hyde_enhancer(config_path: str = "config/config.yaml") -> HyDEEnhancer:
    """Cria instância do HyDE enhancer"""
    return HyDEEnhancer(config_path) 