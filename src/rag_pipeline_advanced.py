"""
Advanced RAG Pipeline - Integração de todas as melhorias de RAG.
Combina Enhanced Corrective RAG, Multi-Query, GraphRAG Enhancement e Adaptive Retrieval.
"""

import logging
from typing import Dict, List, Optional, Any
import asyncio
import time
from datetime import datetime

from src.rag_pipeline_base import APIRAGPipeline
from src.retrieval.corrective_rag import CorrectiveRAG
from src.retrieval.enhanced_corrective_rag import EnhancedCorrectiveRAG, create_enhanced_corrective_rag
from src.retrieval.multi_query_rag import MultiQueryRAG
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.graphrag.enhanced_graph_rag import EnhancedGraphRAG
from src.cache.optimized_rag_cache import OptimizedRAGCache
from src.models.model_router import ModelRouter
from src.augmentation.unified_prompt_system import UnifiedPromptSystem
from .retrieval.raptor_retriever import (
    RaptorRetriever,
    create_raptor_retriever,
    get_default_raptor_config,
    ClusteringStrategy,
    RetrievalStrategy
)


logger = logging.getLogger(__name__)


class AdvancedRAGPipeline(APIRAGPipeline):
    """
    Pipeline RAG avançado que integra:
    1. Adaptive Retrieval - Ajuste dinâmico de parâmetros
    2. Multi-Query RAG - Múltiplas perspectivas
    3. Enhanced Corrective RAG - Auto-correção com T5 e decomposição
    4. Enhanced GraphRAG - Enriquecimento com grafo
    5. Métricas e monitoramento avançado
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Inicializar pipeline base
        super().__init__(config_path)
        
        # Componentes avançados
        self.adaptive_retriever = AdaptiveRetriever()
        self.multi_query_rag = MultiQueryRAG()
        
        # Enhanced Corrective RAG (prioritário sobre o básico)
        self.enhanced_corrective_rag = None  # Lazy loading
        self.corrective_rag = CorrectiveRAG()  # Fallback
        
        self.graph_enhancer = EnhancedGraphRAG()
        
        # FASE 2: Cache otimizado híbrido (L1+L2+L3) para máxima performance
        self.cache = None  # Será inicializado assincronamente
        
        # FASE 1: Model router para fallback local
        self.model_router = ModelRouter()
        
        # FASE 2: Sistema unificado de prompts (Dynamic + Selector)
        self.prompt_system = UnifiedPromptSystem()
        
        # RAPTOR retriever
        self.raptor_retriever: Optional[RaptorRetriever] = None
        self.raptor_tree_built = False
        
        # Configurações avançadas
        self.advanced_config = {
            "enable_adaptive": True,
            "enable_multi_query": True,
            "enable_enhanced_corrective": True,  # Prioritário
            "enable_corrective": True,  # Fallback
            "enable_graph": True,
            "enable_cache": True,  # FASE 1: Cache ativo
            "enable_local_fallback": True,  # FASE 1: Fallback local
            "confidence_threshold": 0.7,
            "max_processing_time": 30.0,  # segundos
            
            # Enhanced Corrective RAG específico
            "enhanced_corrective": {
                "relevance_threshold": 0.75,
                "max_reformulation_attempts": 3,
                "enable_decomposition": True,
                "api_providers": ["openai", "anthropic", "huggingface"],
                "cache_evaluations": True,
                "use_circuit_breaker": True
            }
        }
        
        # Métricas
        self.metrics = {
            "total_advanced_queries": 0,
            "improvements_usage": {
                "adaptive": 0,
                "multi_query": 0,
                "enhanced_corrective": 0,  # Novo
                "corrective": 0,  # Fallback
                "graph": 0,
                "cache": 0,  # FASE 1: Métricas de cache
                "local_fallback": 0,  # FASE 1: Métricas de fallback
                "raptor": 0
            },
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0,
            "cache_hit_rate": 0.0,  # FASE 1: Taxa de cache hits
            "enhanced_corrective_success_rate": 0.0,
            "api_provider_usage": {},
            "components": {
                "raptor": {
                    "initialized": False,
                    "configuration": {}
                }
            },
            "raptor_tree": {
                "built": False,
                "construction_time": 0,
                "stats": {}
            },
            "raptor_retrieval": {
                "total_queries": 0,
                "total_results": 0
            }
        }
    
    async def _initialize_cache(self):
        """Inicializa o cache híbrido otimizado se ainda não foi inicializado"""
        if self.cache is None and self.advanced_config["enable_cache"]:
            try:
                # FASE 2: Cache otimizado com configuração automática via .env
                self.cache = OptimizedRAGCache()
                logger.info("✅ Cache híbrido otimizado inicializado com sucesso")
                
                # Log das configurações carregadas
                if hasattr(self.cache, 'enable_redis'):
                    redis_status = "habilitado" if self.cache.enable_redis else "desabilitado"
                    logger.info(f"   Redis: {redis_status}")
                    logger.info(f"   Max memory entries: {self.cache.max_memory_entries}")
                    logger.info(f"   DB path: {self.cache.db_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao inicializar cache otimizado: {e}")
                self.cache = None
    
    async def _initialize_enhanced_corrective_rag(self):
        """Inicializa o Enhanced Corrective RAG com cache Redis se disponível"""
        if self.enhanced_corrective_rag is None and self.advanced_config["enable_enhanced_corrective"]:
            try:
                # Configuração do Enhanced Corrective RAG
                enhanced_config = {
                    **self.advanced_config["enhanced_corrective"],
                    "cache": self.cache,  # Usar o cache do pipeline
                    "model_router": self.model_router
                }
                
                self.enhanced_corrective_rag = create_enhanced_corrective_rag(enhanced_config)
                logger.info("✅ Enhanced Corrective RAG inicializado com sucesso")
                
                # Log das configurações
                logger.info(f"   Relevance threshold: {enhanced_config['relevance_threshold']}")
                logger.info(f"   Decomposition enabled: {enhanced_config['enable_decomposition']}")
                logger.info(f"   API providers: {enhanced_config['api_providers']}")
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao inicializar Enhanced Corrective RAG: {e}")
                logger.info("   Usando Corrective RAG básico como fallback")
                self.enhanced_corrective_rag = None
    
    async def _initialize_raptor_retriever(self) -> None:
        """Inicializa RAPTOR retriever lazy loading"""
        if self.raptor_retriever is not None:
            return
            
        try:
            raptor_config = self.config.get("raptor", get_default_raptor_config())
            
            if not raptor_config.get("enabled", True):
                logger.info("RAPTOR desabilitado na configuração")
                return
            
            logger.info("Inicializando RAPTOR retriever...")
            self.raptor_retriever = await create_raptor_retriever(raptor_config)
            
            # Atualizar métricas
            self.metrics["components"]["raptor"] = {
                "initialized": True,
                "configuration": {
                    "clustering_strategy": raptor_config.get("clustering_strategy", "global_local"),
                    "retrieval_strategy": raptor_config.get("retrieval_strategy", "collapsed_tree"),
                    "max_levels": raptor_config.get("max_levels", 5),
                    "embedding_model": raptor_config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
                }
            }
            
            logger.info("RAPTOR retriever inicializado com sucesso")
            
        except ImportError as e:
            logger.warning(f"Dependências do RAPTOR não disponíveis: {e}")
            logger.info("Continuando sem RAPTOR retriever")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar RAPTOR retriever: {e}")
            logger.info("Continuando sem RAPTOR retriever")

    async def build_raptor_tree(self, documents: List[str]) -> Dict[str, Any]:
        """Constrói árvore RAPTOR com documentos"""
        
        await self._initialize_raptor_retriever()
        
        if self.raptor_retriever is None:
            return {"error": "RAPTOR retriever não disponível"}
        
        try:
            logger.info(f"Construindo árvore RAPTOR com {len(documents)} documentos")
            
            start_time = time.time()
            stats = await self.raptor_retriever.build_tree(documents)
            construction_time = time.time() - start_time
            
            self.raptor_tree_built = True
            
            # Atualizar métricas
            self.metrics["raptor_tree"] = {
                "built": True,
                "construction_time": construction_time,
                "stats": {
                    "total_nodes": stats.total_nodes,
                    "levels": stats.levels,
                    "nodes_per_level": stats.nodes_per_level,
                    "compression_ratio": stats.compression_ratio,
                    "memory_usage_mb": stats.memory_usage_mb
                }
            }
            
            logger.info(f"Árvore RAPTOR construída: {stats.total_nodes} nós, "
                       f"{stats.levels} níveis, {construction_time:.2f}s")
            
            return {
                "success": True,
                "stats": stats.__dict__,
                "tree_summary": self.raptor_retriever.get_tree_summary()
            }
            
        except Exception as e:
            logger.error(f"Erro ao construir árvore RAPTOR: {e}")
            return {"error": f"Falha na construção: {str(e)}"}

    async def _raptor_retrieve(self, 
                              query: str, 
                              k: int = 10, 
                              **kwargs) -> List[Dict]:
        """Retrieval usando RAPTOR"""
        
        await self._initialize_raptor_retriever()
        
        if self.raptor_retriever is None or not self.raptor_tree_built:
            logger.warning("RAPTOR não disponível ou árvore não construída")
            return []
        
        try:
            # Configuração específica para RAPTOR
            max_tokens = kwargs.get("max_tokens", 2000)
            strategy = kwargs.get("retrieval_strategy")
            
            if strategy:
                strategy = RetrievalStrategy(strategy)
            
            # Busca RAPTOR
            results = self.raptor_retriever.search(
                query=query,
                k=k,
                max_tokens=max_tokens,
                strategy=strategy
            )
            
            # Converter para formato Document (dict simples)
            documents = []
            for result in results:
                doc = {
                    "content": result["content"],
                    "score": result["score"],
                    "metadata": {
                        "source": "raptor",
                        "node_id": result["metadata"]["node_id"],
                        "level": result["metadata"]["level"],
                        "token_count": result["metadata"]["token_count"],
                        "cluster_id": result["metadata"].get("cluster_id"),
                        **result["metadata"]
                    }
                }
                documents.append(doc)
            
            logger.info(f"RAPTOR retrieval: {len(documents)} documentos encontrados")
            
            # Atualizar métricas
            if "raptor_retrieval" not in self.metrics:
                self.metrics["raptor_retrieval"] = {"total_queries": 0, "total_results": 0}
            
            self.metrics["raptor_retrieval"]["total_queries"] += 1
            self.metrics["raptor_retrieval"]["total_results"] += len(documents)
            
            return documents
            
        except Exception as e:
            logger.error(f"Erro no RAPTOR retrieval: {e}")
            return []

    async def retrieve(self, 
                      query: str, 
                      k: int = 5, 
                      retrieval_method: str = "hybrid",
                      **kwargs) -> List[Dict]:
        """
        Retrieval unificado com múltiplas estratégias incluindo RAPTOR
        
        Args:
            query: Pergunta/consulta
            k: Número de documentos para retornar
            retrieval_method: Método de retrieval
                - hybrid: Combinação inteligente (padrão)
                - semantic: Busca semântica pura
                - corrective: Com correção automática
                - enhanced_corrective: Corrective RAG aprimorado
                - multi_query: Múltiplas variações da query
                - adaptive: K adaptativo baseado na query
                - graph: Graph-based retrieval
                - raptor: RAPTOR hierarchical retrieval
        """
        
        start_time = time.time()
        
        try:
            if retrieval_method == "raptor":
                documents = await self._raptor_retrieve(query, k, **kwargs)
            elif retrieval_method == "enhanced_corrective":
                await self._initialize_enhanced_corrective_rag()
                if self.enhanced_corrective_rag:
                    documents = await self._enhanced_corrective_retrieve(query, k, **kwargs)
                else:
                    logger.warning("Enhanced Corrective RAG não disponível, usando básico")
                    documents = await self._basic_retrieval(query, k)
            elif retrieval_method == "hybrid":
                # Hybrid inteligente com RAPTOR se disponível
                if self.raptor_retriever and self.raptor_tree_built:
                    # 50% RAPTOR + 50% outros métodos
                    raptor_docs = await self._raptor_retrieve(query, k//2, **kwargs)
                    other_docs = await self._basic_retrieval(query, k//2)
                    
                    # Combinar e reranquear
                    all_docs = raptor_docs + other_docs
                    # Ordenar por score
                    all_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
                    documents = all_docs[:k]
                else:
                    # Fallback para básico
                    documents = await self._basic_retrieval(query, k)
            else:
                # Fallback para retrieval básico
                documents = await self._basic_retrieval(query, k)
            
            # Métricas
            retrieval_time = time.time() - start_time
            
            self.metrics.setdefault("retrieval_calls", 0)
            self.metrics["retrieval_calls"] += 1
            
            self.metrics.setdefault("retrieval_methods", {})
            self.metrics["retrieval_methods"].setdefault(retrieval_method, 0)
            self.metrics["retrieval_methods"][retrieval_method] += 1
            
            self.metrics.setdefault("avg_retrieval_time", 0)
            self.metrics["avg_retrieval_time"] = (
                (self.metrics["avg_retrieval_time"] * (self.metrics["retrieval_calls"] - 1) + retrieval_time) 
                / self.metrics["retrieval_calls"]
            )
            
            # Atualizar métricas por método
            self.metrics["improvements_usage"][retrieval_method] = self.metrics["improvements_usage"].get(retrieval_method, 0) + 1
            
            logger.info(f"Retrieval {retrieval_method}: {len(documents)} docs em {retrieval_time:.2f}s")
            
            return documents
            
        except Exception as e:
            logger.error(f"Erro no retrieval {retrieval_method}: {e}")
            
            # Fallback para básico
            if retrieval_method != "basic":
                logger.info("Tentando fallback para retrieval básico")
                return await self._basic_retrieval(query, k)
            else:
                return []

    async def _enhanced_corrective_retrieve(self, query: str, k: int, **kwargs) -> List[Dict]:
        """Retrieval usando Enhanced Corrective RAG"""
        if not self.enhanced_corrective_rag:
            return await self._basic_retrieval(query, k)
            
        try:
            # Usar Enhanced Corrective RAG
            result = await self.enhanced_corrective_rag.retrieve_and_correct(
                query=query,
                initial_docs=[],  # Deixar que ele faça a busca inicial
                k=k
            )
            
            # Converter resultado para formato esperado
            documents = []
            if "corrected_docs" in result:
                for doc in result["corrected_docs"]:
                    documents.append({
                        "content": doc.get("content", ""),
                        "score": doc.get("score", 0.0),
                        "metadata": {
                            "source": "enhanced_corrective_rag",
                            **doc.get("metadata", {})
                        }
                    })
            
            logger.info(f"Enhanced Corrective RAG: {len(documents)} documentos")
            return documents
            
        except Exception as e:
            logger.error(f"Erro no Enhanced Corrective RAG: {e}")
            return await self._basic_retrieval(query, k)

    async def query_advanced(self,
                            question: str,
                            config: Optional[Dict] = None,
                            force_improvements: Optional[List[str]] = None) -> Dict:
        """
        Executa query com todas as melhorias avançadas.
        
        Args:
            question: Pergunta do usuário
            config: Configurações específicas para esta query
            force_improvements: Lista de melhorias para forçar uso
            
        Returns:
            Dict com resposta enriquecida e metadados
        """
        start_time = time.time()
        self.metrics["total_advanced_queries"] += 1
        
        # FASE 1: Inicializar cache se necessário
        await self._initialize_cache()
        
        # FASE 2: Inicializar Enhanced Corrective RAG se necessário
        await self._initialize_enhanced_corrective_rag()
        
        # FASE 3: Verificar cache híbrido otimizado primeiro
        cache_result = None
        if self.cache and self.advanced_config["enable_cache"]:
            cache_result, cache_source, metadata = await self.cache.get(question)
            if cache_result:
                self.metrics["improvements_usage"]["cache"] += 1
                self.metrics["cache_hit_rate"] = (
                    self.metrics["improvements_usage"]["cache"] / 
                    self.metrics["total_advanced_queries"]
                )
                
                # Log detalhado do cache hit
                confidence = metadata.get("confidence", 0.0)
                age = metadata.get("age", 0.0)
                access_count = metadata.get("access_count", 0)
                
                logger.info(f"🎯 Cache HIT ({cache_source}): confidence={confidence:.2f}, age={age:.1f}s, accessed={access_count}x")
                
                # Adicionar metadados de cache na resposta
                if isinstance(cache_result, dict):
                    cache_result["cache_metadata"] = {
                        "source": cache_source,
                        "confidence": confidence,
                        "age": age,
                        "access_count": access_count,
                        "cache_hit": True
                    }
                
                return cache_result
        
        # Mesclar configurações
        query_config = {**self.advanced_config}
        if config:
            query_config.update(config)
        
        # Determinar quais melhorias usar
        improvements_to_use = self._determine_improvements(question, force_improvements)
        
        logger.info(f"Processando query avançada: '{question[:50]}...'")
        logger.info(f"Melhorias ativas: {improvements_to_use}")
        
        try:
            # 1. Análise adaptativa da query
            query_analysis = None
            if "adaptive" in improvements_to_use:
                query_analysis = self.adaptive_retriever.analyze_query(question)
                optimal_k = query_analysis.optimal_k
                logger.info(f"Análise adaptativa: tipo={query_analysis.query_type}, k={optimal_k}")
            else:
                optimal_k = 5  # Default
            
            # 2. Multi-query expansion
            queries = [question]
            if "multi_query" in improvements_to_use:
                queries = await self.multi_query_rag.generate_multi_queries(question)
                self.metrics["improvements_usage"]["multi_query"] += 1
                logger.info(f"Multi-query gerou {len(queries)} variações")
            
            # 3. Retrieval com correção
            documents = []
            if "enhanced_corrective" in improvements_to_use:
                # Usar Enhanced Corrective RAG
                enhanced_corrective_result = await self.enhanced_corrective_rag.retrieve_and_correct(
                    queries[0],  # Query principal
                    k=optimal_k
                )
                documents = enhanced_corrective_result["documents"]
                self.metrics["improvements_usage"]["enhanced_corrective"] += 1
                
                # Se multi-query ativo, buscar para outras queries também
                if len(queries) > 1:
                    for extra_query in queries[1:]:
                        extra_result = await self.enhanced_corrective_rag.retrieve_and_correct(
                            extra_query,
                            k=3  # Menos documentos para queries extras
                        )
                        documents.extend(extra_result["documents"])
            elif "corrective" in improvements_to_use:
                # Usar Corrective RAG
                corrective_result = await self.corrective_rag.retrieve_and_correct(
                    queries[0],  # Query principal
                    k=optimal_k
                )
                documents = corrective_result["documents"]
                self.metrics["improvements_usage"]["corrective"] += 1
                
                # Se multi-query ativo, buscar para outras queries também
                if len(queries) > 1:
                    for extra_query in queries[1:]:
                        extra_result = await self.corrective_rag.retrieve_and_correct(
                            extra_query,
                            k=3  # Menos documentos para queries extras
                        )
                        documents.extend(extra_result["documents"])
            else:
                # Retrieval normal
                if "adaptive" in improvements_to_use:
                    retrieval_result = await self.adaptive_retriever.retrieve_adaptive(queries[0])
                    documents = retrieval_result["documents"]
                    self.metrics["improvements_usage"]["adaptive"] += 1
                else:
                    # Fallback para retrieval básico
                    documents = await self._basic_retrieval(queries[0], optimal_k)
            
            # 4. Enriquecimento com grafo
            if "graph" in improvements_to_use and documents:
                documents = await self.graph_enhancer.enrich_with_graph_context(documents)
                self.metrics["improvements_usage"]["graph"] += 1
                logger.info("Documentos enriquecidos com contexto do grafo")
            
            # 5. Preparar contexto final
            context = self._prepare_advanced_context(documents, query_analysis)
            
            # 6. Gerar resposta
            generation_start = time.time()
            response = await self._generate_advanced_response(
                question,
                context,
                query_analysis
            )
            generation_time = time.time() - generation_start
            
            # 7. Avaliar confiança
            confidence = await self._evaluate_response_confidence(
                question,
                response["answer"],
                documents
            )
            
            # 8. Preparar resultado final
            processing_time = time.time() - start_time
            self._update_metrics(confidence, processing_time)
            
            result = {
                "answer": response["answer"],
                "confidence": confidence,
                "sources": self._format_sources(documents[:5]),  # Top 5
                "improvements_used": list(improvements_to_use),
                "query_analysis": query_analysis.__dict__ if query_analysis else None,
                "metrics": {
                    "processing_time": processing_time,
                    "generation_time": generation_time,
                    "total_documents": len(documents),
                    "queries_generated": len(queries)
                },
                "model_used": response.get("model_used"),
                "cost": response.get("cost", 0.0)
            }
            
            # FASE 2: Armazenar no cache híbrido com métricas detalhadas
            if self.cache and confidence > 0.6:  # Threshold um pouco menor para mais cache
                try:
                    # Calcular métricas para o cache
                    tokens_saved = len(response["answer"]) // 4  # Estimativa rough
                    cost_saved = result.get("cost", 0.0)
                    
                    await self.cache.set(
                        question,
                        result,
                        confidence=confidence,
                        tokens_saved=tokens_saved,
                        processing_time_saved=processing_time,
                        cost_savings=cost_saved,
                        ttl_hours=24  # Cache por 24 horas
                    )
                    
                    logger.info(f"💾 Resultado cacheado: confidence={confidence:.2f}, tokens_saved={tokens_saved}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao salvar no cache: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no pipeline avançado: {e}")
            
            # FASE 1: Fallback local se APIs falharem
            if self.advanced_config["enable_local_fallback"]:
                try:
                    logger.info("🔄 Tentando fallback local com ModelRouter...")
                    
                    # Usar retrieval básico para contexto
                    basic_docs = await self._basic_retrieval(question, 5)
                    context = self._prepare_advanced_context(basic_docs)
                    
                    # Gerar resposta com modelo local
                    local_response = self.model_router.generate_hybrid_response(
                        question, context, [doc.get("content", "") for doc in basic_docs]
                    )
                    
                    self.metrics["improvements_usage"]["local_fallback"] += 1
                    
                    return {
                        "answer": local_response,
                        "confidence": 0.6,  # Confiança média para fallback
                        "sources": self._format_sources(basic_docs[:3]),
                        "improvements_used": ["local_fallback"],
                        "query_analysis": None,
                        "metrics": {
                            "processing_time": time.time() - start_time,
                            "generation_time": 0.0,
                            "total_documents": len(basic_docs),
                            "queries_generated": 1
                        },
                        "model_used": "local_fallback",
                        "cost": 0.0
                    }
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback local também falhou: {fallback_error}")
            
            # Fallback final para pipeline básico
            return await self.query(question)
    
    def _determine_improvements(self, 
                                question: str, 
                                force_improvements: Optional[List[str]] = None) -> set:
        """Determina quais melhorias usar baseado na query."""
        if force_improvements:
            return set(force_improvements)
        
        improvements = set()
        
        # Sempre usar adaptive por padrão
        if self.advanced_config["enable_adaptive"]:
            improvements.add("adaptive")
        
        # Multi-query para perguntas complexas
        if self.advanced_config["enable_multi_query"]:
            if len(question.split()) > 10 or "?" in question[:-1]:
                improvements.add("multi_query")
        
        # Corrective para todas as queries por padrão
        if self.advanced_config["enable_corrective"]:
            improvements.add("corrective")
        
        # Graph para queries sobre relações ou arquitetura
        if self.advanced_config["enable_graph"]:
            graph_keywords = ["relaciona", "conecta", "arquitetura", "estrutura", "depende"]
            if any(keyword in question.lower() for keyword in graph_keywords):
                improvements.add("graph")
        
        return improvements
    
    async def _basic_retrieval(self, query: str, k: int) -> List[Dict]:
        """Fallback para retrieval básico."""
        # Usar o retriever híbrido existente
        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        return await retriever.retrieve(query, limit=k)
    
    def _prepare_advanced_context(self, 
                                  documents: List[Dict], 
                                  query_analysis: Optional[Any] = None) -> str:
        """Prepara contexto otimizado baseado na análise."""
        if not documents:
            return "Nenhum documento relevante encontrado."
        
        context_parts = []
        
        # Ajustar formato baseado no tipo de query
        if query_analysis and query_analysis.query_type == "list":
            # Formato de lista
            context_parts.append("Informações relevantes encontradas:")
            for i, doc in enumerate(documents[:8], 1):  # Mais docs para listas
                content = doc.get("enriched_content", doc.get("content", ""))[:300]
                context_parts.append(f"\n{i}. {content}...")
        
        elif query_analysis and query_analysis.query_type == "definition":
            # Formato conciso
            context_parts.append("Definições encontradas:")
            for doc in documents[:3]:  # Menos docs para definições
                content = doc.get("enriched_content", doc.get("content", ""))[:400]
                context_parts.append(f"\n{content}")
        
        else:
            # Formato padrão
            for i, doc in enumerate(documents[:5], 1):
                content = doc.get("enriched_content", doc.get("content", ""))[:500]
                source = doc.get("metadata", {}).get("source", "Unknown")
                
                context_parts.append(f"\n[Documento {i} - {source}]")
                context_parts.append(content)
                
                # Adicionar contexto do grafo se disponível
                if "graph_context" in doc:
                    graph_summary = doc["graph_context"].get("summary", "")
                    if graph_summary:
                        context_parts.append(f"Contexto adicional: {graph_summary}")
        
        return "\n".join(context_parts)
    
    async def _generate_advanced_response(self,
                                          question: str,
                                          context: str,
                                          query_analysis: Optional[Any] = None) -> Dict:
        """Gera resposta otimizada usando sistema unificado de prompts."""
        
        # FASE 2: Usar sistema unificado de prompts para gerar prompt otimizado
        try:
            # Extrair chunks do contexto
            context_chunks = []
            if context and context != "Nenhum documento relevante encontrado.":
                # Dividir contexto em chunks lógicos
                context_parts = context.split("\n[Documento")
                for part in context_parts[:5]:  # Limitar a 5 chunks
                    if part.strip():
                        # Limpar marcadores e manter apenas o conteúdo
                        clean_part = part.replace("[Documento", "").replace("]", "").strip()
                        if len(clean_part) > 50:  # Ignorar chunks muito pequenos
                            context_chunks.append(clean_part[:500])  # Limitar tamanho
            
            # Gerar prompt otimizado usando sistema unificado
            prompt_result = await self.prompt_system.generate_optimal_prompt(
                query=question,
                context_chunks=context_chunks,
                language="Português",
                depth="quick"  # Pode ser ajustado baseado na análise
            )
            
            # Log das informações do prompt gerado
            logger.info(f"🎯 Prompt otimizado: task_type={prompt_result.task_type}, "
                       f"source={prompt_result.prompt_source}, confidence={prompt_result.confidence:.2f}")
            
            # Usar temperature sugerida pelo sistema de prompts
            suggested_temp = prompt_result.metadata.get("temperature_suggestion", 0.5)
            
            # Usar pipeline base para geração com prompt otimizado
            response = await self._call_llm_api(
                prompt=prompt_result.final_prompt,
                system_prompt="",  # Já incluído no prompt final
                temperature=suggested_temp
            )
            
            # Adicionar metadados do prompt ao response
            if isinstance(response, dict):
                response["prompt_metadata"] = {
                    "task_type": prompt_result.task_type,
                    "template_id": prompt_result.template_id,
                    "prompt_source": prompt_result.prompt_source,
                    "prompt_confidence": prompt_result.confidence,
                    "temperature_used": suggested_temp,
                    "context_chunks_used": len(context_chunks),
                    "reasoning_applied": prompt_result.metadata.get("reasoning_required", False)
                }
            
            return response
            
        except Exception as e:
            logger.warning(f"Erro no sistema unificado de prompts: {e}. Usando fallback.")
            
            # Fallback para sistema anterior
            system_prompt = "Você é um assistente especializado em fornecer respostas precisas e contextualizadas."
            
            if query_analysis:
                if query_analysis.query_type == "list":
                    system_prompt += " Formate a resposta como uma lista clara e organizada."
                elif query_analysis.query_type == "definition":
                    system_prompt += " Forneça uma definição concisa e clara."
                elif query_analysis.query_type == "implementation":
                    system_prompt += " Inclua exemplos de código quando apropriado."
                elif query_analysis.query_type == "comparison":
                    system_prompt += " Estruture a comparação de forma clara, destacando diferenças e semelhanças."
            
            return await self._call_llm_api(
                prompt=f"Contexto:\n{context}\n\nPergunta: {question}",
                system_prompt=system_prompt
            )
    
    async def _evaluate_response_confidence(self,
                                            question: str,
                                            answer: str,
                                            documents: List[Dict]) -> float:
        """Avalia a confiança na resposta gerada."""
        # Implementação simplificada
        # Em produção, usar modelo específico para avaliação
        
        confidence = 0.5  # Base
        
        # Boost se muitos documentos relevantes
        if len(documents) > 5:
            confidence += 0.2
        
        # Boost se resposta é substancial
        if len(answer) > 200:
            confidence += 0.1
        
        # Boost se documentos têm scores altos
        avg_score = sum(d.get("score", 0.5) for d in documents[:3]) / min(3, len(documents))
        if avg_score > 0.8:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _format_sources(self, documents: List[Dict]) -> List[Dict]:
        """Formata sources para resposta final."""
        sources = []
        
        for doc in documents:
            source = {
                "content": doc.get("content", "")[:200] + "...",
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", 0.0)
            }
            
            # Adicionar info do grafo se disponível
            if "graph_context" in doc:
                source["graph_entities"] = doc["graph_context"].get("central_entities", [])
            
            sources.append(source)
        
        return sources
    
    def _update_metrics(self, confidence: float, processing_time: float):
        """Atualiza métricas do pipeline."""
        # Média móvel da confiança
        n = self.metrics["total_advanced_queries"]
        self.metrics["avg_confidence"] = (
            (self.metrics["avg_confidence"] * (n - 1) + confidence) / n
        )
        
        # Média móvel do tempo
        self.metrics["avg_processing_time"] = (
            (self.metrics["avg_processing_time"] * (n - 1) + processing_time) / n
        )
    
    def get_advanced_stats(self) -> Dict:
        """Retorna estatísticas do pipeline avançado."""
        base_stats = self.get_stats()
        
        base_stats["advanced_metrics"] = {
            "total_advanced_queries": self.metrics["total_advanced_queries"],
            "improvements_usage": self.metrics["improvements_usage"],
            "avg_confidence": round(self.metrics["avg_confidence"], 3),
            "avg_processing_time": round(self.metrics["avg_processing_time"], 2),
            "cache_hit_rate": round(self.metrics["cache_hit_rate"], 3),  # FASE 1: Taxa de cache
            "improvements_adoption_rate": {
                improvement: count / max(1, self.metrics["total_advanced_queries"])
                for improvement, count in self.metrics["improvements_usage"].items()
            }
        }
        
        # FASE 2: Estatísticas detalhadas do cache híbrido otimizado
        if self.cache:
            try:
                cache_stats = self.cache.get_stats()
                base_stats["cache_metrics"] = {
                    **cache_stats,
                    "efficiency_summary": {
                        "total_savings": f"${cache_stats.get('cost_savings', 0):.4f}",
                        "time_saved_minutes": cache_stats.get('processing_time_saved', 0) / 60,
                        "tokens_saved_formatted": f"{cache_stats.get('tokens_saved', 0):,}",
                        "hit_rate_percentage": f"{cache_stats.get('hit_rate', 0):.1%}",
                        "avg_confidence": f"{cache_stats.get('avg_confidence', 0):.2f}"
                    }
                }
                logger.debug(f"Cache stats: hit_rate={cache_stats.get('hit_rate', 0):.1%}, tokens_saved={cache_stats.get('tokens_saved', 0)}")
            except Exception as e:
                logger.warning(f"Erro ao obter estatísticas do cache: {e}")
        
        # FASE 1: Estatísticas do model router
        if self.model_router:
            try:
                router_stats = self.model_router.get_model_status()
                base_stats["model_router_metrics"] = router_stats
            except Exception as e:
                logger.warning(f"Erro ao obter estatísticas do router: {e}")
        
        return base_stats
    
    async def cleanup(self):
        """FASE 2: Método para limpeza de recursos do cache otimizado"""
        try:
            if self.cache:
                # Log final das estatísticas antes de fechar
                stats = self.cache.get_stats()
                logger.info(f"🧹 Fechando cache otimizado - Hit rate final: {stats.get('hit_rate', 0):.1%}")
                logger.info(f"   Tokens economizados total: {stats.get('tokens_saved', 0):,}")
                logger.info(f"   Economia total: ${stats.get('cost_savings', 0):.4f}")
                
                self.cache.close()
                logger.info("✅ Cache híbrido otimizado fechado com sucesso")
        except Exception as e:
            logger.warning(f"Erro ao fechar cache: {e}")
        
        # Chamar cleanup do pipeline base se existir
        if hasattr(super(), 'cleanup'):
            await super().cleanup() 

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Status completo do pipeline incluindo RAPTOR"""
        
        status = {
            "config_loaded": self.config is not None,
            "vector_store_ready": self.vector_store is not None,
            "retriever_ready": self.retriever is not None,
            "llm_router_ready": self.llm_router is not None,
            "enhanced_corrective_rag_ready": self.enhanced_corrective_rag is not None,
            "raptor_ready": self.raptor_retriever is not None,
            "raptor_tree_built": self.raptor_tree_built,
            "total_documents": len(self.documents) if hasattr(self, 'documents') else 0,
            "configuration": {
                "providers": list(self.config.get("providers", {}).keys()) if self.config else [],
                "advanced_features": list(self.config.get("advanced_features", {}).keys()) if self.config else [],
                "raptor_enabled": self.config.get("raptor", {}).get("enabled", False) if self.config else False
            },
            "metrics": self.metrics
        }
        
        # Status detalhado do RAPTOR
        if self.raptor_retriever:
            raptor_summary = self.raptor_retriever.get_tree_summary()
            status["raptor_status"] = raptor_summary
        
        return status 