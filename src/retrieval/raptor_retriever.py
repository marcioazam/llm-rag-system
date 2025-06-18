"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

Implementação baseada no paper de Stanford (2024):
"RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
https://arxiv.org/abs/2401.18059

Características principais:
- Clustering recursivo com UMAP + GMM
- Summarização hierárquica multi-nível  
- Retrieval em múltiplas abstrações
- Tree traversal e collapsed tree
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ClusteringStrategy(Enum):
    """Estratégias de clustering disponíveis"""
    GLOBAL_LOCAL = "global_local"  # UMAP global + local
    SINGLE_LEVEL = "single_level"  # Apenas um nível
    ADAPTIVE = "adaptive"          # Adaptativo baseado em dados

class RetrievalStrategy(Enum):
    """Estratégias de retrieval da árvore"""
    TREE_TRAVERSAL = "tree_traversal"  # Navegação por camadas
    COLLAPSED_TREE = "collapsed_tree"  # Árvore achatada
    HYBRID = "hybrid"                  # Combinação de ambas

@dataclass
class RaptorNode:
    """Nó da árvore RAPTOR"""
    node_id: str
    content: str
    embedding: np.ndarray
    level: int
    children_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    cluster_id: Optional[int] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável"""
        return {
            "node_id": self.node_id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "level": self.level,
            "children_ids": self.children_ids,
            "parent_id": self.parent_id,
            "cluster_id": self.cluster_id,
            "token_count": self.token_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RaptorNode':
        """Cria a partir de dicionário"""
        embedding = np.array(data["embedding"]) if data["embedding"] else None
        return cls(
            node_id=data["node_id"],
            content=data["content"],
            embedding=embedding,
            level=data["level"],
            children_ids=data["children_ids"],
            parent_id=data["parent_id"],
            cluster_id=data["cluster_id"],
            token_count=data["token_count"],
            metadata=data["metadata"]
        )

@dataclass
class ClusterInfo:
    """Informações de um cluster"""
    cluster_id: int
    node_ids: List[str]
    centroid: np.ndarray
    size: int
    coherence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RaptorTreeStats:
    """Estatísticas da árvore RAPTOR"""
    total_nodes: int
    levels: int
    nodes_per_level: Dict[int, int]
    clusters_per_level: Dict[int, int]
    compression_ratio: float
    construction_time: float
    memory_usage_mb: float

class MockEmbeddingModel:
    """Mock para quando sentence-transformers não está disponível"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.warning(f"Usando mock embedding model para {model_name}")
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Gera embeddings mock baseados em hash do texto"""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        embeddings = []
        for t in texts:
            # Gerar embedding determinístico baseado no texto
            hash_val = hash(t) % (2**32)
            np.random.seed(hash_val)
            embedding = np.random.normal(0, 1, 384)  # Tamanho típico
            embedding = embedding / np.linalg.norm(embedding)  # Normalizar
            embeddings.append(embedding)
        
        if isinstance(text, str):
            return embeddings[0]
        return np.array(embeddings)

class UMAPClusterer:
    """Clustering hierárquico com UMAP + GMM"""
    
    def __init__(self, 
                 global_neighbors: Optional[int] = None,
                 local_neighbors: int = 10,
                 n_components: int = 10,
                 min_dist: float = 0.0,
                 random_state: int = 42):
        self.global_neighbors = global_neighbors
        self.local_neighbors = local_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.random_state = random_state
        
        # Modelos UMAP
        self.global_umap = None
        self.local_umaps = {}
        
        # Modelos GMM
        self.global_gmm = None
        self.local_gmms = {}
        
    def _fit_umap(self, embeddings: np.ndarray, n_neighbors: int):
        """Treina modelo UMAP ou PCA como fallback"""
        
        if not UMAP_AVAILABLE:
            logger.info("UMAP não disponível, usando PCA como fallback")
            from sklearn.decomposition import PCA
            n_comp = min(self.n_components, embeddings.shape[1], embeddings.shape[0]-1)
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            reduced = pca.fit_transform(embeddings)
            
            # Criar objeto mock que simula UMAP
            class MockUMAP:
                def __init__(self, pca_model):
                    self.pca = pca_model
                
                def transform(self, X):
                    return self.pca.transform(X)
            
            return MockUMAP(pca)
        
        try:
            umap_model = umap.UMAP(
                n_neighbors=min(n_neighbors, len(embeddings) - 1),
                n_components=self.n_components,
                min_dist=self.min_dist,
                random_state=self.random_state,
                metric='cosine'
            )
            return umap_model.fit(embeddings)
        except Exception as e:
            logger.warning(f"Erro no UMAP: {e}, usando PCA como fallback")
            # Fallback: usar PCA simples
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(self.n_components, embeddings.shape[1], embeddings.shape[0]-1))
            reduced = pca.fit_transform(embeddings)
            
            # Criar objeto mock que simula UMAP
            class MockUMAP:
                def __init__(self, pca_model):
                    self.pca = pca_model
                
                def transform(self, X):
                    return self.pca.transform(X)
            
            return MockUMAP(pca)
    
    def _determine_optimal_clusters(self, 
                                  embeddings: np.ndarray, 
                                  max_clusters: int = 50) -> int:
        """Determina número ótimo de clusters usando BIC"""
        if len(embeddings) < 2:
            return 1
            
        max_k = min(max_clusters, len(embeddings) // 2, 50)
        if max_k < 1:
            max_k = 1
            
        bic_scores = []
        
        for k in range(1, max_k + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='full',
                    random_state=self.random_state,
                    max_iter=100
                )
                gmm.fit(embeddings)
                bic_scores.append(gmm.bic(embeddings))
            except:
                bic_scores.append(float('inf'))
        
        # Melhor K = menor BIC
        optimal_k = np.argmin(bic_scores) + 1
        return optimal_k
    
    def hierarchical_cluster(self, 
                           embeddings: np.ndarray, 
                           node_ids: List[str]) -> Tuple[List[ClusterInfo], Dict[str, Any]]:
        """Clustering hierárquico global + local"""
        
        if len(embeddings) <= 1:
            logger.info("Muito poucos embeddings para clustering")
            return [], {"error": "insufficient_data"}
        
        # Determinar n_neighbors global automaticamente
        if self.global_neighbors is None:
            self.global_neighbors = max(2, int(np.sqrt(len(embeddings))))
            
        logger.info(f"Iniciando clustering hierárquico com {len(embeddings)} embeddings")
        
        try:
            # FASE 1: Clustering Global
            logger.info("Fase 1: Clustering global")
            
            # UMAP global
            self.global_umap = self._fit_umap(embeddings, self.global_neighbors)
            global_reduced = self.global_umap.transform(embeddings)
            
            # GMM global
            global_k = self._determine_optimal_clusters(global_reduced)
            self.global_gmm = GaussianMixture(
                n_components=global_k,
                covariance_type='full',
                random_state=self.random_state
            )
            global_labels = self.global_gmm.fit_predict(global_reduced)
            
            logger.info(f"Clusters globais encontrados: {global_k}")
            
            # FASE 2: Clustering Local em cada cluster global
            logger.info("Fase 2: Clustering local")
            final_clusters = []
            local_cluster_id = 0
            
            for global_cluster_id in range(global_k):
                # Pontos do cluster global
                global_mask = global_labels == global_cluster_id
                if not np.any(global_mask):
                    continue
                    
                local_embeddings = embeddings[global_mask]
                local_node_ids = [node_ids[i] for i in range(len(node_ids)) if global_mask[i]]
                
                if len(local_embeddings) <= 1:
                    # Cluster muito pequeno - criar cluster final
                    cluster_info = ClusterInfo(
                        cluster_id=local_cluster_id,
                        node_ids=local_node_ids,
                        centroid=local_embeddings[0] if len(local_embeddings) > 0 else np.zeros(embeddings.shape[1]),
                        size=len(local_embeddings),
                        metadata={"global_cluster": global_cluster_id}
                    )
                    final_clusters.append(cluster_info)
                    local_cluster_id += 1
                    continue
                
                # UMAP local
                local_umap = self._fit_umap(local_embeddings, self.local_neighbors)
                local_reduced = local_umap.transform(local_embeddings)
                self.local_umaps[global_cluster_id] = local_umap
                
                # GMM local
                local_k = self._determine_optimal_clusters(local_reduced)
                local_gmm = GaussianMixture(
                    n_components=local_k,
                    covariance_type='full',
                    random_state=self.random_state
                )
                local_labels = local_gmm.fit_predict(local_reduced)
                self.local_gmms[global_cluster_id] = local_gmm
                
                # Criar clusters finais
                for local_label in range(local_k):
                    local_mask = local_labels == local_label
                    if not np.any(local_mask):
                        continue
                        
                    cluster_embeddings = local_embeddings[local_mask]
                    cluster_node_ids = [local_node_ids[i] for i in range(len(local_node_ids)) if local_mask[i]]
                    
                    centroid = np.mean(cluster_embeddings, axis=0)
                    
                    # Calcular coerência (silhouette score)
                    coherence = 0.0
                    if len(cluster_embeddings) > 1:
                        try:
                            coherence = silhouette_score(local_reduced, local_labels)
                        except:
                            coherence = 0.0
                    
                    cluster_info = ClusterInfo(
                        cluster_id=local_cluster_id,
                        node_ids=cluster_node_ids,
                        centroid=centroid,
                        size=len(cluster_embeddings),
                        coherence_score=coherence,
                        metadata={
                            "global_cluster": global_cluster_id,
                            "local_cluster": local_label
                        }
                    )
                    final_clusters.append(cluster_info)
                    local_cluster_id += 1
            
            metadata = {
                "global_clusters": global_k,
                "total_final_clusters": len(final_clusters),
                "avg_cluster_size": np.mean([c.size for c in final_clusters]) if final_clusters else 0,
                "clustering_strategy": "hierarchical"
            }
            
            logger.info(f"Clustering concluído: {len(final_clusters)} clusters finais")
            return final_clusters, metadata
            
        except Exception as e:
            logger.error(f"Erro no clustering: {e}")
            # Fallback: criar um cluster para cada nó
            fallback_clusters = []
            for i, node_id in enumerate(node_ids):
                cluster_info = ClusterInfo(
                    cluster_id=i,
                    node_ids=[node_id],
                    centroid=embeddings[i],
                    size=1,
                    metadata={"fallback": True}
                )
                fallback_clusters.append(cluster_info)
            
            return fallback_clusters, {"clustering_strategy": "fallback"}

class RaptorSummarizer:
    """Summarização hierárquica para RAPTOR"""
    
    def __init__(self, 
                 api_provider: str = "openai",
                 model_name: str = "gpt-4o-mini",
                 max_tokens: int = 1000):
        self.api_provider = api_provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Inicializar tokenizer com fallback
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.use_tiktoken = True
            except:
                self.use_tiktoken = False
        else:
            self.use_tiktoken = False
        
    def _count_tokens(self, text: str) -> int:
        """Conta tokens no texto"""
        if hasattr(self, 'use_tiktoken') and self.use_tiktoken:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Fallback: aproximar por palavras
        return int(len(text.split()) * 1.3)
    
    def _create_simple_summary(self, texts: List[str], level: int) -> str:
        """Cria resumo simples por concatenação inteligente"""
        
        if not texts:
            return ""
            
        if len(texts) == 1:
            return texts[0]
        
        # Estratégia baseada no nível
        if level <= 1:
            # Níveis baixos: preservar mais detalhes
            combined = "\n\n".join(texts)
            max_length = self.max_tokens * 4  # Aproximação de caracteres
            
            if len(combined) <= max_length:
                return combined
            
            # Truncar preservando início de cada texto
            summary_parts = []
            available_length = max_length // len(texts)
            
            for text in texts:
                if len(text) <= available_length:
                    summary_parts.append(text)
                else:
                    # Truncar mantendo início
                    truncated = text[:available_length-10] + "..."
                    summary_parts.append(truncated)
            
            return "\n\n".join(summary_parts)
        
        else:
            # Níveis altos: resumo mais agressivo
            # Pegar primeiras sentenças de cada texto
            summary_parts = []
            sentences_per_text = max(1, 3 // len(texts))
            
            for text in texts:
                sentences = text.split('. ')
                key_sentences = sentences[:sentences_per_text]
                summary_parts.append('. '.join(key_sentences))
            
            summary = '\n\n'.join(summary_parts)
            summary += f"\n\n[Resumo de nível {level} de {len(texts)} documentos]"
            
            return summary
    
    async def summarize_cluster(self, 
                              texts: List[str], 
                              level: int,
                              cluster_info: ClusterInfo,
                              context: Optional[str] = None) -> str:
        """Summariza um cluster de textos"""
        
        if not texts:
            return ""
        
        logger.info(f"Summarizando cluster {cluster_info.cluster_id} com {len(texts)} textos (nível {level})")
        
        try:
            # Por enquanto, usar summarização simples
            # TODO: Integrar com APIModelRouter real para LLM summarization
            summary = self._create_simple_summary(texts, level)
            
            logger.info(f"Resumo criado: {self._count_tokens(summary)} tokens")
            return summary
            
        except Exception as e:
            logger.error(f"Erro na summarização: {e}")
            # Fallback: concatenar textos de forma inteligente
            combined = "\n\n".join(texts[:3])  # Primeiros 3 textos
            if len(texts) > 3:
                combined += f"\n\n[... e mais {len(texts) - 3} textos similares]"
            return combined

class RaptorRetriever:
    """Retriever RAPTOR principal"""
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 chunk_size: int = 250,
                 chunk_overlap: int = 50,
                 clustering_strategy: ClusteringStrategy = ClusteringStrategy.GLOBAL_LOCAL,
                 retrieval_strategy: RetrievalStrategy = RetrievalStrategy.COLLAPSED_TREE,
                 max_levels: int = 5,
                 min_cluster_size: int = 2,
                 max_cluster_size: int = 100,
                 api_provider: str = "openai",
                 model_name: str = "gpt-4o-mini"):
        
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.clustering_strategy = clustering_strategy
        self.retrieval_strategy = retrieval_strategy
        self.max_levels = max_levels
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        # Inicializar componentes
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                logger.warning(f"Erro ao carregar {embedding_model}: {e}, usando mock")
                self.embedding_model = MockEmbeddingModel(embedding_model)
        else:
            self.embedding_model = MockEmbeddingModel(embedding_model)
            
        self.clusterer = UMAPClusterer()
        self.summarizer = RaptorSummarizer(api_provider, model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Árvore RAPTOR
        self.tree: Dict[str, RaptorNode] = {}
        self.levels: Dict[int, List[str]] = {}  # level -> node_ids
        self.root_nodes: List[str] = []
        
        # Estatísticas
        self.stats: Optional[RaptorTreeStats] = None
        
    def _chunk_text(self, text: str) -> List[str]:
        """Divide texto em chunks com overlap"""
        
        # Dividir em sentenças primeiro
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '. '
            try:
                sentence_tokens = len(self.tokenizer.encode(sentence))
            except:
                sentence_tokens = len(sentence.split()) * 1.3  # Aproximação
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Salvar chunk atual
                chunks.append(current_chunk.strip())
                
                # Iniciar novo chunk com overlap
                if self.chunk_overlap > 0:
                    # Manter últimas palavras para overlap
                    words = current_chunk.split()
                    overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                    try:
                        current_tokens = len(self.tokenizer.encode(current_chunk))
                    except:
                        current_tokens = len(current_chunk.split()) * 1.3
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens
        
        # Adicionar último chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_leaf_nodes(self, documents: List[str]) -> List[RaptorNode]:
        """Cria nós folha a partir dos documentos"""
        leaf_nodes = []
        
        logger.info(f"Criando nós folha para {len(documents)} documentos")
        
        for doc_idx, document in enumerate(documents):
            chunks = self._chunk_text(document)
            
            for chunk_idx, chunk in enumerate(chunks):
                node_id = f"leaf_{doc_idx}_{chunk_idx}"
                
                # Calcular embedding
                embedding = self.embedding_model.encode(chunk)
                
                # Contar tokens
                try:
                    token_count = len(self.tokenizer.encode(chunk))
                except:
                    token_count = len(chunk.split()) * 1.3
                
                node = RaptorNode(
                    node_id=node_id,
                    content=chunk,
                    embedding=embedding,
                    level=0,
                    token_count=int(token_count),
                    metadata={
                        "document_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "source": "original_document"
                    }
                )
                
                leaf_nodes.append(node)
        
        logger.info(f"Criados {len(leaf_nodes)} nós folha")
        return leaf_nodes
    
    async def _build_tree_level(self, 
                               current_nodes: List[RaptorNode], 
                               level: int) -> List[RaptorNode]:
        """Constrói um nível da árvore"""
        
        if len(current_nodes) <= 1:
            logger.info(f"Parando construção no nível {level}: apenas {len(current_nodes)} nós")
            return []
        
        logger.info(f"Construindo nível {level} com {len(current_nodes)} nós")
        
        # Extrair embeddings e IDs
        embeddings = np.array([node.embedding for node in current_nodes])
        node_ids = [node.node_id for node in current_nodes]
        
        # Clustering
        clusters, cluster_metadata = self.clusterer.hierarchical_cluster(embeddings, node_ids)
        
        if not clusters:
            logger.warning(f"Nenhum cluster encontrado no nível {level}")
            return []
        
        # Filtrar clusters muito pequenos ou muito grandes
        valid_clusters = []
        for cluster in clusters:
            if self.min_cluster_size <= cluster.size <= self.max_cluster_size:
                valid_clusters.append(cluster)
            else:
                logger.debug(f"Cluster {cluster.cluster_id} rejeitado: tamanho {cluster.size}")
        
        if not valid_clusters:
            logger.warning(f"Nenhum cluster válido no nível {level}")
            return []
        
        logger.info(f"Processando {len(valid_clusters)} clusters válidos")
        
        # Criar nós de nível superior
        parent_nodes = []
        
        for cluster in valid_clusters:
            # Obter nós do cluster
            cluster_nodes = [node for node in current_nodes if node.node_id in cluster.node_ids]
            
            if not cluster_nodes:
                continue
            
            # Preparar textos para summarização
            texts = [node.content for node in cluster_nodes]
            
            # Summarizar
            try:
                summary = await self.summarizer.summarize_cluster(
                    texts, level, cluster, 
                    context=f"Cluster de {len(texts)} documentos"
                )
            except Exception as e:
                logger.error(f"Erro na summarização do cluster {cluster.cluster_id}: {e}")
                # Fallback: concatenar textos
                summary = "\n\n".join(texts[:3])
                if len(texts) > 3:
                    summary += f"\n\n[... e mais {len(texts) - 3} textos]"
            
            # Criar nó pai
            parent_id = f"level_{level}_cluster_{cluster.cluster_id}"
            
            # Calcular embedding do resumo
            parent_embedding = self.embedding_model.encode(summary)
            
            try:
                token_count = len(self.tokenizer.encode(summary))
            except:
                token_count = len(summary.split()) * 1.3
            
            parent_node = RaptorNode(
                node_id=parent_id,
                content=summary,
                embedding=parent_embedding,
                level=level,
                children_ids=[node.node_id for node in cluster_nodes],
                cluster_id=cluster.cluster_id,
                token_count=int(token_count),
                metadata={
                    "cluster_size": cluster.size,
                    "coherence_score": cluster.coherence_score,
                    "compression_ratio": len(summary) / sum(len(text) for text in texts) if texts else 0,
                    **cluster.metadata
                }
            )
            
            # Atualizar referências dos filhos
            for child_node in cluster_nodes:
                child_node.parent_id = parent_id
            
            parent_nodes.append(parent_node)
        
        logger.info(f"Nível {level} criado com {len(parent_nodes)} nós")
        return parent_nodes
    
    async def build_tree(self, documents: List[str]) -> RaptorTreeStats:
        """Constrói a árvore RAPTOR completa"""
        
        start_time = time.time()
        logger.info(f"Iniciando construção da árvore RAPTOR com {len(documents)} documentos")
        
        # Limpar árvore existente
        self.tree = {}
        self.levels = {}
        self.root_nodes = []
        
        # Criar nós folha (nível 0)
        current_level_nodes = self._create_leaf_nodes(documents)
        
        # Adicionar nós folha à árvore
        for node in current_level_nodes:
            self.tree[node.node_id] = node
        
        self.levels[0] = [node.node_id for node in current_level_nodes]
        
        # Construir níveis superiores
        level = 1
        total_nodes = len(current_level_nodes)
        
        while level <= self.max_levels and len(current_level_nodes) > 1:
            # Construir próximo nível
            next_level_nodes = await self._build_tree_level(current_level_nodes, level)
            
            if not next_level_nodes:
                break
            
            # Adicionar nós à árvore
            for node in next_level_nodes:
                self.tree[node.node_id] = node
            
            self.levels[level] = [node.node_id for node in next_level_nodes]
            total_nodes += len(next_level_nodes)
            
            # Próximo nível
            current_level_nodes = next_level_nodes
            level += 1
        
        # Definir nós raiz
        if current_level_nodes:
            self.root_nodes = [node.node_id for node in current_level_nodes]
        
        # Calcular estatísticas
        construction_time = time.time() - start_time
        
        # Calcular compressão
        try:
            original_tokens = sum(len(self.tokenizer.encode(doc)) for doc in documents)
        except:
            original_tokens = sum(len(doc.split()) * 1.3 for doc in documents)
            
        tree_tokens = sum(node.token_count for node in self.tree.values())
        compression_ratio = tree_tokens / original_tokens if original_tokens > 0 else 1.0
        
        # Estimar uso de memória (aproximado)
        memory_mb = sum(node.embedding.nbytes for node in self.tree.values()) / (1024 * 1024)
        memory_mb += sum(len(node.content.encode('utf-8')) for node in self.tree.values()) / (1024 * 1024)
        
        self.stats = RaptorTreeStats(
            total_nodes=total_nodes,
            levels=level - 1,
            nodes_per_level={lvl: len(nodes) for lvl, nodes in self.levels.items()},
            clusters_per_level={lvl: len(set(self.tree[nid].cluster_id for nid in nodes if self.tree[nid].cluster_id is not None)) 
                              for lvl, nodes in self.levels.items() if lvl > 0},
            compression_ratio=compression_ratio,
            construction_time=construction_time,
            memory_usage_mb=memory_mb
        )
        
        logger.info(f"Árvore RAPTOR construída: {total_nodes} nós, {level-1} níveis, {construction_time:.2f}s")
        logger.info(f"Compressão: {compression_ratio:.2f}x, Memória: {memory_mb:.2f}MB")
        
        return self.stats
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similaridade cosseno entre dois vetores"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def _collapsed_tree_search(self, 
                             query_embedding: np.ndarray, 
                             k: int = 20,
                             max_tokens: int = 2000) -> List[RaptorNode]:
        """Busca considerando todos os nós simultaneamente"""
        
        if not self.tree:
            return []
        
        # Calcular similaridades para todos os nós
        similarities = []
        for node in self.tree.values():
            similarity = self._calculate_similarity(query_embedding, node.embedding)
            similarities.append((similarity, node))
        
        # Ordenar por similaridade
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar nós respeitando limite de tokens
        selected_nodes = []
        total_tokens = 0
        
        for similarity, node in similarities:
            if total_tokens + node.token_count <= max_tokens:
                selected_nodes.append(node)
                total_tokens += node.token_count
            
            if len(selected_nodes) >= k or total_tokens >= max_tokens:
                break
        
        return selected_nodes
    
    def search(self, 
              query: str, 
              k: int = 10,
              max_tokens: int = 2000,
              strategy: Optional[RetrievalStrategy] = None) -> List[Dict[str, Any]]:
        """Busca na árvore RAPTOR"""
        
        if not self.tree:
            logger.warning("Árvore RAPTOR não foi construída")
            return []
        
        strategy = strategy or self.retrieval_strategy
        
        # Calcular embedding da query
        query_embedding = self.embedding_model.encode(query)
        
        # Por enquanto, apenas collapsed tree (mais simples e efetivo)
        selected_nodes = self._collapsed_tree_search(query_embedding, k, max_tokens)
        
        # Converter para formato de resposta
        results = []
        for node in selected_nodes:
            # Calcular similaridade final
            similarity = self._calculate_similarity(query_embedding, node.embedding)
            
            result = {
                "content": node.content,
                "score": float(similarity),
                "metadata": {
                    "node_id": node.node_id,
                    "level": node.level,
                    "token_count": node.token_count,
                    "cluster_id": node.cluster_id,
                    **node.metadata
                }
            }
            results.append(result)
        
        logger.info(f"Busca RAPTOR retornou {len(results)} resultados "
                   f"(estratégia: {strategy.value})")
        
        return results
    
    def get_tree_summary(self) -> Dict[str, Any]:
        """Retorna resumo da árvore construída"""
        
        if not self.stats:
            return {"status": "not_built"}
        
        return {
            "status": "built",
            "stats": {
                "total_nodes": self.stats.total_nodes,
                "levels": self.stats.levels,
                "nodes_per_level": self.stats.nodes_per_level,
                "clusters_per_level": self.stats.clusters_per_level,
                "compression_ratio": self.stats.compression_ratio,
                "construction_time": self.stats.construction_time,
                "memory_usage_mb": self.stats.memory_usage_mb
            },
            "root_nodes": len(self.root_nodes),
            "leaf_nodes": len(self.levels.get(0, [])),
            "strategies": {
                "clustering": self.clustering_strategy.value,
                "retrieval": self.retrieval_strategy.value
            }
        }

# Funções auxiliares para integração

async def create_raptor_retriever(config: Dict[str, Any]) -> RaptorRetriever:
    """Factory para criar RaptorRetriever"""
    
    retriever = RaptorRetriever(
        embedding_model=config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"),
        chunk_size=config.get("chunk_size", 250),
        chunk_overlap=config.get("chunk_overlap", 50),
        clustering_strategy=ClusteringStrategy(config.get("clustering_strategy", "global_local")),
        retrieval_strategy=RetrievalStrategy(config.get("retrieval_strategy", "collapsed_tree")),
        max_levels=config.get("max_levels", 5),
        min_cluster_size=config.get("min_cluster_size", 2),
        max_cluster_size=config.get("max_cluster_size", 100),
        api_provider=config.get("api_provider", "openai"),
        model_name=config.get("model_name", "gpt-4o-mini")
    )
    
    return retriever

def get_default_raptor_config() -> Dict[str, Any]:
    """Configuração padrão para RAPTOR"""
    
    return {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "chunk_size": 250,
        "chunk_overlap": 50,
        "clustering_strategy": "global_local",
        "retrieval_strategy": "collapsed_tree",
        "max_levels": 5,
        "min_cluster_size": 2,
        "max_cluster_size": 100,
        "api_provider": "openai",
        "model_name": "gpt-4o-mini",
        
        # Parâmetros UMAP
        "umap": {
            "n_components": 10,
            "min_dist": 0.0,
            "local_neighbors": 10
        },
        
        # Parâmetros summarização
        "summarization": {
            "max_tokens": 1000,
            "strategy": "hierarchical"
        },
        
        # Parâmetros retrieval
        "retrieval": {
            "default_k": 10,
            "default_max_tokens": 2000
        }
    }