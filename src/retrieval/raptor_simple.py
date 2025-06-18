"""
RAPTOR Simplificado - Versão que funciona sem dependências especiais

Implementa os conceitos principais do RAPTOR sem UMAP, sentence-transformers ou tiktoken:
- Clustering simples com K-means
- Summarização por concatenação inteligente
- Retrieval hierárquico
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import hashlib

logger = logging.getLogger(__name__)

class SimpleStrategy(Enum):
    """Estratégias simples de clustering"""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"

class RetrievalStrategy(Enum):
    """Estratégias de retrieval"""
    FLAT = "flat"
    HIERARCHICAL = "hierarchical"

@dataclass
class SimpleNode:
    """Nó simplificado da árvore"""
    node_id: str
    content: str
    embedding: np.ndarray
    level: int
    children_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    cluster_id: Optional[int] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimpleCluster:
    """Informações de cluster simples"""
    cluster_id: int
    node_ids: List[str]
    centroid: np.ndarray
    size: int

@dataclass
class SimpleStats:
    """Estatísticas simples"""
    total_nodes: int
    levels: int
    nodes_per_level: Dict[int, int]
    construction_time: float

class SimpleEmbedder:
    """Embedder simples baseado em hash TF-IDF básico"""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vocab = {}
        self.idf = {}
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenização simples"""
        import re
        # Limpar e dividir texto
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def _build_vocab(self, texts: List[str]):
        """Constrói vocabulário simples"""
        word_counts = {}
        doc_counts = {}
        
        for text in texts:
            words = self._tokenize(text)
            unique_words = set(words)
            
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            for word in unique_words:
                doc_counts[word] = doc_counts.get(word, 0) + 1
        
        # Manter apenas palavras mais frequentes
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        self.vocab = {word: i for i, (word, _) in enumerate(sorted_words[:self.dim//2])}
        
        # Calcular IDF simples
        total_docs = len(texts)
        for word in self.vocab:
            self.idf[word] = np.log(total_docs / (doc_counts.get(word, 1) + 1))
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Codifica texto em embedding"""
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = text
            single = False
        
        if not self.vocab:
            self._build_vocab(texts)
        
        embeddings = []
        for t in texts:
            words = self._tokenize(t)
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Criar vetor TF-IDF simples
            vector = np.zeros(self.dim)
            for word, count in word_counts.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    tf = count / len(words) if words else 0
                    idf = self.idf.get(word, 1.0)
                    vector[idx] = tf * idf
            
            # Adicionar características de hash para diversidade
            hash_features = []
            for i in range(self.dim - len(self.vocab)):
                hash_val = int(hashlib.md5((t + str(i)).encode()).hexdigest()[:8], 16)
                hash_features.append((hash_val % 1000) / 1000.0)
            
            if len(self.vocab) + len(hash_features) < self.dim:
                hash_features.extend([0.0] * (self.dim - len(self.vocab) - len(hash_features)))
            
            vector[len(self.vocab):] = hash_features[:self.dim - len(self.vocab)]
            
            # Normalizar
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            embeddings.append(vector)
        
        if single:
            return embeddings[0]
        return np.array(embeddings)

class SimpleClusterer:
    """Clusterer simples com K-means"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def cluster(self, embeddings: np.ndarray, node_ids: List[str], 
                max_clusters: int = 10) -> Tuple[List[SimpleCluster], Dict[str, Any]]:
        """Clustering simples"""
        
        if len(embeddings) <= 1:
            return [], {"error": "insufficient_data"}
        
        # Determinar número de clusters
        n_clusters = min(max_clusters, max(2, len(embeddings) // 3))
        
        try:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Criar clusters
            clusters = []
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                if not np.any(mask):
                    continue
                
                cluster_embeddings = embeddings[mask]
                cluster_node_ids = [node_ids[i] for i in range(len(node_ids)) if mask[i]]
                centroid = np.mean(cluster_embeddings, axis=0)
                
                cluster = SimpleCluster(
                    cluster_id=cluster_id,
                    node_ids=cluster_node_ids,
                    centroid=centroid,
                    size=len(cluster_node_ids)
                )
                clusters.append(cluster)
            
            metadata = {
                "total_clusters": len(clusters),
                "avg_cluster_size": np.mean([c.size for c in clusters]) if clusters else 0,
                "method": "kmeans"
            }
            
            return clusters, metadata
            
        except Exception as e:
            logger.error(f"Erro no clustering: {e}")
            # Fallback: cada nó é seu próprio cluster
            clusters = []
            for i, node_id in enumerate(node_ids):
                cluster = SimpleCluster(
                    cluster_id=i,
                    node_ids=[node_id],
                    centroid=embeddings[i],
                    size=1
                )
                clusters.append(cluster)
            
            return clusters, {"method": "fallback"}

class SimpleSummarizer:
    """Summarizador simples"""
    
    def __init__(self, max_length: int = 500):
        self.max_length = max_length
    
    def _count_words(self, text: str) -> int:
        """Conta palavras aproximadamente"""
        return len(text.split())
    
    async def summarize(self, texts: List[str], level: int) -> str:
        """Summariza textos"""
        
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Estratégia baseada no nível
        if level <= 1:
            # Níveis baixos: concatenar preservando detalhes
            combined = "\n\n".join(texts)
            if self._count_words(combined) <= self.max_length:
                return combined
            
            # Truncar de forma inteligente
            summary_parts = []
            words_per_text = self.max_length // len(texts)
            
            for text in texts:
                words = text.split()
                if len(words) <= words_per_text:
                    summary_parts.append(text)
                else:
                    truncated = " ".join(words[:words_per_text-1]) + "..."
                    summary_parts.append(truncated)
            
            return "\n\n".join(summary_parts)
        
        else:
            # Níveis altos: resumo mais agressivo
            key_sentences = []
            sentences_per_text = max(1, 2 // len(texts))
            
            for text in texts:
                sentences = text.split('. ')
                # Pegar primeiras sentenças
                selected = sentences[:sentences_per_text]
                key_sentences.extend(selected)
            
            summary = '. '.join(key_sentences)
            if len(summary) > 0 and not summary.endswith('.'):
                summary += '.'
            
            summary += f"\n\n[Resumo nível {level} de {len(texts)} documentos]"
            
            return summary

class SimpleRaptor:
    """RAPTOR simplificado"""
    
    def __init__(self, 
                 chunk_size: int = 200,
                 max_levels: int = 3,
                 min_cluster_size: int = 2,
                 embedding_dim: int = 128):
        
        self.chunk_size = chunk_size
        self.max_levels = max_levels
        self.min_cluster_size = min_cluster_size
        
        # Componentes
        self.embedder = SimpleEmbedder(embedding_dim)
        self.clusterer = SimpleClusterer()
        self.summarizer = SimpleSummarizer()
        
        # Árvore
        self.tree: Dict[str, SimpleNode] = {}
        self.levels: Dict[int, List[str]] = {}
        self.root_nodes: List[str] = []
        self.stats: Optional[SimpleStats] = None
    
    def _chunk_text(self, text: str) -> List[str]:
        """Divide texto em chunks simples"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    def _create_leaf_nodes(self, documents: List[str]) -> List[SimpleNode]:
        """Cria nós folha"""
        leaf_nodes = []
        
        for doc_idx, document in enumerate(documents):
            chunks = self._chunk_text(document)
            
            for chunk_idx, chunk in enumerate(chunks):
                node_id = f"leaf_{doc_idx}_{chunk_idx}"
                
                # Calcular embedding (será feito em lote depois)
                node = SimpleNode(
                    node_id=node_id,
                    content=chunk,
                    embedding=np.zeros(self.embedder.dim),  # Temporário
                    level=0,
                    token_count=len(chunk.split()),
                    metadata={
                        "document_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "source": "original"
                    }
                )
                leaf_nodes.append(node)
        
        # Calcular embeddings em lote
        texts = [node.content for node in leaf_nodes]
        embeddings = self.embedder.encode(texts)
        
        for node, embedding in zip(leaf_nodes, embeddings):
            node.embedding = embedding
        
        return leaf_nodes
    
    async def _build_tree_level(self, current_nodes: List[SimpleNode], level: int) -> List[SimpleNode]:
        """Constrói um nível da árvore"""
        
        if len(current_nodes) <= 1:
            return []
        
        logger.info(f"Construindo nível {level} com {len(current_nodes)} nós")
        
        # Clustering
        embeddings = np.array([node.embedding for node in current_nodes])
        node_ids = [node.node_id for node in current_nodes]
        
        clusters, _ = self.clusterer.cluster(embeddings, node_ids)
        
        if not clusters:
            return []
        
        # Filtrar clusters pequenos
        valid_clusters = [c for c in clusters if c.size >= self.min_cluster_size]
        
        if not valid_clusters:
            return []
        
        # Criar nós pais
        parent_nodes = []
        
        for cluster in valid_clusters:
            cluster_nodes = [node for node in current_nodes if node.node_id in cluster.node_ids]
            
            if not cluster_nodes:
                continue
            
            # Summarizar
            texts = [node.content for node in cluster_nodes]
            summary = await self.summarizer.summarize(texts, level)
            
            # Criar nó pai
            parent_id = f"level_{level}_cluster_{cluster.cluster_id}"
            parent_embedding = self.embedder.encode(summary)
            
            parent_node = SimpleNode(
                node_id=parent_id,
                content=summary,
                embedding=parent_embedding,
                level=level,
                children_ids=[node.node_id for node in cluster_nodes],
                cluster_id=cluster.cluster_id,
                token_count=len(summary.split()),
                metadata={
                    "cluster_size": cluster.size,
                    "compression_ratio": len(summary) / sum(len(text) for text in texts) if texts else 0
                }
            )
            
            # Atualizar filhos
            for child_node in cluster_nodes:
                child_node.parent_id = parent_id
            
            parent_nodes.append(parent_node)
        
        return parent_nodes
    
    async def build_tree(self, documents: List[str]) -> SimpleStats:
        """Constrói árvore completa"""
        
        start_time = time.time()
        logger.info(f"Construindo árvore RAPTOR simples com {len(documents)} documentos")
        
        # Limpar
        self.tree = {}
        self.levels = {}
        self.root_nodes = []
        
        # Criar folhas
        current_nodes = self._create_leaf_nodes(documents)
        
        # Adicionar à árvore
        for node in current_nodes:
            self.tree[node.node_id] = node
        
        self.levels[0] = [node.node_id for node in current_nodes]
        total_nodes = len(current_nodes)
        
        # Construir níveis superiores
        level = 1
        while level <= self.max_levels and len(current_nodes) > 1:
            next_nodes = await self._build_tree_level(current_nodes, level)
            
            if not next_nodes:
                break
            
            # Adicionar à árvore
            for node in next_nodes:
                self.tree[node.node_id] = node
            
            self.levels[level] = [node.node_id for node in next_nodes]
            total_nodes += len(next_nodes)
            
            current_nodes = next_nodes
            level += 1
        
        # Definir raízes
        if current_nodes:
            self.root_nodes = [node.node_id for node in current_nodes]
        
        # Estatísticas
        construction_time = time.time() - start_time
        
        self.stats = SimpleStats(
            total_nodes=total_nodes,
            levels=level - 1,
            nodes_per_level={lvl: len(nodes) for lvl, nodes in self.levels.items()},
            construction_time=construction_time
        )
        
        logger.info(f"Árvore construída: {total_nodes} nós, {level-1} níveis, {construction_time:.2f}s")
        return self.stats
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca na árvore"""
        
        if not self.tree:
            return []
        
        # Embedding da query
        query_embedding = self.embedder.encode(query)
        
        # Calcular similaridades
        similarities = []
        for node in self.tree.values():
            similarity = np.dot(query_embedding, node.embedding)
            similarities.append((similarity, node))
        
        # Ordenar e selecionar
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for similarity, node in similarities[:k]:
            result = {
                "content": node.content,
                "score": float(similarity),
                "metadata": {
                    "node_id": node.node_id,
                    "level": node.level,
                    "token_count": node.token_count,
                    **node.metadata
                }
            }
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas"""
        if not self.stats:
            return {"status": "not_built"}
        
        return {
            "status": "built",
            "total_nodes": self.stats.total_nodes,
            "levels": self.stats.levels,
            "nodes_per_level": self.stats.nodes_per_level,
            "construction_time": self.stats.construction_time,
            "root_nodes": len(self.root_nodes)
        }

# Funções auxiliares

async def create_simple_raptor(config: Dict[str, Any]) -> SimpleRaptor:
    """Cria RAPTOR simples"""
    
    return SimpleRaptor(
        chunk_size=config.get("chunk_size", 200),
        max_levels=config.get("max_levels", 3),
        min_cluster_size=config.get("min_cluster_size", 2),
        embedding_dim=config.get("embedding_dim", 128)
    )

def get_simple_raptor_config() -> Dict[str, Any]:
    """Configuração padrão simplificada"""
    return {
        "chunk_size": 200,
        "max_levels": 3,
        "min_cluster_size": 2,
        "embedding_dim": 128
    }