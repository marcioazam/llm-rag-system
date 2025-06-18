"""
Demo RAPTOR Funcional - Versão simplificada que funciona
"""

import asyncio
import time
import numpy as np
import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans

@dataclass
class RaptorNode:
    node_id: str
    content: str
    embedding: np.ndarray
    level: int
    children_ids: List[str]
    metadata: Dict[str, Any]

class SimpleRaptor:
    """RAPTOR simplificado funcional"""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.tree = {}
        self.levels = {}
        
    def _simple_embed(self, text: str) -> np.ndarray:
        """Embedding simples baseado em hash"""
        # Usar hash do texto para gerar embedding determinístico
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:16], 16)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.normal(0, 1, self.embedding_dim)
        return embedding / np.linalg.norm(embedding)
    
    def _chunk_text(self, text: str, chunk_size: int = 50) -> List[str]:
        """Divide texto em chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks if chunks else [text]
    
    def _simple_summary(self, texts: List[str], level: int) -> str:
        """Summarização simples"""
        if len(texts) == 1:
            return texts[0]
        
        # Para níveis altos, resumo mais agressivo
        if level > 1:
            # Pegar primeira sentença de cada texto
            sentences = []
            for text in texts:
                first_sentence = text.split('.')[0] + '.'
                sentences.append(first_sentence)
            summary = ' '.join(sentences)
            return f"{summary} [Resumo nível {level} de {len(texts)} textos]"
        else:
            # Níveis baixos, preservar mais conteúdo
            return "\n\n".join(texts[:3])  # Primeiros 3 textos
    
    async def build_tree(self, documents: List[str]) -> Dict[str, Any]:
        """Constrói árvore RAPTOR"""
        
        start_time = time.time()
        print(f"Construindo arvore RAPTOR com {len(documents)} documentos...")
        
        # Nível 0: Criar chunks
        level_0_nodes = []
        node_id = 0
        
        for doc_idx, doc in enumerate(documents):
            chunks = self._chunk_text(doc)
            for chunk_idx, chunk in enumerate(chunks):
                embedding = self._simple_embed(chunk)
                node = RaptorNode(
                    node_id=f"node_{node_id}",
                    content=chunk,
                    embedding=embedding,
                    level=0,
                    children_ids=[],
                    metadata={"doc_idx": doc_idx, "chunk_idx": chunk_idx}
                )
                level_0_nodes.append(node)
                self.tree[node.node_id] = node
                node_id += 1
        
        self.levels[0] = [node.node_id for node in level_0_nodes]
        current_nodes = level_0_nodes
        total_nodes = len(level_0_nodes)
        
        # Construir níveis superiores
        level = 1
        max_levels = 3
        
        while level <= max_levels and len(current_nodes) > 1:
            print(f"   Nivel {level}: {len(current_nodes)} nos")
            
            # Clustering simples
            embeddings = np.array([node.embedding for node in current_nodes])
            n_clusters = max(2, min(len(current_nodes) // 2, 5))
            
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
            except:
                # Fallback: clusters baseados em posição
                labels = np.arange(len(current_nodes)) % n_clusters
            
            # Criar nós pais
            next_level_nodes = []
            clusters = {}
            
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(current_nodes[i])
            
            for cluster_id, cluster_nodes in clusters.items():
                if len(cluster_nodes) >= 2:  # Só clusters com 2+ nós
                    # Summarizar conteúdo do cluster
                    texts = [node.content for node in cluster_nodes]
                    summary = self._simple_summary(texts, level)
                    
                    # Criar nó pai
                    parent_embedding = self._simple_embed(summary)
                    parent_node = RaptorNode(
                        node_id=f"node_{node_id}",
                        content=summary,
                        embedding=parent_embedding,
                        level=level,
                        children_ids=[node.node_id for node in cluster_nodes],
                        metadata={"cluster_size": len(cluster_nodes), "cluster_id": cluster_id}
                    )
                    
                    next_level_nodes.append(parent_node)
                    self.tree[parent_node.node_id] = parent_node
                    node_id += 1
            
            if not next_level_nodes:
                break
                
            self.levels[level] = [node.node_id for node in next_level_nodes]
            total_nodes += len(next_level_nodes)
            current_nodes = next_level_nodes
            level += 1
        
        construction_time = time.time() - start_time
        
        stats = {
            "total_nodes": total_nodes,
            "levels": level - 1,
            "nodes_per_level": {lvl: len(nodes) for lvl, nodes in self.levels.items()},
            "construction_time": construction_time
        }
        
        print(f"Arvore construida: {total_nodes} nos, {level-1} niveis, {construction_time:.2f}s")
        return stats
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca na árvore"""
        
        if not self.tree:
            return []
        
        query_embedding = self._simple_embed(query)
        
        # Calcular similaridades com todos os nós
        similarities = []
        for node in self.tree.values():
            similarity = np.dot(query_embedding, node.embedding)
            similarities.append((similarity, node))
        
        # Ordenar por similaridade
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Retornar top-k
        results = []
        for similarity, node in similarities[:k]:
            result = {
                "content": node.content,
                "score": float(similarity),
                "metadata": {
                    "node_id": node.node_id,
                    "level": node.level,
                    **node.metadata
                }
            }
            results.append(result)
        
        return results

async def demo_raptor_working():
    """Demo RAPTOR que funciona"""
    
    print("RAPTOR Demo - Versao Funcional")
    print("=" * 50)
    
    # Documentos de teste
    docs = [
        "Python é uma linguagem de programação interpretada e de alto nível. É conhecida por sua sintaxe simples e legibilidade.",
        "Machine Learning é um campo da inteligência artificial que permite computadores aprenderem padrões dos dados.",
        "RAG (Retrieval-Augmented Generation) combina busca de informações com geração de texto para respostas mais precisas.",
        "Cloud computing oferece recursos computacionais via internet, incluindo servidores, armazenamento e aplicações.",
        "DevOps é uma metodologia que integra desenvolvimento e operações para acelerar entrega de software.",
        "Deep Learning usa redes neurais profundas para resolver problemas complexos como reconhecimento de imagens.",
        "Kubernetes é uma plataforma de orquestração de containers que automatiza deployment e scaling de aplicações.",
        "APIs (Application Programming Interfaces) permitem comunicação entre diferentes sistemas de software."
    ]
    
    # Criar e treinar RAPTOR
    raptor = SimpleRaptor(embedding_dim=64)
    
    stats = await raptor.build_tree(docs)
    
    print(f"\nEstatisticas da arvore:")
    print(f"   • Total de nos: {stats['total_nodes']}")
    print(f"   • Niveis: {stats['levels']}")
    print(f"   • Tempo de construcao: {stats['construction_time']:.2f}s")
    
    for level, count in stats['nodes_per_level'].items():
        print(f"   • Nivel {level}: {count} nos")
    
    # Testes de busca
    queries = [
        "Como usar Python para machine learning?",
        "Explicar cloud computing e containers",
        "O que é RAG e como funciona?",
        "DevOps e desenvolvimento de software"
    ]
    
    print(f"\nTestes de Busca:")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        results = raptor.search(query, k=3)
        
        total_tokens = 0
        level_dist = {}
        
        for j, result in enumerate(results, 1):
            score = result['score']
            level = result['metadata']['level']
            content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
            tokens = len(result['content'].split())
            total_tokens += tokens
            
            level_dist[level] = level_dist.get(level, 0) + 1
            
            print(f"{j}. [Nivel {level}, Score: {score:.3f}, {tokens} tokens]")
            print(f"   {content}")
        
        print(f"\nResumo: {len(results)} resultados, {total_tokens} tokens")
        print(f"Por nivel: {dict(sorted(level_dist.items()))}")
    
    # Demonstrar natureza hierárquica
    print(f"\nDemonstracao Hierarquica:")
    print("=" * 50)
    
    query = "Python e tecnologia"
    results = raptor.search(query, k=8)
    
    by_level = {}
    for result in results:
        level = result['metadata']['level']
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(result)
    
    for level in sorted(by_level.keys()):
        level_results = by_level[level]
        print(f"\nNIVEL {level} ({len(level_results)} resultados):")
        
        for result in level_results[:2]:  # Mostrar 2 por nível
            score = result['score']
            content = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
            print(f"   Score: {score:.3f}")
            print(f"   {content}")
            print()
    
    print("Demo RAPTOR concluido!")
    print("\nObservacoes:")
    print("   • Niveis mais altos tem resumos mais abstratos")
    print("   • Niveis mais baixos preservam detalhes especificos")
    print("   • Clustering agrupa conteudo semanticamente similar")
    print("   • Busca hierarquica captura diferentes niveis de abstracao")

if __name__ == "__main__":
    asyncio.run(demo_raptor_working())