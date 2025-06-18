"""
Enhanced GraphRAG - Otimização do Neo4j para enriquecimento de contexto.
Implementa técnicas avançadas de graph traversal e community detection.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple
import asyncio
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict

from src.graphdb.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """Contexto extraído do grafo."""
    entities: List[Dict]
    relationships: List[Dict]
    communities: List[Set[str]]
    central_entities: List[str]
    context_summary: str


class EnhancedGraphRAG:
    """
    Implementa GraphRAG avançado com:
    1. Multi-hop reasoning otimizado
    2. Community detection (Louvain)
    3. Entity importance scoring
    4. Subgraph caching
    5. Semantic filtering de caminhos
    """
    
    def __init__(self,
                 neo4j_store: Optional[Neo4jStore] = None,
                 max_hops: int = 3,
                 community_min_size: int = 3,
                 cache_ttl: int = 3600):
        # FASE 1: Neo4j opcional - só inicializar se necessário
        try:
            self.neo4j_store = neo4j_store or Neo4jStore()
            self.neo4j_available = True
        except Exception as e:
            logger.warning(f"Neo4j não disponível: {e}. GraphRAG funcionará sem enriquecimento de grafo.")
            self.neo4j_store = None
            self.neo4j_available = False
            
        self.max_hops = max_hops
        self.community_min_size = community_min_size
        self.cache_ttl = cache_ttl
        self.subgraph_cache = {}
        
    async def enrich_with_graph_context(self, 
                                        documents: List[Dict],
                                        query_embedding: Optional[List[float]] = None) -> List[Dict]:
        """
        Enriquece documentos com contexto do grafo.
        
        Args:
            documents: Documentos recuperados
            query_embedding: Embedding da query para guided traversal
            
        Returns:
            Documentos enriquecidos com contexto do grafo
        """
        enriched_docs = []
        
        for doc in documents:
            try:
                # FASE 1: Verificar se Neo4j está disponível
                if not self.neo4j_available:
                    # Retornar documento sem enriquecimento
                    enriched_docs.append(doc)
                    continue
                
                # 1. Extrair entidades do documento
                entities = await self._extract_entities(doc.get("content", ""))
                
                if not entities:
                    enriched_docs.append(doc)
                    continue
                
                # 2. Buscar contexto do grafo
                graph_context = await self._get_graph_context(
                    entities,
                    query_embedding
                )
                
                # 3. Enriquecer documento
                enriched_doc = doc.copy()
                enriched_doc["graph_context"] = {
                    "entities": graph_context.entities,
                    "relationships": graph_context.relationships,
                    "communities": [list(c) for c in graph_context.communities],
                    "central_entities": graph_context.central_entities,
                    "summary": graph_context.context_summary
                }
                
                # 4. Adicionar contexto expandido ao conteúdo
                enriched_doc["enriched_content"] = self._merge_context(
                    doc.get("content", ""),
                    graph_context
                )
                
                enriched_docs.append(enriched_doc)
                
            except Exception as e:
                logger.error(f"Erro ao enriquecer documento: {e}")
                enriched_docs.append(doc)
        
        return enriched_docs
    
    async def _extract_entities(self, text: str) -> List[str]:
        """
        Extrai entidades do texto.
        Por enquanto, usa heurística simples. Pode ser melhorado com NER.
        """
        # Heurística: palavras capitalizadas e termos técnicos
        import re
        
        entities = []
        
        # Palavras capitalizadas (possíveis nomes próprios)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)
        
        # Termos entre aspas
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        
        # Termos técnicos comuns (pode expandir)
        tech_terms = re.findall(r'\b(?:API|REST|GraphQL|Docker|Kubernetes|Python|Java|ML|AI|RAG|LLM)\b', text, re.IGNORECASE)
        entities.extend(tech_terms)
        
        # Limpar e deduplicar
        entities = list(set(e.strip() for e in entities if len(e.strip()) > 2))
        
        return entities[:10]  # Limitar a 10 entidades
    
    async def _get_graph_context(self,
                                 entities: List[str],
                                 query_embedding: Optional[List[float]] = None) -> GraphContext:
        """
        Obtém contexto expandido do grafo para as entidades.
        """
        # Verificar cache
        cache_key = hash(tuple(sorted(entities)))
        if cache_key in self.subgraph_cache:
            logger.debug(f"Cache hit para entidades: {entities[:3]}...")
            return self.subgraph_cache[cache_key]
        
        # 1. Query Neo4j para subgrafo
        cypher_query = """
        MATCH (e:Entity)
        WHERE e.name IN $entities
        OPTIONAL MATCH path = (e)-[r*1..%d]-(related:Entity)
        WITH e, related, relationships(path) as rels
        RETURN 
            collect(DISTINCT e) as source_entities,
            collect(DISTINCT related) as related_entities,
            collect(DISTINCT rels) as relationships
        LIMIT 100
        """ % self.max_hops
        
        try:
            result = await self.neo4j_store.query(cypher_query, {"entities": entities})
            
            if not result:
                return GraphContext([], [], [], [], "Sem contexto de grafo disponível")
            
            # 2. Processar resultados
            all_entities = []
            all_relationships = []
            
            for record in result:
                # Processar entidades
                for e in record.get("source_entities", []) + record.get("related_entities", []):
                    if e:
                        all_entities.append({
                            "id": e.get("id"),
                            "name": e.get("name"),
                            "type": e.get("type"),
                            "properties": e.get("properties", {})
                        })
                
                # Processar relacionamentos
                for rel_list in record.get("relationships", []):
                    if rel_list:
                        for rel in rel_list:
                            all_relationships.append({
                                "source": rel.start_node.get("name"),
                                "target": rel.end_node.get("name"),
                                "type": rel.type,
                                "properties": dict(rel)
                            })
            
            # 3. Detectar comunidades
            communities = await self._detect_communities(all_entities, all_relationships)
            
            # 4. Identificar entidades centrais
            central_entities = self._identify_central_entities(all_entities, all_relationships)
            
            # 5. Gerar resumo do contexto
            context_summary = self._generate_context_summary(
                all_entities,
                all_relationships,
                communities,
                central_entities
            )
            
            # Criar contexto
            graph_context = GraphContext(
                entities=all_entities,
                relationships=all_relationships,
                communities=communities,
                central_entities=central_entities,
                context_summary=context_summary
            )
            
            # Cachear resultado
            self.subgraph_cache[cache_key] = graph_context
            
            return graph_context
            
        except Exception as e:
            logger.error(f"Erro ao buscar contexto do grafo: {e}")
            return GraphContext([], [], [], [], "Erro ao acessar grafo")
    
    async def _detect_communities(self, 
                                  entities: List[Dict], 
                                  relationships: List[Dict]) -> List[Set[str]]:
        """
        Detecta comunidades no subgrafo usando algoritmo de Louvain.
        """
        try:
            # Criar grafo NetworkX
            G = nx.Graph()
            
            # Adicionar nós
            for entity in entities:
                G.add_node(entity["name"], **entity)
            
            # Adicionar arestas
            for rel in relationships:
                G.add_edge(rel["source"], rel["target"], **rel)
            
            # Detectar comunidades
            if len(G.nodes()) > 0:
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(G)
                
                # Agrupar por comunidade
                communities = defaultdict(set)
                for node, comm_id in partition.items():
                    communities[comm_id].add(node)
                
                # Filtrar comunidades pequenas
                valid_communities = [
                    nodes for nodes in communities.values()
                    if len(nodes) >= self.community_min_size
                ]
                
                return valid_communities
            
        except ImportError:
            logger.warning("python-louvain não instalado. Pulando detecção de comunidades.")
        except Exception as e:
            logger.error(f"Erro na detecção de comunidades: {e}")
        
        return []
    
    def _identify_central_entities(self, 
                                   entities: List[Dict], 
                                   relationships: List[Dict]) -> List[str]:
        """
        Identifica entidades mais centrais usando métricas de centralidade.
        """
        try:
            # Criar grafo
            G = nx.Graph()
            
            for entity in entities:
                G.add_node(entity["name"])
            
            for rel in relationships:
                G.add_edge(rel["source"], rel["target"])
            
            if len(G.nodes()) > 0:
                # Calcular diferentes centralidades
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                
                # Combinar scores
                combined_scores = {}
                for node in G.nodes():
                    combined_scores[node] = (
                        degree_centrality.get(node, 0) * 0.5 +
                        betweenness_centrality.get(node, 0) * 0.5
                    )
                
                # Top 5 entidades mais centrais
                central = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                return [node for node, score in central[:5]]
            
        except Exception as e:
            logger.error(f"Erro ao identificar entidades centrais: {e}")
        
        return []
    
    def _generate_context_summary(self,
                                  entities: List[Dict],
                                  relationships: List[Dict],
                                  communities: List[Set[str]],
                                  central_entities: List[str]) -> str:
        """
        Gera um resumo textual do contexto do grafo.
        """
        summary_parts = []
        
        # Estatísticas básicas
        summary_parts.append(f"Contexto do grafo: {len(entities)} entidades, {len(relationships)} relações")
        
        # Entidades centrais
        if central_entities:
            summary_parts.append(f"Entidades principais: {', '.join(central_entities[:3])}")
        
        # Comunidades
        if communities:
            summary_parts.append(f"Identificadas {len(communities)} comunidades de conhecimento")
        
        # Tipos de relacionamento mais comuns
        if relationships:
            rel_types = defaultdict(int)
            for rel in relationships:
                rel_types[rel["type"]] += 1
            
            top_rel_types = sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:3]
            rel_summary = ", ".join([f"{t} ({c})" for t, c in top_rel_types])
            summary_parts.append(f"Relações principais: {rel_summary}")
        
        return ". ".join(summary_parts)
    
    def _merge_context(self, original_content: str, graph_context: GraphContext) -> str:
        """
        Mescla o contexto do grafo com o conteúdo original.
        """
        context_parts = [original_content]
        
        if graph_context.context_summary:
            context_parts.append(f"\n\n[Contexto do Grafo]\n{graph_context.context_summary}")
        
        if graph_context.central_entities:
            context_parts.append(f"\nEntidades relacionadas: {', '.join(graph_context.central_entities)}")
        
        return "\n".join(context_parts) 