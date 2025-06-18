"""
GraphRAG Enhancement para Sistema RAG
Implementa técnicas do Microsoft GraphRAG com community detection e multi-hop reasoning
Baseado em: https://github.com/microsoft/graphrag
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import networkx as nx
from community import community_louvain
import numpy as np
from collections import defaultdict
import json

from ..models.api_model_router import APIModelRouter
from ..graphdb.neo4j_store import Neo4jStore
from ..embeddings.api_embedding_service import APIEmbeddingService

logger = logging.getLogger(__name__)

@dataclass
class CodeEntity:
    """Representa uma entidade de código no grafo"""
    id: str
    name: str
    type: str  # 'class', 'function', 'module', 'variable'
    content: str
    file_path: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    community_id: Optional[int] = None

@dataclass
class CodeRelationship:
    """Representa relacionamento entre entidades"""
    source_id: str
    target_id: str
    relationship_type: str  # 'imports', 'calls', 'inherits', 'uses'
    metadata: Dict[str, Any]
    weight: float = 1.0

@dataclass
class Community:
    """Representa uma comunidade detectada no grafo"""
    id: int
    entities: List[CodeEntity]
    summary: str
    centroid_embedding: List[float]
    metadata: Dict[str, Any]

class GraphRAGEnhancer:
    """
    Implementa GraphRAG com community detection via Louvain algorithm
    e multi-hop reasoning para código
    """
    
    def __init__(self, neo4j_store: Neo4jStore = None):
        self.neo4j_store = neo4j_store or Neo4jStore()
        self.model_router = APIModelRouter()
        self.embedding_service = APIEmbeddingService()
        
        # NetworkX para análise de grafo
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        
        # Cache de comunidades
        self.communities: Dict[int, Community] = {}
        self.entity_to_community: Dict[str, int] = {}
        
        # Configurações
        self.config = {
            'louvain_resolution': 1.0,  # Parâmetro de resolução para Louvain
            'max_hops': 3,  # Máximo de hops para reasoning
            'community_summary_model': 'gpt-4o-mini',
            'semantic_threshold': 0.7,
            'min_community_size': 3
        }
    
    async def build_knowledge_graph(self, code_entities: List[CodeEntity]) -> nx.Graph:
        """
        Constrói knowledge graph a partir de entidades de código
        Implementa construção automática via LLM conforme Microsoft GraphRAG
        """
        logger.info(f"Construindo knowledge graph com {len(code_entities)} entidades")
        
        # Adicionar nós ao grafo
        for entity in code_entities:
            self.graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type,
                content=entity.content,
                file_path=entity.file_path,
                metadata=entity.metadata
            )
        
        # Extrair relacionamentos via LLM
        relationships = await self._extract_relationships_llm(code_entities)
        
        # Adicionar arestas ao grafo
        for rel in relationships:
            self.graph.add_edge(
                rel.source_id,
                rel.target_id,
                type=rel.relationship_type,
                weight=rel.weight,
                metadata=rel.metadata
            )
            
            # Também adicionar ao grafo direcionado
            self.directed_graph.add_edge(
                rel.source_id,
                rel.target_id,
                type=rel.relationship_type,
                weight=rel.weight
            )
        
        logger.info(f"Grafo construído: {self.graph.number_of_nodes()} nós, {self.graph.number_of_edges()} arestas")
        
        # Persistir no Neo4j se disponível
        if self.neo4j_store:
            await self._persist_graph_to_neo4j(code_entities, relationships)
        
        return self.graph
    
    async def _extract_relationships_llm(self, entities: List[CodeEntity]) -> List[CodeRelationship]:
        """
        Usa LLM para extrair relacionamentos entre entidades
        Baseado na abordagem do Microsoft GraphRAG
        """
        relationships = []
        
        # Processar em batches para eficiência
        batch_size = 10
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            
            # Preparar prompt para extração de relacionamentos
            prompt = self._create_relationship_extraction_prompt(batch)
            
            # Chamar LLM
            response = await self.model_router.route_request(
                prompt=prompt,
                task_type="relationship_extraction",
                max_tokens=1000
            )
            
            # Parsear resposta
            extracted_rels = self._parse_relationship_response(response, batch)
            relationships.extend(extracted_rels)
        
        return relationships
    
    def _create_relationship_extraction_prompt(self, entities: List[CodeEntity]) -> str:
        """Cria prompt para extração de relacionamentos"""
        entities_info = []
        for entity in entities:
            entities_info.append(f"- {entity.type} '{entity.name}' in {entity.file_path}")
        
        prompt = f"""Analyze the following code entities and identify relationships between them.

Entities:
{chr(10).join(entities_info)}

For each pair of entities that have a relationship, identify:
1. Source entity
2. Target entity  
3. Relationship type (imports, calls, inherits, uses, references, depends_on)
4. Confidence score (0-1)

Return as JSON array of relationships."""

        return prompt
    
    def _parse_relationship_response(self, response: str, entities: List[CodeEntity]) -> List[CodeRelationship]:
        """Parseia resposta do LLM para extrair relacionamentos"""
        relationships = []
        
        try:
            # Extrair JSON da resposta
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                rel_data = json.loads(json_match.group())
                
                # Criar objetos CodeRelationship
                entity_map = {e.name: e.id for e in entities}
                
                for rel in rel_data:
                    if rel['source'] in entity_map and rel['target'] in entity_map:
                        relationship = CodeRelationship(
                            source_id=entity_map[rel['source']],
                            target_id=entity_map[rel['target']],
                            relationship_type=rel['type'],
                            metadata={'confidence': rel.get('confidence', 0.8)},
                            weight=rel.get('confidence', 0.8)
                        )
                        relationships.append(relationship)
        
        except Exception as e:
            logger.warning(f"Erro ao parsear relacionamentos: {e}")
        
        return relationships
    
    async def detect_communities(self) -> Dict[int, Community]:
        """
        Detecta comunidades usando Louvain algorithm
        Agrupa módulos de código relacionados
        """
        logger.info("Detectando comunidades com algoritmo Louvain")
        
        # Aplicar Louvain algorithm
        partition = community_louvain.best_partition(
            self.graph,
            resolution=self.config['louvain_resolution']
        )
        
        # Agrupar entidades por comunidade
        communities_dict = defaultdict(list)
        for node_id, community_id in partition.items():
            communities_dict[community_id].append(node_id)
            self.entity_to_community[node_id] = community_id
        
        # Criar objetos Community com summaries
        for community_id, entity_ids in communities_dict.items():
            if len(entity_ids) >= self.config['min_community_size']:
                # Coletar entidades da comunidade
                community_entities = []
                for entity_id in entity_ids:
                    node_data = self.graph.nodes[entity_id]
                    entity = CodeEntity(
                        id=entity_id,
                        name=node_data['name'],
                        type=node_data['type'],
                        content=node_data['content'],
                        file_path=node_data['file_path'],
                        metadata=node_data['metadata']
                    )
                    community_entities.append(entity)
                
                # Gerar summary da comunidade via LLM
                summary = await self._generate_community_summary(community_entities)
                
                # Calcular centroid embedding
                centroid_embedding = await self._calculate_centroid_embedding(community_entities)
                
                # Criar objeto Community
                community = Community(
                    id=community_id,
                    entities=community_entities,
                    summary=summary,
                    centroid_embedding=centroid_embedding,
                    metadata={
                        'size': len(community_entities),
                        'modularity': nx.algorithms.community.modularity(
                            self.graph, 
                            [{entity_id} for entity_id in entity_ids]
                        )
                    }
                )
                
                self.communities[community_id] = community
        
        logger.info(f"Detectadas {len(self.communities)} comunidades")
        return self.communities
    
    async def _generate_community_summary(self, entities: List[CodeEntity]) -> str:
        """
        Gera summary de uma comunidade usando LLM
        Técnica do Microsoft GraphRAG para summarização
        """
        # Preparar informações das entidades
        entities_info = []
        for entity in entities[:10]:  # Limitar para não exceder contexto
            entities_info.append(f"- {entity.type} '{entity.name}': {entity.content[:100]}...")
        
        prompt = f"""Analyze this group of related code entities and provide a concise summary of their purpose and relationships.

Entities in this module/community:
{chr(10).join(entities_info)}

Provide a 2-3 sentence summary describing:
1. The main purpose/functionality of this code module
2. How these entities work together
3. Key patterns or architectural decisions"""

        response = await self.model_router.route_request(
            prompt=prompt,
            task_type="summarization",
            model_preference=self.config['community_summary_model']
        )
        
        return response.strip()
    
    async def _calculate_centroid_embedding(self, entities: List[CodeEntity]) -> List[float]:
        """Calcula embedding centróide de uma comunidade"""
        embeddings = []
        
        for entity in entities:
            if entity.embedding is None:
                # Gerar embedding se não existir
                entity.embedding = await self.embedding_service.embed_text(entity.content)
            embeddings.append(entity.embedding)
        
        # Calcular centróide
        if embeddings:
            centroid = np.mean(embeddings, axis=0).tolist()
            return centroid
        
        return []
    
    async def multi_hop_reasoning(
        self, 
        start_entity_id: str, 
        query: str,
        max_hops: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Implementa multi-hop reasoning com semantic filtering
        Navega pelo grafo para responder queries complexas
        """
        max_hops = max_hops or self.config['max_hops']
        
        logger.info(f"Iniciando multi-hop reasoning de {start_entity_id} com max_hops={max_hops}")
        
        # Embedding da query para semantic filtering
        query_embedding = await self.embedding_service.embed_text(query)
        
        # BFS com semantic filtering
        visited = set()
        queue = [(start_entity_id, 0, [start_entity_id])]  # (node, depth, path)
        relevant_paths = []
        
        while queue:
            current_id, depth, path = queue.pop(0)
            
            if current_id in visited or depth > max_hops:
                continue
            
            visited.add(current_id)
            
            # Verificar relevância semântica do nó atual
            node_data = self.graph.nodes[current_id]
            node_embedding = await self._get_or_create_embedding(current_id, node_data)
            
            similarity = self._calculate_similarity(query_embedding, node_embedding)
            
            if similarity >= self.config['semantic_threshold']:
                relevant_paths.append({
                    'path': path,
                    'depth': depth,
                    'final_node': current_id,
                    'similarity': similarity,
                    'node_data': node_data
                })
            
            # Explorar vizinhos se não atingiu max_hops
            if depth < max_hops:
                for neighbor in self.graph.neighbors(current_id):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, depth + 1, new_path))
        
        # Rankear paths por relevância
        relevant_paths.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Construir resposta com reasoning chain
        reasoning_result = await self._construct_reasoning_response(
            query, 
            relevant_paths[:5],  # Top 5 paths
            start_entity_id
        )
        
        return reasoning_result
    
    async def _get_or_create_embedding(self, entity_id: str, node_data: Dict) -> List[float]:
        """Obtém ou cria embedding para um nó"""
        # Verificar cache primeiro
        if 'embedding' in node_data and node_data['embedding']:
            return node_data['embedding']
        
        # Gerar embedding
        content = node_data.get('content', '')
        embedding = await self.embedding_service.embed_text(content)
        
        # Cachear no grafo
        self.graph.nodes[entity_id]['embedding'] = embedding
        
        return embedding
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calcula similaridade coseno entre embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        
        # Normalizar vetores
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Similaridade coseno
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
    
    async def _construct_reasoning_response(
        self, 
        query: str, 
        relevant_paths: List[Dict],
        start_entity_id: str
    ) -> Dict[str, Any]:
        """
        Constrói resposta estruturada do multi-hop reasoning
        """
        # Coletar informações dos paths
        path_summaries = []
        all_entities = set()
        
        for path_info in relevant_paths:
            path = path_info['path']
            path_summary = {
                'path': path,
                'depth': path_info['depth'],
                'similarity': path_info['similarity'],
                'entities': []
            }
            
            # Coletar dados de cada entidade no path
            for entity_id in path:
                node_data = self.graph.nodes[entity_id]
                entity_info = {
                    'id': entity_id,
                    'name': node_data['name'],
                    'type': node_data['type'],
                    'file': node_data['file_path']
                }
                path_summary['entities'].append(entity_info)
                all_entities.add(entity_id)
            
            path_summaries.append(path_summary)
        
        # Gerar explicação via LLM
        explanation = await self._generate_reasoning_explanation(
            query, 
            path_summaries,
            start_entity_id
        )
        
        return {
            'query': query,
            'start_entity': start_entity_id,
            'relevant_paths': path_summaries,
            'total_entities_explored': len(all_entities),
            'explanation': explanation,
            'communities_involved': list(set(
                self.entity_to_community.get(e, -1) 
                for path in relevant_paths 
                for e in path['path']
            ))
        }
    
    async def _generate_reasoning_explanation(
        self, 
        query: str,
        path_summaries: List[Dict],
        start_entity: str
    ) -> str:
        """Gera explicação do reasoning via LLM"""
        paths_info = []
        for i, path in enumerate(path_summaries[:3]):  # Top 3 paths
            entities = " → ".join([e['name'] for e in path['entities']])
            paths_info.append(f"{i+1}. {entities} (similarity: {path['similarity']:.2f})")
        
        prompt = f"""Based on the multi-hop graph traversal for the query, explain the reasoning:

Query: {query}
Starting from: {start_entity}

Relevant paths found:
{chr(10).join(paths_info)}

Provide a concise explanation of:
1. How these code entities relate to the query
2. The key connections discovered
3. Why these paths are relevant"""

        response = await self.model_router.route_request(
            prompt=prompt,
            task_type="reasoning"
        )
        
        return response
    
    async def query_with_graph_context(
        self, 
        query: str,
        initial_entities: List[str]
    ) -> Dict[str, Any]:
        """
        Executa query usando contexto do grafo
        Combina community detection + multi-hop reasoning
        """
        results = {
            'query': query,
            'community_context': [],
            'multi_hop_results': [],
            'combined_insights': None
        }
        
        # 1. Identificar comunidades relevantes
        query_embedding = await self.embedding_service.embed_text(query)
        relevant_communities = []
        
        for comm_id, community in self.communities.items():
            similarity = self._calculate_similarity(
                query_embedding, 
                community.centroid_embedding
            )
            
            if similarity >= self.config['semantic_threshold']:
                relevant_communities.append({
                    'community_id': comm_id,
                    'similarity': similarity,
                    'summary': community.summary,
                    'size': len(community.entities)
                })
        
        results['community_context'] = sorted(
            relevant_communities, 
            key=lambda x: x['similarity'], 
            reverse=True
        )[:3]
        
        # 2. Multi-hop reasoning a partir de entidades iniciais
        for entity_id in initial_entities[:3]:  # Limitar para performance
            if entity_id in self.graph.nodes:
                hop_result = await self.multi_hop_reasoning(entity_id, query)
                results['multi_hop_results'].append(hop_result)
        
        # 3. Combinar insights via LLM
        if results['community_context'] or results['multi_hop_results']:
            combined_insights = await self._generate_combined_insights(
                query,
                results['community_context'],
                results['multi_hop_results']
            )
            results['combined_insights'] = combined_insights
        
        return results
    
    async def _generate_combined_insights(
        self,
        query: str,
        community_context: List[Dict],
        multi_hop_results: List[Dict]
    ) -> str:
        """Gera insights combinados de community + multi-hop"""
        # Preparar contexto das comunidades
        comm_info = []
        for comm in community_context[:2]:
            comm_info.append(f"- Community (similarity {comm['similarity']:.2f}): {comm['summary']}")
        
        # Preparar contexto do multi-hop
        hop_info = []
        for result in multi_hop_results[:2]:
            hop_info.append(f"- From {result['start_entity']}: {result['explanation'][:200]}...")
        
        prompt = f"""Synthesize insights from graph analysis for the query:

Query: {query}

Community Analysis:
{chr(10).join(comm_info) if comm_info else 'No relevant communities found'}

Multi-hop Reasoning:
{chr(10).join(hop_info) if hop_info else 'No relevant paths found'}

Provide integrated insights that:
1. Combine community-level patterns with specific code paths
2. Highlight key architectural insights
3. Suggest relevant code areas to explore"""

        response = await self.model_router.route_request(
            prompt=prompt,
            task_type="synthesis"
        )
        
        return response
    
    async def _persist_graph_to_neo4j(
        self, 
        entities: List[CodeEntity],
        relationships: List[CodeRelationship]
    ):
        """Persiste grafo no Neo4j para queries avançadas"""
        try:
            # Criar nós
            for entity in entities:
                await self.neo4j_store.create_code_entity(
                    entity_id=entity.id,
                    name=entity.name,
                    entity_type=entity.type,
                    content=entity.content,
                    file_path=entity.file_path,
                    metadata=entity.metadata
                )
            
            # Criar relacionamentos
            for rel in relationships:
                await self.neo4j_store.create_relationship(
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    relationship_type=rel.relationship_type,
                    metadata=rel.metadata
                )
            
            logger.info("Grafo persistido no Neo4j com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao persistir grafo no Neo4j: {e}")

# Factory function
def create_graph_rag_enhancer(neo4j_store: Optional[Neo4jStore] = None) -> GraphRAGEnhancer:
    """Cria instância do GraphRAG enhancer"""
    return GraphRAGEnhancer(neo4j_store) 