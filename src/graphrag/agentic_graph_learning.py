"""
Agentic Graph Learning - Sistema Autônomo de Aprendizado de Grafo
Expande conhecimento autonomamente, descobre padrões e aprende continuamente
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import networkx as nx
from pathlib import Path
import json
import pickle
from enum import Enum

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Tipos de padrões descobertos"""
    ENTITY_RELATION = "entity_relation"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CAUSAL_CHAIN = "causal_chain"
    CLUSTER_FORMATION = "cluster_formation"
    ANOMALY = "anomaly"
    EMERGING_TOPIC = "emerging_topic"


@dataclass
class DiscoveredPattern:
    """Padrão descoberto autonomamente"""
    pattern_id: str
    pattern_type: PatternType
    confidence: float
    entities: List[str]
    relations: List[Tuple[str, str, str]]  # (source, relation, target)
    metadata: Dict[str, Any]
    discovered_at: datetime
    usage_count: int = 0
    validation_score: float = 0.0


@dataclass
class LearningFeedback:
    """Feedback para aprendizado contínuo"""
    query: str
    response: str
    user_satisfaction: Optional[float]  # 0-1
    relevance_scores: Dict[str, float]
    patterns_used: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutonomousGraphExpander:
    """Expande grafo de conhecimento autonomamente"""
    
    def __init__(self,
                 neo4j_store,
                 llm_service,
                 embedding_service,
                 expansion_threshold: float = 0.7):
        
        self.neo4j = neo4j_store
        self.llm = llm_service
        self.embeddings = embedding_service
        self.expansion_threshold = expansion_threshold
        
        # Buffer de candidatos para expansão
        self.expansion_candidates = deque(maxlen=1000)
        
        # Cache de entidades conhecidas
        self.known_entities = set()
        self._load_known_entities()
        
        # Estatísticas
        self.stats = {
            "expansions_performed": 0,
            "entities_added": 0,
            "relations_discovered": 0,
            "failed_expansions": 0
        }
    
    def _load_known_entities(self):
        """Carrega entidades existentes do grafo"""
        try:
            with self.neo4j.driver.session() as session:
                result = session.run("MATCH (n) RETURN n.name as name")
                self.known_entities = {record["name"] for record in result if record["name"]}
            logger.info(f"Carregadas {len(self.known_entities)} entidades conhecidas")
        except Exception as e:
            logger.error(f"Erro ao carregar entidades: {e}")
    
    async def identify_expansion_candidates(self, 
                                          documents: List[Dict],
                                          queries: List[str]) -> List[Dict]:
        """Identifica candidatos para expansão do grafo"""
        
        candidates = []
        
        for doc in documents:
            content = doc.get("content", "")
            
            # Extrair entidades potenciais
            entities = await self._extract_entities(content)
            
            # Identificar novas entidades
            new_entities = [e for e in entities if e not in self.known_entities]
            
            if new_entities:
                # Avaliar relevância
                relevance = await self._evaluate_relevance(new_entities, queries)
                
                if relevance > self.expansion_threshold:
                    candidates.append({
                        "entities": new_entities,
                        "source_content": content[:500],
                        "relevance": relevance,
                        "metadata": doc.get("metadata", {})
                    })
        
        # Adicionar ao buffer
        self.expansion_candidates.extend(candidates)
        
        return candidates
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extrai entidades usando LLM"""
        
        prompt = f"""Extract key entities (concepts, technologies, people, organizations) from this text.
Return as a simple list.

Text: {text[:1000]}

Entities:"""
        
        try:
            response = await self.llm.agenerate([prompt])
            entities = [e.strip() for e in response.generations[0][0].text.split('\n') if e.strip()]
            return entities[:10]  # Limitar quantidade
        except Exception as e:
            logger.error(f"Erro ao extrair entidades: {e}")
            return []
    
    async def _evaluate_relevance(self, entities: List[str], queries: List[str]) -> float:
        """Avalia relevância das entidades para as queries"""
        
        if not entities or not queries:
            return 0.0
        
        # Embeddings das entidades
        entity_text = " ".join(entities)
        entity_emb = await self.embeddings.aembed_query(entity_text)
        
        # Embeddings das queries
        relevance_scores = []
        for query in queries:
            query_emb = await self.embeddings.aembed_query(query)
            
            # Similaridade cosseno
            similarity = np.dot(entity_emb, query_emb) / (
                np.linalg.norm(entity_emb) * np.linalg.norm(query_emb) + 1e-8
            )
            relevance_scores.append(similarity)
        
        return np.mean(relevance_scores)
    
    async def expand_graph_autonomously(self, batch_size: int = 10) -> Dict[str, Any]:
        """Expande grafo autonomamente com candidatos do buffer"""
        
        if not self.expansion_candidates:
            return {"status": "no_candidates", "expansions": 0}
        
        # Processar batch de candidatos
        batch = []
        for _ in range(min(batch_size, len(self.expansion_candidates))):
            if self.expansion_candidates:
                batch.append(self.expansion_candidates.popleft())
        
        expansions = []
        for candidate in batch:
            try:
                # Descobrir relações
                relations = await self._discover_relations(
                    candidate["entities"],
                    candidate["source_content"]
                )
                
                # Adicionar ao grafo
                added = await self._add_to_graph(
                    candidate["entities"],
                    relations,
                    candidate["metadata"]
                )
                
                if added:
                    expansions.append({
                        "entities": candidate["entities"],
                        "relations": relations,
                        "success": True
                    })
                    
                    # Atualizar estatísticas
                    self.stats["expansions_performed"] += 1
                    self.stats["entities_added"] += len(candidate["entities"])
                    self.stats["relations_discovered"] += len(relations)
                    
                    # Atualizar cache
                    self.known_entities.update(candidate["entities"])
                    
            except Exception as e:
                logger.error(f"Erro na expansão: {e}")
                self.stats["failed_expansions"] += 1
        
        return {
            "status": "success",
            "expansions": len(expansions),
            "details": expansions,
            "stats": self.stats
        }
    
    async def _discover_relations(self, 
                                entities: List[str],
                                context: str) -> List[Tuple[str, str, str]]:
        """Descobre relações entre entidades"""
        
        if len(entities) < 2:
            return []
        
        prompt = f"""Given these entities and context, identify relationships between them.
Format: Entity1 -> Relationship -> Entity2

Entities: {', '.join(entities)}
Context: {context}

Relationships:"""
        
        try:
            response = await self.llm.agenerate([prompt])
            text = response.generations[0][0].text
            
            relations = []
            for line in text.split('\n'):
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) == 3:
                        source = parts[0].strip()
                        rel_type = parts[1].strip()
                        target = parts[2].strip()
                        
                        if source in entities and target in entities:
                            relations.append((source, rel_type, target))
            
            return relations
            
        except Exception as e:
            logger.error(f"Erro ao descobrir relações: {e}")
            return []
    
    async def _add_to_graph(self,
                          entities: List[str],
                          relations: List[Tuple[str, str, str]],
                          metadata: Dict) -> bool:
        """Adiciona entidades e relações ao grafo Neo4j"""
        
        try:
            with self.neo4j.driver.session() as session:
                # Adicionar entidades
                for entity in entities:
                    session.run(
                        """
                        MERGE (n:Entity {name: $name})
                        SET n.discovered_at = datetime(),
                            n.discovery_method = 'autonomous',
                            n.metadata = $metadata
                        """,
                        name=entity,
                        metadata=json.dumps(metadata)
                    )
                
                # Adicionar relações
                for source, rel_type, target in relations:
                    session.run(
                        """
                        MATCH (a:Entity {name: $source})
                        MATCH (b:Entity {name: $target})
                        MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
                        SET r.discovered_at = datetime(),
                            r.confidence = $confidence
                        """,
                        source=source,
                        target=target,
                        rel_type=rel_type,
                        confidence=0.8  # Confidence inicial
                    )
                
                return True
                
        except Exception as e:
            logger.error(f"Erro ao adicionar ao grafo: {e}")
            return False


class PatternDiscoveryEngine:
    """Motor de descoberta de padrões emergentes"""
    
    def __init__(self, neo4j_store, min_support: float = 0.1):
        self.neo4j = neo4j_store
        self.min_support = min_support
        
        # Cache de padrões descobertos
        self.discovered_patterns: Dict[str, DiscoveredPattern] = {}
        
        # Histórico para análise temporal
        self.interaction_history = deque(maxlen=10000)
        
        # Estatísticas
        self.stats = defaultdict(int)
    
    async def discover_patterns(self) -> List[DiscoveredPattern]:
        """Descobre padrões emergentes no grafo"""
        
        patterns = []
        
        # 1. Padrões de relação entre entidades
        entity_patterns = await self._discover_entity_patterns()
        patterns.extend(entity_patterns)
        
        # 2. Sequências temporais
        temporal_patterns = await self._discover_temporal_patterns()
        patterns.extend(temporal_patterns)
        
        # 3. Cadeias causais
        causal_patterns = await self._discover_causal_patterns()
        patterns.extend(causal_patterns)
        
        # 4. Clusters emergentes
        cluster_patterns = await self._discover_clusters()
        patterns.extend(cluster_patterns)
        
        # 5. Anomalias
        anomalies = await self._detect_anomalies()
        patterns.extend(anomalies)
        
        # Atualizar cache
        for pattern in patterns:
            self.discovered_patterns[pattern.pattern_id] = pattern
            self.stats[f"patterns_{pattern.pattern_type.value}"] += 1
        
        logger.info(f"Descobertos {len(patterns)} novos padrões")
        
        return patterns
    
    async def _discover_entity_patterns(self) -> List[DiscoveredPattern]:
        """Descobre padrões de relações frequentes"""
        
        patterns = []
        
        try:
            with self.neo4j.driver.session() as session:
                # Encontrar padrões de triplas frequentes
                result = session.run("""
                    MATCH (a)-[r]->(b)
                    WITH type(r) as rel_type, 
                         labels(a)[0] as source_type,
                         labels(b)[0] as target_type,
                         count(*) as frequency
                    WHERE frequency > 5
                    RETURN source_type, rel_type, target_type, frequency
                    ORDER BY frequency DESC
                    LIMIT 20
                """)
                
                for record in result:
                    if record["frequency"] / 100 > self.min_support:  # Support threshold
                        pattern = DiscoveredPattern(
                            pattern_id=f"entity_pattern_{len(patterns)}",
                            pattern_type=PatternType.ENTITY_RELATION,
                            confidence=min(record["frequency"] / 100, 1.0),
                            entities=[record["source_type"], record["target_type"]],
                            relations=[(record["source_type"], record["rel_type"], record["target_type"])],
                            metadata={
                                "frequency": record["frequency"],
                                "support": record["frequency"] / 100
                            },
                            discovered_at=datetime.now()
                        )
                        patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Erro ao descobrir padrões de entidade: {e}")
        
        return patterns
    
    async def _discover_temporal_patterns(self) -> List[DiscoveredPattern]:
        """Descobre sequências temporais de interações"""
        
        patterns = []
        
        # Analisar histórico de interações
        if len(self.interaction_history) < 10:
            return patterns
        
        # Agrupar por janelas temporais
        time_windows = defaultdict(list)
        window_size = timedelta(hours=1)
        
        for interaction in self.interaction_history:
            window = interaction["timestamp"] // window_size
            time_windows[window].append(interaction)
        
        # Identificar sequências repetidas
        sequences = defaultdict(int)
        
        for window_interactions in time_windows.values():
            # Extrair sequência de tipos de query
            sequence = tuple(i["query_type"] for i in window_interactions[:5])
            if len(sequence) >= 3:
                sequences[sequence] += 1
        
        # Criar padrões para sequências frequentes
        for sequence, count in sequences.items():
            if count >= 3:  # Mínimo 3 ocorrências
                pattern = DiscoveredPattern(
                    pattern_id=f"temporal_{len(patterns)}",
                    pattern_type=PatternType.TEMPORAL_SEQUENCE,
                    confidence=min(count / 10, 1.0),
                    entities=list(sequence),
                    relations=[],
                    metadata={
                        "sequence": sequence,
                        "occurrences": count
                    },
                    discovered_at=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _discover_causal_patterns(self) -> List[DiscoveredPattern]:
        """Descobre cadeias causais através de análise de caminhos"""
        
        patterns = []
        
        try:
            with self.neo4j.driver.session() as session:
                # Encontrar caminhos comuns
                result = session.run("""
                    MATCH path = (a)-[*2..4]->(b)
                    WHERE a <> b
                    WITH path, length(path) as path_length
                    LIMIT 100
                    RETURN nodes(path) as nodes, 
                           relationships(path) as rels,
                           path_length
                """)
                
                # Analisar caminhos para padrões causais
                path_counts = defaultdict(int)
                
                for record in result:
                    # Simplificar caminho para padrão
                    node_types = [n.labels[0] if n.labels else "Unknown" for n in record["nodes"]]
                    rel_types = [r.type for r in record["rels"]]
                    
                    pattern_key = tuple(zip(node_types[:-1], rel_types, node_types[1:]))
                    path_counts[pattern_key] += 1
                
                # Criar padrões para caminhos frequentes
                for path_pattern, count in path_counts.items():
                    if count >= 3:
                        entities = []
                        relations = []
                        
                        for source_type, rel_type, target_type in path_pattern:
                            if source_type not in entities:
                                entities.append(source_type)
                            if target_type not in entities:
                                entities.append(target_type)
                            relations.append((source_type, rel_type, target_type))
                        
                        pattern = DiscoveredPattern(
                            pattern_id=f"causal_{len(patterns)}",
                            pattern_type=PatternType.CAUSAL_CHAIN,
                            confidence=min(count / 10, 1.0),
                            entities=entities,
                            relations=relations,
                            metadata={
                                "chain_length": len(relations),
                                "occurrences": count
                            },
                            discovered_at=datetime.now()
                        )
                        patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Erro ao descobrir padrões causais: {e}")
        
        return patterns
    
    async def _discover_clusters(self) -> List[DiscoveredPattern]:
        """Descobre clusters emergentes usando community detection"""
        
        patterns = []
        
        try:
            # Construir grafo NetworkX a partir do Neo4j
            G = nx.Graph()
            
            with self.neo4j.driver.session() as session:
                # Obter nós
                nodes_result = session.run("MATCH (n) RETURN n.name as name")
                nodes = [record["name"] for record in nodes_result if record["name"]]
                G.add_nodes_from(nodes)
                
                # Obter arestas
                edges_result = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN a.name as source, b.name as target
                """)
                edges = [(record["source"], record["target"]) 
                        for record in edges_result 
                        if record["source"] and record["target"]]
                G.add_edges_from(edges)
            
            # Detectar comunidades
            if len(G) > 10:
                import community
                communities = community.best_partition(G)
                
                # Agrupar por comunidade
                community_groups = defaultdict(list)
                for node, comm_id in communities.items():
                    community_groups[comm_id].append(node)
                
                # Criar padrões para comunidades significativas
                for comm_id, members in community_groups.items():
                    if len(members) >= 5:  # Mínimo 5 membros
                        pattern = DiscoveredPattern(
                            pattern_id=f"cluster_{comm_id}",
                            pattern_type=PatternType.CLUSTER_FORMATION,
                            confidence=len(members) / len(G),
                            entities=members[:10],  # Top 10 membros
                            relations=[],
                            metadata={
                                "cluster_size": len(members),
                                "cluster_id": comm_id,
                                "modularity": nx.algorithms.community.modularity(G, [members])
                            },
                            discovered_at=datetime.now()
                        )
                        patterns.append(pattern)
        
        except ImportError:
            logger.warning("python-louvain não instalado, pulando detecção de comunidades")
        except Exception as e:
            logger.error(f"Erro ao descobrir clusters: {e}")
        
        return patterns
    
    async def _detect_anomalies(self) -> List[DiscoveredPattern]:
        """Detecta padrões anômalos"""
        
        patterns = []
        
        try:
            with self.neo4j.driver.session() as session:
                # Detectar nós com grau anormalmente alto
                result = session.run("""
                    MATCH (n)
                    WITH n, size((n)--()) as degree
                    WHERE degree > 20
                    RETURN n.name as name, degree
                    ORDER BY degree DESC
                    LIMIT 10
                """)
                
                for record in result:
                    pattern = DiscoveredPattern(
                        pattern_id=f"anomaly_high_degree_{record['name']}",
                        pattern_type=PatternType.ANOMALY,
                        confidence=0.9,
                        entities=[record["name"]],
                        relations=[],
                        metadata={
                            "anomaly_type": "high_degree",
                            "degree": record["degree"]
                        },
                        discovered_at=datetime.now()
                    )
                    patterns.append(pattern)
                
                # Detectar componentes isolados
                result = session.run("""
                    MATCH (n)
                    WHERE NOT (n)--()
                    RETURN n.name as name
                    LIMIT 10
                """)
                
                isolated_nodes = [record["name"] for record in result if record["name"]]
                if isolated_nodes:
                    pattern = DiscoveredPattern(
                        pattern_id="anomaly_isolated_nodes",
                        pattern_type=PatternType.ANOMALY,
                        confidence=0.8,
                        entities=isolated_nodes,
                        relations=[],
                        metadata={
                            "anomaly_type": "isolated_components",
                            "count": len(isolated_nodes)
                        },
                        discovered_at=datetime.now()
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Erro ao detectar anomalias: {e}")
        
        return patterns
    
    def record_interaction(self, query: str, response: str, metadata: Dict):
        """Registra interação para análise de padrões"""
        
        self.interaction_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "response": response[:200],  # Primeiros 200 chars
            "query_type": metadata.get("query_type", "unknown"),
            "metadata": metadata
        })


class ContinuousLearningEngine:
    """Motor de aprendizado contínuo com feedback loops"""
    
    def __init__(self,
                 graph_expander: AutonomousGraphExpander,
                 pattern_engine: PatternDiscoveryEngine,
                 learning_rate: float = 0.1):
        
        self.graph_expander = graph_expander
        self.pattern_engine = pattern_engine
        self.learning_rate = learning_rate
        
        # Buffer de feedback
        self.feedback_buffer: List[LearningFeedback] = []
        
        # Pesos de padrões (aprendidos)
        self.pattern_weights: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Métricas de aprendizado
        self.learning_metrics = {
            "total_feedback": 0,
            "positive_feedback": 0,
            "patterns_reinforced": 0,
            "patterns_weakened": 0,
            "avg_satisfaction": 0.0
        }
    
    async def process_feedback(self, feedback: LearningFeedback):
        """Processa feedback para aprendizado"""
        
        self.feedback_buffer.append(feedback)
        self.learning_metrics["total_feedback"] += 1
        
        # Atualizar satisfação média
        if feedback.user_satisfaction is not None:
            n = self.learning_metrics["total_feedback"]
            self.learning_metrics["avg_satisfaction"] = (
                (self.learning_metrics["avg_satisfaction"] * (n-1) + feedback.user_satisfaction) / n
            )
            
            if feedback.user_satisfaction > 0.7:
                self.learning_metrics["positive_feedback"] += 1
        
        # Ajustar pesos de padrões baseado no feedback
        await self._adjust_pattern_weights(feedback)
        
        # Trigger aprendizado em batch
        if len(self.feedback_buffer) >= 10:
            await self.batch_learning()
    
    async def _adjust_pattern_weights(self, feedback: LearningFeedback):
        """Ajusta pesos dos padrões baseado no feedback"""
        
        if feedback.user_satisfaction is None:
            return
        
        # Calcular ajuste baseado na satisfação
        adjustment = self.learning_rate * (feedback.user_satisfaction - 0.5)
        
        # Atualizar pesos dos padrões usados
        for pattern_id in feedback.patterns_used:
            old_weight = self.pattern_weights[pattern_id]
            new_weight = np.clip(old_weight + adjustment, 0.1, 1.0)
            self.pattern_weights[pattern_id] = new_weight
            
            if adjustment > 0:
                self.learning_metrics["patterns_reinforced"] += 1
            else:
                self.learning_metrics["patterns_weakened"] += 1
            
            logger.debug(f"Peso do padrão {pattern_id}: {old_weight:.3f} -> {new_weight:.3f}")
    
    async def batch_learning(self):
        """Aprendizado em batch com feedback acumulado"""
        
        if not self.feedback_buffer:
            return
        
        logger.info(f"Iniciando aprendizado em batch com {len(self.feedback_buffer)} feedbacks")
        
        # 1. Identificar queries bem-sucedidas para expansão
        successful_queries = [
            fb for fb in self.feedback_buffer
            if fb.user_satisfaction and fb.user_satisfaction > 0.8
        ]
        
        if successful_queries:
            # Extrair documentos relevantes para expansão
            relevant_docs = []
            for fb in successful_queries:
                # Simular documentos dos scores de relevância
                for doc_id, score in fb.relevance_scores.items():
                    if score > 0.7:
                        relevant_docs.append({
                            "content": fb.response,  # Usar resposta como proxy
                            "metadata": {"query": fb.query, "satisfaction": fb.user_satisfaction}
                        })
            
            # Identificar candidatos para expansão
            queries = [fb.query for fb in successful_queries]
            candidates = await self.graph_expander.identify_expansion_candidates(
                relevant_docs, queries
            )
            
            logger.info(f"Identificados {len(candidates)} candidatos para expansão do grafo")
        
        # 2. Descobrir novos padrões
        new_patterns = await self.pattern_engine.discover_patterns()
        logger.info(f"Descobertos {len(new_patterns)} novos padrões")
        
        # 3. Expandir grafo com melhores candidatos
        expansion_result = await self.graph_expander.expand_graph_autonomously(batch_size=5)
        logger.info(f"Expansão do grafo: {expansion_result}")
        
        # 4. Limpar buffer
        self.feedback_buffer = self.feedback_buffer[-50:]  # Manter últimos 50
        
        # 5. Salvar estado de aprendizado
        await self.save_learning_state()
    
    async def save_learning_state(self):
        """Salva estado do aprendizado"""
        
        state = {
            "pattern_weights": dict(self.pattern_weights),
            "learning_metrics": self.learning_metrics,
            "discovered_patterns": list(self.pattern_engine.discovered_patterns.keys()),
            "known_entities": list(self.graph_expander.known_entities)[:1000],  # Limitar tamanho
            "timestamp": datetime.now().isoformat()
        }
        
        state_path = Path("storage/agentic_learning_state.json")
        state_path.parent.mkdir(exist_ok=True)
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info("Estado de aprendizado salvo")
    
    async def load_learning_state(self):
        """Carrega estado de aprendizado anterior"""
        
        state_path = Path("storage/agentic_learning_state.json")
        
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.pattern_weights.update(state.get("pattern_weights", {}))
                self.learning_metrics.update(state.get("learning_metrics", {}))
                
                logger.info(f"Estado de aprendizado carregado: "
                          f"{len(self.pattern_weights)} padrões conhecidos")
                
            except Exception as e:
                logger.error(f"Erro ao carregar estado: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de aprendizado"""
        
        return {
            "learning_metrics": self.learning_metrics,
            "pattern_weights_summary": {
                "total_patterns": len(self.pattern_weights),
                "avg_weight": np.mean(list(self.pattern_weights.values())) if self.pattern_weights else 0,
                "top_patterns": sorted(
                    self.pattern_weights.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            },
            "graph_expansion_stats": self.graph_expander.stats,
            "pattern_discovery_stats": dict(self.pattern_engine.stats),
            "feedback_buffer_size": len(self.feedback_buffer)
        }


class AgenticGraphLearning:
    """Sistema integrado de Agentic Graph Learning"""
    
    def __init__(self,
                 neo4j_store,
                 llm_service,
                 embedding_service,
                 config: Optional[Dict] = None):
        
        self.config = config or {}
        
        # Componentes principais
        self.graph_expander = AutonomousGraphExpander(
            neo4j_store,
            llm_service,
            embedding_service,
            expansion_threshold=self.config.get("expansion_threshold", 0.7)
        )
        
        self.pattern_engine = PatternDiscoveryEngine(
            neo4j_store,
            min_support=self.config.get("min_pattern_support", 0.1)
        )
        
        self.learning_engine = ContinuousLearningEngine(
            self.graph_expander,
            self.pattern_engine,
            learning_rate=self.config.get("learning_rate", 0.1)
        )
        
        # Estado
        self.is_learning_enabled = True
        self.auto_expansion_enabled = self.config.get("auto_expansion", True)
        
        logger.info("Agentic Graph Learning inicializado")
    
    async def initialize(self):
        """Inicializa o sistema de aprendizado"""
        
        # Carregar estado anterior
        await self.learning_engine.load_learning_state()
        
        # Descobrir padrões iniciais
        initial_patterns = await self.pattern_engine.discover_patterns()
        logger.info(f"Padrões iniciais descobertos: {len(initial_patterns)}")
        
        return {
            "status": "initialized",
            "known_entities": len(self.graph_expander.known_entities),
            "discovered_patterns": len(initial_patterns)
        }
    
    async def process_query_with_learning(self,
                                        query: str,
                                        response: str,
                                        documents: List[Dict],
                                        metadata: Dict) -> Dict[str, Any]:
        """Processa query e aprende com a interação"""
        
        # Registrar interação
        self.pattern_engine.record_interaction(query, response, metadata)
        
        # Identificar candidatos para expansão
        if self.auto_expansion_enabled:
            candidates = await self.graph_expander.identify_expansion_candidates(
                documents, [query]
            )
            
            # Expandir incrementalmente
            if candidates:
                expansion = await self.graph_expander.expand_graph_autonomously(batch_size=2)
                logger.debug(f"Expansão incremental: {expansion}")
        
        # Descobrir padrões usados
        patterns_used = []
        for pattern_id, pattern in self.pattern_engine.discovered_patterns.items():
            # Verificar se padrão foi relevante para esta query
            relevance = await self._evaluate_pattern_relevance(pattern, query, documents)
            if relevance > 0.5:
                patterns_used.append(pattern_id)
                pattern.usage_count += 1
        
        return {
            "patterns_discovered": len(self.pattern_engine.discovered_patterns),
            "patterns_used": patterns_used,
            "expansion_candidates": len(self.graph_expander.expansion_candidates),
            "learning_enabled": self.is_learning_enabled
        }
    
    async def _evaluate_pattern_relevance(self,
                                        pattern: DiscoveredPattern,
                                        query: str,
                                        documents: List[Dict]) -> float:
        """Avalia relevância de um padrão para a query"""
        
        # Implementação simplificada
        # Em produção, usar embeddings e análise mais sofisticada
        
        relevance = 0.0
        
        # Verificar se entidades do padrão aparecem na query/docs
        query_lower = query.lower()
        for entity in pattern.entities:
            if entity.lower() in query_lower:
                relevance += 0.3
        
        # Verificar documentos
        for doc in documents[:3]:
            content = doc.get("content", "").lower()
            for entity in pattern.entities:
                if entity.lower() in content:
                    relevance += 0.1
        
        return min(relevance, 1.0)
    
    async def submit_feedback(self,
                            query: str,
                            response: str,
                            satisfaction: float,
                            patterns_used: List[str],
                            relevance_scores: Optional[Dict[str, float]] = None):
        """Submete feedback para aprendizado"""
        
        feedback = LearningFeedback(
            query=query,
            response=response,
            user_satisfaction=satisfaction,
            relevance_scores=relevance_scores or {},
            patterns_used=patterns_used,
            timestamp=datetime.now()
        )
        
        await self.learning_engine.process_feedback(feedback)
        
        return {"status": "feedback_processed", "satisfaction": satisfaction}
    
    async def trigger_batch_learning(self):
        """Força execução de aprendizado em batch"""
        
        if not self.is_learning_enabled:
            return {"status": "learning_disabled"}
        
        await self.learning_engine.batch_learning()
        
        stats = self.learning_engine.get_learning_stats()
        
        return {
            "status": "batch_learning_completed",
            "stats": stats
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema"""
        
        return {
            "learning_enabled": self.is_learning_enabled,
            "auto_expansion": self.auto_expansion_enabled,
            "graph_stats": self.graph_expander.stats,
            "pattern_stats": {
                "total_patterns": len(self.pattern_engine.discovered_patterns),
                "pattern_types": dict(self.pattern_engine.stats)
            },
            "learning_stats": self.learning_engine.get_learning_stats()
        }
    
    def enable_learning(self, enabled: bool = True):
        """Habilita/desabilita aprendizado"""
        self.is_learning_enabled = enabled
        logger.info(f"Aprendizado {'habilitado' if enabled else 'desabilitado'}")
    
    def enable_auto_expansion(self, enabled: bool = True):
        """Habilita/desabilita expansão automática"""
        self.auto_expansion_enabled = enabled
        logger.info(f"Expansão automática {'habilitada' if enabled else 'desabilitada'}")


def create_agentic_graph_learning(neo4j_store,
                                llm_service,
                                embedding_service,
                                config: Optional[Dict] = None) -> AgenticGraphLearning:
    """Factory para criar sistema de Agentic Graph Learning"""
    
    default_config = {
        "expansion_threshold": 0.7,
        "min_pattern_support": 0.1,
        "learning_rate": 0.1,
        "auto_expansion": True
    }
    
    if config:
        default_config.update(config)
    
    return AgenticGraphLearning(
        neo4j_store,
        llm_service,
        embedding_service,
        default_config
    ) 