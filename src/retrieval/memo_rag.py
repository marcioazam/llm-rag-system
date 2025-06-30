"""
MemoRAG - Memory-Enhanced RAG com Memória Global Comprimida
Suporta contextos ultra-longos (até 2M tokens) com geração de clues
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import pickle
import zlib
from collections import deque, defaultdict
import heapq
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemorySegment:
    """Segmento de memória comprimida"""
    segment_id: str
    content: str
    embedding: Optional[np.ndarray]
    token_count: int
    importance_score: float
    creation_time: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    compression_ratio: float = 1.0
    clues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Clue:
    """Clue para guiar retrieval"""
    clue_text: str
    clue_type: str  # keyword, concept, entity, relation
    relevance_score: float
    source_segments: List[str]  # IDs dos segmentos origem
    embedding: Optional[np.ndarray] = None


class GlobalMemoryStore:
    """
    Armazenamento de memória global com compressão
    Suporta até 2M tokens através de compressão e hierarquia
    """
    
    def __init__(self,
                 max_tokens: int = 2_000_000,
                 compression_threshold: int = 10_000,
                 segment_size: int = 1000):
        
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.segment_size = segment_size
        
        # Memória hierárquica
        self.memory_levels = {
            "hot": {},      # Acesso frequente (não comprimido)
            "warm": {},     # Acesso médio (parcialmente comprimido)
            "cold": {}      # Acesso raro (altamente comprimido)
        }
        
        # Índices
        self.segment_index = {}  # ID -> MemorySegment
        self.clue_index = {}     # Clue text -> Clue object
        self.embedding_index = {}  # Para busca vetorial rápida
        
        # Estatísticas
        self.total_tokens = 0
        self.compression_stats = {
            "original_size": 0,
            "compressed_size": 0,
            "compression_ratio": 0.0
        }
        
        logger.info(f"GlobalMemoryStore inicializado: max {max_tokens/1e6:.1f}M tokens")
    
    def add_memory(self, 
                  content: str,
                  importance: float = 0.5,
                  metadata: Optional[Dict] = None) -> str:
        """Adiciona conteúdo à memória global"""
        
        # Estimar tokens (aproximação)
        token_count = len(content.split()) * 1.3
        
        # Verificar limite
        if self.total_tokens + token_count > self.max_tokens:
            self._evict_old_memories(token_count)
        
        # Criar segmento
        segment_id = f"seg_{int(time.time() * 1000)}_{len(self.segment_index)}"
        
        segment = MemorySegment(
            segment_id=segment_id,
            content=content,
            embedding=None,  # Será calculado depois
            token_count=int(token_count),
            importance_score=importance,
            creation_time=time.time(),
            metadata=metadata or {}
        )
        
        # Adicionar ao nível "hot"
        self.memory_levels["hot"][segment_id] = segment
        self.segment_index[segment_id] = segment
        self.total_tokens += token_count
        
        # Comprimir se necessário
        if len(content) > self.compression_threshold:
            self._compress_segment(segment)
        
        logger.debug(f"Memória adicionada: {segment_id} ({token_count:.0f} tokens)")
        
        return segment_id
    
    def _compress_segment(self, segment: MemorySegment):
        """Comprime um segmento de memória"""
        
        original_size = len(segment.content.encode())
        
        # Comprimir conteúdo
        compressed = zlib.compress(segment.content.encode(), level=6)
        compressed_size = len(compressed)
        
        # Substituir conteúdo se compressão for efetiva
        if compressed_size < original_size * 0.8:
            segment.metadata["original_content"] = segment.content
            segment.content = compressed
            segment.compression_ratio = original_size / compressed_size
            segment.metadata["compressed"] = True
            
            # Atualizar estatísticas
            self.compression_stats["original_size"] += original_size
            self.compression_stats["compressed_size"] += compressed_size
            self.compression_stats["compression_ratio"] = (
                self.compression_stats["original_size"] / 
                max(self.compression_stats["compressed_size"], 1)
            )
            
            logger.debug(f"Segmento comprimido: {segment.segment_id} "
                        f"({segment.compression_ratio:.1f}x)")
    
    def _decompress_segment(self, segment: MemorySegment) -> str:
        """Descomprime um segmento se necessário"""
        
        if segment.metadata.get("compressed", False):
            # Descomprimir
            decompressed = zlib.decompress(segment.content).decode()
            return decompressed
        else:
            return segment.content
    
    def _evict_old_memories(self, needed_tokens: int):
        """Remove memórias antigas para liberar espaço"""
        
        # Calcular score de evicção (menor = evict primeiro)
        eviction_candidates = []
        
        for level_name, level_segments in self.memory_levels.items():
            level_weight = {"hot": 3.0, "warm": 2.0, "cold": 1.0}[level_name]
            
            for seg_id, segment in level_segments.items():
                # Score baseado em: importância, idade, frequência de acesso
                age = time.time() - segment.creation_time
                access_rate = segment.access_count / max(age, 1)
                
                eviction_score = (
                    segment.importance_score * level_weight * 
                    (1 + access_rate) / (1 + age / 3600)  # Idade em horas
                )
                
                heapq.heappush(eviction_candidates, (eviction_score, seg_id, segment))
        
        # Remover até ter espaço suficiente
        freed_tokens = 0
        removed_segments = []
        
        while freed_tokens < needed_tokens and eviction_candidates:
            score, seg_id, segment = heapq.heappop(eviction_candidates)
            
            # Remover de todos os índices
            for level_segments in self.memory_levels.values():
                level_segments.pop(seg_id, None)
            
            self.segment_index.pop(seg_id, None)
            self.embedding_index.pop(seg_id, None)
            
            freed_tokens += segment.token_count
            removed_segments.append(seg_id)
        
        self.total_tokens -= freed_tokens
        
        logger.info(f"Evicted {len(removed_segments)} segments, freed {freed_tokens:.0f} tokens")
    
    def promote_segment(self, segment_id: str):
        """Promove segmento para nível mais quente"""
        
        segment = self.segment_index.get(segment_id)
        if not segment:
            return
        
        # Encontrar nível atual
        current_level = None
        for level_name, level_segments in self.memory_levels.items():
            if segment_id in level_segments:
                current_level = level_name
                break
        
        # Promover se possível
        if current_level == "cold":
            self.memory_levels["cold"].pop(segment_id)
            self.memory_levels["warm"][segment_id] = segment
        elif current_level == "warm":
            self.memory_levels["warm"].pop(segment_id)
            self.memory_levels["hot"][segment_id] = segment
        
        # Atualizar acesso
        segment.access_count += 1
        segment.last_accessed = time.time()
    
    def demote_cold_segments(self, age_threshold_hours: float = 24):
        """Move segmentos antigos para níveis mais frios"""
        
        current_time = time.time()
        age_threshold = age_threshold_hours * 3600
        
        # Hot -> Warm
        to_demote = []
        for seg_id, segment in self.memory_levels["hot"].items():
            if current_time - segment.last_accessed > age_threshold:
                to_demote.append(seg_id)
        
        for seg_id in to_demote:
            segment = self.memory_levels["hot"].pop(seg_id)
            self.memory_levels["warm"][seg_id] = segment
            
            # Comprimir se ainda não estiver
            if not segment.metadata.get("compressed", False):
                self._compress_segment(segment)
        
        # Warm -> Cold
        to_demote = []
        for seg_id, segment in self.memory_levels["warm"].items():
            if current_time - segment.last_accessed > age_threshold * 2:
                to_demote.append(seg_id)
        
        for seg_id in to_demote:
            segment = self.memory_levels["warm"].pop(seg_id)
            self.memory_levels["cold"][seg_id] = segment
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da memória"""
        
        level_stats = {}
        for level_name, level_segments in self.memory_levels.items():
            level_stats[level_name] = {
                "segments": len(level_segments),
                "tokens": sum(s.token_count for s in level_segments.values())
            }
        
        return {
            "total_tokens": self.total_tokens,
            "total_segments": len(self.segment_index),
            "levels": level_stats,
            "compression": self.compression_stats,
            "utilization": self.total_tokens / self.max_tokens
        }


class ClueGenerator:
    """Gerador de clues para guiar retrieval"""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.clue_templates = {
            "keyword": "Palavras-chave importantes: {keywords}",
            "concept": "Conceito principal: {concept}",
            "entity": "Entidades mencionadas: {entities}",
            "relation": "Relação: {subject} -> {relation} -> {object}",
            "question": "Pergunta relacionada: {question}",
            "summary": "Resumo: {summary}"
        }
    
    async def generate_clues(self, 
                           content: str,
                           max_clues: int = 5) -> List[Clue]:
        """Gera clues a partir do conteúdo"""
        
        clues = []
        
        # 1. Extrair keywords (heurística simples)
        keywords = self._extract_keywords(content)
        if keywords:
            clue = Clue(
                clue_text=self.clue_templates["keyword"].format(
                    keywords=", ".join(keywords[:3])
                ),
                clue_type="keyword",
                relevance_score=0.8,
                source_segments=[]
            )
            clues.append(clue)
        
        # 2. Gerar conceitos e entidades via LLM (se disponível)
        if self.llm_service:
            llm_clues = await self._generate_llm_clues(content)
            clues.extend(llm_clues[:max_clues-1])
        
        # 3. Extrair perguntas potenciais
        questions = self._extract_potential_questions(content)
        for q in questions[:2]:
            clue = Clue(
                clue_text=self.clue_templates["question"].format(question=q),
                clue_type="question",
                relevance_score=0.7,
                source_segments=[]
            )
            clues.append(clue)
        
        return clues[:max_clues]
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extrai keywords usando heurísticas"""
        
        # Simplificado - em produção usar TF-IDF ou similar
        words = content.lower().split()
        
        # Remover stopwords básicas
        stopwords = {"o", "a", "de", "que", "e", "é", "em", "para", "com", "um", "uma"}
        
        # Contar frequência
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3 and word not in stopwords:
                word_freq[word] += 1
        
        # Top palavras por frequência
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in top_words[:5]]
    
    async def _generate_llm_clues(self, content: str) -> List[Clue]:
        """Gera clues usando LLM"""
        
        try:
            prompt = f"""Analise o seguinte texto e extraia:
1. Conceito principal (uma frase)
2. Entidades importantes (pessoas, lugares, organizações)
3. Relação principal (sujeito -> relação -> objeto)

Texto: {content[:500]}...

Formato da resposta:
CONCEITO: [conceito]
ENTIDADES: [lista]
RELAÇÃO: [sujeito] -> [relação] -> [objeto]"""
            
            response = await self.llm_service.generate(prompt)
            
            # Parse resposta (simplificado)
            clues = []
            
            lines = response.split('\n')
            for line in lines:
                if line.startswith("CONCEITO:"):
                    concept = line.replace("CONCEITO:", "").strip()
                    clue = Clue(
                        clue_text=self.clue_templates["concept"].format(concept=concept),
                        clue_type="concept",
                        relevance_score=0.9,
                        source_segments=[]
                    )
                    clues.append(clue)
                    
                elif line.startswith("ENTIDADES:"):
                    entities = line.replace("ENTIDADES:", "").strip()
                    clue = Clue(
                        clue_text=self.clue_templates["entity"].format(entities=entities),
                        clue_type="entity",
                        relevance_score=0.85,
                        source_segments=[]
                    )
                    clues.append(clue)
            
            return clues
            
        except Exception as e:
            logger.warning(f"Erro ao gerar clues com LLM: {e}")
            return []
    
    def _extract_potential_questions(self, content: str) -> List[str]:
        """Extrai possíveis perguntas que o conteúdo responde"""
        
        questions = []
        
        # Padrões simples
        if "é um" in content or "é uma" in content:
            # Possível definição
            questions.append("O que é...?")
        
        if "porque" in content.lower() or "pois" in content.lower():
            # Possível explicação
            questions.append("Por que...?")
        
        if "passo" in content.lower() or "primeiro" in content.lower():
            # Possível tutorial
            questions.append("Como fazer...?")
        
        return questions


class MemoRAG:
    """
    Sistema RAG com memória global e geração de clues
    Suporta contextos ultra-longos através de compressão e hierarquia
    """
    
    def __init__(self,
                 embedding_service,
                 llm_service,
                 max_memory_tokens: int = 2_000_000,
                 clue_guided_retrieval: bool = True,
                 memory_persistence_path: Optional[str] = None):
        
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.clue_guided = clue_guided_retrieval
        self.persistence_path = memory_persistence_path
        
        # Componentes principais
        self.global_memory = GlobalMemoryStore(max_tokens=max_memory_tokens)
        self.clue_generator = ClueGenerator(llm_service)
        
        # Cache de embeddings
        self.embedding_cache = {}
        
        # Buffer de documentos recentes
        self.recent_buffer = deque(maxlen=100)
        
        # Estatísticas
        self.stats = {
            "total_queries": 0,
            "clue_guided_retrievals": 0,
            "memory_hits": 0,
            "average_retrieval_time": 0.0,
            "total_tokens_processed": 0
        }
        
        # Carregar memória persistente se existir
        if self.persistence_path:
            self._load_persistent_memory()
        
        logger.info(f"MemoRAG inicializado: {max_memory_tokens/1e6:.1f}M tokens capacity")
    
    async def add_document(self, 
                         document: str,
                         metadata: Optional[Dict] = None,
                         importance: float = 0.5) -> Dict[str, Any]:
        """Adiciona documento à memória global"""
        
        start_time = time.time()
        
        # Dividir em segmentos se muito grande
        segments = self._segment_document(document)
        segment_ids = []
        
        for segment in segments:
            # Adicionar à memória
            seg_id = self.global_memory.add_memory(
                content=segment,
                importance=importance,
                metadata=metadata
            )
            segment_ids.append(seg_id)
            
            # Gerar embedding
            embedding = await self._get_embedding(segment)
            self.global_memory.segment_index[seg_id].embedding = embedding
            self.global_memory.embedding_index[seg_id] = embedding
            
            # Gerar clues
            if self.clue_guided:
                clues = await self.clue_generator.generate_clues(segment)
                self.global_memory.segment_index[seg_id].clues = [c.clue_text for c in clues]
                
                # Indexar clues
                for clue in clues:
                    clue.source_segments.append(seg_id)
                    self.global_memory.clue_index[clue.clue_text] = clue
        
        # Adicionar ao buffer recente
        self.recent_buffer.append({
            "segment_ids": segment_ids,
            "timestamp": time.time(),
            "metadata": metadata
        })
        
        # Atualizar estatísticas
        self.stats["total_tokens_processed"] += sum(
            self.global_memory.segment_index[sid].token_count 
            for sid in segment_ids
        )
        
        processing_time = time.time() - start_time
        
        # Reorganizar memória periodicamente
        if len(self.global_memory.segment_index) % 100 == 0:
            self.global_memory.demote_cold_segments()
        
        return {
            "segment_ids": segment_ids,
            "segments_created": len(segment_ids),
            "processing_time": processing_time,
            "memory_stats": self.global_memory.get_memory_stats()
        }
    
    def _segment_document(self, document: str, max_segment_size: int = 1000) -> List[str]:
        """Divide documento em segmentos menores"""
        
        words = document.split()
        segments = []
        
        current_segment = []
        current_size = 0
        
        for word in words:
            current_segment.append(word)
            current_size += 1
            
            if current_size >= max_segment_size:
                segments.append(" ".join(current_segment))
                current_segment = []
                current_size = 0
        
        if current_segment:
            segments.append(" ".join(current_segment))
        
        return segments
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Obtém embedding com cache"""
        
        text_hash = hash(text)
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Gerar embedding
        embedding = await self.embedding_service.aembed_query(text)
        embedding_array = np.array(embedding)
        
        # Cache
        self.embedding_cache[text_hash] = embedding_array
        
        return embedding_array
    
    async def retrieve(self,
                      query: str,
                      k: int = 10,
                      use_clues: bool = True,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieval com suporte a clues e memória global
        """
        
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Gerar embedding da query
        query_embedding = await self._get_embedding(query)
        
        # Buscar clues relevantes se habilitado
        relevant_segments = []
        
        if use_clues and self.clue_guided:
            clue_segments = await self._clue_guided_retrieval(query, query_embedding)
            relevant_segments.extend(clue_segments)
            self.stats["clue_guided_retrievals"] += 1
        
        # Busca vetorial direta
        vector_segments = await self._vector_retrieval(query_embedding, k=k*2)
        relevant_segments.extend(vector_segments)
        
        # Combinar e ranquear
        final_segments = self._combine_and_rank_segments(
            relevant_segments, 
            query_embedding,
            k=k
        )
        
        # Converter para formato de documento
        documents = []
        for segment, score in final_segments:
            # Promover segmento (indicar uso)
            self.global_memory.promote_segment(segment.segment_id)
            
            # Descomprimir conteúdo se necessário
            content = self.global_memory._decompress_segment(segment)
            
            doc = {
                "content": content,
                "score": float(score),
                "metadata": {
                    **segment.metadata,
                    "segment_id": segment.segment_id,
                    "importance": segment.importance_score,
                    "clues": segment.clues,
                    "compression_ratio": segment.compression_ratio,
                    "access_count": segment.access_count
                }
            }
            documents.append(doc)
        
        # Atualizar estatísticas
        retrieval_time = time.time() - start_time
        self.stats["memory_hits"] += len(documents)
        
        # Média móvel do tempo
        self.stats["average_retrieval_time"] = (
            self.stats["average_retrieval_time"] * 0.9 + retrieval_time * 0.1
        )
        
        logger.info(f"MemoRAG retrieval: {len(documents)} docs em {retrieval_time:.2f}s "
                   f"(clues: {use_clues})")
        
        return documents
    
    async def _clue_guided_retrieval(self, 
                                   query: str,
                                   query_embedding: np.ndarray) -> List[MemorySegment]:
        """Retrieval guiado por clues"""
        
        # Gerar clues da query
        query_clues = await self.clue_generator.generate_clues(query, max_clues=3)
        
        relevant_segments = set()
        
        # Buscar segmentos relacionados aos clues
        for clue in query_clues:
            # Buscar clues similares no índice
            for stored_clue_text, stored_clue in self.global_memory.clue_index.items():
                # Similaridade simples de texto (em produção usar embeddings)
                similarity = self._text_similarity(clue.clue_text, stored_clue_text)
                
                if similarity > 0.7:
                    # Adicionar segmentos fonte
                    for seg_id in stored_clue.source_segments:
                        if seg_id in self.global_memory.segment_index:
                            relevant_segments.add(self.global_memory.segment_index[seg_id])
        
        return list(relevant_segments)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade entre textos (simplificado)"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union
    
    async def _vector_retrieval(self,
                              query_embedding: np.ndarray,
                              k: int) -> List[MemorySegment]:
        """Busca vetorial nos segmentos"""
        
        # Calcular similaridades
        similarities = []
        
        for seg_id, segment_embedding in self.global_memory.embedding_index.items():
            if segment_embedding is not None:
                # Similaridade cosseno
                similarity = np.dot(query_embedding, segment_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(segment_embedding) + 1e-8
                )
                
                segment = self.global_memory.segment_index[seg_id]
                similarities.append((segment, similarity))
        
        # Top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [seg for seg, _ in similarities[:k]]
    
    def _combine_and_rank_segments(self,
                                  segments: List[MemorySegment],
                                  query_embedding: np.ndarray,
                                  k: int) -> List[Tuple[MemorySegment, float]]:
        """Combina e ranqueia segmentos únicos"""
        
        # Deduplicar
        unique_segments = {}
        for segment in segments:
            if segment.segment_id not in unique_segments:
                unique_segments[segment.segment_id] = segment
        
        # Calcular scores finais
        scored_segments = []
        
        for segment in unique_segments.values():
            # Score baseado em múltiplos fatores
            base_score = 0.0
            
            # Similaridade vetorial
            if segment.embedding is not None:
                vector_sim = np.dot(query_embedding, segment.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(segment.embedding) + 1e-8
                )
                base_score += vector_sim * 0.6
            
            # Importância
            base_score += segment.importance_score * 0.2
            
            # Recência (boost para segmentos recentes)
            age_hours = (time.time() - segment.creation_time) / 3600
            recency_boost = 1.0 / (1.0 + age_hours / 24)  # Decai em 24h
            base_score += recency_boost * 0.1
            
            # Frequência de acesso
            access_boost = min(segment.access_count / 10, 1.0) * 0.1
            base_score += access_boost
            
            scored_segments.append((segment, base_score))
        
        # Ordenar por score
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        
        return scored_segments[:k]
    
    async def query_with_memory(self,
                              query: str,
                              k: int = 10,
                              **kwargs) -> Dict[str, Any]:
        """
        Query completa com contexto da memória global
        """
        
        # Recuperar documentos relevantes
        documents = await self.retrieve(query, k=k, **kwargs)
        
        if not documents:
            return {
                "answer": "Não encontrei informações relevantes na memória.",
                "sources": [],
                "memory_stats": self.global_memory.get_memory_stats()
            }
        
        # Construir contexto
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"[Fonte {i+1}] {doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Gerar resposta
        prompt = f"""Baseado no contexto da memória global, responda a pergunta.

Contexto da Memória:
{context}

Pergunta: {query}

Resposta:"""
        
        answer = await self.llm_service.generate(prompt, max_tokens=500)
        
        return {
            "answer": answer,
            "sources": documents,
            "memory_stats": self.global_memory.get_memory_stats(),
            "retrieval_metadata": {
                "documents_retrieved": len(documents),
                "average_importance": np.mean([d["metadata"]["importance"] for d in documents]),
                "used_clues": kwargs.get("use_clues", True)
            }
        }
    
    def _save_persistent_memory(self):
        """Salva memória em disco"""
        
        if not self.persistence_path:
            return
        
        try:
            Path(self.persistence_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Preparar dados para serialização
            save_data = {
                "memory_levels": self.global_memory.memory_levels,
                "segment_index": self.global_memory.segment_index,
                "clue_index": self.global_memory.clue_index,
                "stats": self.stats,
                "total_tokens": self.global_memory.total_tokens
            }
            
            # Salvar com pickle (em produção usar formato mais robusto)
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Memória salva: {self.persistence_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar memória: {e}")
    
    def _load_persistent_memory(self):
        """Carrega memória do disco"""
        
        if not self.persistence_path or not Path(self.persistence_path).exists():
            return
        
        try:
            with open(self.persistence_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restaurar estado
            self.global_memory.memory_levels = save_data["memory_levels"]
            self.global_memory.segment_index = save_data["segment_index"]
            self.global_memory.clue_index = save_data["clue_index"]
            self.stats = save_data["stats"]
            self.global_memory.total_tokens = save_data["total_tokens"]
            
            # Reconstruir índice de embeddings
            for seg_id, segment in self.global_memory.segment_index.items():
                if segment.embedding is not None:
                    self.global_memory.embedding_index[seg_id] = segment.embedding
            
            logger.info(f"Memória carregada: {len(self.global_memory.segment_index)} segmentos")
            
        except Exception as e:
            logger.error(f"Erro ao carregar memória: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do MemoRAG"""
        
        return {
            "query_stats": dict(self.stats),
            "memory_stats": self.global_memory.get_memory_stats(),
            "embedding_cache_size": len(self.embedding_cache),
            "recent_buffer_size": len(self.recent_buffer),
            "clue_index_size": len(self.global_memory.clue_index)
        }
    
    def __del__(self):
        """Salva memória ao destruir objeto"""
        if hasattr(self, 'persistence_path'):
            self._save_persistent_memory()


def create_memo_rag(embedding_service,
                   llm_service,
                   config: Optional[Dict] = None) -> MemoRAG:
    """Factory para criar MemoRAG"""
    
    default_config = {
        "max_memory_tokens": 2_000_000,
        "clue_guided_retrieval": True,
        "memory_persistence_path": "storage/memo_rag_memory.pkl"
    }
    
    if config:
        default_config.update(config)
    
    return MemoRAG(
        embedding_service=embedding_service,
        llm_service=llm_service,
        **default_config
    ) 