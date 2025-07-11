from __future__ import annotations

import logging
import re
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk import sent_tokenize  # type: ignore
except ImportError:  # pragma: no cover
    sent_tokenize = None  # type: ignore

from ..preprocessing.intelligent_preprocessor import IntelligentPreprocessor
from .recursive_chunker import RecursiveChunker
from .semantic_chunker_enhanced import EnhancedSemanticChunker  # FASE 1: Enhanced chunker

logger = logging.getLogger(__name__)


class AdvancedChunker:
    """Chunker multimodal que combina várias estratégias.

    A interface segue a do Semantic/Recursive para compatibilidade.
    """

    def __init__(self, embedding_service, max_chunk_size: int = 800, chunk_overlap: int = 50):
        self.embedding_service = embedding_service
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.preprocessor = None
        try:
            self.preprocessor = IntelligentPreprocessor()
        except Exception as exc:  # pragma: no cover
            logger.debug("IntelligentPreprocessor indisponível: %s", exc)

        self.recursive = RecursiveChunker(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        
        # FASE 1: Enhanced semantic chunker como padrão
        self.enhanced_semantic = EnhancedSemanticChunker(
            similarity_threshold=0.6,
            min_chunk_size=50,
            max_chunk_size=max_chunk_size,
            language="portuguese"
        )

        # Mapeia nomes → métodos
        self.strategies = {
            "semantic": self.enhanced_semantic_chunk,  # FASE 1: Usar enhanced
            "semantic_basic": self.semantic_chunk,  # Manter básico para compatibilidade
            "structural": self.structural_chunk,
            "sliding_window": self.sliding_window_chunk,
            "recursive": self.recursive_chunk,
            "topic_based": self.topic_based_chunk,
            "entity_aware": self.entity_aware_chunk,
        }

    # -------------------------------------------------------------
    # API pública
    # -------------------------------------------------------------

    def chunk(self, document: Dict[str, Any], strategy: str = "hybrid") -> List[Dict[str, Any]]:
        if strategy == "hybrid":
            return self._hybrid_chunk(document)
        if strategy in self.strategies:
            return self.strategies[strategy](document)
        raise ValueError(f"Estratégia desconhecida: {strategy}")

    # -------------------------------------------------------------
    # Estratégias
    # -------------------------------------------------------------

    def _hybrid_chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        structural_chunks = self.structural_chunk(document)
        refined = []
        for chunk in structural_chunks:
            if len(chunk["content"]) > self.max_chunk_size:
                refined.extend(self.enhanced_semantic_chunk(chunk))  # FASE 1: Usar enhanced
            else:
                refined.append(chunk)
        enriched = self._enrich_with_entities(refined, document)
        return self._add_contextual_overlap(enriched)
    
    # FASE 1: Enhanced semantic chunking (novo método padrão)
    def enhanced_semantic_chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunking semântico enhanced com NLTK e centroides"""
        try:
            # Usar o enhanced chunker
            chunks = self.enhanced_semantic.chunk(
                text=document["content"],
                metadata=document.get("metadata", {})
            )
            
            # Converter para formato compatível
            result = []
            for chunk in chunks:
                result.append({
                    "content": chunk.content,
                    "metadata": {
                        **document.get("metadata", {}),
                        **chunk.metadata,
                        "chunk_method": "enhanced_semantic"
                    }
                })
            
            return result
            
        except Exception as e:
            logger.warning(f"Enhanced semantic chunking falhou: {e}, usando fallback")
            # Fallback para semantic básico
            return self.semantic_chunk(document)

    # ------------------------
    def semantic_chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = document["content"]
        sentences = self._split_sentences(text)
        if not sentences:
            return [document]
        embeddings = self.embedding_service.embed_texts(sentences, show_progress=False)
        chunks: List[List[str]] = []
        cur_sents: List[str] = []
        cur_emb = None
        for sent, emb in zip(sentences, embeddings):
            if not cur_sents:
                cur_sents.append(sent)
                cur_emb = emb
                continue
            similarity = cosine_similarity([cur_emb], [emb])[0][0]
            cur_size = sum(len(s) for s in cur_sents)
            if cur_size + len(sent) <= self.max_chunk_size and similarity > 0.65:
                cur_sents.append(sent)
                cur_emb = np.mean([cur_emb, emb], axis=0)
            else:
                chunks.append(cur_sents)
                cur_sents = [sent]
                cur_emb = emb
        if cur_sents:
            chunks.append(cur_sents)
        return [
            {
                "content": " ".join(sent_group),
                "metadata": {**document.get("metadata", {}), "chunk_method": "semantic"},
            }
            for sent_group in chunks
        ]

    # ------------------------
    def structural_chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = document["content"]
        paragraphs = re.split(r"\n{2,}", text)
        chunks = []
        current = []
        for para in paragraphs:
            if sum(len(p) for p in current) + len(para) <= self.max_chunk_size:
                current.append(para)
            else:
                chunks.append("\n\n".join(current))
                current = [para]
        if current:
            chunks.append("\n\n".join(current))
        return [
            {"content": c, "metadata": {**document.get("metadata", {}), "chunk_method": "structural"}}
            for c in chunks
        ]

    # ------------------------
    def sliding_window_chunk(self, document: Dict[str, Any], window: int = 400, stride: int = 200) -> List[Dict[str, Any]]:
        text = document["content"]
        chunks = [text[i : i + window] for i in range(0, len(text), stride)]
        return [
            {
                "content": c,
                "metadata": {**document.get("metadata", {}), "chunk_method": "sliding_window"},
            }
            for c in chunks
        ]

    # ------------------------
    def recursive_chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            # Tentar API chunk_text primeiro
            text_chunks = self.recursive.chunk_text(document["content"])
            return [
                {
                    "content": chunk,
                    "metadata": {**document.get("metadata", {}), "chunk_method": "recursive"},
                }
                for chunk in text_chunks
            ]
        except AttributeError:
            try:
                # Fallback para API chunk
                chunks = self.recursive.chunk(document["content"], document.get("metadata", {}))
                return [
                    {
                        "content": chunk.content if hasattr(chunk, 'content') else str(chunk),
                        "metadata": {**document.get("metadata", {}), "chunk_method": "recursive"},
                    }
                    for chunk in chunks
                ]
            except Exception:
                # Se tudo falhar, usar fallback semântico
                return self.semantic_chunk(document)

    # ------------------------
    def topic_based_chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Placeholder: fallback para semantic
        return self.semantic_chunk(document)

    # ------------------------
    def entity_aware_chunk(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.preprocessor is None:
            return self.semantic_chunk(document)
        processed = self.preprocessor.process(document["content"])
        entities_map = processed["entities"]
        sentences = self._split_sentences(document["content"])
        chunks: List[List[str]] = []
        current: List[str] = []
        current_entities: set[str] = set()
        for sent in sentences:
            # Tratar entities_map como lista ou dicionário
            if isinstance(entities_map, dict):
                sent_entities = {
                    ent
                    for ent_list in entities_map.values()
                    for ent in ent_list
                    if ent.lower() in sent.lower()
                }
            else:
                # Se for lista, iterar diretamente
                sent_entities = {
                    ent for ent in entities_map
                    if isinstance(ent, str) and ent.lower() in sent.lower()
                }
            if not current:
                current.append(sent)
                current_entities.update(sent_entities)
                continue
            combined_len = len(" ".join(current + [sent]))
            if combined_len <= self.max_chunk_size and (current_entities & sent_entities):
                current.append(sent)
                current_entities.update(sent_entities)
            else:
                chunks.append((current, current_entities))
                current = [sent]
                current_entities = sent_entities
        if current:
            chunks.append((current, current_entities))

        return [
            {
                "content": " ".join(sent_group),
                "metadata": {
                    **document.get("metadata", {}),
                    "chunk_method": "entity_aware",
                    "entities": list(ent_set),
                },
            }
            for sent_group, ent_set in chunks
        ]

    # -------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        if sent_tokenize is not None:
            try:
                return sent_tokenize(text)
            except Exception:  # pragma: no cover
                pass
        return re.split(r"(?<=[.!?]) +", text)

    def _enrich_with_entities(self, chunks: List[Dict[str, Any]], original_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.preprocessor is None:
            return chunks
        entities = self.preprocessor.process(original_doc["content"])["entities"]
        for chunk in chunks:
            chunk["metadata"].setdefault("entities", entities)
        return chunks

    def _add_contextual_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.chunk_overlap <= 0:
            return chunks
        augmented: List[Dict[str, Any]] = []
        prev_tail = ""
        for chunk in chunks:
            content = (prev_tail + " " + chunk["content"]).strip()
            augmented.append({**chunk, "content": content})
            prev_tail = chunk["content"][-self.chunk_overlap :]
        return augmented
