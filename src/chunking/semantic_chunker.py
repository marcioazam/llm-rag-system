import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import functools

from .base_chunker import BaseChunker, Chunk

class SemanticChunker(BaseChunker):
    """Chunking baseado em similaridade semântica entre sentenças"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.5,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1000):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        # Dividir em sentenças
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        # Gerar embeddings (cache LRU por sentença)
        @functools.lru_cache(maxsize=10000)
        def _encode_sentence(sent: str):
            return self.model.encode([sent])[0]

        embeddings = np.stack([_encode_sentence(s) for s in sentences])
        
        # Agrupar sentenças semanticamente similares
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            # Calcular similaridade
            similarity = cosine_similarity(
                [current_embedding], 
                [embeddings[i]]
            )[0][0]
            
            # Verificar condições para criar novo chunk
            current_length = sum(len(s) for s in current_chunk)
            
            if (similarity < self.similarity_threshold or 
                current_length + len(sentences[i]) > self.max_chunk_size):
                
                # Salvar chunk atual se tiver tamanho mínimo
                if current_length >= self.min_chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(Chunk(
                        content=chunk_text,
                        metadata=metadata,
                        chunk_id=str(uuid.uuid4()),
                        document_id=metadata.get("document_id", ""),
                        position=len(chunks)
                    ))
                
                # Iniciar novo chunk
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                # Adicionar ao chunk atual
                current_chunk.append(sentences[i])
                # Atualizar embedding como média
                current_embedding = np.mean(
                    [current_embedding, embeddings[i]], 
                    axis=0
                )
        
        # Adicionar último chunk
        if current_chunk and sum(len(s) for s in current_chunk) >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text,
                metadata=metadata,
                chunk_id=str(uuid.uuid4()),
                document_id=metadata.get("document_id", ""),
                position=len(chunks)
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Dividir texto em sentenças"""
        import re
        
        # Regex melhorado para divisão de sentenças
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filtrar sentenças vazias
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
