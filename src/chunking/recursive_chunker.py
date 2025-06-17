from typing import List, Dict, Any
import uuid

from .base_chunker import BaseChunker, Chunk

class RecursiveChunker(BaseChunker):
    """Chunking recursivo com múltiplos separadores"""
    
    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        chunks = self._split_text_recursively(text, self.separators)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                result.append(Chunk(
                    content=chunk_text,
                    metadata=metadata,
                    chunk_id=str(uuid.uuid4()),
                    document_id=metadata.get("document_id", ""),
                    position=i
                ))
        
        return result
    
    def _split_text_recursively(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        
        separator = separators[0]
        chunks = []
        
        if separator:
            splits = text.split(separator)
        else:
            # Último recurso: dividir por caracteres
            return self._split_by_char_count(text)
        
        for split in splits:
            if len(split) <= self.chunk_size:
                chunks.append(split)
            else:
                # Recursão com próximo separador
                sub_chunks = self._split_text_recursively(
                    split, 
                    separators[1:]
                )
                chunks.extend(sub_chunks)
        
        # Combinar chunks pequenos e adicionar overlap
        return self._combine_chunks(chunks)
    
    def _split_by_char_count(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks
    
    def _combine_chunks(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []
        
        combined = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            if len(current) + len(chunk) + 1 <= self.chunk_size:
                current += " " + chunk
            else:
                combined.append(current)
                # Adicionar overlap
                if self.chunk_overlap > 0:
                    overlap_text = current[-self.chunk_overlap:]
                    current = overlap_text + " " + chunk
                else:
                    current = chunk
        
        if current:
            combined.append(current)
        
        return combined
