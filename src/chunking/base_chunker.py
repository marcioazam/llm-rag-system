from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    position: int
    
class BaseChunker(ABC):
    """Base class for all chunking strategies"""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        pass
