"""
Enhanced Semantic Chunker - Incorpora melhorias da proposta mantendo compatibilidade
"""

import numpy as np
import nltk
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import functools

from .base_chunker import BaseChunker, Chunk

# Download necessário do NLTK (executar uma vez)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        # Fallback silencioso se download falhar
        pass

class EnhancedSemanticChunker(BaseChunker):
    """
    Chunking semântico aprimorado que combina:
    - NLTK para divisão de sentenças mais precisa
    - Cálculo de centroides para melhor representação
    - Suporte a português e outras línguas
    - Interface compatível com sistema existente
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.6,
                 min_chunk_size: int = 50,  # Menor para ser mais flexível
                 max_chunk_size: int = 512,
                 language: str = "portuguese",
                 use_centroids: bool = True):
        """
        Args:
            model_name: Modelo sentence-transformers
            similarity_threshold: Limiar de similaridade (0.0-1.0)
            min_chunk_size: Tamanho mínimo do chunk em caracteres
            max_chunk_size: Tamanho máximo do chunk em caracteres
            language: Idioma para tokenização NLTK
            use_centroids: Se usar centroides (mais preciso) ou média simples
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.language = language
        self.use_centroids = use_centroids
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunking semântico principal"""
        # Dividir em sentenças usando NLTK (mais preciso)
        sentences = self._split_sentences_nltk(text)
        if not sentences:
            return []
        
        # Gerar embeddings com cache
        embeddings = self._get_embeddings(sentences)
        
        # Agrupar sentenças semanticamente
        chunks = self._group_sentences_semantic(sentences, embeddings)
        
        # Converter para objetos Chunk
        return self._create_chunk_objects(chunks, metadata)
    
    def _split_sentences_nltk(self, text: str) -> List[str]:
        """Divisão de sentenças usando NLTK - mais precisa que regex"""
        # Limpar texto primeiro
        text = text.strip()
        if not text:
            return []
            
        try:
            sentences = nltk.sent_tokenize(text, language=self.language)
        except LookupError:
            try:
                # Fallback para português se language não disponível
                sentences = nltk.sent_tokenize(text, language='portuguese')
            except LookupError:
                # Fallback final para regex se NLTK não disponível
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filtrar sentenças muito pequenas e limpar
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    @functools.lru_cache(maxsize=10000)
    def _get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """Cache de embeddings por sentença"""
        return self.model.encode([sentence])[0]
    
    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Obter embeddings para todas as sentenças"""
        embeddings = [self._get_sentence_embedding(s) for s in sentences]
        return np.stack(embeddings)
    
    def _group_sentences_semantic(self, sentences: List[str], embeddings: np.ndarray) -> List[List[str]]:
        """
        Agrupamento semântico melhorado com centroides
        Baseado na proposta, mas otimizado
        """
        chunks = []
        current_chunk = [sentences[0]]
        current_embeddings = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            # Calcular representação do chunk atual
            if self.use_centroids:
                # Usar centroide (mais representativo)
                chunk_representation = np.mean(current_embeddings, axis=0)
            else:
                # Usar último embedding (mais simples)
                chunk_representation = current_embeddings[-1]
            
            # Calcular similaridade
            similarity = cosine_similarity(
                [chunk_representation], 
                [embeddings[i]]
            )[0][0]
            
            # Verificar condições para novo chunk
            current_length = sum(len(s) for s in current_chunk)
            potential_length = current_length + len(sentences[i])
            
            if (similarity >= self.similarity_threshold and 
                potential_length <= self.max_chunk_size):
                # Adicionar à chunk atual
                current_chunk.append(sentences[i])
                current_embeddings.append(embeddings[i])
            else:
                # Finalizar chunk atual (sempre adicionar para evitar perda)
                if current_length >= self.min_chunk_size:
                    chunks.append(current_chunk.copy())
                elif len(current_chunk) > 0:
                    # Adicionar mesmo se menor que mínimo para não perder conteúdo
                    chunks.append(current_chunk.copy())
                
                # Iniciar novo chunk
                current_chunk = [sentences[i]]
                current_embeddings = [embeddings[i]]
        
        # Adicionar último chunk (sempre adicionar se não vazio)
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_chunk_objects(self, sentence_groups: List[List[str]], metadata: Dict[str, Any]) -> List[Chunk]:
        """Converter grupos de sentenças em objetos Chunk"""
        chunks = []
        
        for i, sentence_group in enumerate(sentence_groups):
            chunk_text = " ".join(sentence_group)
            
            # Metadados enriquecidos
            chunk_metadata = {
                **metadata,
                "chunk_method": "enhanced_semantic",
                "sentence_count": len(sentence_group),
                "similarity_threshold": self.similarity_threshold,
                "language": self.language
            }
            
            chunk = Chunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=str(uuid.uuid4()),
                document_id=metadata.get("document_id", ""),
                position=i
            )
            
            chunks.append(chunk)
        
        return chunks
    
    # Método de compatibilidade com proposta original
    def semantic_chunking(self, text: str, max_chunk_size: int = None) -> List[str]:
        """
        Método compatível com a proposta original
        Retorna lista de strings ao invés de objetos Chunk
        """
        if max_chunk_size:
            # Temporariamente ajustar max_chunk_size
            original_max = self.max_chunk_size
            self.max_chunk_size = max_chunk_size
        
        try:
            chunks = self.chunk(text, {})
            result = [chunk.content for chunk in chunks]
            return result
        finally:
            if max_chunk_size:
                self.max_chunk_size = original_max


# Função de conveniência para compatibilidade com proposta
def create_semantic_chunker(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                           similarity_threshold: float = 0.6) -> EnhancedSemanticChunker:
    """Cria chunker compatível com a interface proposta"""
    return EnhancedSemanticChunker(
        model_name=model_name,
        similarity_threshold=similarity_threshold
    ) 