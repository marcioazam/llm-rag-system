"""
Sparse Vector Service para Qdrant 1.8.0+
Implementa BM25-style sparse vectors para hybrid search
Performance otimizada: 16x improvement conforme Qdrant 1.8.0
"""

import re
import math
import asyncio
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class SparseVector:
    """Representa um sparse vector otimizado para Qdrant 1.8.0"""
    indices: List[int]
    values: List[float]
    dimension: int = 0
    
    def to_dict(self) -> Dict:
        """Converte para formato Qdrant"""
        return {
            "indices": self.indices,
            "values": self.values
        }

class BM25SparseEncoder:
    """
    Encoder BM25 otimizado para Qdrant 1.8.0 sparse vectors
    Baseado no paper original de Robertson & Zaragoza (2009)
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Vocabulário e estatísticas
        self.vocabulary: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.corpus_size: int = 0
        self.avg_doc_length: float = 0.0
        
        # Cache para performance
        self._token_cache: Dict[str, List[str]] = {}
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenização otimizada com cache"""
        if text in self._token_cache:
            return self._token_cache[text]
            
        # Normalização e tokenização
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [token for token in text.split() if len(token) > 2]
        
        # Cache resultado
        if len(self._token_cache) < 10000:  # Limitar cache
            self._token_cache[text] = tokens
            
        return tokens
    
    def fit(self, documents: List[str]) -> None:
        """
        Treina o encoder BM25 no corpus
        Otimizado para Qdrant 1.8.0 performance
        """
        logger.info(f"Treinando BM25 encoder em {len(documents)} documentos")
        
        # Tokenizar todos os documentos
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        doc_lengths = [len(tokens) for tokens in tokenized_docs]
        
        self.corpus_size = len(documents)
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths)
        
        # Construir vocabulário
        all_tokens = set()
        for tokens in tokenized_docs:
            all_tokens.update(tokens)
            
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        
        # Calcular document frequencies
        self.doc_freqs = defaultdict(int)
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # Calcular IDF scores
        for token in self.vocabulary:
            df = self.doc_freqs[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5))
            self.idf_scores[token] = max(self.epsilon, idf)
            
        logger.info(f"Vocabulário construído: {len(self.vocabulary)} tokens únicos")
    
    def encode(self, text: str) -> SparseVector:
        """
        Converte texto em sparse vector BM25
        Otimizado para Qdrant 1.8.0 sparse index
        """
        tokens = self._tokenize(text)
        doc_length = len(tokens)
        
        # Contar frequências dos tokens
        token_freqs = Counter(tokens)
        
        # Calcular scores BM25
        indices = []
        values = []
        
        for token, freq in token_freqs.items():
            if token not in self.vocabulary:
                continue
                
            # Score BM25
            idf = self.idf_scores[token]
            tf_component = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            )
            score = idf * tf_component
            
            if score > 0:  # Apenas valores positivos
                indices.append(self.vocabulary[token])
                values.append(float(score))
        
        # Ordenar por índice para otimização Qdrant
        if indices:
            sorted_pairs = sorted(zip(indices, values))
            indices, values = zip(*sorted_pairs)
            indices, values = list(indices), list(values)
        
        return SparseVector(
            indices=indices,
            values=values,
            dimension=len(self.vocabulary)
        )
    
    def batch_encode(self, texts: List[str]) -> List[SparseVector]:
        """Encoding em batch para melhor performance"""
        return [self.encode(text) for text in texts]

class AdvancedSparseVectorService:
    """
    Serviço avançado de sparse vectors para Qdrant 1.8.0+
    Combina múltiplas estratégias de sparse encoding
    """
    
    def __init__(self, config_path: str = "config/hybrid_search_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Encoders
        self.bm25_encoder = BM25SparseEncoder(
            k1=self.config["embedding_providers"]["sparse"]["k1"],
            b=self.config["embedding_providers"]["sparse"]["b"],
            epsilon=self.config["embedding_providers"]["sparse"]["epsilon"]
        )
        
        # TF-IDF para keywords importantes
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.is_fitted = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config não encontrado: {config_path}, usando defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Configuração padrão"""
        return {
            "embedding_providers": {
                "sparse": {
                    "k1": 1.2,
                    "b": 0.75,
                    "epsilon": 0.25
                }
            },
            "chunking": {
                "preserve_keywords": True,
                "keyword_extraction": {
                    "max_keywords": 20,
                    "min_keyword_freq": 2
                }
            }
        }
    
    async def fit(self, documents: List[str]) -> None:
        """
        Treina todos os encoders no corpus
        Assíncrono para melhor performance
        """
        logger.info("Iniciando treinamento de sparse encoders")
        
        # Treinar BM25 (CPU intensivo)
        await asyncio.get_event_loop().run_in_executor(
            None, self.bm25_encoder.fit, documents
        )
        
        # Treinar TF-IDF para keywords
        await asyncio.get_event_loop().run_in_executor(
            None, self.tfidf_vectorizer.fit, documents
        )
        
        self.is_fitted = True
        logger.info("Treinamento de sparse encoders concluído")
    
    def encode_text(self, text: str) -> SparseVector:
        """
        Codifica texto em sparse vector otimizado
        Combina BM25 + keyword extraction
        """
        if not self.is_fitted:
            raise ValueError("Encoder não foi treinado. Execute fit() primeiro.")
        
        # Encoding BM25 principal
        sparse_vector = self.bm25_encoder.encode(text)
        
        # Boost para keywords importantes (se configurado)
        if self.config["chunking"]["preserve_keywords"]:
            sparse_vector = self._boost_keywords(text, sparse_vector)
        
        return sparse_vector
    
    def _boost_keywords(self, text: str, sparse_vector: SparseVector) -> SparseVector:
        """
        Aplica boost em keywords importantes usando TF-IDF
        """
        try:
            # Extrair keywords importantes
            tfidf_matrix = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Encontrar top keywords
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_indices = np.argsort(tfidf_scores)[-self.config["chunking"]["keyword_extraction"]["max_keywords"]:]
            
            # Aplicar boost nos índices correspondentes
            boost_factor = 1.5
            indices_to_boost = set()
            
            for idx in top_indices:
                if tfidf_scores[idx] > 0:
                    keyword = feature_names[idx]
                    # Encontrar índice no vocabulário BM25
                    if keyword in self.bm25_encoder.vocabulary:
                        bm25_idx = self.bm25_encoder.vocabulary[keyword]
                        indices_to_boost.add(bm25_idx)
            
            # Aplicar boost
            boosted_values = []
            for i, (idx, value) in enumerate(zip(sparse_vector.indices, sparse_vector.values)):
                if idx in indices_to_boost:
                    boosted_values.append(value * boost_factor)
                else:
                    boosted_values.append(value)
            
            return SparseVector(
                indices=sparse_vector.indices,
                values=boosted_values,
                dimension=sparse_vector.dimension
            )
            
        except Exception as e:
            logger.warning(f"Erro no boost de keywords: {e}")
            return sparse_vector
    
    async def batch_encode(self, texts: List[str]) -> List[SparseVector]:
        """
        Encoding em batch assíncrono
        Otimizado para throughput máximo
        """
        if not self.is_fitted:
            raise ValueError("Encoder não foi treinado. Execute fit() primeiro.")
        
        # Processar em chunks para balancear memória vs performance
        chunk_size = 100
        results = []
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_results = await asyncio.get_event_loop().run_in_executor(
                None, self._encode_chunk, chunk
            )
            results.extend(chunk_results)
        
        return results
    
    def _encode_chunk(self, texts: List[str]) -> List[SparseVector]:
        """Processa chunk de textos"""
        return [self.encode_text(text) for text in texts]
    
    def get_vocabulary_size(self) -> int:
        """Retorna tamanho do vocabulário"""
        return len(self.bm25_encoder.vocabulary)
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do encoder"""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "vocabulary_size": len(self.bm25_encoder.vocabulary),
            "corpus_size": self.bm25_encoder.corpus_size,
            "avg_doc_length": self.bm25_encoder.avg_doc_length,
            "cache_size": len(self.bm25_encoder._token_cache)
        }

# Factory function para facilitar uso
def create_sparse_vector_service(config_path: str = None) -> AdvancedSparseVectorService:
    """Cria instância do serviço de sparse vectors"""
    if config_path is None:
        config_path = "config/hybrid_search_config.yaml"
    return AdvancedSparseVectorService(config_path) 