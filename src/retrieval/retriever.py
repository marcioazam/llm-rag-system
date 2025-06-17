from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever:
    """Sistema de retrieval híbrido com reranking"""
    
    def __init__(
        self,
        vector_store,
        embedding_service,
        rerank: bool = True,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        mmr_lambda: float = 0.5,
    ):

        self.vector_store = vector_store
        self.embedding_service = embedding_service

        # Reranker cross-encoder
        self.rerank = rerank
        self.reranker = CrossEncoder(rerank_model) if rerank else None

        # BM25
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_corpus_tokens: List[List[str]] = []
        self._bm25_docs: List[Dict[str, Any]] = []

        # Pesos configuráveis
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.mmr_lambda = mmr_lambda
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: float = 0.7,
        filter: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        search_type: str = "hybrid",
        use_mmr: bool = False,
    ) -> List[Dict[str, Any]]:
        """Recuperar documentos relevantes"""
        
        if project_id:
            filter = {**(filter or {}), "project_id": project_id}

        if search_type == "semantic":
            return self._semantic_search(query, k, similarity_threshold, filter)
        if search_type == "keyword":
            return self._keyword_search(query, k, filter)
        if search_type == "mmr":
            return self._mmr_search(query, k, similarity_threshold, filter)
        # híbrido
        return self._hybrid_search(query, k, similarity_threshold, filter, use_mmr)
    
    # ------------------------------------------------------------------
    # Indexação BM25
    # ------------------------------------------------------------------

    def index_bm25(self, documents: List[Dict[str, Any]]):
        tokens = [d["content"].lower().split() for d in documents]
        self.bm25 = BM25Okapi(tokens)
        self._bm25_corpus_tokens = tokens
        self._bm25_docs = documents

    # ------------------------------------------------------------------
    # Métodos de busca
    # ------------------------------------------------------------------

    def _semantic_search(self, query: str, k: int, threshold: float, filter: Optional[Dict[str, Any]]):
        query_embedding = self.embedding_service.embed_query(query)
        results = self.vector_store.search(query_embedding=query_embedding.tolist(), k=k * 3, filter=filter)
        good = [r for r in results if r["distance"] is None or (1 - r["distance"]) >= threshold]
        for r in good:
            r["score"] = 1 - r.get("distance", 1)
        return good[:k]

    def _keyword_search(self, query: str, k: int, filter: Optional[Dict[str, Any]]):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][: k * 3]
        results = []
        for idx in top_idx:
            doc = self._bm25_docs[idx]
            res = {"id": doc.get("id", idx), "content": doc["content"], "metadata": doc.get("metadata", {}), "score": scores[idx] / max(scores)}
            results.append(res)
        return results[:k]

    def _hybrid_search(self, query: str, k: int, threshold: float, filter: Optional[Dict[str, Any]], use_mmr: bool):
        semantic = self._semantic_search(query, k * 2, threshold, filter)
        keyword = self._keyword_search(query, k * 2, filter)
        scores_sem = {r["id"]: r["score"] * self.vector_weight for r in semantic}
        scores_kw = {r["id"]: r["score"] * self.bm25_weight for r in keyword}
        ids = set(scores_sem) | set(scores_kw)
        merged = []
        for did in ids:
            score = scores_sem.get(did, 0) + scores_kw.get(did, 0)
            doc = next((r for r in semantic + keyword if r.get("id") == did), None)
            if doc:
                doc["score"] = score
                merged.append(doc)
        merged.sort(key=lambda x: x["score"], reverse=True)
        top = merged[: k * 2]
        if use_mmr:
            return self._mmr(query, top, k)
        if self.rerank and self.reranker and top:
            return self._rerank(query, top, k)
        return top[:k]

    # ------------------------------------------------------------------
    # Rerankers / MMR
    # ------------------------------------------------------------------

    def _rerank(self, query: str, results: List[Dict[str, Any]], k: int):
        pairs = [[query, r["content"]] for r in results]
        scores = self.reranker.predict(pairs)
        for r, s in zip(results, scores):
            r["score"] = float(s)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def _mmr(self, query: str, results: List[Dict[str, Any]], k: int):
        if not results:
            return []
        embeddings = [r.get("embedding") or self.embedding_service.embed_texts([r["content"]], show_progress=False)[0] for r in results]
        query_emb = self.embedding_service.embed_query(query)
        selected, selected_embs = [], []
        while len(selected) < k and results:
            mmr_scores: List[Tuple[int, float]] = []
            for idx, (r, emb) in enumerate(zip(results, embeddings)):
                relevance = cosine_similarity([query_emb], [emb])[0][0]
                diversity = max(
                    cosine_similarity([emb], selected_embs)[0].max() if selected_embs else 0,
                    0,
                )
                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * diversity
                mmr_scores.append((idx, mmr_score))
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(results.pop(best_idx))
            selected_embs.append(embeddings.pop(best_idx))
        return selected

# Alias para compatibilidade com versões que esperam 'Retriever'
Retriever = HybridRetriever
