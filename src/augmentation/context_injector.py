from __future__ import annotations

from typing import List, Dict
import re

class ContextInjector:
    def __init__(self, relevance_threshold: float = 0.7, max_tokens: int = 3000, max_symbols: int = 10, max_relations: int = 10):
        self.relevance_threshold = relevance_threshold
        self.max_tokens = max_tokens
        self.max_symbols = max_symbols
        self.max_relations = max_relations

    def inject_context(self, query: str, retrieved_docs: List[Dict]) -> List[str]:
        """Processa docs → snippets chave → limita tokens."""
        filtered = [d for d in retrieved_docs if d.get("score", 1) >= self.relevance_threshold]
        filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Extrai top-sentences por doc
        snippets_docs: List[Dict] = self._extract_key_snippets(query, filtered)

        # Flatten e cortar por tokens
        snippets: List[str] = []
        current_tokens = 0
        for sn in snippets_docs:
            txt = sn["content"]
            extras = []
            if sn.get("symbols"):
                sym_list = ", ".join(s.get("name") for s in sn["symbols"][: self.max_symbols])
                extras.append(f"Símbolos: {sym_list}")
            if sn.get("relations"):
                rel_list = ", ".join(r.get("target") for r in sn["relations"][: self.max_relations])
                extras.append(f"Relações: {rel_list}")
            extra_block = "\n".join(extras)
            tokens = len(txt.split())
            if current_tokens + tokens > self.max_tokens:
                break
            snippet_text = f"Fonte: {sn['source']}\n{txt}"
            if extra_block:
                snippet_text += f"\n{extra_block}"
            snippets.append(snippet_text)
            current_tokens += tokens

        return snippets

    def _extract_key_snippets(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Retorna top sentenças para cada doc baseado em overlap de termos."""
        query_terms = set(query.lower().split())
        snippets = []
        for doc in docs:
            sentences = self._split_sentences(doc["content"])
            scored = []
            for idx, sent in enumerate(sentences):
                sent_terms = set(sent.lower().split())
                overlap = len(query_terms & sent_terms)
                position_score = 1.0 / (idx + 1)
                score = overlap * 0.7 + position_score * 0.3
                scored.append((sent, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s for s, _ in scored[:3]]
            snippets.append({
                "source": doc.get("metadata", {}).get("source", "Desconhecido"),
                "content": " ... ".join(top_sentences),
                "symbols": doc.get("metadata", {}).get("symbols"),
                "relations": doc.get("metadata", {}).get("relations"),
                "id": doc.get("id"),
                "relevance": doc.get("score", 0)
            })
        return snippets

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return re.split(r"(?<=[.!?]) +", text) 