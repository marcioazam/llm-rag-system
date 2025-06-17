from __future__ import annotations

from typing import List
import logging

try:
    import spacy  # type: ignore
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore

try:
    from nltk.corpus import wordnet as wn  # type: ignore
except ImportError:  # pragma: no cover
    wn = None  # type: ignore

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """Gera variações simples de query via sinônimos e decomposição por vírgulas."""

    def __init__(self, max_expansions: int = 3):
        self.max_expansions = max_expansions
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as exc:  # pragma: no cover
                logger.debug("spaCy modelo indisponível: %s", exc)
                self.nlp = None
        else:
            self.nlp = None

    # ---------------------------------------------------------
    def enhance_query(self, query: str) -> List[str]:
        variants = [query]
        variants.extend(self._expand_with_synonyms(query))
        variants.extend(self._split_clauses(query))
        # dedup mantendo ordem
        seen = set()
        uniq = []
        for q in variants:
            if q not in seen:
                seen.add(q)
                uniq.append(q)
        return uniq[: self.max_expansions]

    # ---------------------------------------------------------
    def _expand_with_synonyms(self, query: str) -> List[str]:
        if wn is None:
            return []
        words = query.split()
        expansions = []
        for i, w in enumerate(words):
            syns = wn.synsets(w)
            lemmas = {l.name().replace("_", " ") for s in syns for l in s.lemmas()}
            for lemma in list(lemmas)[:1]:  # pega 1 sinonimo
                new_q = words.copy()
                new_q[i] = lemma
                expansions.append(" ".join(new_q))
        return expansions

    def _split_clauses(self, query: str) -> List[str]:
        if self.nlp is None:
            return []
        doc = self.nlp(query)
        subs = []
        clause = []
        for token in doc:
            clause.append(token.text)
            if token.dep_ in {"cc", "punct"} and len(clause) > 3:
                subs.append(" ".join(clause[:-1]))
                clause = []
        if clause:
            subs.append(" ".join(clause))
        return subs 