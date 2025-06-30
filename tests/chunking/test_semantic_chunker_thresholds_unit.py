import importlib, sys, types

MODULE = "src.chunking.semantic_chunker_enhanced"
sys.modules.pop(MODULE, None)
mod = importlib.import_module(MODULE)

# ----------------- Patch dependências pesadas -----------------
class _ST:  # Dummy SentenceTransformer
    def encode(self, sents):
        # Vetor simples: index encoded in first value
        return [[float(i)] for i, _ in enumerate(sents)]

mod.SentenceTransformer = _ST  # type: ignore
mod.np.stack = lambda seq: list(seq)  # type: ignore

# Vamos substituir cosine_similarity para retornar valor controlado

def _cosine_factory(val: float):
    def _cos(a, b):  # noqa: D401
        return [[val]]
    return _cos

mod.nltk = types.ModuleType("nltk")
mod.nltk.sent_tokenize = lambda txt, language="portuguese": [s.strip() + "." for s in txt.split(".") if s.strip()]  # type: ignore

Chunker = getattr(mod, "EnhancedSemanticChunker", None)
import pytest
if not callable(Chunker):
    pytest.skip("EnhancedSemanticChunker indisponível", allow_module_level=True)


def _build_chunker(threshold: float, cos_val: float):
    mod.cosine_similarity = _cosine_factory(cos_val)  # type: ignore
    return Chunker(similarity_threshold=threshold, min_chunk_size=1, max_chunk_size=100, use_centroids=False)


def test_similarity_equal_threshold_merges():
    chunker = _build_chunker(0.6, 0.6)
    text = "A primeira frase longa o suficiente. Segunda frase também longa."  # duas sentenças
    chunks = chunker.chunk(text, {})
    # Espera-se que ambas sentenças fiquem no mesmo chunk
    assert len(chunks) == 1


def test_similarity_below_threshold_splits():
    chunker = _build_chunker(0.6, 0.59)
    text = "Frase um longa. Frase dois longa."  # duas sentenças
    chunks = chunker.chunk(text, {})
    assert len(chunks) == 2 