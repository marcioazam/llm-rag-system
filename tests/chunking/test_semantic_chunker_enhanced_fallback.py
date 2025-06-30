import importlib, sys, types
import pytest

sys.modules.pop("src.chunking.semantic_chunker_enhanced", None)
mod = importlib.import_module("src.chunking.semantic_chunker_enhanced")

# Patch heavy deps
class _DummyST:
    def __init__(self, *a, **k):
        pass
    def encode(self, sents):
        return [[0.0]] * len(sents)
mod.SentenceTransformer = _DummyST  # type: ignore
mod.cosine_similarity = lambda a, b: [[1.0]]  # type: ignore
mod.np.stack = lambda seq: list(seq)  # type: ignore
mod.nltk = None  # type: ignore

if not callable(getattr(mod, "EnhancedSemanticChunker", None)):
    pytest.skip("EnhancedSemanticChunker indisponível", allow_module_level=True)

ESC = mod.EnhancedSemanticChunker(similarity_threshold=0.5, min_chunk_size=5, max_chunk_size=50)

def test_fallback_short_sentences():
    text = "A. B. C."
    chunks = ESC.chunk(text, {})
    # Como sentenças <10 chars filtradas, deve retornar []
    assert chunks == [] 