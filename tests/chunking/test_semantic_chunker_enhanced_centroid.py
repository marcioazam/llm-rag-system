import importlib, sys, types

MODULE = "src.chunking.semantic_chunker_enhanced"
sys.modules.pop(MODULE, None)
mod = importlib.import_module(MODULE)

class _ST:  # no heavy
    def encode(self, sents):
        return [[0.1, 0.2]] * len(sents)

mod.SentenceTransformer = _ST  # type: ignore
mod.cosine_similarity = lambda a, b: [[0.99]]  # high similarity
mod.np.stack = lambda seq: list(seq)  # type: ignore
nltk_stub = types.ModuleType("nltk"); nltk_stub.sent_tokenize = lambda txt, language="portuguese": [txt]
mod.nltk = nltk_stub  # type: ignore

Chunker = getattr(mod, "EnhancedSemanticChunker", None)
if not callable(Chunker):
    import pytest
    pytest.skip("EnhancedSemanticChunker indisponível", allow_module_level=True)

chunker = Chunker(use_centroids=True, min_chunk_size=1, max_chunk_size=100)

def test_centroid_path_runs():
    text = "Uma frase suficientemente longa para teste."
    chunks = chunker.chunk(text, {})
    assert chunks and chunks[0].metadata["chunk_method"] == "enhanced_semantic" 