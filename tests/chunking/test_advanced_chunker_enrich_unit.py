import types, sys, numpy as np
from src.chunking import AdvancedChunker

class _Embed:
    def embed_texts(self, sentences, show_progress=False):
        return [np.array([0.1])] * len(sentences)


def test_enrich_with_entities():
    chunker = AdvancedChunker(_Embed(), max_chunk_size=50)

    # Stub preprocessor
    class _Prep:
        def process(self, text):
            return {"entities": ["E1", "E2"]}
    chunker.preprocessor = _Prep()

    chunks = [
        {"content": "ABC", "metadata": {}},
        {"content": "DEF", "metadata": {}},
    ]
    enriched = chunker._enrich_with_entities(chunks, {"content": "ABC DEF"})
    assert all(c["metadata"].get("entities") == ["E1", "E2"] for c in enriched) 