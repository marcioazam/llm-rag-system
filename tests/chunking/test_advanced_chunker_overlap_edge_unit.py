from src.chunking import AdvancedChunker
import numpy as np

class _Embed:
    def embed_texts(self, sentences, show_progress=False):
        return [np.array([0.1])] * len(sentences)

def test_no_overlap():
    chunker = AdvancedChunker(_Embed(), max_chunk_size=50, chunk_overlap=0)
    chunks = [
        {"content": "AAA", "metadata": {}},
        {"content": "BBB", "metadata": {}},
    ]
    result = chunker._add_contextual_overlap(chunks)
    assert result == chunks  # deve ser idêntico 