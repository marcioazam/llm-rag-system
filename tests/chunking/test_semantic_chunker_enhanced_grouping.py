import importlib, sys, types
from typing import List

import pytest

MODULE_NAME = "src.chunking.semantic_chunker_enhanced"

# Garantir recarga limpa
sys.modules.pop(MODULE_NAME, None)
sc_mod = importlib.import_module(MODULE_NAME)

# ------------------------------------------------------------------
# Patches para dependências pesadas
# ------------------------------------------------------------------
class _DummyST:  # noqa: D401
    def __init__(self, *a, **k):
        self.calls: List[str] = []
    def encode(self, sents):  # noqa: D401
        # Retornar vetores ortogonais determinísticos baseados no índice
        vecs = []
        for idx, _ in enumerate(sents):
            base = [0.0] * 4
            if idx % 2 == 1:
                base[0] = 1.0  # Diferente para forçar baixa similaridade
            vecs.append(base)
            self.calls.append("call")
        return vecs

sc_mod.SentenceTransformer = _DummyST  # type: ignore[attr-defined]

# cosine_similarity simples: produto interno normalizado
def _simple_cosine(a, b):  # noqa: D401
    # Vetores pequenos e já normalizados (0 ou 1). Calculo manual simples.
    return [[sum(x * y for x, y in zip(a[0], b[0]))]]

sc_mod.cosine_similarity = _simple_cosine  # type: ignore

# Simplificar numpy stack para lista
sc_mod.np.stack = lambda seq: list(seq)  # type: ignore

# Stub do nltk com sent_tokenize trivial (divide por ponto)
nltk_stub = types.ModuleType("nltk")

def _sent_tokenize(text, language="portuguese"):
    return [s.strip() + "." for s in text.strip().split(".") if s.strip()]

nltk_stub.sent_tokenize = _sent_tokenize  # type: ignore
sc_mod.nltk = nltk_stub  # type: ignore

# ------------------------------------------------------------------
# Criar instância do chunker sob teste
# ------------------------------------------------------------------
ChunkerCls = getattr(sc_mod, "EnhancedSemanticChunker", None)
if not callable(ChunkerCls):
    pytest.skip("EnhancedSemanticChunker indisponível", allow_module_level=True)

chunker = ChunkerCls(similarity_threshold=0.9, min_chunk_size=1, max_chunk_size=50, use_centroids=False)


def test_grouping_creates_multiple_chunks():
    text = "Primeira frase longa e detalhada. Segunda frase distinta." * 1
    chunks = chunker.chunk(text, {})
    # Devido à similaridade baixa (0), deve produzir 2 chunks
    assert len(chunks) == 2
    assert all(isinstance(c.content, str) for c in chunks)


def test_semantic_chunking_respects_max_chunk_size():
    orig_max = chunker.max_chunk_size
    txt = "Frase." * 100  # texto grande
    out = chunker.semantic_chunking(txt, max_chunk_size=20)
    # Nenhum chunk pode exceder 20 caracteres
    assert all(len(c) <= 20 for c in out)
    # max_chunk_size restaurado ao original
    assert chunker.max_chunk_size == orig_max 