import sys, types, importlib
import numpy as np
import pytest

# ====== Stubs para dependências pesadas ======
# sentence_transformers (criadas pelo __init__ do EnhancedSemanticChunker)
st_mod = types.ModuleType("sentence_transformers")
class _DummyST:
    def __init__(self, *a, **k):
        pass
    def encode(self, sentences, **k):
        return [np.array([0.1])] * len(sentences)
st_mod.SentenceTransformer = _DummyST
sys.modules.setdefault("sentence_transformers", st_mod)

# sklearn cosine_similarity usado pelo AdvancedChunker
sk_mod = types.ModuleType("sklearn")
metrics_mod = types.ModuleType("sklearn.metrics")
pairwise_mod = types.ModuleType("sklearn.metrics.pairwise")

def _cosine_similarity(a, b):
    # 1.0 se vetores idênticos, 0.0 caso contrário
    return np.array([[1.0 if np.allclose(a, b) else 0.0]])

pairwise_mod.cosine_similarity = _cosine_similarity
metrics_mod.pairwise = pairwise_mod
sk_mod.metrics = metrics_mod
sys.modules.setdefault("sklearn", sk_mod)
sys.modules.setdefault("sklearn.metrics", metrics_mod)
sys.modules.setdefault("sklearn.metrics.pairwise", pairwise_mod)

# nltk stub (evitar download)
nltk_mod = types.ModuleType("nltk")

def _sent_tokenize(text, language="portuguese"):
    raise LookupError  # força fallback regex
nltk_mod.sent_tokenize = _sent_tokenize
class _Data:
    @staticmethod
    def find(name):
        raise LookupError
nltk_mod.data = _Data()
nltk_mod.download = lambda *a, **k: None
sys.modules.setdefault("nltk", nltk_mod)

# ====== Importar AdvancedChunker ======
sys.modules.pop("src.chunking.advanced_chunker", None)
from src.chunking import AdvancedChunker


class _EmbedSame:
    """Retorna o mesmo vetor para todas as sentenças (similaridade 1)."""
    def embed_texts(self, sentences, show_progress=False):
        return [np.array([0.5])] * len(sentences)

class _EmbedDiff:
    """Retorna vetores distintos, forçando similaridade 0."""
    def embed_texts(self, sentences, show_progress=False):
        return [np.array([i]) for i, _ in enumerate(sentences)]


def _make_document(sentences):
    return {"content": " ".join(sentences), "metadata": {}}

# =====================================================
# Testes
# =====================================================

def test_size_limit_enforced():
    sentences = ["A frase um texto.", "Outra frase com conteúdo suficiente.", "Mais uma sentença para teste."]
    doc = _make_document(sentences)

    chunker = AdvancedChunker(_EmbedSame(), max_chunk_size=40, chunk_overlap=0)  # limite pequeno
    chunks = chunker.semantic_chunk(doc)

    # Devido ao limite de tamanho baixo, devem ser criados múltiplos chunks
    assert len(chunks) >= 2
    # Nenhum chunk deve ultrapassar tamanho máximo
    assert all(len(c["content"]) <= chunker.max_chunk_size for c in chunks)


def test_similarity_threshold_split():
    sentences = ["Primeira frase longa.", "Segunda frase longa.", "Terceira frase igualmente longa."]
    doc = _make_document(sentences)

    # Usar limite menor para forçar divisão se similaridade baixa
    chunker = AdvancedChunker(_EmbedDiff(), max_chunk_size=60, chunk_overlap=0)
    chunks = chunker.semantic_chunk(doc)

    # Como os vetores diferem, deve haver mais de um chunk
    assert len(chunks) > 1


def test_contextual_overlap():
    # Preparar chunks simulados
    chunks = [
        {"content": "AAAA", "metadata": {}},
        {"content": "BBBB", "metadata": {}},
        {"content": "CCCC", "metadata": {}},
    ]
    chunker = AdvancedChunker(_EmbedSame(), max_chunk_size=100, chunk_overlap=2)
    overlapped = chunker._add_contextual_overlap(chunks)

    # Deve manter mesmo número de chunks
    assert len(overlapped) == len(chunks)
    # O início do segundo chunk deve conter os 2 últimos caracteres do primeiro
    assert overlapped[1]["content"].startswith(chunks[0]["content"][-2:]) 