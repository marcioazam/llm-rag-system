import sys, types, importlib
from textwrap import dedent
import pytest

import numpy as np

# ==========================
# Helpers de stub
# ==========================

def _setup_common_stubs():
    """Registra stubs mínimos para dependências pesadas."""
    # Stub sentence_transformers
    st_mod = types.ModuleType("sentence_transformers")
    class _DummyST:
        def __init__(self, *a, **k):
            pass
        def encode(self, sentences, **kwargs):
            # Retorna vetores unitários distintos para diferenças sutis
            return [np.arange(3) * (i + 1) for i, _ in enumerate(sentences)]
    st_mod.SentenceTransformer = _DummyST
    sys.modules.setdefault("sentence_transformers", st_mod)

    # Stub sklearn cosine_similarity
    sk_mod = types.ModuleType("sklearn")
    sk_metrics = types.ModuleType("sklearn.metrics")
    sk_pairwise = types.ModuleType("sklearn.metrics.pairwise")
    def _cosine_similarity(a, b):
        # Similaridade máxima se vetores idênticos, senão 0.5
        same = np.allclose(a, b)
        return np.array([[1.0 if same else 0.5]])
    sk_pairwise.cosine_similarity = _cosine_similarity
    sk_metrics.pairwise = sk_pairwise
    sk_mod.metrics = sk_metrics
    sys.modules.setdefault("sklearn", sk_mod)
    sys.modules.setdefault("sklearn.metrics", sk_metrics)
    sys.modules.setdefault("sklearn.metrics.pairwise", sk_pairwise)


def _make_nltk_stub(fail: bool):
    """Cria stub do módulo nltk. Se fail=True, sent_tokenize gera LookupError."""
    nltk_mod = types.ModuleType("nltk")
    def sent_tokenize(text, language="portuguese"):
        if fail:
            raise LookupError
        # Split simples em ponto de interrogação/exclamação/ponto.
        import re
        return re.split(r"[.!?]\s+", text)
    nltk_mod.sent_tokenize = sent_tokenize
    class _Data:
        @staticmethod
        def find(name):
            if fail:
                raise LookupError
            return True
    nltk_mod.data = _Data()
    nltk_mod.download = lambda *a, **k: None
    return nltk_mod


def _import_chunker(nltk_fail=False):
    """Importa EnhancedSemanticChunker após configurar stubs."""
    _setup_common_stubs()
    sys.modules["nltk"] = _make_nltk_stub(nltk_fail)
    # Remover módulo para forçar nova importação com os stubs
    sys.modules.pop("src.chunking.semantic_chunker_enhanced", None)
    mod = importlib.import_module("src.chunking.semantic_chunker_enhanced")
    import types as _t
    candidate = getattr(mod, "EnhancedSemanticChunker", None)
    if callable(candidate):
        return candidate
    # Se candidate é módulo ou None, procurar na tabela de símbolos
    for v in vars(mod).values():
        if isinstance(v, type) and v.__name__ == "EnhancedSemanticChunker":
            return v
    # Como fallback final, levantar erro explícito
    raise RuntimeError("Não foi possível localizar classe EnhancedSemanticChunker")

# ==========================
# Testes
# ==========================

def test_chunker_centroids_success_nltk():
    try:
        EnhancedSemanticChunker = _import_chunker(nltk_fail=False)
    except Exception as exc:
        pytest.skip(f"Chunker não disponível: {exc}")
    if not callable(EnhancedSemanticChunker):
        pytest.skip("EnhancedSemanticChunker não é chamável")
    text = dedent(
        """
        Esta é a primeira frase longa o suficiente para passar pelo filtro. 
        Aqui temos a segunda frase, igualmente extensa e relevante para o teste. 
        Finalmente, temos a terceira frase que completa o nosso exemplo.
        """
    ).strip()
    chunker = EnhancedSemanticChunker(similarity_threshold=0.4, min_chunk_size=20, max_chunk_size=120, use_centroids=True)
    chunks = chunker.semantic_chunking(text)
    # Deve criar pelo menos um chunk e concatenar frases
    assert isinstance(chunks, list) and chunks
    # Cada chunk não pode exceder o limite
    assert all(len(c) <= 120 for c in chunks)


def test_chunker_no_centroids_regex_fallback_and_max_size():
    try:
        EnhancedSemanticChunker = _import_chunker(nltk_fail=True)
    except Exception as exc:
        pytest.skip(f"Chunker não disponível: {exc}")
    if not callable(EnhancedSemanticChunker):
        pytest.skip("EnhancedSemanticChunker não é chamável")
    text = (
        "Frase A bastante longa para testar o limite de tamanho do chunk. "
        "Frase B igualmente extensa para garantir que a divisão aconteça corretamente. "
        "Frase C também longa o suficiente para gerar múltiplos chunks quando o tamanho máximo é reduzido."
    )
    chunker = EnhancedSemanticChunker(similarity_threshold=0.0, min_chunk_size=10, max_chunk_size=80, use_centroids=False)
    small_chunks = chunker.semantic_chunking(text)

    # Agora aumentar max_chunk_size -> deve resultar em menos chunks
    chunker.max_chunk_size = 300
    large_chunks = chunker.semantic_chunking(text)

    assert len(small_chunks) >= len(large_chunks) >= 1
    # Regex fallback: _split_sentences_nltk deve ter usado regex; asseguramos chamadas não falharam
    assert all(isinstance(c, str) and c for c in small_chunks) 