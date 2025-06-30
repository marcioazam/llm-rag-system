import sys, types
from src.chunking import AdvancedChunker
import numpy as np

class _EmbedStub:
    def embed_texts(self, sentences, show_progress=False):
        return [np.array([0.1])] * len(sentences)


def _make_chunker():
    return AdvancedChunker(_EmbedStub(), max_chunk_size=100)


def test_split_sentences_with_nltk():
    # Stub nltk com sent_tokenize funcional
    nltk_mod = types.ModuleType("nltk")
    nltk_mod.sent_tokenize = lambda text, language="portuguese": text.split(".")
    class _Data:
        @staticmethod
        def find(name):
            return True
    nltk_mod.data = _Data()
    nltk_mod.download = lambda *a, **k: None
    sys.modules["nltk"] = nltk_mod

    sys.modules.pop("src.chunking.advanced_chunker", None)  # garantir reload não necessário
    chunker = _make_chunker()
    sents = chunker._split_sentences("Um. Dois. Três.")
    assert len(sents) >= 3


def test_split_sentences_regex_fallback():
    # Stub nltk para falhar
    nltk_mod = types.ModuleType("nltk")
    def _sent_tokenize_fail(text, language="portuguese"):
        raise LookupError
    nltk_mod.sent_tokenize = _sent_tokenize_fail
    class _DataF:
        @staticmethod
        def find(name):
            raise LookupError
    nltk_mod.data = _DataF()
    nltk_mod.download = lambda *a, **k: None
    sys.modules["nltk"] = nltk_mod

    sys.modules.pop("src.chunking.advanced_chunker", None)
    chunker = _make_chunker()
    sents = chunker._split_sentences("A? B! C.")
    # Regex separa por pontuação => 3 sentenças
    assert len(sents) >= 3 