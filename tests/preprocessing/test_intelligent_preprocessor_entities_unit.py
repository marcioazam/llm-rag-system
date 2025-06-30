import types, sys
from importlib import reload

# ---------------------------------------------------------------------------
# Criar stubs leves para spaCy e transformers.pipeline ----------------------
# ---------------------------------------------------------------------------

# Stub spaCy ---------------------------------------------------------------
spacy_stub = types.ModuleType("spacy")

def _blank(lang="en"):
    class _DummyNLP:  # noqa: D401
        def __call__(self, text):  # noqa: D401
            class _Doc:  # noqa: D401
                # Uma entidade simulada
                ents = [types.SimpleNamespace(text="Python", label_="LANG")]
            return _Doc()
    return _DummyNLP()

def _load(model):  # noqa: D401
    return _blank()

spacy_stub.blank = _blank  # type: ignore
spacy_stub.load = _load  # type: ignore
sys.modules["spacy"] = spacy_stub

# Stub transformers.pipeline ----------------------------------------------
transformers_stub = types.ModuleType("transformers")

def _pipeline(task, *a, **k):  # noqa: D401
    def _ner(text):  # noqa: D401
        return [{
            "word": "Java",
            "entity": "LANG",
            "start": 0,
            "end": 4,
            "score": 0.9
        }]
    return _ner if task == "ner" else None

transformers_stub.pipeline = _pipeline  # type: ignore
sys.modules["transformers"] = transformers_stub

# ---------------------------------------------------------------------------
# Agora importar módulo alvo e criar instância ------------------------------
# ---------------------------------------------------------------------------
from src.preprocessing import intelligent_preprocessor as ip  # noqa: E402

reload(ip)  # Recarrega para usar stubs

IP = ip.IntelligentPreprocessor()


def test_extract_entities_with_stubs():
    text = "Python e Java são linguagens."
    ents = IP.extract_entities(text)
    # Deve existir pelo menos uma entidade reconhecida
    labels = {e["label"] for e in ents}
    assert "LANG" in labels
    assert len(ents) >= 1


def test_process_includes_entities_and_summary():
    result = IP.process(text="Python. Java. Golang.")
    assert result["entities"]
    assert result["summary"].startswith("Python") 