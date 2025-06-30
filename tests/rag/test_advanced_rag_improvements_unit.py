import sys, types, pytest

# ==== stubs reutilizados de dependências pesadas (se não existirem) ====
for name in [
    "sklearn", "sklearn.metrics", "sklearn.metrics.pairwise",
    "sentence_transformers", "nltk",
]:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
# Cosméticos para evitar AttributeError em pipeline import
if not hasattr(sys.modules["sklearn.metrics.pairwise"], "cosine_similarity"):
    import numpy as np
    sys.modules["sklearn.metrics.pairwise"].cosine_similarity = lambda a, b: np.array([[0.0]])
if not hasattr(sys.modules["sentence_transformers"], "SentenceTransformer"):
    class _DummyST:
        def __init__(self, *a, **k):
            pass
        def encode(self, s, **k):
            return [[0.1]] * len(s)
    sys.modules["sentence_transformers"].SentenceTransformer = _DummyST

# Stubs para módulos internos pesados (se ainda não carregados)
_stub_internal = [
    "src.retrieval.corrective_rag",
    "src.retrieval.enhanced_corrective_rag",
    "src.retrieval.multi_query_rag",
    "src.retrieval.adaptive_retriever",
    "src.graphrag.enhanced_graph_rag",
    "src.cache.optimized_rag_cache",
    "src.models.model_router",
    "src.augmentation.unified_prompt_system",
    "src.retrieval.raptor_retriever",
]
for name in _stub_internal:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        setattr(mod, name.split(".")[-1].title().replace("_", ""), object)
        sys.modules[name] = mod

from src.rag_pipeline_advanced import AdvancedRAGPipeline


def _build_pipeline():
    pl = object.__new__(AdvancedRAGPipeline)
    # Config mínima necessária
    pl.advanced_config = {
        "enable_adaptive": True,
        "enable_multi_query": True,
        "enable_corrective": True,
        "enable_graph": True,
    }
    return pl


def test_determine_improvements_basic():
    pl = _build_pipeline()
    improvements = pl._determine_improvements("Explique a arquitetura do sistema em detalhes")
    # Deve conter adaptive e graph
    assert "adaptive" in improvements
    assert "graph" in improvements

    # Forçar override
    forced = pl._determine_improvements("qualquer", force_improvements=["multi_query"])
    assert forced == {"multi_query"}


def test_prepare_advanced_context_list_format():
    pl = _build_pipeline()

    docs = [
        {"content": "Conteúdo A" * 20, "metadata": {"source": "a.txt"}},
        {"content": "Conteúdo B" * 20, "metadata": {"source": "b.txt"}},
    ]

    class QA:
        query_type = "list"
    ctx = pl._prepare_advanced_context(docs, query_analysis=QA())
    # Deve incluir marcador de lista e pelo menos 2 itens
    assert "Informações relevantes" in ctx
    assert ctx.count("\n1.") == 1 