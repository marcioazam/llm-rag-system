import importlib, sys, types, asyncio
from types import SimpleNamespace

import pytest

# Stubs para dependências pesadas já semelhantes ao arquivo de unidade anterior
_stub_defs = {
    "src.retrieval.corrective_rag": {"CorrectiveRAG": object},
    "src.retrieval.multi_query_rag": {"MultiQueryRAG": object},
    "src.retrieval.enhanced_corrective_rag": {"create_enhanced_corrective_rag": lambda cfg: None, "EnhancedCorrectiveRAG": object},
    "src.retrieval.hybrid_retriever": {"HybridRetriever": object},
    "src.cache.optimized_rag_cache": {"OptimizedRAGCache": object},
    "src.graphrag.enhanced_graph_rag": {"EnhancedGraphRAG": object},
    "src.augmentation.unified_prompt_system": {"UnifiedPromptSystem": object},
    "src.retrieval.adaptive_retriever": {"AdaptiveRetriever": object},
    "src.retrieval.raptor_retriever": {
        "RaptorRetriever": object,
        "create_raptor_retriever": lambda cfg: None,
        "get_default_raptor_config": lambda: {},
    },
    "src.retrieval.raptor_module": {},
}
for mod_name, attrs in _stub_defs.items():
    mod = types.ModuleType(mod_name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[mod_name] = mod

AdvancedRAGPipeline = importlib.import_module("src.rag_pipeline_advanced").AdvancedRAGPipeline


@pytest.fixture
def pipeline():
    return AdvancedRAGPipeline()


class _FakeAnalysis(SimpleNamespace):
    def __init__(self, query_type: str):
        super().__init__(query_type=query_type)


def _make_docs(n: int, enriched=False, high_score=False):
    docs = []
    for i in range(n):
        d = {
            "content": f"conteudo {i}" * 30,
            "metadata": {"source": f"src{i}"},
            "score": 0.9 if high_score else 0.5,
        }
        if enriched:
            d["enriched_content"] = d["content"] + " extra"
        docs.append(d)
    return docs


def test_prepare_context_list_format(pipeline):
    docs = _make_docs(10, enriched=True)
    ctx = pipeline._prepare_advanced_context(docs, _FakeAnalysis("list"))  # type: ignore
    assert "Informações relevantes" in ctx
    assert ctx.count("\n1.") == 1  # lista enumerada


def test_prepare_context_definition_format(pipeline):
    docs = _make_docs(3)
    ctx = pipeline._prepare_advanced_context(docs, _FakeAnalysis("definition"))  # type: ignore
    assert "Definições encontradas" in ctx


def test_evaluate_response_confidence_max(pipeline):
    docs = _make_docs(6, high_score=True)
    long_answer = "x" * 250
    conf = asyncio.run(pipeline._evaluate_response_confidence("q", long_answer, docs))  # type: ignore
    assert conf == 1.0  # limitado a 1.0


def test_evaluate_response_confidence_base(pipeline):
    docs = _make_docs(1)
    conf = asyncio.run(pipeline._evaluate_response_confidence("q", "curto", docs))  # type: ignore
    assert conf == 0.5 