import importlib, sys, types
from typing import List, Dict

import pytest

# ---------------------------------------------------------
# Patches mínimos antes da importação para evitar dependências pesadas
# ---------------------------------------------------------
# Stub para AdaptiveRetriever e outros componentes pesados
retrieval_stub = types.ModuleType("src.retrieval.adaptive_retriever")
retrieval_stub.AdaptiveRetriever = lambda *a, **k: None  # type: ignore
sys.modules["src.retrieval.adaptive_retriever"] = retrieval_stub

# Map de módulos e atributos necessários
_stub_defs = {
    "src.retrieval.corrective_rag": {"CorrectiveRAG": object},
    "src.retrieval.multi_query_rag": {"MultiQueryRAG": object},
    "src.retrieval.enhanced_corrective_rag": {"create_enhanced_corrective_rag": lambda cfg: None, "EnhancedCorrectiveRAG": object},
    "src.retrieval.hybrid_retriever": {"HybridRetriever": object},
    "src.cache.optimized_rag_cache": {"OptimizedRAGCache": object},
    "src.graphrag.enhanced_graph_rag": {"EnhancedGraphRAG": object},
    "src.augmentation.unified_prompt_system": {"UnifiedPromptSystem": object},
    "src.retrieval.raptor_retriever": {
        "RaptorRetriever": object,
        "create_raptor_retriever": lambda cfg: None,
        "get_default_raptor_config": lambda: {},
    },
    "src.retrieval.raptor_module": {},
}

for mod_name, attrs in _stub_defs.items():
    stub = types.ModuleType(mod_name)
    for attr, val in attrs.items():
        setattr(stub, attr, val)
    sys.modules[mod_name] = stub

# Stubs que expõem funções esperadas
sys.modules["src.retrieval.enhanced_corrective_rag"].create_enhanced_corrective_rag = lambda cfg: None  # type: ignore
sys.modules["src.retrieval.raptor_retriever"].RaptorRetriever = object  # type: ignore
sys.modules["src.retrieval.raptor_retriever"].create_raptor_retriever = lambda cfg: None  # type: ignore
sys.modules["src.retrieval.raptor_retriever"].get_default_raptor_config = lambda: {}

# Importar módulo agora que stubs estão prontos
rp_mod = importlib.import_module("src.rag_pipeline_advanced")
AdvancedRAGPipeline = rp_mod.AdvancedRAGPipeline


@pytest.fixture
def pipeline():
    # Instância com dependências já stubadas
    return AdvancedRAGPipeline()


def test_determine_improvements_defaults(pipeline):
    """Verifica heurística de seleção das melhorias."""
    question = "Qual é a arquitetura do sistema e como as camadas se relacionam?"
    improvements = pipeline._determine_improvements(question)  # type: ignore
    assert {"adaptive", "corrective", "graph"}.issubset(improvements)


def test_determine_improvements_force(pipeline):
    forced = ["adaptive", "multi_query"]
    improvements = pipeline._determine_improvements("irrelevante", force_improvements=forced)  # type: ignore
    assert set(improvements) == set(forced)


def test_format_sources_includes_graph_entities(pipeline):
    docs: List[Dict] = [
        {
            "content": "Texto" * 10,
            "metadata": {"source": "unit_test"},
            "score": 0.9,
            "graph_context": {"central_entities": ["NodeA", "NodeB"]},
        }
    ]
    sources = pipeline._format_sources(docs)  # type: ignore
    assert sources[0]["metadata"]["source"] == "unit_test"
    assert sources[0]["graph_entities"] == ["NodeA", "NodeB"]


def test_update_metrics_moving_average(pipeline):
    pipeline.metrics["total_advanced_queries"] = 2
    pipeline.metrics["avg_confidence"] = 0.5
    pipeline.metrics["avg_processing_time"] = 1.0
    pipeline._update_metrics(confidence=0.8, processing_time=2.0)  # type: ignore
    # Nova média deve estar entre valores originais e novo input
    assert 0.5 <= pipeline.metrics["avg_confidence"] <= 0.8
    assert 1.0 <= pipeline.metrics["avg_processing_time"] <= 2.0 