import importlib, sys, types

import pytest

# Reuse stubs minimal as before
required = {
    "src.retrieval.hybrid_retriever": {"HybridRetriever": object},
}
for mn, attrs in required.items():
    mod = types.ModuleType(mn)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[mn] = mod

Adv = importlib.import_module("src.rag_pipeline_advanced").AdvancedRAGPipeline

@pytest.fixture
def pipeline():
    return Adv()

def test_prepare_context_with_graph_summary(pipeline):
    docs = [
        {
            "content": "conteudo bem extensivo" * 20,
            "metadata": {"source": "paper"},
            "graph_context": {"summary": "Este nó conecta A e B"},
        }
    ]
    ctx = pipeline._prepare_advanced_context(docs, None)  # type: ignore
    assert "Contexto adicional" in ctx
    assert "Documento 1" in ctx 