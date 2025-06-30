import os, importlib, sys, types
import pytest

pytest.skip("Raptor flag test pulado: dependências pesadas não stubadas", allow_module_level=True)

def test_pipeline_without_raptor(monkeypatch):
    monkeypatch.delenv("ENABLE_RAPTOR", raising=False)
    # Stub sklearn para evitar dependência pesada durante import do pipeline
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    fe_mod = types.ModuleType("sklearn.feature_extraction")
    fe_text_mod = types.ModuleType("sklearn.feature_extraction.text")
    # Stub da classe requerida pelo código
    fe_text_mod.TfidfVectorizer = object  # type: ignore
    fe_mod.text = fe_text_mod  # type: ignore
    sys.modules.setdefault("sklearn.feature_extraction", fe_mod)
    sys.modules["sklearn.feature_extraction.text"] = fe_text_mod
    # Stubar outros módulos que o pipeline importa indiretamente
    hyb_mod = types.ModuleType("src.retrieval.hybrid_retriever")
    hyb_mod.HybridRetriever = object  # type: ignore
    sys.modules.setdefault("src.retrieval.hybrid_retriever", hyb_mod)

    hq_mod = types.ModuleType("src.vectordb.hybrid_qdrant_store")
    hq_mod.HybridQdrantStore = object  # type: ignore
    sys.modules.setdefault("src.vectordb.hybrid_qdrant_store", hq_mod)

    from importlib import reload
    mod = importlib.import_module("src.rag_pipeline_advanced")
    reload(mod)
    pipeline = mod.AdvancedRAGPipeline()
    assert pipeline.raptor_enabled is False
    assert pipeline.raptor is None 