import sys, types

# ==== stubs para dependências pesadas antes do import do pipeline ====

# sklearn stub (evita import pesado)
sk_stub = types.ModuleType("sklearn")
feature_stub = types.ModuleType("sklearn.feature_extraction")
text_stub = types.ModuleType("sklearn.feature_extraction.text")
feature_stub.text = text_stub
sk_stub.feature_extraction = feature_stub
sys.modules.setdefault("sklearn", sk_stub)
sys.modules.setdefault("sklearn.feature_extraction", feature_stub)
sys.modules.setdefault("sklearn.feature_extraction.text", text_stub)

# sentence_transformers stub (alguns submódulos aguardados por retrievers)
st_stub = types.ModuleType("sentence_transformers")
class _DummyST:
    def __init__(self, *a, **k):
        pass
    def encode(self, s, **k):
        return [[0.1] * 3] * len(s)
setattr(st_stub, "SentenceTransformer", _DummyST)
sys.modules.setdefault("sentence_transformers", st_stub)

import types  # já temos sys
# ==== stubs para submódulos internos da pipeline ==== 
_stub_names = [
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
for name in _stub_names:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        # Add minimal attributes expected later
        if name.endswith("corrective_rag"):
            class _Dummy: pass
            setattr(mod, "CorrectiveRAG", _Dummy)
            setattr(mod, "EnhancedCorrectiveRAG", _Dummy)
            setattr(mod, "create_enhanced_corrective_rag", lambda cfg=None: _Dummy())
        elif name.endswith("adaptive_retriever"):
            class _D: pass
            setattr(mod, "AdaptiveRetriever", _D)
        elif name.endswith("multi_query_rag"):
            class _D: pass
            setattr(mod, "MultiQueryRAG", _D)
        elif name.endswith("enhanced_graph_rag"):
            class _D: pass
            setattr(mod, "EnhancedGraphRAG", _D)
        elif name.endswith("optimized_rag_cache"):
            class _D:
                def get_stats(self): return {}
                def close(self): pass
            setattr(mod, "OptimizedRAGCache", _D)
        elif name.endswith("model_router"):
            class _D:
                def get_model_status(self): return {}
            setattr(mod, "ModelRouter", _D)
        elif name.endswith("unified_prompt_system"):
            class _D:
                async def generate_optimal_prompt(self, **kwargs):
                    class _Res:
                        def __init__(self):
                            self.task_type = "qa"
                            self.prompt_source = "stub"
                            self.confidence = 0.9
                            self.template_id = "0"
                            self.metadata = {}
                            self.final_prompt = "prompt"
                    return _Res()
            setattr(mod, "UnifiedPromptSystem", _D)
        elif name.endswith("raptor_retriever"):
            class _RR: pass
            setattr(mod, "RaptorRetriever", _RR)
            setattr(mod, "create_raptor_retriever", lambda cfg=None: _RR())
            setattr(mod, "get_default_raptor_config", lambda: {})
        sys.modules[name] = mod

from src.rag_pipeline_advanced import AdvancedRAGPipeline

def _build_pipeline_stub(total_queries: int = 1):
    """Cria instância de AdvancedRAGPipeline sem executar __init__."""
    pipeline = object.__new__(AdvancedRAGPipeline)  # Ignora __init__ pesado
    # Métricas mínimas necessárias para métodos helpers
    pipeline.metrics = {
        "total_advanced_queries": total_queries,
        "avg_confidence": 0.5,
        "avg_processing_time": 1.0,
    }
    return pipeline


def test_format_sources_basic():
    pipeline = _build_pipeline_stub()
    docs = [
        {
            "content": "A" * 250,
            "metadata": {"source": "doc1.txt"},
            "score": 0.9,
            "graph_context": {"central_entities": ["Entidade1"]},
        },
        {
            "content": "B" * 50,
            "metadata": {},
            "score": 0.3,
        },
    ]

    sources = pipeline._format_sources(docs)
    assert len(sources) == 2
    # Conteúdo deve estar truncado (+= '...')
    assert sources[0]["content"].endswith("...")
    # Deve preservar score
    assert sources[0]["score"] == 0.9
    # Deve incluir entidades do grafo
    assert sources[0]["graph_entities"] == ["Entidade1"]


def test_update_metrics_average():
    pipeline = _build_pipeline_stub(total_queries=2)  # n=2 para evitar divisão zero
    prev_conf = pipeline.metrics["avg_confidence"]
    prev_proc = pipeline.metrics["avg_processing_time"]

    pipeline._update_metrics(confidence=0.8, processing_time=2.0)

    # Novas médias devem estar entre antigo e novo valor
    assert prev_conf < pipeline.metrics["avg_confidence"] <= 0.8
    assert prev_proc < pipeline.metrics["avg_processing_time"] <= 2.0 