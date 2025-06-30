import numpy as np
import sys, types, asyncio

# Stubs mínimos para evitar imports pesados
if "sklearn.metrics.pairwise" not in sys.modules:
    pm = types.ModuleType("sklearn.metrics.pairwise")
    pm.cosine_similarity = lambda a, b: np.array([[0.0]])
    sys.modules["sklearn.metrics.pairwise"] = pm
sys.modules.setdefault("sklearn.metrics", types.ModuleType("sklearn.metrics"))

from src.rag_pipeline_advanced import AdvancedRAGPipeline


def _pipeline():
    return object.__new__(AdvancedRAGPipeline)


def _make_docs(n, score):
    return [{"content": "x", "score": score}] * n


def test_confidence_base():
    pl = _pipeline()
    conf = asyncio.run(pl._evaluate_response_confidence("q", "short", _make_docs(1, 0.5)))
    assert conf == 0.5


def test_confidence_many_docs_long_answer_high_scores():
    pl = _pipeline()
    answer = "A" * 250  # >200 chars
    docs = _make_docs(6, 0.9)  # len>5 and high score
    conf = asyncio.run(pl._evaluate_response_confidence("q", answer, docs))
    # 0.5 +0.2 +0.1 +0.2 = 1.0 (capped)
    assert conf == 1.0 