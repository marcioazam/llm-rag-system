import sys
import types
import json
import pytest
import numpy as np

# -----------------------------------------------------------------------------
# Stub para Neo4jStore e dependências pesadas antes de importar o módulo alvo
# -----------------------------------------------------------------------------

def _inject_stubs():
    # Stub Neo4jStore
    neo4j_store_mod = types.ModuleType("src.graphdb.neo4j_store")
    class _DummyStore:  # noqa: D401
        def __init__(self, *_, **__):
            pass
    neo4j_store_mod.Neo4jStore = _DummyStore  # type: ignore
    sys.modules["src.graphdb.neo4j_store"] = neo4j_store_mod

    # Stub APIModelRouter
    api_router_mod = types.ModuleType("src.models.api_model_router")
    class _DummyRouter:  # noqa: D401
        async def route_request(self, *_, **__):  # always returns empty string
            return "[]"
    api_router_mod.APIModelRouter = _DummyRouter  # type: ignore
    sys.modules["src.models.api_model_router"] = api_router_mod

    # Stub APIEmbeddingService
    embed_mod = types.ModuleType("src.embeddings.api_embedding_service")
    class _DummyEmbed:  # noqa: D401
        async def embed_text(self, text):  # returns small vector based on len
            return [float(len(text))]
    embed_mod.APIEmbeddingService = _DummyEmbed  # type: ignore
    sys.modules["src.embeddings.api_embedding_service"] = embed_mod

_injected = False

def _ensure_stubs():
    global _injected  # pylint: disable=global-statement
    if not _injected:
        _inject_stubs()
        _injected = True


# -----------------------------------------------------------------------------
# Helpers para criar entidades
# -----------------------------------------------------------------------------
from dataclasses import asdict
from typing import List, Dict, Any


def _entity(idx: int):
    from src.graphrag.graph_rag_enhancer import CodeEntity  # type: ignore
    return CodeEntity(
        id=f"E{idx}",
        name=f"Func{idx}",
        type="function",
        content="def f(): pass",
        file_path=f"file{idx}.py",
        metadata={},
    )


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def stubs():
    _ensure_stubs()
    yield


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_relationship_prompt_contains_entities():
    _ensure_stubs()
    from src.graphrag.graph_rag_enhancer import GraphRAGEnhancer  # import after stubs

    enhancer = GraphRAGEnhancer()
    ents = [_entity(1), _entity(2)]
    prompt = enhancer._create_relationship_extraction_prompt(ents)  # type: ignore

    # Deve conter cada nome e file_path
    for e in ents:
        assert e.name in prompt and e.file_path in prompt


def test_parse_relationship_response_extracts_relations():
    _ensure_stubs()
    from src.graphrag.graph_rag_enhancer import GraphRAGEnhancer  # type: ignore

    enhancer = GraphRAGEnhancer()
    ents = [_entity(1), _entity(2)]

    # Criar resposta fake JSON
    rel_json = json.dumps([
        {
            "source": "Func1",
            "target": "Func2",
            "type": "calls",
            "confidence": 0.9
        }
    ])
    response = f"LLM output before\n{rel_json}\nafter"

    rels = enhancer._parse_relationship_response(response, ents)  # type: ignore
    assert len(rels) == 1
    rel = rels[0]
    assert rel.source_id == "E1" and rel.target_id == "E2" and rel.relationship_type == "calls"


def test_calculate_similarity_basic():
    _ensure_stubs()
    from src.graphrag.graph_rag_enhancer import GraphRAGEnhancer  # type: ignore

    enhancer = GraphRAGEnhancer()
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    sim = enhancer._calculate_similarity(v1, v2)  # type: ignore
    # ortogonal => ~0
    assert abs(sim) < 1e-6

    sim_identical = enhancer._calculate_similarity([1, 1], [1, 1])  # type: ignore
    assert abs(sim_identical - 1.0) < 1e-6 