import os
import sys
import types
from typing import Dict, List

import pytest


# ---------------------------------------------------------------------------
# Stub Neo4j driver – armazena nós/relacionamentos em memória
# ---------------------------------------------------------------------------

def _inject_neo4j_stub(tmp_path):
    """Injeta stub de `neo4j.GraphDatabase` em `sys.modules`."""
    if any(m.startswith("neo4j") for m in sys.modules):
        # Limpar para recarregar com stub
        for m in list(sys.modules):
            if m.startswith("neo4j"):
                sys.modules.pop(m, None)

    neo4j_mod = types.ModuleType("neo4j")

    class _DummyResult:
        def __init__(self, records):
            self._records = records

        def single(self):
            return self._records[0] if self._records else None

        def __iter__(self):
            return iter(self._records)

    class _DummySession:
        def __init__(self, store):
            self._store = store  # Dicts: nodes, rels

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        # Simplistic parser for queries used in tests
        def run(self, query: str, **params):
            nodes = self._store["nodes"]
            rels = self._store["rels"]

            if query.startswith("MERGE (e:CodeElement") or query.startswith("MERGE (n:" ):
                # add/update node using params or **params
                node_id = params.get("id") or params["id"]
                props = params.get("properties") or params
                nodes[node_id] = dict(props)
                return _DummyResult([])

            if query.startswith("MATCH (a {id: $source_id})"):
                # add relationship
                rels.append((params["source_id"], params["target_id"], query.split("MERGE (a)-[r:")[1].split("]")[0]))
                return _DummyResult([])

            if query.startswith("MATCH (n {id: $node_id}) RETURN n"):
                n = nodes.get(params["node_id"])
                return _DummyResult([{"n": n}] if n else [])

            if query.startswith("MATCH (n) RETURN count(n)"):
                return _DummyResult([{"total_nodes": len(nodes)}])

            if query.startswith("MATCH ()-[r]->() RETURN count(r)"):
                return _DummyResult([{"total_relations": len(rels)}])

            if query.startswith("MATCH (n) DETACH DELETE n"):
                nodes.clear(); rels.clear();
                return _DummyResult([])

            # Default empty result
            return _DummyResult([])

    class _DummyDriver:
        def __init__(self):
            self._store = {"nodes": {}, "rels": []}

        def session(self, database=None):  # noqa: D401
            return _DummySession(self._store)

        def close(self):
            pass

    class _GraphDatabase:
        @staticmethod
        def driver(uri, auth):  # noqa: D401
            return _DummyDriver()

    neo4j_mod.GraphDatabase = _GraphDatabase  # type: ignore
    sys.modules["neo4j"] = neo4j_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _neo4j_stub(tmp_path):
    _inject_neo4j_stub(tmp_path)
    # Definir senha para evitar erro de inicialização
    os.environ["NEO4J_PASSWORD"] = "stub"

    # Recarregar módulo alvo
    if "src.graphdb.neo4j_store" in sys.modules:
        del sys.modules["src.graphdb.neo4j_store"]
    yield
    # Cleanup env
    os.environ.pop("NEO4J_PASSWORD", None)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


def _make_store():
    from src.graphdb.neo4j_store import Neo4jStore

    store = Neo4jStore(password="stub")
    return store


def test_add_and_get_node():
    store = _make_store()
    element = {
        "id": "code::func1",
        "name": "func1",
        "type": "Function",
        "file_path": "file.py",
        "content": "def func1(): pass",
        "metadata": {},
    }
    store.add_code_element(element)

    node = store.get_node("code::func1")
    assert node is not None and node["name"] == "func1"
    store.close()


def test_add_relationship_and_stats():
    store = _make_store()
    # Adicionar dois nós
    for i in range(2):
        store.add_code_element({
            "id": f"code::{i}",
            "name": f"f{i}",
            "type": "Function",
            "file_path": "f.py",
            "content": "",
            "metadata": {},
        })

    from src.graphdb.graph_models import GraphRelation, RelationType

    rel = GraphRelation(source_id="code::0", target_id="code::1", type=RelationType.CALLS.value)
    store.add_relationship(rel)

    stats = store.get_stats()
    assert stats["total_nodes"] == 2
    assert stats["total_relations"] == 1

    # Limpar tudo
    store.clear_all()
    stats2 = store.get_stats()
    assert stats2["total_nodes"] == 0
    store.close()


def test_find_related_nodes():
    store = _make_store()
    # Add nodes and relation
    store.add_code_element({"id": "A", "name": "A", "type": "Concept", "file_path": "", "content": "", "metadata": {}})
    store.add_code_element({"id": "B", "name": "B", "type": "Concept", "file_path": "", "content": "", "metadata": {}})
    from src.graphdb.graph_models import GraphRelation, RelationType

    store.add_relationship(GraphRelation("A", "B", RelationType.IMPORTS.value))
    related = store.find_related_nodes("A", RelationType.IMPORTS.value, depth=1)
    assert any(n.get("id") == "B" for n in related)
    store.close() 