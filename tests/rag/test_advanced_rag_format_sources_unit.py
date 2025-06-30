from src.rag_pipeline_advanced import AdvancedRAGPipeline


def _pipeline():
    return object.__new__(AdvancedRAGPipeline)


def test_format_sources_with_graph():
    pl = _pipeline()
    docs = [
        {
            "content": "X" * 250,
            "metadata": {"source": "src1"},
            "score": 0.9,
            "graph_context": {"central_entities": ["Ent1"]},
        }
    ]
    sources = pl._format_sources(docs)
    assert sources[0]["graph_entities"] == ["Ent1"]
    assert sources[0]["content"].endswith("...")


def test_format_sources_without_graph():
    pl = _pipeline()
    docs = [{"content": "Y" * 20, "metadata": {}, "score": 0.1}]
    sources = pl._format_sources(docs)
    assert "graph_entities" not in sources[0] 