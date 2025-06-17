import pytest

from src.rag_pipeline import RAGPipeline


def test_multiple_pipelines_without_metric_duplication():
    """Instancia o RAGPipeline duas vezes seguidas.

    Se o CollectorRegistry estiver isolado por instância, nenhuma exceção
    `ValueError: Duplicated timeseries` deverá ser lançada.
    """
    # 1ª inicialização
    p1 = RAGPipeline(config_path=None)
    # 2ª inicialização – deve reutilizar registry interno sem colidir
    p2 = RAGPipeline(config_path=None)

    # Asserções de sanidade
    assert p1 is not None and p2 is not None
    # Cada pipeline possui seu próprio registry com as métricas
    for pipe in (p1, p2):
        metrics = {m.name for m in pipe._registry.collect()}
        assert any(n.startswith("rag_queries") for n in metrics)
        assert any(n.startswith("rag_query_latency") for n in metrics) 