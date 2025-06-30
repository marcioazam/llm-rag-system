import asyncio

from src.rag_pipeline_base import BaseRAGPipeline


async def _run_query(pipeline):
    return await pipeline.query("Pergunta teste?")


def test_add_documents_and_stats():
    pipeline = BaseRAGPipeline()
    # add_documents deve atualizar métricas
    ok = asyncio.run(pipeline.add_documents(["doc1", "doc2"]))
    assert ok is True
    # Executar duas queries para manipular métricas
    asyncio.run(_run_query(pipeline))
    asyncio.run(_run_query(pipeline))

    stats = pipeline.get_stats()
    base = stats["base_metrics"]
    assert base["total_queries"] == 2
    assert base["successful_queries"] == 2
    assert base["total_documents_processed"] == 2
    # Taxa de sucesso deve ser 1.0
    assert base["success_rate"] == 1.0 