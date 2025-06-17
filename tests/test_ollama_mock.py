from unittest import mock

from src.rag_pipeline import RAGPipeline


def test_ollama_fallback_to_mock(monkeypatch):
    """Simula indisponibilidade do Ollama e verifica uso do _Mock."""
    # Faz com que o construtor Client levante exceção de conexão
    with mock.patch("ollama.Client", side_effect=ConnectionError("offline")):
        pipeline = RAGPipeline(config_path=None)

    # A instância _Mock possui método generate que retorna dict fixo
    resp = pipeline.ollama_client.generate(prompt="hi")
    assert "Resposta mock" in resp["response"] 