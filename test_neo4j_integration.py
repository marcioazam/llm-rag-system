from pathlib import Path

from config import get_rag_config
from src.rag_pipeline import RAGPipeline
from src.graphdb.code_analyzer import CodeAnalyzer


def test_graph_integration():
    """Teste básico de integração entre RAGPipeline, Neo4jStore e CodeAnalyzer."""

    # ------------------------------------------------------------------
    # Inicializar pipeline com configurações do projeto
    # ------------------------------------------------------------------
    rag = RAGPipeline(config_path=None)
    rag.config.update(get_rag_config())
    rag._initialize_components()

    assert rag.graph_store is not None, "Graph store não inicializado"

    # ------------------------------------------------------------------
    # Analisar o próprio código-fonte do projeto (pasta src)
    # ------------------------------------------------------------------
    project_dir = Path(__file__).resolve().parent / "src"

    analyzer = CodeAnalyzer(rag.graph_store)
    analyzer.analyze_project(str(project_dir))

    # ------------------------------------------------------------------
    # Executar consulta utilizando contexto de grafo
    # ------------------------------------------------------------------
    question = "Como a classe RAGPipeline se relaciona com outros componentes?"
    result = rag.query(query_text=question, k=5)

    print(result)

    assert "answer" in result and result["answer"], "Resposta não foi gerada" 