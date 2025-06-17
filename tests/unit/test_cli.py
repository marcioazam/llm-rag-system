import sys
import types
from click.testing import CliRunner

# -----------------------------------------------------------------------
# Stubs para evitar inicialização pesada durante import do rag_cli
# -----------------------------------------------------------------------

# Stub do RAGClient
rag_client_mod = types.ModuleType("src.client.rag_client")

class _StubClient:  # pylint: disable=too-few-public-methods
    def query(self, question, k=5):  # noqa: D401
        return {"answer": "stub", "sources": [], "model": "stub"}

    def query_llm_only(self, question):  # noqa: D401
        return {"answer": "stub", "model": "stub"}

    def index_documents(self, file_paths):  # noqa: D401
        return {"total_documents": len(file_paths), "total_chunks": 1}

rag_client_mod.RAGClient = _StubClient
sys.modules["src.client.rag_client"] = rag_client_mod

# Stub do RAGPipeline
rag_pipeline_mod = types.ModuleType("rag_pipeline")

class _StubPipeline:  # pylint: disable=too-few-public-methods
    def __init__(self, *args, **kwargs):
        self.config = {}

    def query(self, query_text, k=5, use_hybrid=True):  # noqa: D401
        return {"answer": "stub", "sources": [], "model": "stub"}

    def query_llm_only(self, question, system_prompt=None):  # noqa: D401
        return {"answer": "stub", "model": "stub"}

    def index_documents(self, paths):  # noqa: D401
        return {"total_documents": len(paths), "total_chunks": 1}

    def _initialize_components(self):
        # No-op para stub
        return None

rag_pipeline_mod.RAGPipeline = _StubPipeline
sys.modules["rag_pipeline"] = rag_pipeline_mod

# Stub DocumentLoader para comando add (não usado aqui, mas garante import)
doc_loader_mod = types.ModuleType("src.utils.document_loader")
class _StubLoader:  # pylint: disable=too-few-public-methods
    pass
doc_loader_mod.DocumentLoader = _StubLoader
sys.modules["src.utils.document_loader"] = doc_loader_mod

# Agora podemos importar rag_cli com segurança
from src.cli import rag_cli  # noqa: E402  pylint: disable=wrong-import-order

runner = CliRunner()


def test_cli_help():
    """Comando --help deve sair sem erros."""
    result = runner.invoke(rag_cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "Sistema RAG" in result.output


def test_cli_query_llm_only():
    """Verifica comando query com --llm-only e --json-output."""
    result = runner.invoke(
        rag_cli.cli,
        [
            "query",
            "Qual a capital da França?",
            "--llm-only",
            "--json-output",
        ],
    )
    assert result.exit_code == 0
    assert "\"answer\": \"stub\"" in result.output 