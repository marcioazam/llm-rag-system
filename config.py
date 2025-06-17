import os

# ---------------------------------------------------------------------------
# Carregar variáveis de ambiente (.env)
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:  # pragma: no cover
    # dotenv não é obrigatório em produção; apenas avisa se faltar.
    import logging

    logging.getLogger(__name__).warning("python-dotenv não instalado; variáveis de ambiente não serão carregadas de .env")

# Configurações da API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = False  # Desabilitado para evitar problemas

# Configurações do Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"

# Diretórios
UPLOAD_DIR = "uploads"
STORAGE_DIR = "storage"
LOG_DIR = "logs"

# Configurações do RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 2000

# ---------------------------------------------------------------------------
# Configurações do RAG / Grafo
# ---------------------------------------------------------------------------

RAG_CONFIG = {
    # Parâmetros existentes/legacy (podem ser expandidos conforme necessário)
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "max_tokens": MAX_TOKENS,

    # Configurações Neo4j (podem ser sobrescritas por variáveis de ambiente)
    "use_graph_store": os.getenv("USE_GRAPH_STORE", "true").lower() == "true",
    "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
    "neo4j_password": os.getenv("NEO4J_PASSWORD", "arrozefeijao13"),
}


def get_rag_config() -> dict:
    """Retorna cópia da configuração para ser usada no *RAGPipeline*.

    Ideal para evitar mutações globais acidentais.
    """

    return RAG_CONFIG.copy()

# Criar diretórios se não existirem
for dir_path in [UPLOAD_DIR, STORAGE_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)
