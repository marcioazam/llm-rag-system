import os
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
# Removido import ollama - usando apenas LLMs em nuvem
from prometheus_client import Counter, Histogram
import time
import uuid
import hashlib
import concurrent.futures, functools
from prometheus_client import CollectorRegistry

# Imports condicionais para compatibilidade
try:
    from .chunking.semantic_chunker import SemanticChunker
    from .chunking.recursive_chunker import RecursiveChunker
    from .embeddings.embedding_service import EmbeddingService
    from .retrieval.retriever import HybridRetriever
    from .utils.document_loader import DocumentLoader
    from src.models.model_router import ModelRouter, AdvancedModelRouter
    from .chunking.base_chunker import Chunk
    # Integração com grafo
    from .graphdb.neo4j_store import Neo4jStore
    from .graphdb.graph_models import NodeType, RelationType
except ImportError:
    # Fallback para imports absolutos (compatibilidade)
    from chunking.semantic_chunker import SemanticChunker
    from chunking.recursive_chunker import RecursiveChunker
    from src.embeddings.embedding_service import EmbeddingService
    from src.retrieval.retriever import Retriever as HybridRetriever
    from src.utils.document_loader import DocumentLoader
    from src.models.model_router import ModelRouter, AdvancedModelRouter
    from graphdb.neo4j_store import Neo4jStore
    from graphdb.graph_models import NodeType, RelationType

PROMETHEUS_STARTED = False

class RAGPipeline:
    """
    Pipeline completo de RAG com suporte a:
    - Configuração via YAML ou parâmetros
    - Roteamento inteligente de modelos (simples e avançado)
    - Fallback para LLM
    - Chunking semântico e recursivo
    - Retrieval híbrido
    - Compatibilidade com versões anteriores
    """

    def __init__(self, 
                 config_path: str = "config/config.yaml",
                 collection_name: Optional[str] = None,
                 persist_directory: str = "./chroma_db",
                 use_advanced_routing: bool = True,
                 settings: Dict[str, Any] = None,
                 graph_store: Neo4jStore = None,
                 embedding_type: str = "sentence-transformers"):
        
        global PROMETHEUS_STARTED
        # Configuração flexível - YAML ou parâmetros
        self.use_config_file = os.path.exists(config_path) if config_path else False
        
        if self.use_config_file:
            self._load_config(config_path, collection_name)
        else:
            self._setup_default_config(collection_name, persist_directory)
        
        # Configurações RAG
        self.rag_config = self.config.get("rag", {})
        self.fallback_to_llm = self.rag_config.get("fallback_to_llm", True)
        self.min_relevance_score = self.rag_config.get("min_relevance_score", 0.5)
        self.hybrid_mode = self.rag_config.get("hybrid_mode", False)
        self.enable_model_routing = self.rag_config.get("enable_model_routing", True)
        self.use_advanced_routing = use_advanced_routing
        self.embedding_type = embedding_type
        
        # Configurar logging
        self._setup_logging()
        
        # ------------------------------
        # Métricas Prometheus (isoladas por pipeline para evitar duplicação)
        # ------------------------------
        self._registry = CollectorRegistry()

        self._metric_query_count = Counter(
            "rag_queries_total", "Total de queries RAG", registry=self._registry
        )
        self._metric_query_latency = Histogram(
            "rag_query_latency_seconds", "Latência das queries RAG", registry=self._registry
        )
        self._metric_errors = Counter(
            "rag_errors_total", "Total de erros em operações críticas", registry=self._registry
        )

        # Métricas adicionais com labels
        self._metric_prompt_usage = Counter(
            "rag_prompt_usage_total", "Total de queries por prompt id", ["prompt_id"], registry=self._registry
        )
        self._metric_prompt_variant = Counter(
            "rag_prompt_variant_total", "Total de queries por variante de prompt", ["variant"], registry=self._registry
        )

        # Expor endpoint Prometheus se habilitado
        if self.config.get("monitoring", {}).get("enabled") and not PROMETHEUS_STARTED:
            port = self.config["monitoring"].get("port", 8001)
            try:
                from prometheus_client import start_http_server
                start_http_server(port, registry=self._registry)
                PROMETHEUS_STARTED = True
                self.logger.info("Prometheus HTTP server iniciado na porta %s", port)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Falha ao iniciar servidor Prometheus: %s", exc)

        # Inicializar componentes
        self._initialize_components()

        # Adicionar settings
        if settings:
            self.config.update(settings)

        # Adicionar graph_store
        self.graph_store = graph_store

        self.logger.info(f"RAG Pipeline initialized successfully (routing: {self.routing_mode})")

        # ------------------------------------------------------------------
        # Registrar arquitetura ativa
        # ------------------------------------------------------------------
        try:
            from architecture.registry import (
                RAGArchitecture,
                register_architecture,
            )

            arch_info = RAGArchitecture(
                document_loaders=[self.document_loader.__class__.__name__],
                preprocessors=[],
                chunkers=[self._get_chunker().__class__.__name__],
                embedding_models=[embed_config.get("model_name", "unknown")],
                vector_databases=[self.vector_store.__class__.__name__],
                retrieval_strategies=[self.retriever.__class__.__name__],
                rerankers=["HybridReranker"] if self.rerank_enabled else [],
                llm_models=[self.model, self.code_model],
                prompt_optimizers=[],
                metrics_collectors=["prometheus_client"],
                feedback_loops=[],
            )

            register_architecture(arch_info)
            self.logger.debug("RAGArchitecture registrada com sucesso")
        except Exception as e:
            self.logger.debug(f"Falha ao registrar RAGArchitecture: {e}")

    def _load_config(self, config_path: str, collection_name: Optional[str]):
        """Carrega configuração do arquivo YAML"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Override collection name se fornecido
        if collection_name:
            if "vectordb" not in self.config:
                self.config["vectordb"] = {}
            self.config["vectordb"]["collection_name"] = collection_name

    def _setup_default_config(self, collection_name: str, persist_directory: str):
        """Configuração padrão quando não há arquivo YAML"""
        self.config = {
            "chunking": {
                "method": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "min_chunk_size": 100,
                "max_chunk_size": 500
            },
            "embeddings": {
                "model_name": "all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32
            },
            "vectordb": {
                "type": "qdrant",
                "persist_directory": persist_directory,
                "collection_name": collection_name or "documents"
            },
            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.5,
                "rerank": False,
                "rerank_model": None
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "code_model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "rag": {
                "fallback_to_llm": True,
                "min_relevance_score": 0.5,
                "hybrid_mode": False,
                "enable_model_routing": True
            },
            # Configurações de grafo / Neo4j
            "use_graph_store": False,
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password"
        }

    def _setup_logging(self):
        """Configurar sistema de logging"""
        # Criar diretório de logs se não existir
        os.makedirs('logs', exist_ok=True)
        
        from pythonjsonlogger import jsonlogger

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        log_handler = logging.FileHandler('logs/rag_pipeline.json')
        formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
        # Stream handler simples texto
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)
        self.logger = logging.getLogger(__name__)

    def _initialize_components(self):
        """Inicializar todos os componentes do pipeline"""
        
        # Embedding Service
        embed_config = self.config["embeddings"]

        if self.config["embeddings"].get("hierarchical", False):
            from .embeddings.hierarchical_embedding_service import HierarchicalEmbeddingService
            self.embedding_service = HierarchicalEmbeddingService(
                device=embed_config.get("device", "cpu"),
                batch_size=embed_config.get("batch_size", 16)
            )
        elif self.embedding_type == "sentence-transformers":
            self.embedding_service = EmbeddingService(
                model_name=embed_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                device=embed_config.get("device", "cpu"),
                batch_size=embed_config.get("batch_size", 16)
            )
        else:
            raise NotImplementedError("Somente 'sentence-transformers' está implementado no momento.")

        # Vector Store
        vectordb_config = self.config["vectordb"]
        vectordb_type = vectordb_config.get("type", "chromadb").lower()

        if vectordb_type == "qdrant":
            # Importação lazy para evitar dependência obrigatória quando não usada
            try:
                from .vectordb.qdrant_store import QdrantVectorStore  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Dependência 'qdrant-client' não instalada. Adicione-a ao requirements.txt."
                ) from exc

            self.vector_store = QdrantVectorStore(
                host=vectordb_config.get("host", "localhost"),
                port=vectordb_config.get("port", 6333),
                collection_name=vectordb_config.get("collection_name", "rag_chunks"),
                dim=embed_config.get("embedding_dim", 768) if isinstance(embed_config, dict) else 768,
            )
            self.logger.info("Usando QdrantVectorStore como backend vetorial")
        else:
            raise ValueError("vectordb.type diferente de 'qdrant' não é suportado neste build.")

        # Metadata Store (SQLite)
        from .metadata.sqlite_store import SQLiteMetadataStore  # type: ignore
        try:
            db_path = os.path.join(vectordb_config["persist_directory"], "metadata", "chunks.db")
            self.metadata_store = SQLiteMetadataStore(db_path=db_path)
            self.logger.info("SQLiteMetadataStore inicializado em %s", db_path)
        except Exception as exc:
            self.logger.warning("Falha ao inicializar SQLiteMetadataStore: %s", exc)
            self.metadata_store = None

        # Reranker (CrossEncoder) – opcional
        from .retrieval.reranker import HybridReranker  # import interno para evitar dependência circular

        retrieval_config = self.config["retrieval"]
        self.rerank_enabled = retrieval_config.get("rerank", False)
        self.reranker: Optional[HybridReranker] = None
        if self.rerank_enabled:
            try:
                self.reranker = HybridReranker()
                self.logger.info("HybridReranker inicializado para reranking CrossEncoder")
            except Exception as e:
                self.logger.warning(f"Falha ao inicializar HybridReranker (desabilitando rerank): {e}")
                self.rerank_enabled = False

        # Retriever (sem rerank interno, pois faremos externamente)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            rerank=False,
            rerank_model=None
        )

        # Document Loader (multi-formato)
        try:
            from .utils.smart_document_loader import SmartDocumentLoader
            self.document_loader = SmartDocumentLoader()
            self.logger.info("SmartDocumentLoader habilitado (multi-formato)")
        except Exception as e:
            self.logger.warning(f"Falha ao carregar SmartDocumentLoader, usando DocumentLoader básico: {e}")
            self.document_loader = DocumentLoader()

        # ------------------------------------------------------------------
        # Integração com Neo4j / Graph Store
        # ------------------------------------------------------------------

        self.graph_store = None  # Garantir atributo sempre presente
        if self.config.get("use_graph_store", False):
            try:
                self.graph_store = Neo4jStore(
                    uri=self.config.get("neo4j_uri", "bolt://localhost:7687"),
                    user=self.config.get("neo4j_user", "neo4j"),
                    password=self.config.get("neo4j_password", "password"),
                )
                self.logger.info("Graph store (Neo4j) inicializado com sucesso")
            except Exception as e:  # pragma: no cover
                self.logger.warning(
                    f"Não foi possível inicializar o Neo4jStore, prosseguindo sem grafo: {e}"
                )

        # Model Router com suporte a ambos os tipos
        if self.enable_model_routing:
            if self.use_advanced_routing:
                try:
                    self.model_router = AdvancedModelRouter()
                    self.routing_mode = 'advanced'
                    self.logger.info("Usando roteamento avançado multi-modelo")
                except Exception as e:
                    self.logger.warning(f"Erro ao inicializar roteador avançado: {e}")
                    self.logger.info("Voltando para roteador simples")
                    self.model_router = ModelRouter()
                    self.routing_mode = 'simple'
            else:
                self.model_router = ModelRouter()
                self.routing_mode = 'simple'
        else:
            self.routing_mode = 'disabled'

        # Cliente LLM em nuvem (removido Ollama)
        self.cloud_llm_client = self._initialize_cloud_llm()

        # Modelos disponíveis (compatibilidade)
        self.model = self.config["llm"]["model"]
        self.code_model = self.config["llm"].get("code_model", "gpt-4")
    
    def _initialize_cloud_llm(self):
        """Inicializa cliente para LLM em nuvem"""
        llm_config = self.config["llm"]
        provider = llm_config.get("provider", "openai")
        
        if provider == "openai":
            try:
                import openai
                return openai.OpenAI()
            except ImportError:
                self.logger.warning("OpenAI não instalado. Instale com: pip install openai")
                return None
        elif provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic()
            except ImportError:
                self.logger.warning("Anthropic não instalado. Instale com: pip install anthropic")
                return None
        else:
            self.logger.warning(f"Provider {provider} não suportado")
            return None

    def _get_chunker(self, chunking_strategy: str = None, chunk_size: int = None, chunk_overlap: int = None):
        """Obter chunker baseado na estratégia"""
        chunking_config = self.config["chunking"]
        strategy = chunking_strategy or chunking_config.get("method", "recursive")
        
        if strategy == "semantic":
            return SemanticChunker(
                min_chunk_size=chunking_config.get("min_chunk_size", 100),
                max_chunk_size=chunk_size or chunking_config.get("max_chunk_size", 500)
            )
        elif strategy == "advanced" or strategy == "hybrid":
            from .chunking.advanced_chunker import AdvancedChunker
            return AdvancedChunker(
                embedding_service=self.embedding_service,
                max_chunk_size=chunk_size or chunking_config.get("max_chunk_size", 800),
                chunk_overlap=chunk_overlap or chunking_config.get("chunk_overlap", 50)
            )
        else:
            return RecursiveChunker(
                chunk_size=chunk_size or chunking_config.get("chunk_size", 500),
                chunk_overlap=chunk_overlap or chunking_config.get("chunk_overlap", 50)
            )

    def index_documents(self, 
                       document_paths: List[str],
                       batch_size: int = 10) -> Dict[str, Any]:
        """Indexar documentos no banco vetorial usando document loader"""
        
        total_chunks = 0
        errors = []
        chunker = self._get_chunker()

        for doc_path in document_paths:
            try:
                self.logger.info(f"Processing document: {doc_path}")

                # Carregar documento
                document = self.document_loader.load(doc_path)

                # Chunking
                if hasattr(chunker, 'chunk'):
                    chunks = chunker.chunk(
                        text=document["content"],
                        metadata=document["metadata"]
                    )
                else:
                    # Compatibilidade com método antigo
                    chunk_texts = chunker.chunk_text(document["content"])
                    chunks = [
                        Chunk(content=text, metadata=document["metadata"])
                        for text in chunk_texts
                    ]

                # Gerar embeddings
                texts = [chunk.content for chunk in chunks]
                embeddings = self.embedding_service.embed_texts(texts)

                # Adicionar ao vector store
                if hasattr(self.vector_store, 'add_chunks'):
                    self.vector_store.add_chunks(chunks, embeddings.tolist())
                else:
                    # Compatibilidade com método antigo
                    metadatas = [chunk.metadata for chunk in chunks]
                    self.vector_store.add_documents(
                        documents=texts,
                        embeddings=embeddings.tolist(),
                        metadata=metadatas
                    )

                total_chunks += len(chunks)
                self.logger.info(f"Added {len(chunks)} chunks from {doc_path}")

            except Exception as e:
                self.logger.error(f"Error processing {doc_path}: {str(e)}")
                self._metric_errors.inc()
                errors.append({"document": doc_path, "error": str(e)})

        return {
            "total_chunks": total_chunks,
            "total_documents": len(document_paths),
            "errors": errors
        }

    def add_documents(self,
                     documents: List[Dict[str, str]],
                     project_id: str | None = None,
                     chunking_strategy: str = 'recursive',
                     chunk_size: int = 500,
                     chunk_overlap: int = 50) -> None:
        """
        Adiciona documentos ao vector store (compatibilidade com ambas as versões)
        
        Args:
            documents: Lista de documentos com 'content' e opcionalmente 'metadata' e 'source'
            project_id: Identificador do projeto
            chunking_strategy: 'semantic' ou 'recursive'
            chunk_size: Tamanho do chunk
            chunk_overlap: Sobreposição entre chunks
        """
        
        chunker = self._get_chunker(chunking_strategy, chunk_size, chunk_overlap)
        
        # Preprocessor inteligente
        try:
            from .preprocessing.intelligent_preprocessor import IntelligentPreprocessor
            preprocessor = IntelligentPreprocessor()
        except Exception as e:
            preprocessor = None
            self.logger.debug(f"IntelligentPreprocessor indisponível: {e}")

        all_chunks = []
        all_metadatas = []
        all_texts = []

        for doc in documents:
            # Pré-processar conteúdo se preprocessor disponível
            content_to_use = doc['content']
            if preprocessor is not None:
                processed = preprocessor.process(doc['content'])
                content_to_use = processed['cleaned']
                # Merge metadata
                doc_metadata = doc.get('metadata', {})
                doc_metadata.update(processed.get('metadata', {}))
                doc_metadata['entities'] = processed.get('entities')
                doc['metadata'] = doc_metadata

            # Usar o método apropriado baseado no chunker
            if hasattr(chunker, 'chunk'):
                # Método novo (SemanticChunker ou RecursiveChunker atualizado)
                base_metadata = doc.get('metadata', {})
                base_metadata['source'] = doc.get('source', 'unknown')
                if project_id:
                    base_metadata['project_id'] = project_id
                
                chunks = chunker.chunk(
                    text=content_to_use,
                    metadata=base_metadata
                )
                
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk.content)
                    metadata = chunk.metadata.copy()
                    metadata['chunk_index'] = i
                    # -----------------------------
                    # Persistência em FS e SQLite
                    # -----------------------------
                    chunk_id = metadata.get('id') or str(uuid.uuid4())
                    chunk_hash = hashlib.sha1(chunk.content.encode('utf-8')).hexdigest()
                    # Diretório para chunks brutos
                    chunk_dir = os.path.join(self.config['vectordb']['persist_directory'], 'chunks')
                    os.makedirs(chunk_dir, exist_ok=True)
                    file_path = os.path.join(chunk_dir, f"{chunk_id}.txt")
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(chunk.content)
                    except Exception as exc:
                        self.logger.debug(f"Falha ao salvar chunk {chunk_id} em disco: {exc}")

                    metadata['file_path'] = file_path
                    metadata['chunk_hash'] = chunk_hash
                    # Salvar no SQLiteMetadataStore
                    try:
                        if hasattr(self, 'metadata_store') and self.metadata_store is not None:
                            self.metadata_store.upsert_metadata({
                                'id': chunk_id,
                                'file_path': file_path,
                                'language': metadata.get('language'),
                                'symbols': metadata.get('symbols'),
                                'relations': metadata.get('relations'),
                                'coverage': metadata.get('coverage'),
                                'source': metadata.get('source', 'unknown'),
                                'chunk_hash': chunk_hash,
                                'project_id': project_id,
                            })
                    except Exception as exc:
                        self.logger.debug(f"Falha ao persistir metadados no SQLite: {exc}")

                    metadata['id'] = chunk_id
                    all_metadatas.append(metadata)
                    all_chunks.append(chunk)
                    
            else:
                # Método antigo (compatibilidade)
                chunk_texts = chunker.chunk_text(content_to_use)
                for i, chunk_text in enumerate(chunk_texts):
                    all_texts.append(chunk_text)
                    metadata = doc.get('metadata', {}).copy()
                    metadata['chunk_index'] = i
                    metadata['source'] = doc.get('source', 'unknown')
                    if project_id:
                        metadata['project_id'] = project_id
                    chunk_id = metadata.get('id') or str(uuid.uuid4())
                    chunk_hash = hashlib.sha1(chunk_text.encode('utf-8')).hexdigest()
                    chunk_dir = os.path.join(self.config['vectordb']['persist_directory'], 'chunks')
                    os.makedirs(chunk_dir, exist_ok=True)
                    file_path = os.path.join(chunk_dir, f"{chunk_id}.txt")
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(chunk_text)
                    except Exception as exc:
                        self.logger.debug(f"Falha ao salvar chunk {chunk_id} em disco: {exc}")

                    metadata['file_path'] = file_path
                    metadata['chunk_hash'] = chunk_hash
                    try:
                        if hasattr(self, 'metadata_store') and self.metadata_store is not None:
                            self.metadata_store.upsert_metadata({
                                'id': chunk_id,
                                'file_path': file_path,
                                'language': metadata.get('language'),
                                'symbols': metadata.get('symbols'),
                                'relations': metadata.get('relations'),
                                'coverage': metadata.get('coverage'),
                                'source': metadata.get('source', 'unknown'),
                                'chunk_hash': chunk_hash,
                                'project_id': project_id,
                            })
                    except Exception as exc:
                        self.logger.debug(f"Falha ao persistir metadados no SQLite: {exc}")

                    metadata['id'] = chunk_id
                    all_metadatas.append(metadata)
                    
                    # Criar chunk objects se necessário
                    if hasattr(self.vector_store, 'add_chunks'):
                        chunk = Chunk(content=chunk_text, metadata=metadata)
                        all_chunks.append(chunk)

        # Gerar embeddings
        embeddings = self.embedding_service.embed_texts(all_texts)

        # Adicionar ao vector store usando o método apropriado
        if hasattr(self.vector_store, 'add_chunks') and all_chunks:
            self.vector_store.add_chunks(all_chunks, embeddings.tolist())
        else:
            # Compatibilidade com método antigo
            self.vector_store.add_documents(
                documents=all_texts,
                embeddings=embeddings.tolist(),
                metadata=all_metadatas
            )

        self.logger.info(f"Adicionados {len(all_texts)} chunks ao vector store")

    def query(self,
              query_text: str,
              k: int = None,
              system_prompt: Optional[str] = None,
              force_use_context: bool = False,
              use_hybrid: bool = None,
              force_models: Optional[List[str]] = None,
              project_id: str | None = None) -> Dict[str, Any]:
        """
        Executar query RAG com todas as funcionalidades combinadas
        
        Args:
            query_text: Pergunta do usuário
            k: Número de documentos a recuperar
            system_prompt: Prompt customizado do sistema
            force_use_context: Se True, sempre tenta usar contexto
            use_hybrid: Se True, usa roteamento inteligente
            force_models: Lista de modelos específicos (para modo avançado)
            project_id: Identificador do projeto
        """
        
        start_t = time.perf_counter()
        status_label = "ok"

        selected_prompt_id = None  # <- track for diagnostics

        if k is None:
            k = self.config["retrieval"]["top_k"]
        
        if use_hybrid is None:
            use_hybrid = self.enable_model_routing

        # -------- Prompt Selection (Step 6) --------
        from src.ab_test import decide_variant
        variant = decide_variant(query_text)
        self._metric_prompt_variant.labels(variant=variant).inc()

        use_prompt_variant = variant == "with_prompt"

        if system_prompt is None and use_prompt_variant:
            try:
                from src.prompt_selector import select_prompt
                _pid, system_prompt = select_prompt(query_text, depth="quick")
                selected_prompt_id = _pid
                self.logger.debug(f"Prompt '{_pid}' selecionado para a query.")
            except Exception as _e:  # noqa: E501
                self.logger.debug(f"Prompt selection falhou: {_e}")
                use_prompt_variant = False  # fallback

        # --------------------------------------------------
        # Query Enhancement (sinônimos, reformulações)
        # --------------------------------------------------
        enhanced_queries = [query_text]
        if self.config["retrieval"].get("query_enhancement", False):
            try:
                from .retrieval.query_enhancer import QueryEnhancer
                enhancer = QueryEnhancer(max_expansions=self.config["retrieval"].get("max_expansions", 3))
                enhanced_queries = enhancer.enhance_query(query_text)
                self.logger.info(f"Query enhancement gerou {len(enhanced_queries)} variações")
            except Exception as e:
                self.logger.debug(f"Falha em QueryEnhancer: {e}")

        # Recuperar documentos relevantes
        self.logger.info(f"Retrieving context for query: {query_text}")
        retrieved_docs = []

        try:
            if hasattr(self.retriever, 'retrieve'):
                all_results = []
                for qvar in enhanced_queries:
                    res = self.retriever.retrieve(
                        query=qvar,
                        k=k,
                        similarity_threshold=self.config["retrieval"]["similarity_threshold"],
                        search_type=self.config["retrieval"].get("search_type", "hybrid"),
                        use_mmr=self.config["retrieval"].get("use_mmr", False),
                        project_id=project_id
                    )
                    all_results.extend(res)
                # dedup por conteúdo/id
                seen = set()
                retrieved_docs = []
                for r in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
                    uid = r.get("id") or r["content"][:50]
                    if uid not in seen:
                        seen.add(uid)
                        retrieved_docs.append(r)
                        if len(retrieved_docs) >= k:
                            break
            else:
                # Compatibilidade com método antigo
                results = self.retriever.retrieve(query_text, k=k, project_id=project_id)
                retrieved_docs = [
                    {
                        "content": r['document'],
                        "metadata": r['metadata'],
                        "distance": r.get('distance', 0)
                    }
                    for r in results
                ]
        except Exception as e:
            self.logger.warning(f"Error retrieving documents: {e}")
            self._metric_errors.inc()

        # -------------------------------------------------
        # Reranking opcional com CrossEncoder
        # -------------------------------------------------
        if self.rerank_enabled and self.reranker is not None and retrieved_docs:
            try:
                retrieved_docs = self.reranker.rerank(query_text, retrieved_docs, k=k)
                self.logger.info("Resultados rerankeados via CrossEncoder")
            except Exception as e:
                self.logger.warning(f"Falha ao executar reranking: {e}")

        # Verificar se encontrou documentos
        if not retrieved_docs:
            if self.fallback_to_llm:
                self._last_prompt_id = selected_prompt_id
                return self.query_llm_only(query_text, system_prompt)
            else:
                self._last_prompt_id = selected_prompt_id
                return {
                    "answer": "Não encontrei informações relevantes nos documentos disponíveis para responder sua pergunta.",
                    "sources": [],
                    "model": self.config["llm"]["model"],
                    "response_mode": "no_context",
                    "models_used": [],
                    "strategy": "no_results",
                    "routing_mode": self.routing_mode,
                    "prompt_variant": variant,
                }

        # Filtrar por relevância
        relevant_docs = []
        for doc in retrieved_docs:
            score = 1 - doc.get("distance", 1) if doc.get("distance") is not None else 0
            if score >= self.min_relevance_score:
                relevant_docs.append(doc)

        # Decidir quais documentos usar
        has_relevant_context = len(relevant_docs) > 0
        docs_to_use = relevant_docs if has_relevant_context else retrieved_docs

        if not has_relevant_context and not force_use_context and not self.fallback_to_llm:
            self._last_prompt_id = selected_prompt_id
            return {
                "answer": "Não encontrei informações suficientemente relevantes para responder sua pergunta.",
                "sources": [],
                "model": self.config["llm"]["model"],
                "response_mode": "low_relevance",
                "models_used": [],
                "strategy": "insufficient_relevance",
                "routing_mode": self.routing_mode,
                "prompt_variant": variant,
            }

        # Extrair contexto e fontes
        retrieved_texts = [doc["content"] for doc in docs_to_use]
        sources = [
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": 1 - doc["distance"] if doc["distance"] is not None else None
            }
            for doc in docs_to_use
        ]

        # -----------------------------------------------
        # Enriquecimento com contexto de grafo (Neo4j)
        # -----------------------------------------------
        if self.graph_store is not None and docs_to_use:
            try:
                # Extrair possíveis IDs de embedding dos metadados dos documentos
                embedding_ids = []
                for doc in docs_to_use:
                    meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                    eid = meta.get("id") or meta.get("embedding_id") or doc.get("id")
                    if eid:
                        embedding_ids.append(str(eid))

                if embedding_ids:
                    graph_results = self.graph_store.find_by_embedding_ids(embedding_ids)

                    for item in graph_results:
                        node_info = item.get("node", {})
                        node_summary = f"[GraphNode] {node_info.get('type', '')}: {node_info}"
                        docs_to_use.append({
                            "content": node_summary,
                            "metadata": {"source": "graph", "node_id": node_info.get("id")},
                            "distance": None,
                        })

                        for conn in item.get("connections", []):
                            conn_node = conn.get("node")
                            rel_type = conn.get("relationship")
                            if conn_node:
                                conn_summary = f"[Graph-{rel_type}] {conn_node}"
                                docs_to_use.append({
                                    "content": conn_summary,
                                    "metadata": {"source": "graph", "node_id": conn_node.get("id")},
                                    "distance": None,
                                })
            except Exception as e:
                self.logger.warning(f"Erro ao enriquecer contexto via grafo: {e}")

        # Usar roteamento inteligente se habilitado
        if use_hybrid and self.enable_model_routing and hasattr(self, 'model_router'):
            
            context = " ".join(retrieved_texts)
            
            # Roteamento avançado
            if self.routing_mode == 'advanced':
                self.logger.info("Using advanced model routing")
                
                result = self.model_router.generate_advanced_response(
                    query=query_text,
                    context=context,
                    retrieved_docs=retrieved_texts
                )
                
                self._last_prompt_id = selected_prompt_id
                return {
                    'answer': result['answer'],
                    'sources': sources,
                    'model': result.get('primary_model', 'unknown'),
                    'response_mode': 'rag_with_advanced_routing',
                    'models_used': result['models_used'],
                    'strategy': 'multi-model',
                    'tasks_performed': result.get('tasks_performed', []),
                    'routing_mode': self.routing_mode,
                    'retrieved_chunks': retrieved_texts,
                    'prompt_variant': variant,
                }
            
            # Roteamento simples
            elif self.routing_mode == 'simple':
                self.logger.info("Using simple model routing")
                
                routing_info = self.model_router.route_query(query_text, context)
                
                answer = self.model_router.generate_hybrid_response(
                    query=query_text,
                    context=context,
                    retrieved_docs=retrieved_texts
                )
                
                models_used = [routing_info['primary_model']]
                if routing_info.get('secondary_model'):
                    models_used.append(routing_info['secondary_model'])
                
                self._last_prompt_id = selected_prompt_id
                return {
                    "answer": answer,
                    "sources": sources,
                    "model": routing_info['primary_model'],
                    "response_mode": "rag_with_simple_routing",
                    "models_used": models_used,
                    "strategy": routing_info['strategy'],
                    "needs_code": routing_info.get('needs_code', False),
                    "routing_mode": self.routing_mode,
                    "retrieved_chunks": retrieved_texts,
                    "prompt_variant": variant,
                }

        # --------------------------------------------------
        # Context Injector – seleciona e trunca snippets
        # --------------------------------------------------
        from .augmentation.context_injector import ContextInjector
        injector = ContextInjector(
            relevance_threshold=self.config["retrieval"].get("similarity_threshold", 0.7),
            max_tokens=self.config["retrieval"].get("max_context_tokens", 3000),
        )
        context_snippets = injector.inject_context(query_text, docs_to_use)

        # Dynamic Prompt
        from .augmentation.dynamic_prompt_system import DynamicPromptSystem
        prompt_builder = DynamicPromptSystem()
        context = "\n\n".join(context_snippets)

        prompt = prompt_builder.generate_prompt(
            query=query_text,
            context=context_snippets,
            task_type="qa",
            language="Português",
        )

        # -------- Combine selected template (if any) --------
        if system_prompt is not None:
            try:
                from src.template_renderer import render_template
                system_prompt_rendered = render_template(system_prompt, query=query_text, context_snippets=context_snippets)
            except Exception:
                system_prompt_rendered = system_prompt
            prompt = f"{system_prompt_rendered}\n\n{prompt}"

        self.logger.info("Generating response with dynamic prompt")
        def _llm_call():
            return self.ollama_client.generate(
                model=self.config["llm"]["model"],
                prompt=prompt,
                options={
                    "temperature": self.config["llm"]["temperature"],
                    "num_predict": self.config["llm"]["max_tokens"]
                }
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _exec:
            future = _exec.submit(_llm_call)
            try:
                response = future.result(timeout=self.config["llm"].get("timeout", 300))
            except concurrent.futures.TimeoutError:
                self._metric_errors.inc()
                raise TimeoutError("LLM response timed out")

        result = {
            "answer": response["response"],
            "sources": sources,
            "model": self.config["llm"]["model"],
            "response_mode": "rag_with_context",
            "models_used": [self.config["llm"]["model"]],
            "strategy": "traditional_rag",
            "needs_code": False,
            "routing_mode": self.routing_mode,
            "retrieved_chunks": retrieved_texts,
            "prompt_id": selected_prompt_id,
            "prompt_variant": variant,
        }

        # Adicionar nota sobre relevância se necessário
        if len(relevant_docs) < len(retrieved_docs):
            result["note"] = f"Encontrados {len(retrieved_docs)} documentos, mas apenas {len(relevant_docs)} foram considerados relevantes (score >= {self.min_relevance_score})"

        # -----------------------------
        # Pós-processamento: citações
        # -----------------------------
        try:
            from .generation.response_optimizer import ResponseOptimizer
            optimizer = ResponseOptimizer()
            result["answer"] = optimizer.add_citations(result["answer"], sources)
        except Exception:
            pass

        self._metric_query_count.labels(status=status_label).inc()
        self._metric_query_latency.observe(time.perf_counter() - start_t)
        self._last_prompt_id = selected_prompt_id  # for use in fallback paths

        if selected_prompt_id:
            try:
                self._metric_prompt_usage.labels(prompt_id=selected_prompt_id).inc()
            except Exception:
                pass

        return result

    def query_with_specific_models(self, 
                                  query_text: str, 
                                  models: List[str], 
                                  k: int = 5) -> Dict[str, Any]:
        """
        Faz query usando modelos específicos (apenas para modo avançado)
        """
        if self.routing_mode != 'advanced':
            self.logger.warning("Specific model routing only available in advanced mode, falling back to regular query")
            return self.query(query_text, k=k, use_hybrid=True)
        
        # Recupera documentos
        results = self.retriever.retrieve(query_text, k=k)
        retrieved_docs = [r['document'] for r in results]
        sources = [r['metadata'].get('source', 'unknown') for r in results]
        
        # TODO: Implementar lógica para forçar modelos específicos no AdvancedModelRouter
        result = self.model_router.generate_advanced_response(
            query=query_text,
            context=" ".join(retrieved_docs),
            retrieved_docs=retrieved_docs
        )
        
        return {
            'answer': result['answer'],
            'sources': list(set(sources)),
            'models_used': result['models_used'],
            'strategy': 'forced-models',
            'requested_models': models,
            'tasks_performed': result.get('tasks_performed', []),
            'retrieved_chunks': retrieved_docs
        }

    def query_with_code_examples(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """Método de conveniência que sempre usa roteamento híbrido"""
        return self.query(query_text, k=k, use_hybrid=True)

    def query_llm_only(self,
                       question: str,
                       system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Fazer uma query usando apenas o LLM, sem contexto"""

        if system_prompt is None:
            system_prompt = """Você é um assistente útil que responde perguntas com base em seu conhecimento geral.
            Seja preciso, informativo e indique quando estiver menos certo sobre alguma informação."""

        prompt = f"""{system_prompt}

Pergunta: {question}

Resposta:"""

        # Propaga prompt_id se atributo existir no caller frame
        prompt_id = getattr(self, "_last_prompt_id", None)
        selected_prompt_id = prompt_id  # compatibilidade
        variant = "no_prompt"

        self.logger.info("Generating LLM-only response")
        def _llm_call():
            if self.cloud_llm_client is None:
                return {"response": "[LLM em nuvem não disponível]"}
            
            # Usar OpenAI API
            response = self.cloud_llm_client.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=self.config["llm"]["temperature"],
                max_tokens=self.config["llm"]["max_tokens"]
            )
            return {"response": response.choices[0].message.content}

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _exec:
            future = _exec.submit(_llm_call)
            try:
                response = future.result(timeout=self.config["llm"].get("timeout", 300))
            except concurrent.futures.TimeoutError:
                self._metric_errors.inc()
                raise TimeoutError("LLM response timed out")

        self._last_prompt_id = selected_prompt_id
        return {
            "answer": response["response"],
            "sources": [],
            "model": self.config["llm"]["model"],
            "response_mode": "llm_only",
            "models_used": [self.config["llm"]["model"]],
            "strategy": "knowledge_only",
            "needs_code": False,
            "routing_mode": self.routing_mode,
            "prompt_id": prompt_id,
            "prompt_variant": variant,
        }

    def get_available_models(self) -> Dict[str, Any]:
        """Retorna informações sobre modelos disponíveis"""
        if hasattr(self.model_router, 'get_model_status'):
            return self.model_router.get_model_status()
        else:
            return {
                'routing_mode': self.routing_mode,
                'available': {
                    'general': self.model,
                    'code': self.code_model
                }
            }

    def benchmark_models(self, query_text: str, k: int = 3) -> Dict[str, Any]:
        """Compara respostas de diferentes configurações de modelos"""
        results = {}
        
        # Teste 1: Modelo único (baseline)
        single_result = self.query(query_text, k=k, use_hybrid=False)
        results['single_model'] = {
            'answer': single_result['answer'][:200] + '...',
            'models': single_result['models_used'],
            'time': 'N/A'  # TODO: Adicionar medição de tempo
        }
        
        # Teste 2: Modo híbrido baseado no routing_mode
        if self.routing_mode == 'simple':
            hybrid_result = self.query(query_text, k=k, use_hybrid=True)
            results['hybrid_simple'] = {
                'answer': hybrid_result['answer'][:200] + '...',
                'models': hybrid_result['models_used'],
                'strategy': hybrid_result['strategy']
            }
        elif self.routing_mode == 'advanced':
            advanced_result = self.query(query_text, k=k, use_hybrid=True)
            results['hybrid_advanced'] = {
                'answer': advanced_result['answer'][:200] + '...',
                'models': advanced_result['models_used'],
                'tasks': advanced_result.get('tasks_performed', [])
            }
        
        return results

    def clear_index(self):
        """Limpar todo o índice"""
        if hasattr(self.vector_store, 'delete_collection'):
            self.vector_store.delete_collection()
        else:
            # Compatibilidade com método antigo
            self.vector_store.clear()
        self.logger.info("Vector store cleared")

    def clear_database(self) -> None:
        """Limpa o banco de dados vetorial (método de compatibilidade)"""
        self.clear_index()

    def get_collection_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da coleção"""
        try:
            if hasattr(self.vector_store, 'get_collection_size'):
                size = self.vector_store.get_collection_size()
            else:
                size = 0  # Fallback se método não existir
            
            stats = {
                'total_documents': size,
                'routing_mode': self.routing_mode,
                'fallback_to_llm': self.fallback_to_llm,
                'hybrid_mode': self.hybrid_mode,
                'min_relevance_score': self.min_relevance_score
            }
            
            # Adiciona informações de modelos se disponível
            if hasattr(self.model_router, 'available_models'):
                stats['available_models'] = list(self.model_router.available_models)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {
                'total_documents': 0,
                'routing_mode': self.routing_mode,
                'error': str(e)
            }
