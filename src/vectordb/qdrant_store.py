from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)


class QdrantVectorStore:  # pylint: disable=too-few-public-methods
    """Implementação de vector store usando Qdrant.

    A interface replica os métodos principais utilizados no sistema
    (add_documents, search, delete_documents, etc.). Para manter
    compatibilidade com o código existente, os nomes e assinaturas
    seguem o mesmo padrão da classe ``ChromaVectorStore``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "rag_chunks",
        distance: str = "Cosine",
        dim: int = 768,
    ) -> None:
        self.collection_name = collection_name
        self.dim = dim
        self.client = QdrantClient(host=host, port=port)

        # Distância suportada pelo Qdrant
        distance_enum: rest.Distance = getattr(
            rest.Distance,
            distance.upper(),
            rest.Distance.COSINE,
        )

        # Tentar verificar/ criar coleção; se falhar, usar modo em memória
        try:
            if collection_name not in {
                coll.name for coll in self.client.get_collections().collections
            }:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(size=dim, distance=distance_enum),
                )
                logger.info("Coleção Qdrant '%s' criada", collection_name)
            else:
                logger.info("Coleção Qdrant '%s' carregada", collection_name)
            self._in_memory = False
        except Exception as exc:  # pragma: no cover
            logger.warning("Qdrant indisponível (%s). Usando armazenamento em memória.", exc)
            self.client = None  # type: ignore
            self._in_memory = True
            # Estrutura simples em memória: lista de dicts
            self._mem_store: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # API compatível com ChromaVectorStore
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> bool:
        """Adicionar documentos/chunks à coleção.

        Args:
            documents: Texto dos documentos/chunks.
            embeddings: Vetores pré-computados (opcional).
            metadata: Metadados associados.
            ids: Identificadores únicos (string ou int).
        """
        try:
            if not documents:
                logger.warning("Nenhum documento fornecido para add_documents")
                return False

            # Geração de IDs se ausentes
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]

            # Placeholders de metadados se faltarem
            if metadata is None:
                metadata = [{"source": "unknown"} for _ in documents]

            # Se embeddings não forem fornecidos, isso significa que o
            # serviço de embeddings será chamado depois; para manter a
            # compatibilidade, rejeitamos a operação para evitar vetores
            # vazios.
            if embeddings is None:
                logger.error(
                    "Embeddings obrigatórios para QdrantVectorStore.add_documents"
                )
                return False

            # -------------------------------
            # Fallback em memória (Qdrant off)
            # -------------------------------
            if getattr(self, "_in_memory", False):
                for idx, doc, vec, meta in zip(ids, documents, embeddings, metadata):
                    self._mem_store.append(
                        {
                            "content": doc,
                            "embedding": vec,
                            "metadata": meta,
                            "id": idx,
                        }
                    )
                logger.info("%d documentos armazenados em memória", len(documents))
                return True

            vectors = [vec for vec in embeddings]

            points: List[rest.PointStruct] = []
            for idx, vector, text, meta in zip(ids, vectors, documents, metadata):
                payload = dict(meta)
                # guardar conteúdo do chunk para retorno futuro
                payload["document"] = text
                points.append(
                    rest.PointStruct(id=idx, vector=vector, payload=payload)
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info("%d documentos inseridos em '%s'", len(points), self.collection_name)
            return True
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Erro ao adicionar documentos no Qdrant: %s", err)
            return False

    def search(
        self,
        query: str | np.ndarray | None = None,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Buscar documentos semelhantes.

        Accepta *query* texto ou embedding pré-computado (np.ndarray).
        """
        try:
            # Modo em memória – retornar lista vazia (ranking local não implementado)
            if getattr(self, "_in_memory", False):
                return []

            # Se for uma string, não temos embeddings aqui; a pipeline deve
            # garantir que a chamada passe o embedding como np.ndarray.
            if isinstance(query, str):
                if query_embedding is None:
                    raise ValueError(
                        "Para busca por string, forneça 'query_embedding' já calculado."
                    )
                vector = query_embedding
            else:
                vector = query.tolist() if isinstance(query, np.ndarray) else query

            # Filtro payload mapeado para Qdrant
            query_filter = None
            if filter:
                conditions = [
                    rest.FieldCondition(key=k, match=rest.MatchValue(value=v))
                    for k, v in filter.items()
                ]
                query_filter = rest.Filter(must=conditions)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=k,
                query_filter=query_filter,
            )

            formatted: List[Dict[str, Any]] = []
            for hit in results:
                payload = hit.payload or {}
                formatted.append(
                    {
                        "content": payload.get("document", ""),
                        "metadata": {k: v for k, v in payload.items() if k != "document"},
                        "distance": hit.score,
                        "id": hit.id,
                    }
                )

            return formatted
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Erro na busca Qdrant: %s", err)
            return []

    # Métodos auxiliares -------------------------------------------------

    def get_document_count(self) -> int:
        """Número total de pontos na coleção."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Erro ao obter contagem Qdrant: %s", err)
            return 0

    def delete_documents(self, ids: List[str]) -> bool:
        """Remover pontos pelo ID."""
        if not ids:
            return True
        # Modo em memória
        if getattr(self, "_in_memory", False):
            original_len = len(self._mem_store)
            self._mem_store = [p for p in self._mem_store if p["id"] not in ids]  # type: ignore[attr-defined]
            return True

        try:
            self.client.delete(collection_name=self.collection_name, points_selector=rest.PointIdsList(points=ids))
            logger.info("%d documentos removidos de '%s'", len(ids), self.collection_name)
            return True
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Erro ao deletar documentos Qdrant: %s", err)
            return False

    def clear_collection(self) -> bool:
        """Remove e recria a coleção."""
        try:
            if getattr(self, "_in_memory", False):
                self._mem_store = []  # type: ignore[attr-defined]
                return True

            self.client.delete_collection(collection_name=self.collection_name)
            logger.info("Coleção '%s' deletada", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=self.dim, distance=rest.Distance.COSINE),
            )
            logger.info("Coleção '%s' recriada", self.collection_name)
            return True
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Erro ao limpar coleção Qdrant: %s", err)
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "count": info.points_count,
                "host": self.client._config.host,  # type: ignore
                "port": self.client._config.port,  # type: ignore
            }
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Erro ao obter info da coleção: %s", err)
            return {}

    def update_document(
        self, doc_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Atualizar vetor/metadados de ponto."""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest.PointStruct(id=doc_id, vector=embedding, payload=metadata or {})
                ],
            )
            logger.info("Documento %s atualizado em Qdrant", doc_id)
            return True
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Erro ao atualizar documento Qdrant: %s", err)
            return False

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            rec = self.client.retrieve(self.collection_name, ids=[doc_id])
            if not rec:
                return None
            hit = rec[0]
            return {"id": hit.id, "metadata": hit.payload or {}, "vector": hit.vector}
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Erro ao recuperar documento: %s", err)
            return None

    def close(self) -> None:
        """QdrantClient não exige fechamento explícito, mas definimos stub."""
        logger.info("Conexão Qdrant encerrada (noop)") 