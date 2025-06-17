from __future__ import annotations

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

__all__ = [
    "NodeType",
    "RelationType",
    "GraphNode",
    "GraphRelation",
    "CodeStructure",
]


class NodeType(str, Enum):
    """Tipos de nó utilizados no grafo."""

    DOCUMENT = "Document"
    CODE_FILE = "CodeFile"
    CLASS = "Class"
    FUNCTION = "Function"
    CONCEPT = "Concept"
    ENTITY = "Entity"
    VALUE_OBJECT = "ValueObject"
    AGGREGATE = "Aggregate"
    BOUNDED_CONTEXT = "BoundedContext"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


class RelationType(str, Enum):
    """Tipos de relacionamento entre nós."""

    CONTAINS = "CONTAINS"
    IMPORTS = "IMPORTS"
    EXTENDS = "EXTENDS"
    IMPLEMENTS = "IMPLEMENTS"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


@dataclass
class GraphNode:
    """Representa genericamente um nó no grafo."""

    id: str
    type: NodeType
    properties: Dict[str, Any]
    embedding_id: Optional[str] = None  # Link opcional com vetor store ou embeddings

    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Converte o nó em dicionário compatível com parâmetros do driver."""

        data = {
            "id": self.id,
            "type": str(self.type),
            **self.properties,
        }
        if self.embedding_id is not None:
            data["embedding_id"] = self.embedding_id
        return data


@dataclass
class GraphRelation:
    """Representa uma relação entre dois nós."""

    source_id: str
    target_id: str
    type: RelationType
    properties: Optional[Dict[str, Any]] = None

    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Converte dados de propriedades da relação para dicionário."""

        return self.properties or {}


@dataclass
class CodeStructure:
    """Estrutura de código extraída para análise de dependências e navegação."""

    file_path: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    imports: List[str]
    exports: List[str]

    def __str__(self) -> str:  # pragma: no cover
        return self.file_path 