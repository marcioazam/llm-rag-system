from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import json
import pandas as pd
import magic  # type: ignore
import logging
from datetime import datetime
import git  # type: ignore
try:
    from src.code_analysis.code_context_detector import CodeContextDetector
except ImportError:
    CodeContextDetector = None

from .document_loader import DocumentLoader  # Reuso de loaders básicos

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Loader Abstrato Base
# ------------------------------------------------------------------

class BaseLoader(ABC):
    """Define interface de loaders especializados"""

    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        ...


# ------------------------------------------------------------------
# Loaders Específicos
# ------------------------------------------------------------------

class CSVLoader(BaseLoader):
    def load(self, path: str) -> Dict[str, Any]:
        df = pd.read_csv(path)
        content = df.to_csv(index=False)
        return {
            "content": content,
            "metadata": {
                "rows": len(df),
                "columns": list(df.columns),
                "source": path,
                "loader": "CSVLoader",
            },
        }


class JSONLoader(BaseLoader):
    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        content = json.dumps(data, indent=2)
        return {
            "content": content,
            "metadata": {
                "source": path,
                "loader": "JSONLoader",
            },
        }


class CodeLoader(BaseLoader):
    def __init__(self, language: str = "python"):
        self.language = language

    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        return {
            "content": code,
            "metadata": {
                "source": path,
                "language": self.language,
                "loader": "CodeLoader",
            },
        }


class GitRepoLoader(BaseLoader):
    """Extrai conteúdo relevante de um repositório Git"""

    def __init__(self, important_exts: List[str] | None = None):
        if important_exts is None:
            important_exts = [
                ".md",
                ".py",
                ".cs",
                ".js",
                ".tsx",
                ".yaml",
                ".json",
            ]
        self.important_exts = important_exts

    def _get_repo_info(self, repo: git.Repo) -> Dict[str, Any]:
        return {
            "branches": [h.name for h in repo.heads],
            "latest_commit": repo.head.commit.hexsha,
            "commit_date": datetime.fromtimestamp(repo.head.commit.committed_date).isoformat(),
            "author": repo.head.commit.author.name,
        }

    def load(self, repo_path: str) -> Dict[str, Any]:
        repo = git.Repo(repo_path)
        documents = []

        for item in repo.tree().traverse():
            if any(item.path.endswith(ext) for ext in self.important_exts):
                try:
                    content = item.data_stream.read().decode("utf-8", errors="ignore")
                except Exception as exc:
                    logger.debug("Erro ao ler %s: %s", item.path, exc)
                    continue

                last_commit = next(repo.iter_commits(paths=item.path, max_count=1))

                documents.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": item.path,
                            "last_modified": datetime.fromtimestamp(last_commit.committed_date).isoformat(),
                            "author": last_commit.author.name,
                            "commit_message": last_commit.message.strip(),
                            "file_type": Path(item.path).suffix.lstrip("."),
                            "loader": "GitRepoLoader",
                        },
                    }
                )

        return {
            "documents": documents,
            "repo_info": self._get_repo_info(repo),
        }


# ------------------------------------------------------------------
# SmartDocumentLoader – roteia para loaders especializados
# ------------------------------------------------------------------

class SmartDocumentLoader:
    def __init__(self):
        # Usar DocumentLoader como fallback e para tipos básicos
        self.basic_loader = DocumentLoader()

        # Inicializar CodeContextDetector se disponível
        self.context_detector = CodeContextDetector() if CodeContextDetector else None

        self.loaders: list[tuple[str, BaseLoader]] = [
            ("application/pdf", self.basic_loader),
            ("application/vnd.openxmlformats-officedocument", self.basic_loader),  # docx
            ("text/plain", self.basic_loader),
            ("text/x-python", CodeLoader(language="python")),
            ("text/x-csharp", CodeLoader(language="csharp")),
            ("text/csv", CSVLoader()),
            ("application/json", JSONLoader()),
            ("text/markdown", self.basic_loader),
            ("application/x-git", GitRepoLoader()),
        ]

    def detect_and_load(self, file_path: str) -> Dict[str, Any]:
        """Detecta o MIME e delega ao loader apropriado."""
        try:
            file_type = magic.from_file(file_path, mime=True)
        except Exception as exc:
            logger.debug("python-magic falhou (%s); fallback por extensão", exc)
            file_type = None

        # Busca loader por MIME match parcial
        if file_type:
            for mime_pattern, loader in self.loaders:
                if mime_pattern in file_type:
                    result = loader.load(file_path)
                    # Enriquecer metadados via CodeContextDetector se disponível
                    if self.context_detector:
                        ctx = self.context_detector.detect_context(file_path=file_path, code=result["content"])
                        if ctx.get("language"):
                            result["metadata"].update(ctx)
                    return result

        # Fallback por extensão usando basic_loader
        result = self.basic_loader.load(file_path)  # fallback
        # Adicionar metadado do loader usado
        result["metadata"]["loader"] = "DocumentLoader"
        # Tentar enriquecer com contexto de código mesmo se language ausente
        if self.context_detector:
            ctx = self.context_detector.detect_context(file_path=file_path, code=result["content"])
            if ctx.get("language"):
                result["metadata"].update(ctx)
        return result