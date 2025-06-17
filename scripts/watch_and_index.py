#!/usr/bin/env python
"""watch_and_index.py – Observa diretórios e re-indexa arquivos modificados.

Uso:
    python scripts/watch_and_index.py /caminho/projeto

Ctrl+C para interromper.
"""
from __future__ import annotations

import argparse
import sys
import signal
from pathlib import Path
from typing import List
import asyncio

# Importes do projeto
sys.path.append(str(Path(__file__).resolve().parent.parent))  # garantir path raiz

from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.devtools.file_watcher import FileWatcher  # noqa: E402
from src.code_analysis.dependency_analyzer import DependencyAnalyzer  # noqa: E402
from src.graphdb.graph_models import GraphRelation  # noqa: E402
from src.devtools.index_queue import IndexQueue  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch path(s) and index on change")
    parser.add_argument("paths", nargs="+", help="Diretórios ou arquivos a observar")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    project_root = str(Path(args.paths[0]).resolve())
    dep_analyzer = DependencyAnalyzer(project_root=project_root)

    def on_change(raw_path: str):
        # Detect delete marker
        is_deleted = False
        if raw_path.endswith("::deleted"):
            raw_path = raw_path[:-9]
            is_deleted = True

        file_path = raw_path
        try:
            if is_deleted:
                # descobrir id real no SQLite
                chunk_id = None
                if hasattr(pipeline, 'metadata_store') and pipeline.metadata_store:
                    chunk_id = pipeline.metadata_store.get_id_by_filepath(file_path)
                if chunk_id is None:
                    chunk_id = Path(file_path).stem
                if pipeline.vector_store:
                    try:
                        pipeline.vector_store.delete_documents([chunk_id])
                    except Exception:
                        pass
                if hasattr(pipeline, 'metadata_store') and pipeline.metadata_store:
                    try:
                        pipeline.metadata_store.delete_by_id(chunk_id)
                    except Exception:
                        pass
                return

            if not Path(file_path).is_file():
                return
            print(f"[Watcher] Alteração detectada: {file_path}")
            doc = pipeline.document_loader.load(file_path)
            pipeline.add_documents([doc], project_id=project_id)
            print(f"[Watcher] Indexado {file_path}")

            # Call-graph no Neo4j, se ativo
            if pipeline.graph_store is not None:
                relations = dep_analyzer.analyze(doc["content"])
                file_id = f"file::{file_path}"
                for rel in relations:
                    src_id = f"func::{rel['source']}@{file_id}"
                    tgt_id = f"func::{rel['target']}@{file_id}"
                    pipeline.graph_store.add_relationship(
                        GraphRelation(source_id=src_id, target_id=tgt_id, type="CALLS")
                    )
        except Exception as exc:  # pragma: no cover
            print(f"[Watcher] Erro ao indexar {file_path}: {exc}")

    index_queue = IndexQueue(on_change)
    index_queue.start()

    def enqueue(path: str):
        asyncio.get_event_loop().call_soon_threadsafe(index_queue.put, path)

    watcher = FileWatcher(args.paths, enqueue)
    watcher.start()

    async def _shutdown():
        print("Encerrando watcher e fila...")
        watcher.stop()
        await index_queue.stop()

    def _graceful(sig, frame):
        asyncio.get_event_loop().create_task(_shutdown())

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    try:
        asyncio.get_event_loop().run_forever()
    finally:
        asyncio.get_event_loop().run_until_complete(_shutdown())


if __name__ == "__main__":
    main() 