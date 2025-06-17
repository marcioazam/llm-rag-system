from __future__ import annotations

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable


class SQLiteMetadataStore:
    """Armazenamento leve de metadados em SQLite para chunks de documentos."""

    def __init__(self, db_path: str = "./data/metadata/chunks.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_schema()

    # ------------------------------------------------------------
    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                language TEXT,
                symbols TEXT,
                relations TEXT,
                coverage TEXT,
                chunk_hash TEXT,
                project_id TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

        # Migração: garantir coluna 'coverage' em bases antigas
        cur.execute("PRAGMA table_info(chunks)")
        cols = {row[1] for row in cur.fetchall()}
        if "coverage" not in cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN coverage TEXT")
            self.conn.commit()

    # ------------------------------------------------------------
    def upsert_metadata(self, item: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks(id, file_path, language, symbols, relations, coverage, source, chunk_hash, project_id)
            VALUES (:id, :file_path, :language, :symbols, :relations, :coverage, :source, :chunk_hash, :project_id)
            ON CONFLICT(id) DO UPDATE SET
                file_path=excluded.file_path,
                language=excluded.language,
                symbols=excluded.symbols,
                relations=excluded.relations,
                coverage=excluded.coverage,
                source=excluded.source,
                chunk_hash=excluded.chunk_hash,
                project_id=excluded.project_id
            """,
            {
                "id": item.get("id"),
                "file_path": item.get("file_path"),
                "language": item.get("language"),
                "symbols": json.dumps(item.get("symbols")),
                "relations": json.dumps(item.get("relations")),
                "coverage": item.get("coverage"),
                "source": item.get("source"),
                "chunk_hash": item.get("chunk_hash"),
                "project_id": item.get("project_id"),
            },
        )
        self.conn.commit()

    # ------------------------------------------------------------
    def query_by_language(self, language: str) -> Iterable[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE language=?", (language,))
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            yield {col: row[idx] for idx, col in enumerate(cols)}

    def close(self) -> None:
        self.conn.close()

    # -----------------------------------------------
    def get_id_by_filepath(self, file_path: str) -> str | None:
        cur = self.conn.cursor()
        res = cur.execute("SELECT id FROM chunks WHERE file_path=?", (file_path,)).fetchone()
        return res[0] if res else None

    def delete_by_id(self, chunk_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunks WHERE id=?", (chunk_id,))
        self.conn.commit()

    def query_by_project(self, project_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE project_id=?", (project_id,))
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            yield {col: row[idx] for idx, col in enumerate(cols)}

    # ------------------------------------------------------------
    def distinct_coverage(self):
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT DISTINCT coverage FROM chunks WHERE coverage IS NOT NULL AND coverage != ''")
            return sorted([row[0] for row in cur.fetchall() if row[0] is not None])
        except Exception:
            return [] 