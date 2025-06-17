from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Dict

class SnippetManager:
    def __init__(self, db_path: str = "~/.cursor_snippets.db") -> None:
        self.db_path = str(Path(db_path).expanduser())
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_schema()

    def _create_schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                code TEXT,
                tags TEXT
            )"""
        )
        self.conn.commit()

    def save_snippet(self, title: str, code: str, tags: List[str] | None = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO snippets(title, code, tags) VALUES (?, ?, ?)",
            (title, code, ",".join(tags or [])),
        )
        self.conn.commit()

    def search(self, query: str) -> List[Dict]:
        cur = self.conn.cursor()
        wildcard = f"%{query}%"
        res = cur.execute(
            "SELECT id, title, code, tags FROM snippets WHERE title LIKE ? OR tags LIKE ?",
            (wildcard, wildcard),
        ).fetchall()
        return [
            {"id": r[0], "title": r[1], "code": r[2], "tags": r[3].split(",") if r[3] else []}
            for r in res
        ]

    def close(self):
        self.conn.close() 