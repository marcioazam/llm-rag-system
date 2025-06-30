from __future__ import annotations

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, List
from datetime import datetime


class ProjectValidationError(Exception):
    """Erro de validação de projeto"""
    pass


class SQLiteMetadataStore:
    """Armazenamento leve de metadados em SQLite para chunks de documentos com gerenciamento de projetos."""

    def __init__(self, db_path: str = "./data/metadata/chunks.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")  # Habilitar foreign keys
        self._create_schema()

    # ------------------------------------------------------------
    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        
        # Tabela de projetos
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'archived')),
                metadata JSON,
                UNIQUE(name)
            )
            """
        )
        
        # Tabela de chunks com foreign key para projetos
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
            )
            """
        )
        
        # Índices para performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_project_id ON chunks(project_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)")
        
        self.conn.commit()

        # Migração: garantir coluna 'coverage' em bases antigas
        cur.execute("PRAGMA table_info(chunks)")
        cols = {row[1] for row in cur.fetchall()}
        if "coverage" not in cols:
            cur.execute("ALTER TABLE chunks ADD COLUMN coverage TEXT")
            self.conn.commit()

    # ------------------------------------------------------------
    # CRUD DE PROJETOS
    # ------------------------------------------------------------
    
    def create_project(self, project_id: str, name: str, description: str = None, 
                      metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Criar um novo projeto"""
        cur = self.conn.cursor()
        
        try:
            cur.execute(
                """
                INSERT INTO projects (id, name, description, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, name, description, json.dumps(metadata) if metadata else None)
            )
            self.conn.commit()
            
            # Retornar projeto criado
            return self.get_project(project_id)
            
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                if "projects.id" in str(e):
                    raise ProjectValidationError(f"Projeto com ID '{project_id}' já existe")
                elif "projects.name" in str(e):
                    raise ProjectValidationError(f"Projeto com nome '{name}' já existe")
            raise ProjectValidationError(f"Erro ao criar projeto: {e}")
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Obter projeto por ID"""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cur.fetchone()
        
        if row:
            cols = [description[0] for description in cur.description]
            project = dict(zip(cols, row))
            if project['metadata']:
                project['metadata'] = json.loads(project['metadata'])
            return project
        return None
    
    def get_project_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Obter projeto por nome"""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM projects WHERE name = ?", (name,))
        row = cur.fetchone()
        
        if row:
            cols = [description[0] for description in cur.description]
            project = dict(zip(cols, row))
            if project['metadata']:
                project['metadata'] = json.loads(project['metadata'])
            return project
        return None
    
    def list_projects(self, status: str = None, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Listar projetos com filtros opcionais"""
        cur = self.conn.cursor()
        
        query = "SELECT * FROM projects"
        params = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        projects = []
        if rows:
            cols = [description[0] for description in cur.description]
            for row in rows:
                project = dict(zip(cols, row))
                if project['metadata']:
                    project['metadata'] = json.loads(project['metadata'])
                projects.append(project)
        
        return projects
    
    def update_project(self, project_id: str, name: str = None, description: str = None,
                      status: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Atualizar projeto existente"""
        cur = self.conn.cursor()
        
        # Verificar se projeto existe
        if not self.project_exists(project_id):
            raise ProjectValidationError(f"Projeto '{project_id}' não encontrado")
        
        # Construir query de update dinamicamente
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        
        if status is not None:
            if status not in ['active', 'inactive', 'archived']:
                raise ProjectValidationError(f"Status inválido: {status}")
            updates.append("status = ?")
            params.append(status)
        
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(project_id)
            
            query = f"UPDATE projects SET {', '.join(updates)} WHERE id = ?"
            
            try:
                cur.execute(query, params)
                self.conn.commit()
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed: projects.name" in str(e):
                    raise ProjectValidationError(f"Projeto com nome '{name}' já existe")
                raise ProjectValidationError(f"Erro ao atualizar projeto: {e}")
        
        return self.get_project(project_id)
    
    def delete_project(self, project_id: str, force: bool = False) -> bool:
        """Deletar projeto e opcionalmente seus chunks"""
        cur = self.conn.cursor()
        
        # Verificar se projeto existe
        if not self.project_exists(project_id):
            raise ProjectValidationError(f"Projeto '{project_id}' não encontrado")
        
        # Verificar se há chunks associados
        cur.execute("SELECT COUNT(*) FROM chunks WHERE project_id = ?", (project_id,))
        chunk_count = cur.fetchone()[0]
        
        if chunk_count > 0 and not force:
            raise ProjectValidationError(
                f"Projeto '{project_id}' possui {chunk_count} chunks. "
                f"Use force=True para deletar ou remova os chunks primeiro."
            )
        
        # Deletar projeto (chunks serão deletados automaticamente por CASCADE)
        cur.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self.conn.commit()
        
        return cur.rowcount > 0
    
    def project_exists(self, project_id: str) -> bool:
        """Verificar se projeto existe"""
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM projects WHERE id = ? LIMIT 1", (project_id,))
        return cur.fetchone() is not None
    
    def get_project_stats(self, project_id: str) -> Dict[str, Any]:
        """Obter estatísticas do projeto"""
        if not self.project_exists(project_id):
            raise ProjectValidationError(f"Projeto '{project_id}' não encontrado")
        
        cur = self.conn.cursor()
        
        # Estatísticas básicas
        cur.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT language) as languages_count,
                COUNT(DISTINCT file_path) as files_count,
                MIN(created_at) as first_chunk_date,
                MAX(created_at) as last_chunk_date
            FROM chunks 
            WHERE project_id = ?
        """, (project_id,))
        
        stats = dict(zip([col[0] for col in cur.description], cur.fetchone()))
        
        # Estatísticas por linguagem
        cur.execute("""
            SELECT language, COUNT(*) as count
            FROM chunks 
            WHERE project_id = ? AND language IS NOT NULL
            GROUP BY language
            ORDER BY count DESC
        """, (project_id,))
        
        stats['languages'] = {row[0]: row[1] for row in cur.fetchall()}
        
        return stats

    # ------------------------------------------------------------
    # MÉTODOS DE CHUNKS COM VALIDAÇÃO DE PROJETO
    # ------------------------------------------------------------
    
    def upsert_metadata(self, item: Dict[str, Any]) -> None:
        """Inserir/atualizar metadata com validação de projeto"""
        project_id = item.get("project_id")
        
        # Validar se project_id é obrigatório e existe
        if not project_id:
            raise ProjectValidationError("project_id é obrigatório para indexação")
        
        if not self.project_exists(project_id):
            raise ProjectValidationError(f"Projeto '{project_id}' não existe. Crie o projeto primeiro.")
        
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
                "project_id": project_id,
            },
        )
        self.conn.commit()

    # ------------------------------------------------------------
    # MÉTODOS EXISTENTES (mantidos para compatibilidade)
    # ------------------------------------------------------------
    
    def query_by_language(self, language: str) -> Iterable[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE language=?", (language,))
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            yield {col: row[idx] for idx, col in enumerate(cols)}

    def close(self) -> None:
        self.conn.close()

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

    def distinct_coverage(self):
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT DISTINCT coverage FROM chunks WHERE coverage IS NOT NULL AND coverage != ''")
            return sorted([row[0] for row in cur.fetchall() if row[0] is not None])
        except Exception:
            return [] 