#!/usr/bin/env python3
"""
Exemplo Prático - Sistema de Gerenciamento de Projetos RAG
Demonstra uso real do sistema com cenários práticos
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.metadata.sqlite_store import SQLiteMetadataStore, ProjectValidationError
import json

def main():
    """Exemplo prático de uso do sistema de projetos"""
    
    print("🚀 Sistema RAG com Gerenciamento de Projetos")
    print("=" * 50)
    
    # Conectar ao banco
    store = SQLiteMetadataStore(db_path="./data/metadata/chunks.db")
    
    try:
        # Scenario 1: Configurar projetos para uma empresa
        print("\n📋 CENÁRIO 1: Configurar Projetos de uma Empresa")
        print("-" * 50)
        
        projetos_empresa = [
            {
                "id": "web-frontend",
                "name": "Portal Web - Frontend",
                "description": "Interface do usuário em React/TypeScript",
                "metadata": {
                    "team": "frontend",
                    "technology": "react",
                    "priority": "high",
                    "environment": "production"
                }
            },
            {
                "id": "api-backend",
                "name": "API REST - Backend",
                "description": "API principal em Python/FastAPI",
                "metadata": {
                    "team": "backend",
                    "technology": "python",
                    "priority": "high",
                    "environment": "production"
                }
            },
            {
                "id": "mobile-app",
                "name": "App Mobile",
                "description": "Aplicativo móvel React Native",
                "metadata": {
                    "team": "mobile",
                    "technology": "react-native",
                    "priority": "medium",
                    "environment": "development"
                }
            }
        ]
        
        for projeto_config in projetos_empresa:
            try:
                project = store.create_project(
                    project_id=projeto_config["id"],
                    name=projeto_config["name"],
                    description=projeto_config["description"],
                    metadata=projeto_config["metadata"]
                )
                print(f"✅ Projeto criado: {project['name']}")
            except ProjectValidationError:
                print(f"ℹ️  Projeto {projeto_config['id']} já existe")
        
        # Scenario 2: Adicionar conhecimento aos projetos
        print("\n📋 CENÁRIO 2: Indexar Documentos por Projeto")
        print("-" * 50)
        
        # Documentos do Frontend
        frontend_docs = [
            {
                "id": "front_001",
                "file_path": "src/components/Header.tsx",
                "language": "typescript",
                "symbols": ["Header", "useState", "useAuth"],
                "relations": ["React", "AuthContext"],
                "coverage": "component",
                "source": "frontend",
                "chunk_hash": "f1a2b3c4",
                "project_id": "web-frontend"
            },
            {
                "id": "front_002", 
                "file_path": "src/pages/Dashboard.tsx",
                "language": "typescript",
                "symbols": ["Dashboard", "useEffect", "fetchData"],
                "relations": ["React", "API"],
                "coverage": "page",
                "source": "frontend",
                "chunk_hash": "d5e6f7g8",
                "project_id": "web-frontend"
            }
        ]
        
        # Documentos do Backend
        backend_docs = [
            {
                "id": "back_001",
                "file_path": "app/api/users.py",
                "language": "python",
                "symbols": ["get_users", "create_user", "UserModel"],
                "relations": ["FastAPI", "SQLAlchemy", "Pydantic"],
                "coverage": "api",
                "source": "backend",
                "chunk_hash": "h9i0j1k2",
                "project_id": "api-backend"
            },
            {
                "id": "back_002",
                "file_path": "app/auth/jwt_handler.py", 
                "language": "python",
                "symbols": ["create_token", "verify_token", "JWTHandler"],
                "relations": ["PyJWT", "FastAPI"],
                "coverage": "authentication",
                "source": "backend",
                "chunk_hash": "l3m4n5o6",
                "project_id": "api-backend"
            }
        ]
        
        # Adicionar documentos
        all_docs = frontend_docs + backend_docs
        for doc in all_docs:
            store.upsert_metadata(doc)
            print(f"✅ Documento indexado: {doc['file_path']} -> {doc['project_id']}")
        
        # Scenario 3: Consultar por projeto
        print("\n📋 CENÁRIO 3: Consultar Conhecimento por Projeto")
        print("-" * 50)
        
        # Buscar chunks do frontend
        frontend_chunks = list(store.query_by_project("web-frontend"))
        print(f"📊 Frontend tem {len(frontend_chunks)} chunks:")
        for chunk in frontend_chunks:
            print(f"   - {chunk['file_path']} ({chunk['language']})")
        
        # Buscar chunks do backend
        backend_chunks = list(store.query_by_project("api-backend"))
        print(f"📊 Backend tem {len(backend_chunks)} chunks:")
        for chunk in backend_chunks:
            print(f"   - {chunk['file_path']} ({chunk['language']})")
        
        # Scenario 4: Estatísticas e monitoramento
        print("\n📋 CENÁRIO 4: Monitoramento e Estatísticas")
        print("-" * 50)
        
        projects = store.list_projects(status="active")
        print(f"📈 Total de projetos ativos: {len(projects)}")
        
        total_chunks = 0
        for project in projects:
            try:
                stats = store.get_project_stats(project["id"])
                total_chunks += stats["total_chunks"]
                print(f"📊 {project['name']}:")
                print(f"   - Chunks: {stats['total_chunks']}")
                print(f"   - Linguagens: {stats['languages']}")
                print(f"   - Arquivos: {stats['files_count']}")
            except:
                print(f"   - Sem dados ainda")
        
        print(f"📈 Total de chunks no sistema: {total_chunks}")
        
        # Scenario 5: Simular query com filtro por projeto
        print("\n📋 CENÁRIO 5: Simulação de Query por Projeto")
        print("-" * 50)
        
        print("💭 Query: 'Como implementar autenticação JWT?'")
        print("🎯 Filtro: Apenas projeto 'api-backend'")
        
        # Simular busca filtrada
        auth_chunks = [chunk for chunk in backend_chunks 
                      if 'auth' in chunk['file_path'].lower() or 'jwt' in chunk['file_path'].lower()]
        
        if auth_chunks:
            print("🔍 Chunks relevantes encontrados:")
            for chunk in auth_chunks:
                symbols = json.loads(chunk['symbols']) if chunk['symbols'] else []
                print(f"   - {chunk['file_path']}")
                print(f"     Símbolos: {', '.join(symbols)}")
        else:
            print("🔍 Nenhum chunk específico de auth encontrado")
        
        # Scenario 6: Exemplo de uso prático com API
        print("\n📋 CENÁRIO 6: Como Usar com a API")
        print("-" * 50)
        
        api_examples = [
            "# 1. Criar projeto via API",
            "POST /projects",
            '{"id": "novo-projeto", "name": "Meu Projeto"}',
            "",
            "# 2. Adicionar documentos",
            "POST /add_documents", 
            '{"project_id": "novo-projeto", "documents": [...]}',
            "",
            "# 3. Query filtrada",
            "POST /query",
            '{"question": "Como fazer X?", "project_id": "novo-projeto"}',
            "",
            "# 4. Estatísticas",
            "GET /projects/novo-projeto/stats"
        ]
        
        for line in api_examples:
            print(f"   {line}")
        
        # Summary
        print("\n📋 RESUMO FINAL")
        print("-" * 50)
        print("✅ Sistema de projetos configurado")
        print("✅ Documentos indexados com validação")
        print("✅ Consultas isoladas por projeto")
        print("✅ Monitoramento e estatísticas funcionando")
        print("✅ Pronto para uso em produção!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        store.close()

if __name__ == "__main__":
    main() 