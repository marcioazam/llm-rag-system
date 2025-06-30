#!/usr/bin/env python3
"""
Script de Teste - Sistema de CRUD de Projetos RAG
Demonstra o uso completo do novo sistema de gerenciamento de projetos
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.metadata.sqlite_store import SQLiteMetadataStore, ProjectValidationError
import json
import time
from datetime import datetime

def print_header(title):
    """Imprimir cabeçalho formatado"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """Imprimir passo do teste"""
    print(f"\n📋 PASSO {step}: {description}")
    print("-" * 50)

def print_success(message):
    """Imprimir mensagem de sucesso"""
    print(f"✅ {message}")

def print_error(message):
    """Imprimir mensagem de erro"""
    print(f"❌ {message}")

def print_info(message):
    """Imprimir informação"""
    print(f"ℹ️  {message}")

def test_project_crud():
    """Teste completo do CRUD de projetos"""
    
    print_header("TESTE COMPLETO - SISTEMA DE PROJETOS RAG")
    
    # Usar banco de teste
    test_db_path = "./data/test/test_projects.db"
    os.makedirs(os.path.dirname(test_db_path), exist_ok=True)
    
    # Limpar banco anterior se existir
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    store = SQLiteMetadataStore(db_path=test_db_path)
    
    try:
        # ================================================================
        # TESTE 1: CRIAR PROJETOS
        # ================================================================
        print_step(1, "CRIAÇÃO DE PROJETOS")
        
        # Projeto 1: E-commerce
        project1 = store.create_project(
            project_id="ecommerce-frontend",
            name="E-commerce Frontend",
            description="Sistema de e-commerce desenvolvido em React",
            metadata={
                "team": "frontend",
                "technology": "react",
                "priority": "high",
                "environment": "production"
            }
        )
        print_success(f"Projeto criado: {project1['name']}")
        print_info(f"ID: {project1['id']}, Status: {project1['status']}")
        
        # Projeto 2: Backend API
        project2 = store.create_project(
            project_id="ecommerce-backend",
            name="E-commerce Backend API",
            description="API REST desenvolvida em Python/FastAPI",
            metadata={
                "team": "backend",
                "technology": "python",
                "priority": "high",
                "environment": "production"
            }
        )
        print_success(f"Projeto criado: {project2['name']}")
        
        # Projeto 3: DevOps
        project3 = store.create_project(
            project_id="devops-tools",
            name="DevOps Tools & Scripts",
            description="Scripts e ferramentas de automação",
            metadata={
                "team": "devops",
                "technology": "bash",
                "priority": "medium"
            }
        )
        print_success(f"Projeto criado: {project3['name']}")
        
        # ================================================================
        # TESTE 2: TENTAR CRIAR PROJETO DUPLICADO
        # ================================================================
        print_step(2, "VALIDAÇÃO - PROJETO DUPLICADO")
        
        try:
            store.create_project(
                project_id="ecommerce-frontend",  # ID já existe
                name="Projeto Duplicado"
            )
            print_error("FALHA: Deveria ter impedido projeto duplicado!")
        except ProjectValidationError as e:
            print_success(f"Validação funcionou: {e}")
        
        # ================================================================
        # TESTE 3: LISTAR PROJETOS
        # ================================================================
        print_step(3, "LISTAR PROJETOS")
        
        all_projects = store.list_projects()
        print_success(f"Total de projetos: {len(all_projects)}")
        
        for project in all_projects:
            print_info(f"- {project['id']}: {project['name']} ({project['status']})")
        
        # Listar apenas ativos
        active_projects = store.list_projects(status="active")
        print_success(f"Projetos ativos: {len(active_projects)}")
        
        # ================================================================
        # TESTE 4: OBTER PROJETO ESPECÍFICO
        # ================================================================
        print_step(4, "OBTER PROJETO ESPECÍFICO")
        
        project = store.get_project("ecommerce-frontend")
        print_success(f"Projeto encontrado: {project['name']}")
        print_info(f"Descrição: {project['description']}")
        print_info(f"Metadados: {json.dumps(project['metadata'], indent=2)}")
        
        # Tentar obter projeto inexistente
        non_existent = store.get_project("projeto-inexistente")
        if non_existent is None:
            print_success("Retornou None para projeto inexistente")
        
        # ================================================================
        # TESTE 5: ATUALIZAR PROJETO
        # ================================================================
        print_step(5, "ATUALIZAR PROJETO")
        
        updated_project = store.update_project(
            project_id="devops-tools",
            name="DevOps Tools & Automation",
            description="Scripts, ferramentas e automação completa de CI/CD",
            status="active",
            metadata={
                "team": "devops",
                "technology": "bash",
                "priority": "high",  # Mudou de medium para high
                "environment": "all",
                "updated_reason": "Prioridade aumentada"
            }
        )
        print_success(f"Projeto atualizado: {updated_project['name']}")
        print_info(f"Nova prioridade: {updated_project['metadata']['priority']}")
        
        # ================================================================
        # TESTE 6: ADICIONAR CHUNKS COM VALIDAÇÃO
        # ================================================================
        print_step(6, "ADICIONAR CHUNKS COM VALIDAÇÃO DE PROJETO")
        
        # Chunks válidos (projeto existe)
        valid_chunks = [
            {
                "id": "chunk_001",
                "file_path": "src/components/ProductCard.jsx",
                "language": "javascript",
                "symbols": ["ProductCard", "useState", "useEffect"],
                "relations": ["React", "PropTypes"],
                "coverage": "component",
                "source": "frontend",
                "chunk_hash": "abc123",
                "project_id": "ecommerce-frontend"
            },
            {
                "id": "chunk_002",
                "file_path": "src/api/products.py",
                "language": "python",
                "symbols": ["get_products", "create_product", "ProductModel"],
                "relations": ["FastAPI", "SQLAlchemy"],
                "coverage": "api",
                "source": "backend",
                "chunk_hash": "def456",
                "project_id": "ecommerce-backend"
            },
            {
                "id": "chunk_003",
                "file_path": "scripts/deploy.sh",
                "language": "bash",
                "symbols": ["deploy_app", "check_health"],
                "relations": ["docker", "kubernetes"],
                "coverage": "deployment",
                "source": "devops",
                "chunk_hash": "ghi789",
                "project_id": "devops-tools"
            }
        ]
        
        for chunk in valid_chunks:
            store.upsert_metadata(chunk)
            print_success(f"Chunk adicionado: {chunk['id']} -> {chunk['project_id']}")
        
        # Tentar adicionar chunk com projeto inexistente
        try:
            invalid_chunk = {
                "id": "chunk_invalid",
                "file_path": "test.py",
                "project_id": "projeto-inexistente"
            }
            store.upsert_metadata(invalid_chunk)
            print_error("FALHA: Deveria ter impedido chunk com projeto inexistente!")
        except ProjectValidationError as e:
            print_success(f"Validação funcionou: {e}")
        
        # Tentar adicionar chunk sem project_id
        try:
            chunk_no_project = {
                "id": "chunk_no_project",
                "file_path": "test.py"
                # project_id ausente
            }
            store.upsert_metadata(chunk_no_project)
            print_error("FALHA: Deveria ter exigido project_id!")
        except ProjectValidationError as e:
            print_success(f"Validação funcionou: {e}")
        
        # ================================================================
        # TESTE 7: CONSULTAR CHUNKS POR PROJETO
        # ================================================================
        print_step(7, "CONSULTAR CHUNKS POR PROJETO")
        
        frontend_chunks = list(store.query_by_project("ecommerce-frontend"))
        print_success(f"Chunks do frontend: {len(frontend_chunks)}")
        for chunk in frontend_chunks:
            print_info(f"- {chunk['id']}: {chunk['file_path']}")
        
        backend_chunks = list(store.query_by_project("ecommerce-backend"))
        print_success(f"Chunks do backend: {len(backend_chunks)}")
        
        devops_chunks = list(store.query_by_project("devops-tools"))
        print_success(f"Chunks do devops: {len(devops_chunks)}")
        
        # ================================================================
        # TESTE 8: ESTATÍSTICAS DOS PROJETOS
        # ================================================================
        print_step(8, "ESTATÍSTICAS DOS PROJETOS")
        
        for project_id in ["ecommerce-frontend", "ecommerce-backend", "devops-tools"]:
            stats = store.get_project_stats(project_id)
            print_success(f"Estatísticas de {project_id}:")
            print_info(f"  - Total chunks: {stats['total_chunks']}")
            print_info(f"  - Linguagens: {stats['languages_count']}")
            print_info(f"  - Arquivos: {stats['files_count']}")
            print_info(f"  - Linguagens por chunk: {stats['languages']}")
        
        # ================================================================
        # TESTE 9: VALIDAR EXISTÊNCIA DE PROJETOS
        # ================================================================
        print_step(9, "VALIDAR EXISTÊNCIA DE PROJETOS")
        
        existing_projects = ["ecommerce-frontend", "ecommerce-backend", "devops-tools"]
        for project_id in existing_projects:
            exists = store.project_exists(project_id)
            print_success(f"Projeto {project_id} existe: {exists}")
        
        non_existing = ["projeto-fake", "outro-inexistente"]
        for project_id in non_existing:
            exists = store.project_exists(project_id)
            if not exists:
                print_success(f"Projeto {project_id} não existe: correto")
            else:
                print_error(f"FALHA: Projeto {project_id} deveria não existir")
        
        # ================================================================
        # TESTE 10: DELETAR PROJETO COM CHUNKS (deve falhar)
        # ================================================================
        print_step(10, "TENTAR DELETAR PROJETO COM CHUNKS")
        
        try:
            store.delete_project("ecommerce-frontend")  # sem force
            print_error("FALHA: Deveria ter impedido deleção com chunks!")
        except ProjectValidationError as e:
            print_success(f"Validação funcionou: {e}")
        
        # ================================================================
        # TESTE 11: DELETAR PROJETO COM FORCE
        # ================================================================
        print_step(11, "DELETAR PROJETO COM FORCE=TRUE")
        
        # Primeiro verificar quantos chunks tem
        stats_before = store.get_project_stats("devops-tools")
        print_info(f"Projeto devops-tools tem {stats_before['total_chunks']} chunks")
        
        # Deletar com force
        success = store.delete_project("devops-tools", force=True)
        print_success(f"Projeto deletado com sucesso: {success}")
        
        # Verificar se chunks foram deletados também
        try:
            stats_after = store.get_project_stats("devops-tools")
            print_error("FALHA: Projeto ainda existe após deleção!")
        except ProjectValidationError:
            print_success("Projeto foi completamente removido")
        
        # Verificar se chunks órfãos foram removidos
        devops_chunks_after = list(store.query_by_project("devops-tools"))
        if len(devops_chunks_after) == 0:
            print_success("Chunks foram removidos automaticamente (CASCADE)")
        else:
            print_error(f"FALHA: {len(devops_chunks_after)} chunks órfãos encontrados")
        
        # ================================================================
        # TESTE 12: RESUMO FINAL
        # ================================================================
        print_step(12, "RESUMO FINAL DOS TESTES")
        
        final_projects = store.list_projects()
        print_success(f"Projetos restantes: {len(final_projects)}")
        
        total_chunks = 0
        for project in final_projects:
            try:
                stats = store.get_project_stats(project["id"])
                total_chunks += stats["total_chunks"]
                print_info(f"- {project['id']}: {stats['total_chunks']} chunks")
            except:
                print_info(f"- {project['id']}: 0 chunks (erro ao obter stats)")
        
        print_success(f"Total de chunks no sistema: {total_chunks}")
        
        print_header("TODOS OS TESTES CONCLUÍDOS COM SUCESSO! ✅")
        
    except Exception as e:
        print_error(f"Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        store.close()
        print_info("Conexão com banco fechada")

def test_api_integration():
    """Teste de integração com API (requer servidor rodando)"""
    
    print_header("TESTE DE INTEGRAÇÃO COM API")
    
    try:
        import requests
        base_url = "http://localhost:8000"
        
        # Testar se API está rodando
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print_success("API está rodando!")
        else:
            print_error("API não está respondendo corretamente")
            return
            
    except requests.exceptions.RequestException:
        print_error("API não está rodando. Inicie o servidor primeiro:")
        print_info("cd src && python -m api.main")
        return
    
    try:
        # Teste 1: Criar projeto via API
        print_step(1, "CRIAR PROJETO VIA API")
        
        project_data = {
            "id": "api-test-project",
            "name": "Projeto de Teste API",
            "description": "Projeto criado via API para teste",
            "metadata": {
                "test": True,
                "created_by": "test_script"
            }
        }
        
        response = requests.post(f"{base_url}/projects", json=project_data)
        if response.status_code == 200:
            project = response.json()
            print_success(f"Projeto criado: {project['name']}")
        else:
            print_error(f"Falha ao criar projeto: {response.status_code} - {response.text}")
            return
        
        # Teste 2: Listar projetos
        print_step(2, "LISTAR PROJETOS VIA API")
        
        response = requests.get(f"{base_url}/projects")
        if response.status_code == 200:
            projects = response.json()
            print_success(f"Projetos encontrados: {len(projects)}")
            for proj in projects:
                print_info(f"- {proj['id']}: {proj['name']}")
        else:
            print_error(f"Falha ao listar projetos: {response.status_code}")
        
        # Teste 3: Adicionar documentos
        print_step(3, "ADICIONAR DOCUMENTOS VIA API")
        
        documents_data = {
            "project_id": "api-test-project",
            "documents": [
                {
                    "content": "def hello_world(): print('Hello, World!')",
                    "metadata": {"file": "hello.py", "type": "python"}
                },
                {
                    "content": "function greetUser(name) { return `Hello, ${name}!`; }",
                    "metadata": {"file": "greet.js", "type": "javascript"}
                }
            ]
        }
        
        response = requests.post(f"{base_url}/add_documents", json=documents_data)
        if response.status_code == 200:
            result = response.json()
            print_success(f"Documentos adicionados: {result['documents_processed']}")
        else:
            print_error(f"Falha ao adicionar documentos: {response.status_code} - {response.text}")
        
        # Teste 4: Query com filtro por projeto
        print_step(4, "QUERY COM FILTRO POR PROJETO")
        
        query_data = {
            "question": "Como fazer um Hello World?",
            "project_id": "api-test-project",
            "k": 5
        }
        
        response = requests.post(f"{base_url}/query", json=query_data)
        if response.status_code == 200:
            result = response.json()
            print_success("Query executada com sucesso!")
            print_info(f"Resposta: {result.get('answer', 'N/A')[:100]}...")
            print_info(f"Sources: {len(result.get('sources', []))}")
        else:
            print_error(f"Falha na query: {response.status_code} - {response.text}")
        
        # Teste 5: Obter estatísticas
        print_step(5, "OBTER ESTATÍSTICAS DO PROJETO")
        
        response = requests.get(f"{base_url}/projects/api-test-project/stats")
        if response.status_code == 200:
            stats = response.json()
            print_success("Estatísticas obtidas:")
            print_info(f"- Total chunks: {stats['total_chunks']}")
            print_info(f"- Linguagens: {stats['languages']}")
        else:
            print_error(f"Falha ao obter estatísticas: {response.status_code}")
        
        # Teste 6: Tentar adicionar documento a projeto inexistente
        print_step(6, "TENTAR ADICIONAR A PROJETO INEXISTENTE")
        
        invalid_data = {
            "project_id": "projeto-que-nao-existe",
            "documents": [{"content": "teste", "metadata": {}}]
        }
        
        response = requests.post(f"{base_url}/add_documents", json=invalid_data)
        if response.status_code == 400:
            print_success("Validação funcionou: rejeitou projeto inexistente")
            print_info(f"Erro: {response.json().get('detail')}")
        else:
            print_error("FALHA: Deveria ter rejeitado projeto inexistente")
        
        # Teste 7: Deletar projeto de teste
        print_step(7, "DELETAR PROJETO DE TESTE")
        
        response = requests.delete(f"{base_url}/projects/api-test-project?force=true")
        if response.status_code == 200:
            result = response.json()
            print_success(f"Projeto deletado: {result['message']}")
        else:
            print_error(f"Falha ao deletar projeto: {response.status_code}")
        
        print_header("TESTE DE INTEGRAÇÃO CONCLUÍDO COM SUCESSO! ✅")
        
    except Exception as e:
        print_error(f"Erro durante teste de integração: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 INICIANDO TESTES DO SISTEMA DE PROJETOS RAG")
    print("=" * 60)
    
    # Teste 1: CRUD básico
    test_project_crud()
    
    # Teste 2: Integração com API (opcional)
    print("\n" + "="*60)
    print("🌐 TESTE DE INTEGRAÇÃO COM API")
    print("="*60)
    
    choice = input("\nDeseja testar integração com API? (y/n): ").lower().strip()
    if choice in ['y', 'yes', 's', 'sim']:
        test_api_integration()
    else:
        print_info("Teste de integração com API pulado")
    
    print("\n" + "🎉" * 20)
    print("TODOS OS TESTES FINALIZADOS!")
    print("🎉" * 20) 