from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import os
import shutil
import sys
from pathlib import Path
import re

# Adiciona o diretório pai ao path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.pipeline_dependency import get_pipeline
from src.metadata.sqlite_store import ProjectValidationError

app = FastAPI(title="RAG System API", version="2.0.0")

# Expor métricas Prometheus
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)
except ImportError:
    import logging
    logging.warning("prometheus-fastapi-instrumentator não instalado – métricas desativadas.")

# Dependency

# Models
class Document(BaseModel):
    content: str
    source: Optional[str] = None
    metadata: Optional[Dict] = None

class AddDocumentsRequest(BaseModel):
    documents: List[Document]
    project_id: str = Field(..., min_length=1, max_length=100, description="ID do projeto (obrigatório)")
    chunking_strategy: str = 'recursive'
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    @field_validator('project_id')
    def validate_project_id(cls, v):
        if not v or not v.strip():
            raise ValueError('project_id não pode estar vazio')
        # Remover caracteres potencialmente perigosos
        cleaned = re.sub(r'[<>"\'/\\]', '', v.strip())
        if len(cleaned) < 1:
            raise ValueError('project_id inválido')
        return cleaned

# Project Models
class CreateProjectRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=100, description="ID único do projeto")
    name: str = Field(..., min_length=1, max_length=200, description="Nome do projeto")
    description: Optional[str] = Field(None, max_length=1000, description="Descrição do projeto")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadados customizados")
    
    @field_validator('id')
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError('ID do projeto não pode estar vazio')
        # Permitir apenas caracteres alfanuméricos, hífens e underscores
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '', v.strip())
        if len(cleaned) < 1:
            raise ValueError('ID do projeto deve conter apenas letras, números, hífens e underscores')
        return cleaned
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Nome do projeto não pode estar vazio')
        return v.strip()

class UpdateProjectRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Nome do projeto")
    description: Optional[str] = Field(None, max_length=1000, description="Descrição do projeto")
    status: Optional[str] = Field(None, description="Status do projeto")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadados customizados")
    
    @field_validator('status')
    def validate_status(cls, v):
        if v is not None and v not in ['active', 'inactive', 'archived']:
            raise ValueError('Status deve ser: active, inactive ou archived')
        return v

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    created_at: str
    updated_at: str
    metadata: Optional[Dict[str, Any]]

class ProjectStatsResponse(BaseModel):
    total_chunks: int
    languages_count: int
    files_count: int
    first_chunk_date: Optional[str]
    last_chunk_date: Optional[str]
    languages: Dict[str, int]

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Pergunta principal")
    query: Optional[str] = Field(None, max_length=2000, description="Compatibilidade com versão 2.0")
    project_id: Optional[str] = Field(None, description="Filtrar por projeto específico")
    k: Optional[int] = Field(5, ge=1, le=50, description="Número de chunks a recuperar")
    system_prompt: Optional[str] = Field(None, max_length=5000, description="Prompt do sistema")
    force_use_context: Optional[bool] = Field(False, description="Forçar uso de contexto")
    llm_only: Optional[bool] = Field(False, description="Usar apenas LLM")
    use_hybrid: Optional[bool] = Field(True, description="Usar busca híbrida")
    
    # Parâmetros específicos para Cursor IDE
    context: Optional[str] = Field(None, description="Contexto do código atual (Cursor)")
    file_type: Optional[str] = Field(None, description="Tipo de arquivo (.py, .js, etc)")
    project_context: Optional[str] = Field(None, description="Contexto do projeto (Cursor)")
    quick_mode: Optional[bool] = Field(False, description="Modo rápido para Cursor")
    allow_hybrid: Optional[bool] = Field(True, description="Permitir híbrido (Cursor)")
    max_response_time: Optional[int] = Field(15, description="Timeout máximo em segundos")
    
    @field_validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Pergunta não pode estar vazia')
        # Remover caracteres potencialmente perigosos
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        if len(cleaned) < 3:
            raise ValueError('Pergunta deve ter pelo menos 3 caracteres')
        return cleaned
    
    @field_validator('system_prompt')
    def validate_system_prompt(cls, v):
        if v is not None:
            # Limitar tamanho e remover caracteres perigosos
            cleaned = re.sub(r'[<>]', '', v.strip())
            return cleaned if cleaned else None
        return v
    
    @field_validator('project_id')
    def validate_project_id(cls, v):
        if v is not None:
            cleaned = re.sub(r'[^a-zA-Z0-9_-]', '', v.strip())
            return cleaned if cleaned else None
        return v

class IndexRequest(BaseModel):
    document_paths: List[str]
    project_id: str = Field(..., min_length=1, max_length=100, description="ID do projeto (obrigatório)")
    
    @field_validator('project_id')
    def validate_project_id(cls, v):
        if not v or not v.strip():
            raise ValueError('project_id é obrigatório')
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '', v.strip())
        if len(cleaned) < 1:
            raise ValueError('project_id inválido')
        return cleaned

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    model: str
    models_used: Optional[List[str]] = None
    strategy: Optional[str] = None
    needs_code: Optional[bool] = None
    retrieved_chunks: Optional[List[str]] = None

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "RAG System API v2.0", 
        "status": "running",
        "features": [
            "Semantic search",
            "Hybrid model routing",
            "Document upload and indexing",
            "Code generation with CodeLlama",
            "General responses with Llama 3.1",
            "LLM-only mode",
            "File upload support",
            "Project management and isolation"
        ]
    }

# ============================================================================
# ENDPOINTS DE PROJETOS
# ============================================================================

@app.post("/projects", response_model=ProjectResponse)
async def create_project(request: CreateProjectRequest):
    """Criar um novo projeto"""
    try:
        pipeline = get_pipeline()
        if not hasattr(pipeline, 'metadata_store'):
            raise HTTPException(status_code=500, detail="Sistema de metadados não disponível")
        
        project = pipeline.metadata_store.create_project(
            project_id=request.id,
            name=request.name,
            description=request.description,
            metadata=request.metadata
        )
        
        return ProjectResponse(**project)
        
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/projects", response_model=List[ProjectResponse])
async def list_projects(
    status: Optional[str] = Query(None, description="Filtrar por status"),
    limit: Optional[int] = Query(None, ge=1, le=100, description="Limite de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginação")
):
    """Listar projetos com filtros opcionais"""
    try:
        pipeline = get_pipeline()
        if not hasattr(pipeline, 'metadata_store'):
            raise HTTPException(status_code=500, detail="Sistema de metadados não disponível")
        
        projects = pipeline.metadata_store.list_projects(
            status=status,
            limit=limit,
            offset=offset
        )
        
        return [ProjectResponse(**project) for project in projects]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Obter projeto por ID"""
    try:
        pipeline = get_pipeline()
        if not hasattr(pipeline, 'metadata_store'):
            raise HTTPException(status_code=500, detail="Sistema de metadados não disponível")
        
        project = pipeline.metadata_store.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Projeto '{project_id}' não encontrado")
        
        return ProjectResponse(**project)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.put("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, request: UpdateProjectRequest):
    """Atualizar projeto existente"""
    try:
        pipeline = get_pipeline()
        if not hasattr(pipeline, 'metadata_store'):
            raise HTTPException(status_code=500, detail="Sistema de metadados não disponível")
        
        project = pipeline.metadata_store.update_project(
            project_id=project_id,
            name=request.name,
            description=request.description,
            status=request.status,
            metadata=request.metadata
        )
        
        return ProjectResponse(**project)
        
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.delete("/projects/{project_id}")
async def delete_project(project_id: str, force: bool = Query(False, description="Forçar deleção mesmo com chunks")):
    """Deletar projeto"""
    try:
        pipeline = get_pipeline()
        if not hasattr(pipeline, 'metadata_store'):
            raise HTTPException(status_code=500, detail="Sistema de metadados não disponível")
        
        success = pipeline.metadata_store.delete_project(project_id, force=force)
        
        if success:
            return {"message": f"Projeto '{project_id}' deletado com sucesso"}
        else:
            raise HTTPException(status_code=404, detail=f"Projeto '{project_id}' não encontrado")
        
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/projects/{project_id}/stats", response_model=ProjectStatsResponse)
async def get_project_stats(project_id: str):
    """Obter estatísticas do projeto"""
    try:
        pipeline = get_pipeline()
        if not hasattr(pipeline, 'metadata_store'):
            raise HTTPException(status_code=500, detail="Sistema de metadados não disponível")
        
        stats = pipeline.metadata_store.get_project_stats(project_id)
        return ProjectStatsResponse(**stats)
        
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# ============================================================================
# ENDPOINTS MODIFICADOS COM VALIDAÇÃO DE PROJETO
# ============================================================================

# Query endpoints
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Executar query no sistema RAG com suporte a filtro por projeto"""
    import time
    start_time = time.time()
    
    try:
        # Compatibilidade: usar 'query' se 'question' não estiver presente
        question = request.question or request.query
        if not question:
            raise HTTPException(status_code=400, detail="Either 'question' or 'query' must be provided")
        
        # Validar projeto se especificado
        if request.project_id:
            pipeline = get_pipeline()
            if hasattr(pipeline, 'metadata_store'):
                if not pipeline.metadata_store.project_exists(request.project_id):
                    raise HTTPException(status_code=400, detail=f"Projeto '{request.project_id}' não existe")
        
        # 🎯 OTIMIZAÇÃO CURSOR: Construir system prompt otimizado
        system_prompt = request.system_prompt
        if request.context or request.file_type or request.project_context:
            # Cursor está enviando contexto - otimizar prompt
            cursor_prompt = """Você é um assistente de código especializado para IDEs.
Forneça respostas CONCISAS, PRÁTICAS e DIRETAS.
Foque em:
- Exemplos de código funcionais
- Soluções implementáveis
- Explicações claras e breves
- Contexto do projeto atual"""
            
            if request.file_type:
                if request.file_type in [".py", ".python"]:
                    cursor_prompt += "\nFoque em Python: PEP8, type hints, docstrings."
                elif request.file_type in [".js", ".ts", ".jsx", ".tsx"]:
                    cursor_prompt += "\nFoque em JavaScript/TypeScript: ES6+, async/await, tipos."
                elif request.file_type in [".java"]:
                    cursor_prompt += "\nFoque em Java: OOP, Spring Boot, boas práticas."
                elif request.file_type in [".cs"]:
                    cursor_prompt += "\nFoque em C#: .NET, LINQ, async patterns."
            
            if request.project_context:
                cursor_prompt += f"\nContexto do projeto: {request.project_context[:200]}"
            
            if request.context:
                cursor_prompt += f"\nCódigo atual: {request.context[:500]}"
            
            system_prompt = cursor_prompt
        
        # 🎯 OTIMIZAÇÃO CURSOR: Ajustar K baseado no modo
        k = request.k
        if request.quick_mode:
            k = min(3, k)  # Modo rápido = menos chunks
        
        # 🎯 OTIMIZAÇÃO CURSOR: Decidir estratégia baseada na complexidade
        use_hybrid = request.use_hybrid
        if request.quick_mode and not request.allow_hybrid:
            use_hybrid = False  # Força modo rápido
        
        # Preparar filtros para busca por projeto
        filters = {}
        if request.project_id:
            filters["project_id"] = request.project_id
        
        # Executar query baseado no tipo
        if request.llm_only:
            result = get_pipeline().query_llm_only(
                question=question,
                system_prompt=system_prompt
            )
        elif hasattr(get_pipeline(), 'query') and use_hybrid:
            # TODO: Implementar busca com filtros nos pipelines
            result = get_pipeline().query(
                query_text=question,
                k=k,
                use_hybrid=use_hybrid
            )
            # Converter formato se necessário
            if isinstance(result, dict) and 'sources' in result and isinstance(result['sources'], list):
                if result['sources'] and isinstance(result['sources'][0], str):
                    result['sources'] = [{"content": source, "source": "hybrid_search"} for source in result['sources']]
        else:
            result = get_pipeline().query(
                question=question,
                k=k,
                system_prompt=system_prompt,
                force_use_context=request.force_use_context
            )
        
        # 🎯 OTIMIZAÇÃO CURSOR: Adicionar métricas de performance
        processing_time = time.time() - start_time
        if isinstance(result, dict):
            result["processing_time"] = round(processing_time, 3)
            result["mode"] = "cursor_optimized" if (request.context or request.quick_mode) else "standard"
            result["k_used"] = k
            if request.project_id:
                result["project_id"] = request.project_id
        
        return result
        
    except HTTPException:
        raise
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_with_code", response_model=QueryResponse)
async def query_with_code(request: QueryRequest):
    """Realiza uma consulta que sempre inclui exemplos de código"""
    try:
        question = request.question or request.query
        if not question:
            raise HTTPException(status_code=400, detail="Either 'question' or 'query' must be provided")
        
        # Validar projeto se especificado
        if request.project_id:
            pipeline = get_pipeline()
            if hasattr(pipeline, 'metadata_store'):
                if not pipeline.metadata_store.project_exists(request.project_id):
                    raise HTTPException(status_code=400, detail=f"Projeto '{request.project_id}' não existe")
            
        if hasattr(get_pipeline(), 'query_with_code_examples'):
            result = get_pipeline().query_with_code_examples(
                query_text=question,
                k=request.k
            )
            return QueryResponse(**result)
        else:
            # Fallback para query normal
            return await query(request)
            
    except HTTPException:
        raise
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_documents")
async def add_documents(request: AddDocumentsRequest):
    """Adicionar documentos ao sistema com validação de projeto"""
    try:
        # Validar se projeto existe
        pipeline = get_pipeline()
        if hasattr(pipeline, 'metadata_store'):
            if not pipeline.metadata_store.project_exists(request.project_id):
                raise HTTPException(status_code=400, detail=f"Projeto '{request.project_id}' não existe. Crie o projeto primeiro.")
        
        # Adicionar project_id aos metadados de cada documento
        for doc in request.documents:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["project_id"] = request.project_id
        
        result = get_pipeline().add_documents(
            documents=[
                {"content": doc.content, "metadata": doc.metadata or {}}
                for doc in request.documents
            ],
                chunking_strategy=request.chunking_strategy,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap
            )
        
        return {
            "message": f"Adicionados {len(request.documents)} documentos ao projeto '{request.project_id}'",
            "project_id": request.project_id,
            "documents_processed": len(request.documents),
            "chunking_strategy": request.chunking_strategy,
            "result": result
        }
        
    except HTTPException:
        raise
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_documents(request: IndexRequest):
    """Indexar documentos de paths específicos com validação de projeto"""
    try:
        # Validar se projeto existe
        pipeline = get_pipeline()
        if hasattr(pipeline, 'metadata_store'):
            if not pipeline.metadata_store.project_exists(request.project_id):
                raise HTTPException(status_code=400, detail=f"Projeto '{request.project_id}' não existe. Crie o projeto primeiro.")
        
        # TODO: Implementar indexação com project_id nos pipelines
        result = get_pipeline().index_documents(request.document_paths)
        
        return {
            "message": f"Documentos indexados no projeto '{request.project_id}'",
            "project_id": request.project_id,
            "paths": request.document_paths,
            "result": result
        }
        
    except HTTPException:
        raise
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    project_id: str = Query(..., description="ID do projeto para upload")
):
    """Upload de arquivo com validação de projeto"""
    try:
        # Validar se projeto existe
        pipeline = get_pipeline()
        if hasattr(pipeline, 'metadata_store'):
            if not pipeline.metadata_store.project_exists(project_id):
                raise HTTPException(status_code=400, detail=f"Projeto '{project_id}' não existe. Crie o projeto primeiro.")
        
        # Criar diretório para o projeto se não existir
        project_upload_dir = f"./uploads/{project_id}"
        os.makedirs(project_upload_dir, exist_ok=True)
        
        # Salvar arquivo no diretório do projeto
        file_path = os.path.join(project_upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Indexar arquivo automaticamente
        result = get_pipeline().index_documents([file_path])

        return {
            "message": f"Arquivo '{file.filename}' enviado e indexado no projeto '{project_id}'",
            "project_id": project_id,
            "filename": file.filename,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "indexing_result": result
        }
        
    except HTTPException:
        raise
    except ProjectValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENDPOINTS EXISTENTES (mantidos para compatibilidade)
# ============================================================================

@app.delete("/index")
async def clear_index():
    """Limpar índice completo (cuidado: remove todos os projetos!)"""
    try:
        result = get_pipeline().clear_index()
        return {
            "message": "Índice limpo com sucesso",
            "warning": "Todos os projetos e chunks foram removidos",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear_database")
async def clear_database():
    """Limpar banco de dados completo (cuidado: remove todos os projetos!)"""
    try:
        result = get_pipeline().clear_database()
        return {
            "message": "Banco de dados limpo com sucesso",
            "warning": "Todos os projetos e dados foram removidos",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def get_info():
    """Informações do sistema"""
    try:
        pipeline = get_pipeline()
        info = {
            "system": "RAG System API v2.0",
            "features": [
                "Project management",
                "Document indexing per project",
                "Semantic search with project filtering",
                "Hybrid model routing",
                "Code generation",
                "Project isolation"
            ]
        }
        
        # Adicionar informações dos projetos se disponível
        if hasattr(pipeline, 'metadata_store'):
            try:
                projects = pipeline.metadata_store.list_projects(limit=10)
                info["projects_count"] = len(projects)
                info["recent_projects"] = [{"id": p["id"], "name": p["name"]} for p in projects[:5]]
            except Exception:
                pass
        
        return info
        
    except Exception as e:
        return {"error": str(e), "system": "RAG System API v2.0"}

@app.get("/stats")
async def get_stats():
    """Estatísticas do sistema"""
    try:
        pipeline = get_pipeline()
        stats = {"system": "RAG System API v2.0"}
        
        # Adicionar estatísticas dos projetos se disponível
        if hasattr(pipeline, 'metadata_store'):
            try:
                projects = pipeline.metadata_store.list_projects()
                stats["total_projects"] = len(projects)
                stats["active_projects"] = len([p for p in projects if p["status"] == "active"])
                
                # Estatísticas agregadas
                total_chunks = 0
                for project in projects:
                    try:
                        project_stats = pipeline.metadata_store.get_project_stats(project["id"])
                        total_chunks += project_stats["total_chunks"]
                    except Exception:
                        pass
                
                stats["total_chunks"] = total_chunks
            except Exception:
                pass
        
        return stats
        
    except Exception as e:
        return {"error": str(e), "system": "RAG System API v2.0"}

@app.get("/health")
async def health_check():
    """Health check do sistema"""
    try:
        pipeline = get_pipeline()
        health = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "2.0.0"
        }
        
        # Verificar saúde do sistema de projetos
        if hasattr(pipeline, 'metadata_store'):
            try:
                # Teste simples: listar projetos
                pipeline.metadata_store.list_projects(limit=1)
                health["project_system"] = "healthy"
            except Exception as e:
                health["project_system"] = f"unhealthy: {str(e)}"
                health["status"] = "degraded"
        
        return health
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "2.0.0"
        }

@app.get("/query_stream")
async def query_stream(q: str, k: int = 5, project_id: Optional[str] = None):
    """Stream de resposta de query com filtro opcional por projeto"""
    try:
        pipeline = get_pipeline()

        # Validar projeto se especificado
        if project_id and hasattr(pipeline, "metadata_store"):
            if not pipeline.metadata_store.project_exists(project_id):
                raise HTTPException(status_code=400, detail=f"Projeto '{project_id}' não existe")

        async def _generator():
            try:
                # TODO: Implementar streaming real; por enquanto resposta única
                result = pipeline.query(q, k=k)
                yield f"data: {result}\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(_generator(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/coverage_options")
async def coverage_options():
    """Opções de coverage disponíveis"""
    try:
        pipeline = get_pipeline()
        if hasattr(pipeline, 'metadata_store'):
            options = pipeline.metadata_store.distinct_coverage()
            return {"coverage_options": options}
        else:
            return {"coverage_options": []}
    except Exception as e:
        return {"error": str(e), "coverage_options": []}

# Sistema unificado - Cursor usa endpoint /query com parâmetros específicos
