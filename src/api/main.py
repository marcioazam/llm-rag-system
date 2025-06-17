from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import sys
from pathlib import Path

# Adiciona o diretório pai ao path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.pipeline_dependency import get_pipeline

app = FastAPI(title="RAG System API", version="2.0.0")

# Dependency

# Models
class Document(BaseModel):
    content: str
    source: Optional[str] = None
    metadata: Optional[Dict] = None

class AddDocumentsRequest(BaseModel):
    documents: List[Document]
    chunking_strategy: str = 'recursive'
    chunk_size: int = 500
    chunk_overlap: int = 50

class QueryRequest(BaseModel):
    question: str  # Mantém 'question' como campo principal
    query: Optional[str] = None  # Compatibilidade com versão 2.0
    k: Optional[int] = 5
    system_prompt: Optional[str] = None
    force_use_context: Optional[bool] = False
    llm_only: Optional[bool] = False
    use_hybrid: Optional[bool] = True

class IndexRequest(BaseModel):
    document_paths: List[str]

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
            "File upload support"
        ]
    }

# Query endpoints
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Executar query no sistema RAG com suporte a modo híbrido"""
    try:
        # Compatibilidade: usar 'query' se 'question' não estiver presente
        question = request.question or request.query
        if not question:
            raise HTTPException(status_code=400, detail="Either 'question' or 'query' must be provided")
        
        if request.llm_only:
            # Usar apenas LLM sem buscar contexto
            result = get_pipeline().query_llm_only(
                question=question,
                system_prompt=request.system_prompt
            )
        elif hasattr(get_pipeline(), 'query') and request.use_hybrid:
            # Usar modo híbrido se disponível
            result = get_pipeline().query(
                query_text=question,
                k=request.k,
                use_hybrid=request.use_hybrid
            )
            # Converter formato de resposta se necessário
            if isinstance(result, dict) and 'sources' in result and isinstance(result['sources'], list):
                if result['sources'] and isinstance(result['sources'][0], str):
                    result['sources'] = [{"content": source, "source": "hybrid_search"} for source in result['sources']]
        else:
            # Usar RAG tradicional
            result = get_pipeline().query(
                question=question,
                k=request.k,
                system_prompt=request.system_prompt,
                force_use_context=request.force_use_context
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_with_code", response_model=QueryResponse)
async def query_with_code(request: QueryRequest):
    """Realiza uma consulta que sempre inclui exemplos de código"""
    try:
        question = request.question or request.query
        if not question:
            raise HTTPException(status_code=400, detail="Either 'question' or 'query' must be provided")
            
        if hasattr(get_pipeline(), 'query_with_code_examples'):
            result = get_pipeline().query_with_code_examples(
                query_text=question,
                k=request.k
            )
            return QueryResponse(**result)
        else:
            # Fallback para query normal com prompt específico para código
            result = get_pipeline().query(
                question=question,
                k=request.k,
                system_prompt="Focus on providing code examples and technical implementation details."
            )
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document management endpoints
@app.post("/add_documents")
async def add_documents(request: AddDocumentsRequest):
    """Adiciona documentos ao sistema RAG"""
    try:
        documents = [doc.dict() for doc in request.documents]
        if hasattr(get_pipeline(), 'add_documents'):
            get_pipeline().add_documents(
                documents=documents,
                chunking_strategy=request.chunking_strategy,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap
            )
        else:
            # Fallback para método de indexação tradicional
            # Salvar documentos temporariamente e indexar
            temp_paths = []
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for i, doc in enumerate(documents):
                temp_path = temp_dir / f"doc_{i}.txt"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(doc['content'])
                temp_paths.append(str(temp_path))
            
            result = get_pipeline().index_documents(temp_paths)
            
            # Limpar arquivos temporários
            for path in temp_paths:
                os.unlink(path)
                
            return {"message": f"Adicionados {len(documents)} documentos com sucesso", "result": result}
        
        return {"message": f"Adicionados {len(documents)} documentos com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_documents(request: IndexRequest):
    """Indexar documentos a partir de caminhos de arquivo"""
    try:
        result = get_pipeline().index_documents(request.document_paths)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload de arquivo para indexação"""
    try:
        # Salvar arquivo
        upload_dir = Path("data/raw")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Indexar arquivo
        result = get_pipeline().index_documents([str(file_path)])

        return {
            "filename": file.filename,
            "path": str(file_path),
            "indexing_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Database management endpoints
@app.delete("/index")
async def clear_index():
    """Limpar índice"""
    try:
        if hasattr(get_pipeline(), 'clear_index'):
            get_pipeline().clear_index()
        elif hasattr(get_pipeline(), 'clear_database'):
            get_pipeline().clear_database()
        else:
            raise HTTPException(status_code=501, detail="Clear index method not implemented")
        return {"message": "Index cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear_database")
async def clear_database():
    """Limpa o banco de dados vetorial"""
    try:
        if hasattr(get_pipeline(), 'clear_database'):
            get_pipeline().clear_database()
        elif hasattr(get_pipeline(), 'clear_index'):
            get_pipeline().clear_index()
        else:
            raise HTTPException(status_code=501, detail="Clear database method not implemented")
        return {"message": "Banco de dados limpo com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Info and stats endpoints
@app.get("/info")
async def get_info():
    """Obter informações sobre o sistema"""
    try:
        info = {
            "status": "operational",
            "version": "2.0.0"
        }
        
        # Tentar obter informações do vector store
        if hasattr(get_pipeline(), 'vector_store') and hasattr(get_pipeline().vector_store, 'get_collection_info'):
            info["vector_store"] = get_pipeline().vector_store.get_collection_info()
        
        # Tentar obter configurações
        if hasattr(get_pipeline(), 'config'):
            info["config"] = {
                "llm_model": get_pipeline().config.get("llm", {}).get("model", "unknown"),
                "embedding_model": get_pipeline().config.get("embeddings", {}).get("model_name", "unknown"),
                "chunking_method": get_pipeline().config.get("chunking", {}).get("method", "unknown")
            }
        
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Retorna estatísticas do sistema"""
    try:
        stats = {}
        
        if hasattr(get_pipeline(), 'get_collection_stats'):
            stats = get_pipeline().get_collection_stats()
        
        stats['models'] = {
            'general': 'llama3.1:8b-instruct-q4_K_M',
            'code': 'codellama:7b-instruct'
        }
        stats['version'] = '2.0.0'
        stats['features'] = ['hybrid_mode', 'code_generation', 'file_upload']
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "hybrid_mode": "enabled",
        "version": "2.0.0"
    }

@app.get("/query_stream")
async def query_stream(q: str, k: int = 5):
    """Endpoint que faz streaming da resposta RAG em tempo real (SSE)."""

    import json, asyncio

    pipeline = get_pipeline()

    async def _generator():
        result = pipeline.query(query_text=q, k=k)
        # chunk simples linha por linha
        for line in result["answer"].split("\n"):
            yield f"data: {json.dumps({'chunk': line})}\n\n"
            await asyncio.sleep(0)
        yield "event: end\n data: done\n\n"

    return StreamingResponse(_generator(), media_type="text/event-stream")

@app.get("/coverage_options")
async def coverage_options():
    """Retorna lista de valores únicos do campo 'coverage' presentes nos metadados."""
    pipe = get_pipeline()
    if not hasattr(pipe, "metadata_store") or pipe.metadata_store is None:
        return {"coverage": []}
    try:
        values = pipe.metadata_store.distinct_coverage()
        return {"coverage": values}
    except Exception as exc:  # pylint: disable=broad-except
        return JSONResponse(status_code=500, content={"error": str(exc)})
