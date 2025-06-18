#!/usr/bin/env python3
"""
Endpoint otimizado para integração com Cursor
Foca em velocidade, eficiência e respostas práticas
"""

from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

class CursorQueryRequest(BaseModel):
    """Request otimizado para Cursor"""
    question: str
    context: Optional[str] = None  # Contexto do código atual
    file_type: Optional[str] = None  # Tipo de arquivo (.py, .js, etc)
    project_context: Optional[str] = None  # Contexto do projeto
    quick_mode: bool = True  # Modo rápido por padrão
    allow_hybrid: bool = False  # Híbrido apenas se explicitamente permitido
    max_response_time: int = 5  # Timeout em segundos

class CursorQueryResponse(BaseModel):
    """Response otimizado para Cursor"""
    answer: str
    sources: list
    processing_time: float
    model_used: str
    provider_used: str
    cost: float
    mode: str  # "fast" ou "hybrid"
    cached: bool = False

class CursorEndpoint:
    """Endpoint especializado para Cursor"""
    
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.cache = {}  # Cache simples para respostas frequentes
        self.cache_ttl = 300  # 5 minutos
        
    def should_use_hybrid_for_cursor(self, query: str, context_length: int) -> bool:
        """Decide se usar híbrido baseado na complexidade - muito restritivo para Cursor"""
        
        # Critérios MUITO específicos para híbrido no Cursor
        complex_indicators = [
            len(query) > 500,  # Query muito longa
            "analise completa" in query.lower(),
            "refatoração completa" in query.lower(),
            "arquitetura completa" in query.lower(),
            context_length > 10000,  # Contexto muito grande
            query.count("e também") >= 2,  # Múltiplas tarefas explícitas
            "code review completo" in query.lower()
        ]
        
        # Só usar híbrido se >= 4 indicadores (muito restritivo)
        return sum(complex_indicators) >= 4

    def get_cache_key(self, query: str, context: str = "") -> str:
        """Gera chave de cache"""
        return f"{hash(query + context)}"

    def is_cache_valid(self, timestamp: float) -> bool:
        """Verifica se cache ainda é válido"""
        return (time.time() - timestamp) < self.cache_ttl

    async def cursor_optimized_query(self, request: CursorQueryRequest) -> CursorQueryResponse:
        """Query otimizada para Cursor com foco em velocidade"""
        
        start_time = time.time()
        
        try:
            # 1. Verificar cache primeiro
            cache_key = self.get_cache_key(request.question, request.context or "")
            
            if cache_key in self.cache:
                cached_result, timestamp = self.cache[cache_key]
                if self.is_cache_valid(timestamp):
                    logger.info(f"Cache hit para query: {request.question[:50]}...")
                    cached_result["processing_time"] = time.time() - start_time
                    cached_result["cached"] = True
                    return CursorQueryResponse(**cached_result)
            
            # 2. Determinar modo baseado na complexidade
            context_length = len(request.context or "")
            use_hybrid = (
                request.allow_hybrid and 
                not request.quick_mode and
                self.should_use_hybrid_for_cursor(request.question, context_length)
            )
            
            # 3. Preparar contexto otimizado para Cursor
            system_prompt = self._build_cursor_system_prompt(request)
            
            # 4. Executar query
            if use_hybrid:
                result = await self._execute_hybrid_query(request, system_prompt)
                mode = "hybrid"
            else:
                result = await self._execute_fast_query(request, system_prompt)
                mode = "fast"
            
            # 5. Preparar resposta
            processing_time = time.time() - start_time
            
            response_data = {
                "answer": result["answer"],
                "sources": result.get("sources", [])[:3],  # Limitar sources para Cursor
                "processing_time": processing_time,
                "model_used": result.get("model_used", "unknown"),
                "provider_used": result.get("provider_used", "unknown"),
                "cost": result.get("cost", 0.0),
                "mode": mode,
                "cached": False
            }
            
            # 6. Cache resultado se foi rápido e bem-sucedido
            if processing_time < 3.0 and mode == "fast":
                self.cache[cache_key] = (response_data.copy(), time.time())
            
            logger.info(f"Cursor query processada em {processing_time:.2f}s - Modo: {mode}")
            
            return CursorQueryResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Erro no endpoint Cursor: {e}")
            
            # Fallback simples para não quebrar o Cursor
            return CursorQueryResponse(
                answer=f"Erro ao processar pergunta: {str(e)}",
                sources=[],
                processing_time=time.time() - start_time,
                model_used="error",
                provider_used="none",
                cost=0.0,
                mode="error"
            )

    def _build_cursor_system_prompt(self, request: CursorQueryRequest) -> str:
        """Constrói system prompt otimizado para Cursor"""
        
        base_prompt = """Você é um assistente de código especializado para IDEs.
Forneça respostas CONCISAS, PRÁTICAS e DIRETAS.
Foque em:
- Exemplos de código funcionais
- Soluções implementáveis
- Explicações claras e breves
- Contexto do projeto atual"""

        # Adicionar contexto específico do tipo de arquivo
        if request.file_type:
            if request.file_type in [".py", ".python"]:
                base_prompt += "\nFoque em Python: PEP8, type hints, docstrings."
            elif request.file_type in [".js", ".ts", ".jsx", ".tsx"]:
                base_prompt += "\nFoque em JavaScript/TypeScript: ES6+, async/await, tipos."
            elif request.file_type in [".java"]:
                base_prompt += "\nFoque em Java: OOP, Spring Boot, boas práticas."
            elif request.file_type in [".cs"]:
                base_prompt += "\nFoque em C#: .NET, LINQ, async patterns."
        
        # Adicionar contexto do projeto se disponível
        if request.project_context:
            base_prompt += f"\nContexto do projeto: {request.project_context[:200]}"
        
        return base_prompt

    async def _execute_fast_query(self, request: CursorQueryRequest, system_prompt: str) -> Dict[str, Any]:
        """Executa query rápida (modo padrão para Cursor)"""
        
        # Configurações otimizadas para velocidade
        result = self.pipeline.query(
            question=request.question,
            k=3,  # Poucos chunks para velocidade
            system_prompt=system_prompt,
            force_use_context=True,  # Sempre usar contexto local
            # force_model="openai.gpt4o_mini"  # Modelo mais rápido
        )
        
        return result

    async def _execute_hybrid_query(self, request: CursorQueryRequest, system_prompt: str) -> Dict[str, Any]:
        """Executa query híbrida (apenas para casos muito complexos)"""
        
        # Avisar que vai demorar mais
        logger.info("Executando query híbrida para Cursor - tempo estendido")
        
        # Usar pipeline híbrido se disponível
        if hasattr(self.pipeline, 'query_hybrid'):
            result = self.pipeline.query_hybrid(
                question=request.question,
                context=request.context,
                system_prompt=system_prompt,
                max_time=request.max_response_time * 2  # Tempo estendido
            )
        else:
            # Fallback para query normal com mais contexto
            result = self.pipeline.query(
                question=request.question,
                k=8,  # Mais contexto para query complexa
                system_prompt=system_prompt,
                force_use_context=True
            )
        
        return result

    def get_cursor_stats(self) -> Dict[str, Any]:
        """Estatísticas específicas para Cursor"""
        return {
            "cache_size": len(self.cache),
            "cache_hit_rate": getattr(self, '_cache_hits', 0) / max(getattr(self, '_total_queries', 1), 1),
            "avg_response_time": getattr(self, '_avg_response_time', 0.0),
            "total_queries": getattr(self, '_total_queries', 0),
            "fast_mode_usage": getattr(self, '_fast_mode_count', 0),
            "hybrid_mode_usage": getattr(self, '_hybrid_mode_count', 0)
        }

# Função para integrar no FastAPI
def add_cursor_endpoints(app, rag_pipeline):
    """Adiciona endpoints otimizados para Cursor na aplicação FastAPI"""
    
    cursor_endpoint = CursorEndpoint(rag_pipeline)
    
    @app.post("/cursor/query", response_model=CursorQueryResponse)
    async def cursor_query(request: CursorQueryRequest):
        """Endpoint principal otimizado para Cursor"""
        return await cursor_endpoint.cursor_optimized_query(request)
    
    @app.get("/cursor/stats")
    async def cursor_stats():
        """Estatísticas específicas do Cursor"""
        return cursor_endpoint.get_cursor_stats()
    
    @app.post("/cursor/quick")
    async def cursor_quick_query(question: str, context: str = ""):
        """Endpoint ultra-rápido para queries simples do Cursor"""
        request = CursorQueryRequest(
            question=question,
            context=context,
            quick_mode=True,
            allow_hybrid=False,
            max_response_time=3
        )
        return await cursor_endpoint.cursor_optimized_query(request)
    
    return cursor_endpoint 