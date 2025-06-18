"""
Sistema de Logging Estruturado para RAG
Implementa logging com estrutura JSON para melhor observabilidade
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import sys


class StructuredLogger:
    """
    Logger estruturado para o sistema RAG
    Fornece logging consistente e estruturado em JSON
    """

    def __init__(self, name: str = "rag_system", level: str = "INFO"):
        self.name = name
        self.level = getattr(logging, level.upper())
        self.setup_logging()

    def setup_logging(self):
        """Configura o sistema de logging estruturado"""
        
        # Configurar logger básico
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Handler para arquivo JSON
        json_handler = logging.FileHandler(log_dir / "rag_structured.jsonl")
        json_handler.setLevel(self.level)
        
        # Formatter JSON personalizado
        json_formatter = JsonFormatter()
        json_handler.setFormatter(json_formatter)
        
        # Handler para console (desenvolvimento)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Configurar logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.logger.addHandler(json_handler)
        self.logger.addHandler(console_handler)

    def log_query(self, 
                  query_id: str,
                  query_text: str,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  **kwargs):
        """Log de query recebida"""
        data = {
            "event": "query_received",
            "query_id": query_id,
            "query_text": query_text[:100] + "..." if len(query_text) > 100 else query_text,
            "query_length": len(query_text),
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        self.logger.info(json.dumps(data))

    def log_retrieval(self,
                     query_id: str,
                     chunks_found: int,
                     retrieval_time: float,
                     similarity_scores: list,
                     retrieval_method: str = "hybrid",
                     **kwargs):
        """Log de processo de retrieval"""
        data = {
            "event": "retrieval_completed",
            "query_id": query_id,
            "chunks_found": chunks_found,
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "avg_similarity": round(sum(similarity_scores) / len(similarity_scores), 3) if similarity_scores else 0,
            "max_similarity": round(max(similarity_scores), 3) if similarity_scores else 0,
            "min_similarity": round(min(similarity_scores), 3) if similarity_scores else 0,
            "retrieval_method": retrieval_method,
            **kwargs
        }
        self.logger.info(json.dumps(data))

    def log_model_call(self,
                      query_id: str,
                      model_name: str,
                      provider: str,
                      tokens_input: int,
                      tokens_output: int,
                      processing_time: float,
                      cost: float = 0.0,
                      **kwargs):
        """Log de chamada para modelo LLM"""
        data = {
            "event": "model_call_completed",
            "query_id": query_id,
            "model_name": model_name,
            "provider": provider,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "total_tokens": tokens_input + tokens_output,
            "processing_time_ms": round(processing_time * 1000, 2),
            "cost_usd": round(cost, 6),
            "tokens_per_second": round((tokens_input + tokens_output) / processing_time, 2) if processing_time > 0 else 0,
            **kwargs
        }
        self.logger.info(json.dumps(data))

    def log_response(self,
                    query_id: str,
                    response_length: int,
                    total_time: float,
                    success: bool = True,
                    error_type: Optional[str] = None,
                    **kwargs):
        """Log de resposta final"""
        event = "response_success" if success else "response_error"
        
        data = {
            "event": event,
            "query_id": query_id,
            "response_length": response_length,
            "total_time_ms": round(total_time * 1000, 2),
            "success": success,
            **kwargs
        }
        
        if error_type:
            data["error_type"] = error_type
        
        if success:
            self.logger.info(json.dumps(data))
        else:
            self.logger.error(json.dumps(data))

    def log_error(self,
                 error_type: str,
                 error_message: str,
                 query_id: Optional[str] = None,
                 stack_trace: Optional[str] = None,
                 **kwargs):
        """Log de erro estruturado"""
        data = {
            "event": "system_error",
            "error_type": error_type,
            "error_message": error_message,
            "query_id": query_id,
            "stack_trace": stack_trace,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        self.logger.error(json.dumps(data))


class JsonFormatter(logging.Formatter):
    """Formatter customizado para JSON"""
    
    def format(self, record):
        # Se a mensagem já é JSON, retorna como está
        try:
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, ValueError):
            # Se não é JSON, formata como JSON
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            
            return json.dumps(log_data)


class QueryContext:
    """
    Context manager para tracking de queries
    Automatiza logging de início e fim de operações
    """

    def __init__(self, logger: StructuredLogger, operation: str, query_id: Optional[str] = None):
        self.logger = logger
        self.operation = operation
        self.query_id = query_id or str(uuid.uuid4())
        self.start_time = None
        self.metadata = {}

    def __enter__(self):
        self.start_time = time.time()
        data = {
            "event": f"{self.operation}_started",
            "query_id": self.query_id,
            "operation": self.operation,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.logger.info(json.dumps(data))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            data = {
                "event": f"{self.operation}_completed",
                "query_id": self.query_id,
                "operation": self.operation,
                "duration_ms": round(duration * 1000, 2),
                "timestamp": datetime.utcnow().isoformat(),
                **self.metadata
            }
            self.logger.logger.info(json.dumps(data))
        else:
            self.logger.log_error(
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                query_id=self.query_id,
                stack_trace=str(exc_tb) if exc_tb else None
            )

    def add_metadata(self, **kwargs):
        """Adiciona metadados ao contexto"""
        self.metadata.update(kwargs)


# Instância global do logger
_global_logger = None

def get_logger(name: str = "rag_system") -> StructuredLogger:
    """Obtém instância global do logger estruturado"""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name)
    return _global_logger

def log_query_context(operation: str, query_id: Optional[str] = None) -> QueryContext:
    """Cria contexto de query para logging automático"""
    return QueryContext(get_logger(), operation, query_id)


# Exemplo de uso
if __name__ == "__main__":
    logger = get_logger()
    
    # Exemplo de logging de query
    with log_query_context("example_query") as ctx:
        ctx.add_metadata(user_id="test_user", model="gpt-4")
        time.sleep(0.1)  # Simular processamento 