"""
Sistema de Paralelização Massiva para RAG
Execução paralela de estratégias, batch processing otimizado e pipeline assíncrono
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict, deque
import numpy as np
from functools import partial
import aiofiles
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParallelTask:
    """Tarefa para execução paralela"""
    task_id: str
    task_type: str  # retrieval, embedding, generation, evaluation
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0  # Maior = mais prioritário
    dependencies: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0


@dataclass
class BatchRequest:
    """Requisição em batch"""
    batch_id: str
    requests: List[Dict[str, Any]]
    strategy: str  # parallel, sequential, adaptive
    max_concurrent: int = 10
    timeout: float = 30.0
    created_at: datetime = field(default_factory=datetime.now)


class ParallelExecutor:
    """Executor para paralelização de tarefas"""
    
    def __init__(self,
                 max_workers: int = None,
                 use_process_pool: bool = False):
        
        self.max_workers = max_workers or mp.cpu_count()
        self.use_process_pool = use_process_pool
        
        # Pools de execução
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers) if use_process_pool else None
        
        # Filas de tarefas por prioridade
        self.task_queues = defaultdict(deque)  # priority -> deque of tasks
        
        # Tarefas em execução
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Cache de resultados
        self.result_cache: Dict[str, Any] = {}
        
        # Estatísticas
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cache_hits": 0,
            "avg_execution_time": 0.0,
            "tasks_by_type": defaultdict(int)
        }
        
        # Semáforo para controle de concorrência
        self.semaphore = asyncio.Semaphore(self.max_workers * 2)
        
        logger.info(f"ParallelExecutor inicializado com {self.max_workers} workers")
    
    async def submit_task(self, task: ParallelTask) -> str:
        """Submete tarefa para execução"""
        
        # Verificar cache
        cache_key = self._get_cache_key(task)
        if cache_key in self.result_cache:
            task.result = self.result_cache[cache_key]
            self.stats["cache_hits"] += 1
            return task.task_id
        
        # Adicionar à fila apropriada
        self.task_queues[task.priority].append(task)
        self.stats["total_tasks"] += 1
        self.stats["tasks_by_type"][task.task_type] += 1
        
        # Iniciar processamento se não estiver rodando
        if not self.running_tasks:
            asyncio.create_task(self._process_task_queues())
        
        return task.task_id
    
    async def submit_batch(self, 
                         tasks: List[ParallelTask],
                         wait_for_completion: bool = True) -> List[str]:
        """Submete múltiplas tarefas em batch"""
        
        task_ids = []
        
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        
        if wait_for_completion:
            # Aguardar conclusão de todas as tarefas
            await self.wait_for_tasks(task_ids)
        
        return task_ids
    
    async def wait_for_tasks(self, task_ids: List[str], timeout: float = None) -> Dict[str, Any]:
        """Aguarda conclusão de tarefas específicas"""
        
        start_time = time.time()
        results = {}
        
        while True:
            all_completed = True
            
            for task_id in task_ids:
                if task_id in self.running_tasks:
                    all_completed = False
                    break
                
                # Verificar se tarefa foi completada
                for priority_queue in self.task_queues.values():
                    for task in priority_queue:
                        if task.task_id == task_id and task.result is not None:
                            results[task_id] = task.result
                            break
            
            if all_completed:
                break
            
            # Verificar timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout aguardando tarefas: {task_ids}")
                break
            
            await asyncio.sleep(0.1)
        
        return results
    
    async def _process_task_queues(self):
        """Processa filas de tarefas continuamente"""
        
        while any(self.task_queues.values()) or self.running_tasks:
            # Processar por prioridade (maior primeiro)
            for priority in sorted(self.task_queues.keys(), reverse=True):
                queue = self.task_queues[priority]
                
                while queue and len(self.running_tasks) < self.max_workers * 2:
                    task = queue.popleft()
                    
                    # Verificar dependências
                    if await self._check_dependencies(task):
                        # Executar tarefa
                        asyncio_task = asyncio.create_task(self._execute_task(task))
                        self.running_tasks[task.task_id] = asyncio_task
                    else:
                        # Recolocar na fila se dependências não satisfeitas
                        queue.append(task)
                        break
            
            await asyncio.sleep(0.01)
    
    async def _check_dependencies(self, task: ParallelTask) -> bool:
        """Verifica se dependências da tarefa foram satisfeitas"""
        
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            # Verificar se dependência foi completada
            dep_completed = False
            
            for priority_queue in self.task_queues.values():
                for t in priority_queue:
                    if t.task_id == dep_id and t.result is not None:
                        dep_completed = True
                        break
            
            if not dep_completed and dep_id not in self.result_cache:
                return False
        
        return True
    
    async def _execute_task(self, task: ParallelTask):
        """Executa uma tarefa individual"""
        
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Determinar tipo de execução
                if asyncio.iscoroutinefunction(task.function):
                    # Função assíncrona
                    result = await task.function(*task.args, **task.kwargs)
                else:
                    # Função síncrona - executar em thread/process pool
                    if self.use_process_pool and task.task_type in ["embedding", "evaluation"]:
                        # Usar process pool para tarefas CPU-intensivas
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.process_pool,
                            partial(task.function, *task.args, **task.kwargs)
                        )
                    else:
                        # Usar thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.thread_pool,
                            partial(task.function, *task.args, **task.kwargs)
                        )
                
                task.result = result
                task.execution_time = time.time() - start_time
                
                # Cache resultado
                cache_key = self._get_cache_key(task)
                self.result_cache[cache_key] = result
                
                # Atualizar estatísticas
                self.stats["completed_tasks"] += 1
                self._update_avg_execution_time(task.execution_time)
                
            except Exception as e:
                logger.error(f"Erro executando tarefa {task.task_id}: {e}")
                task.error = e
                self.stats["failed_tasks"] += 1
            
            finally:
                # Remover das tarefas em execução
                self.running_tasks.pop(task.task_id, None)
    
    def _get_cache_key(self, task: ParallelTask) -> str:
        """Gera chave de cache para tarefa"""
        
        # Combinar tipo, função e argumentos
        key_parts = [
            task.task_type,
            task.function.__name__,
            str(task.args),
            str(sorted(task.kwargs.items()))
        ]
        
        return "|".join(key_parts)
    
    def _update_avg_execution_time(self, execution_time: float):
        """Atualiza tempo médio de execução"""
        
        n = self.stats["completed_tasks"]
        self.stats["avg_execution_time"] = (
            (self.stats["avg_execution_time"] * (n - 1) + execution_time) / n
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de execução"""
        
        return {
            **self.stats,
            "running_tasks": len(self.running_tasks),
            "queued_tasks": sum(len(q) for q in self.task_queues.values()),
            "cache_size": len(self.result_cache),
            "workers": self.max_workers
        }
    
    def shutdown(self):
        """Encerra pools de execução"""
        
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("ParallelExecutor encerrado")


class BatchProcessor:
    """Processador otimizado para operações em batch"""
    
    def __init__(self,
                 batch_size: int = 32,
                 max_concurrent_batches: int = 4):
        
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        
        # Buffer de requisições
        self.request_buffer = defaultdict(list)  # strategy -> requests
        
        # Batches em processamento
        self.processing_batches: Dict[str, asyncio.Task] = {}
        
        # Resultados
        self.results: Dict[str, Any] = {}
        
        # Estatísticas
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_time": 0.0,
            "requests_per_second": 0.0
        }
        
        self.start_time = time.time()
    
    async def add_request(self,
                        request_id: str,
                        strategy: str,
                        data: Dict[str, Any]) -> str:
        """Adiciona requisição ao buffer"""
        
        request = {
            "id": request_id,
            "data": data,
            "timestamp": time.time()
        }
        
        self.request_buffer[strategy].append(request)
        self.stats["total_requests"] += 1
        
        # Verificar se deve processar batch
        if len(self.request_buffer[strategy]) >= self.batch_size:
            await self._trigger_batch_processing(strategy)
        
        return request_id
    
    async def _trigger_batch_processing(self, strategy: str):
        """Dispara processamento de batch"""
        
        if len(self.processing_batches) >= self.max_concurrent_batches:
            # Aguardar slot disponível
            await self._wait_for_batch_slot()
        
        # Criar batch
        requests = self.request_buffer[strategy][:self.batch_size]
        self.request_buffer[strategy] = self.request_buffer[strategy][self.batch_size:]
        
        batch = BatchRequest(
            batch_id=f"batch_{int(time.time() * 1000)}",
            requests=requests,
            strategy=strategy
        )
        
        # Processar batch
        task = asyncio.create_task(self._process_batch(batch))
        self.processing_batches[batch.batch_id] = task
        
        self.stats["total_batches"] += 1
    
    async def _wait_for_batch_slot(self):
        """Aguarda slot para novo batch"""
        
        while len(self.processing_batches) >= self.max_concurrent_batches:
            # Aguardar qualquer batch terminar
            done, pending = await asyncio.wait(
                self.processing_batches.values(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Remover batches completos
            for task in done:
                batch_id = None
                for bid, t in self.processing_batches.items():
                    if t == task:
                        batch_id = bid
                        break
                
                if batch_id:
                    del self.processing_batches[batch_id]
    
    async def _process_batch(self, batch: BatchRequest):
        """Processa um batch de requisições"""
        
        start_time = time.time()
        
        try:
            if batch.strategy == "parallel":
                results = await self._process_parallel(batch)
            elif batch.strategy == "sequential":
                results = await self._process_sequential(batch)
            elif batch.strategy == "adaptive":
                results = await self._process_adaptive(batch)
            else:
                results = await self._process_parallel(batch)  # Default
            
            # Armazenar resultados
            for request, result in zip(batch.requests, results):
                self.results[request["id"]] = result
            
            # Atualizar estatísticas
            batch_time = time.time() - start_time
            self._update_batch_stats(batch_time, len(batch.requests))
            
        except Exception as e:
            logger.error(f"Erro processando batch {batch.batch_id}: {e}")
            
            # Marcar todas as requisições como falhadas
            for request in batch.requests:
                self.results[request["id"]] = {"error": str(e)}
    
    async def _process_parallel(self, batch: BatchRequest) -> List[Any]:
        """Processa batch em paralelo"""
        
        tasks = []
        
        for request in batch.requests:
            # Criar tarefa para cada requisição
            task = asyncio.create_task(
                self._process_single_request(request["data"])
            )
            tasks.append(task)
        
        # Executar todas em paralelo com limite
        results = []
        
        for i in range(0, len(tasks), batch.max_concurrent):
            chunk = tasks[i:i + batch.max_concurrent]
            chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
            results.extend(chunk_results)
        
        return results
    
    async def _process_sequential(self, batch: BatchRequest) -> List[Any]:
        """Processa batch sequencialmente"""
        
        results = []
        
        for request in batch.requests:
            try:
                result = await self._process_single_request(request["data"])
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
    
    async def _process_adaptive(self, batch: BatchRequest) -> List[Any]:
        """Processa batch com estratégia adaptativa"""
        
        # Analisar complexidade das requisições
        simple_requests = []
        complex_requests = []
        
        for request in batch.requests:
            complexity = self._estimate_complexity(request["data"])
            
            if complexity < 0.5:
                simple_requests.append(request)
            else:
                complex_requests.append(request)
        
        results = {}
        
        # Processar simples em paralelo
        if simple_requests:
            simple_tasks = [
                asyncio.create_task(self._process_single_request(r["data"]))
                for r in simple_requests
            ]
            
            simple_results = await asyncio.gather(*simple_tasks, return_exceptions=True)
            
            for req, res in zip(simple_requests, simple_results):
                results[req["id"]] = res
        
        # Processar complexas com menos paralelismo
        if complex_requests:
            for i in range(0, len(complex_requests), 2):  # 2 por vez
                chunk = complex_requests[i:i+2]
                chunk_tasks = [
                    asyncio.create_task(self._process_single_request(r["data"]))
                    for r in chunk
                ]
                
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                
                for req, res in zip(chunk, chunk_results):
                    results[req["id"]] = res
        
        # Ordenar resultados pela ordem original
        return [results[req["id"]] for req in batch.requests]
    
    async def _process_single_request(self, data: Dict[str, Any]) -> Any:
        """Processa uma requisição individual"""
        
        # Simulação - substituir com lógica real
        await asyncio.sleep(0.1)
        
        # Processar baseado no tipo
        request_type = data.get("type", "unknown")
        
        if request_type == "embedding":
            return {"embedding": np.random.randn(768).tolist()}
        elif request_type == "retrieval":
            return {"documents": [{"content": "Doc sample", "score": 0.9}]}
        elif request_type == "generation":
            return {"response": "Generated response"}
        else:
            return {"result": "Processed"}
    
    def _estimate_complexity(self, data: Dict[str, Any]) -> float:
        """Estima complexidade da requisição"""
        
        # Heurísticas simples
        text_length = len(data.get("text", ""))
        num_operations = len(data.get("operations", []))
        
        complexity = (text_length / 1000) + (num_operations / 10)
        
        return min(1.0, complexity)
    
    def _update_batch_stats(self, batch_time: float, num_requests: int):
        """Atualiza estatísticas de batch"""
        
        n = self.stats["total_batches"]
        
        # Tempo médio por batch
        self.stats["avg_batch_time"] = (
            (self.stats["avg_batch_time"] * (n - 1) + batch_time) / n
        )
        
        # Taxa de requisições por segundo
        elapsed = time.time() - self.start_time
        self.stats["requests_per_second"] = self.stats["total_requests"] / elapsed
    
    async def flush(self):
        """Processa todas as requisições pendentes"""
        
        for strategy, requests in self.request_buffer.items():
            while requests:
                await self._trigger_batch_processing(strategy)
        
        # Aguardar todos os batches
        if self.processing_batches:
            await asyncio.gather(*self.processing_batches.values())
    
    def get_result(self, request_id: str) -> Optional[Any]:
        """Obtém resultado de uma requisição"""
        
        return self.results.get(request_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do processador"""
        
        return {
            **self.stats,
            "buffered_requests": sum(len(reqs) for reqs in self.request_buffer.values()),
            "processing_batches": len(self.processing_batches),
            "completed_requests": len(self.results)
        }


class ParallelRAGPipeline:
    """Pipeline RAG com paralelização massiva"""
    
    def __init__(self,
                 retrieval_strategies: List[str],
                 max_parallel_strategies: int = 5):
        
        self.strategies = retrieval_strategies
        self.max_parallel = max_parallel_strategies
        
        # Componentes de paralelização
        self.executor = ParallelExecutor(max_workers=mp.cpu_count())
        self.batch_processor = BatchProcessor()
        
        # Cache de estratégias
        self.strategy_cache = {}
        
        # Estatísticas
        self.stats = {
            "total_queries": 0,
            "parallel_executions": 0,
            "avg_strategies_per_query": 0.0,
            "strategy_performance": defaultdict(lambda: {"count": 0, "avg_time": 0.0})
        }
    
    async def execute_query(self,
                          query: str,
                          strategies: Optional[List[str]] = None,
                          parallel_mode: str = "adaptive") -> Dict[str, Any]:
        """Executa query com múltiplas estratégias em paralelo"""
        
        self.stats["total_queries"] += 1
        
        # Determinar estratégias a usar
        selected_strategies = strategies or self.strategies[:self.max_parallel]
        
        if parallel_mode == "adaptive":
            selected_strategies = await self._select_adaptive_strategies(query)
        
        # Criar tarefas para cada estratégia
        tasks = []
        
        for strategy in selected_strategies:
            task = ParallelTask(
                task_id=f"{strategy}_{int(time.time() * 1000)}",
                task_type="retrieval",
                function=self._execute_strategy,
                args=(strategy, query),
                kwargs={},
                priority=self._get_strategy_priority(strategy)
            )
            
            tasks.append(task)
        
        # Executar em paralelo
        start_time = time.time()
        
        task_ids = await self.executor.submit_batch(tasks, wait_for_completion=True)
        
        # Coletar resultados
        results = {}
        
        for task_id, task in zip(task_ids, tasks):
            if task.result is not None:
                strategy_name = task.args[0]
                results[strategy_name] = task.result
                
                # Atualizar estatísticas da estratégia
                self._update_strategy_stats(strategy_name, task.execution_time)
        
        # Consolidar resultados
        final_result = await self._consolidate_results(results, query)
        
        # Estatísticas
        total_time = time.time() - start_time
        self.stats["parallel_executions"] += 1
        
        n = self.stats["total_queries"]
        self.stats["avg_strategies_per_query"] = (
            (self.stats["avg_strategies_per_query"] * (n-1) + len(selected_strategies)) / n
        )
        
        return {
            **final_result,
            "metadata": {
                "strategies_used": selected_strategies,
                "parallel_execution_time": total_time,
                "individual_times": {
                    s: self.stats["strategy_performance"][s]["avg_time"]
                    for s in selected_strategies
                }
            }
        }
    
    async def _select_adaptive_strategies(self, query: str) -> List[str]:
        """Seleciona estratégias adaptivamente baseado na query"""
        
        # Analisar características da query
        query_length = len(query.split())
        has_code = "code" in query.lower() or "function" in query.lower()
        is_complex = query_length > 20 or "?" in query[:-1]
        
        selected = []
        
        # Sempre incluir estratégia base
        selected.append("standard")
        
        # Adicionar baseado em características
        if has_code:
            selected.append("code_specific")
        
        if is_complex:
            selected.extend(["multi_query", "graph_enhanced"])
        
        # Adicionar estratégias com bom desempenho histórico
        top_performers = sorted(
            self.stats["strategy_performance"].items(),
            key=lambda x: x[1]["avg_time"] if x[1]["count"] > 0 else float('inf')
        )[:2]
        
        for strategy, _ in top_performers:
            if strategy not in selected:
                selected.append(strategy)
        
        return selected[:self.max_parallel]
    
    async def _execute_strategy(self, strategy: str, query: str) -> Dict[str, Any]:
        """Executa uma estratégia específica"""
        
        # Simulação - substituir com implementação real
        await asyncio.sleep(np.random.uniform(0.1, 0.5))
        
        return {
            "strategy": strategy,
            "documents": [
                {"content": f"Doc from {strategy}", "score": np.random.uniform(0.5, 1.0)}
                for _ in range(5)
            ]
        }
    
    def _get_strategy_priority(self, strategy: str) -> int:
        """Determina prioridade da estratégia"""
        
        # Prioridades baseadas em tipo
        priorities = {
            "standard": 5,
            "multi_query": 4,
            "graph_enhanced": 3,
            "code_specific": 4,
            "semantic": 5
        }
        
        return priorities.get(strategy, 1)
    
    def _update_strategy_stats(self, strategy: str, execution_time: float):
        """Atualiza estatísticas da estratégia"""
        
        stats = self.stats["strategy_performance"][strategy]
        n = stats["count"]
        
        stats["avg_time"] = (stats["avg_time"] * n + execution_time) / (n + 1)
        stats["count"] = n + 1
    
    async def _consolidate_results(self,
                                  strategy_results: Dict[str, Any],
                                  query: str) -> Dict[str, Any]:
        """Consolida resultados de múltiplas estratégias"""
        
        if not strategy_results:
            return {"documents": [], "answer": "Nenhum resultado encontrado"}
        
        # Coletar todos os documentos
        all_documents = []
        
        for strategy, result in strategy_results.items():
            docs = result.get("documents", [])
            
            # Adicionar metadado da estratégia
            for doc in docs:
                doc["source_strategy"] = strategy
            
            all_documents.extend(docs)
        
        # Deduplicar e ranquear
        unique_docs = self._deduplicate_documents(all_documents)
        ranked_docs = sorted(unique_docs, key=lambda x: x.get("score", 0), reverse=True)
        
        # Gerar resposta consolidada (simulação)
        answer = f"Resposta consolidada de {len(strategy_results)} estratégias"
        
        return {
            "documents": ranked_docs[:10],  # Top 10
            "answer": answer,
            "total_documents": len(all_documents),
            "unique_documents": len(unique_docs)
        }
    
    def _deduplicate_documents(self, documents: List[Dict]) -> List[Dict]:
        """Remove documentos duplicados mantendo melhor score"""
        
        seen = {}
        
        for doc in documents:
            content_hash = hash(doc.get("content", ""))
            
            if content_hash not in seen:
                seen[content_hash] = doc
            else:
                # Manter documento com maior score
                if doc.get("score", 0) > seen[content_hash].get("score", 0):
                    seen[content_hash] = doc
        
        return list(seen.values())
    
    async def process_batch_queries(self,
                                  queries: List[str],
                                  batch_strategy: str = "parallel") -> List[Dict[str, Any]]:
        """Processa múltiplas queries em batch"""
        
        # Adicionar ao batch processor
        request_ids = []
        
        for query in queries:
            request_id = await self.batch_processor.add_request(
                request_id=f"query_{int(time.time() * 1000)}_{len(request_ids)}",
                strategy=batch_strategy,
                data={"type": "rag_query", "query": query}
            )
            request_ids.append(request_id)
        
        # Processar batches
        await self.batch_processor.flush()
        
        # Coletar resultados
        results = []
        
        for request_id in request_ids:
            result = self.batch_processor.get_result(request_id)
            results.append(result or {"error": "No result"})
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Gera relatório de performance"""
        
        executor_stats = self.executor.get_stats()
        batch_stats = self.batch_processor.get_stats()
        
        return {
            "pipeline_stats": self.stats,
            "executor_stats": executor_stats,
            "batch_processor_stats": batch_stats,
            "performance_summary": {
                "queries_per_second": self.stats["total_queries"] / (time.time() - self.executor.stats.get("start_time", time.time())),
                "avg_parallel_efficiency": executor_stats["completed_tasks"] / max(executor_stats["total_tasks"], 1),
                "cache_efficiency": executor_stats["cache_hits"] / max(executor_stats["total_tasks"], 1),
                "batch_efficiency": batch_stats["requests_per_second"]
            }
        }
    
    def shutdown(self):
        """Encerra pipeline"""
        
        self.executor.shutdown()
        logger.info("ParallelRAGPipeline encerrado")


def create_parallel_rag_pipeline(strategies: List[str],
                               max_parallel: int = 5) -> ParallelRAGPipeline:
    """Factory para criar pipeline RAG paralelo"""
    
    return ParallelRAGPipeline(
        retrieval_strategies=strategies,
        max_parallel_strategies=max_parallel
    ) 