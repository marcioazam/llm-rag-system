"""Utilitários para testes do sistema RAG.

Este módulo fornece:
- Fixtures reutilizáveis
- Helpers para criação de dados de teste
- Validadores de resultados
- Mocks configurados
"""

import tempfile
import os
import yaml
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock
from pathlib import Path
import pytest


class TestDataFactory:
    """
    Factory para criação de dados de teste padronizados.
    
    Fornece métodos para criar:
    - Configurações de teste
    - Documentos de exemplo
    - Chunks simulados
    - Resultados de busca
    """
    
    @staticmethod
    def create_base_config() -> Dict[str, Any]:
        """
        Cria configuração base para testes.
        
        Returns:
            Dict[str, Any]: Configuração padrão
        """
        return {
            "chunking": {
                "method": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""]
            },
            "embeddings": {
                "model_name": "all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32
            },
            "vectordb": {
                "type": "qdrant",
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "vector_size": 384
            },
            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.5,
                "rerank": False
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "model_router": {
                "type": "simple",
                "default_model": "gpt-3.5-turbo",
                "routing_strategy": "content_based"
            },
            "neo4j": {
                "enabled": False,
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "test"
            },
            "rag": {
                "fallback_to_llm": True,
                "min_relevance_score": 0.5,
                "hybrid_mode": False,
                "enable_model_routing": True,
                "max_context_length": 4000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    @staticmethod
    def create_test_documents(count: int = 3) -> List[Dict[str, Any]]:
        """
        Cria documentos de teste com conteúdo variado.
        
        Args:
            count: Número de documentos a criar
            
        Returns:
            List[Dict[str, Any]]: Lista de documentos de teste
        """
        documents = []
        
        templates = [
            {
                "content": "Este é um documento sobre inteligência artificial e machine learning. "
                          "Contém informações sobre algoritmos, redes neurais e processamento de linguagem natural. "
                          "A IA tem revolucionado diversos setores da economia e sociedade.",
                "metadata": {
                    "source": "ai_document.pdf",
                    "category": "Technology",
                    "author": "Dr. AI Expert",
                    "date": "2024-01-15",
                    "tags": ["AI", "ML", "NLP"]
                }
            },
            {
                "content": "Sustentabilidade ambiental é um tema crucial para o futuro do planeta. "
                          "Envolve práticas de conservação, energia renovável e redução de emissões. "
                          "Empresas e governos devem trabalhar juntos para alcançar metas climáticas.",
                "metadata": {
                    "source": "sustainability_report.txt",
                    "category": "Environment",
                    "author": "Green Institute",
                    "date": "2024-02-01",
                    "tags": ["sustainability", "climate", "environment"]
                }
            },
            {
                "content": "A economia digital tem transformado modelos de negócio tradicionais. "
                          "E-commerce, fintech e plataformas digitais são exemplos dessa transformação. "
                          "A digitalização acelera processos e melhora a experiência do cliente.",
                "metadata": {
                    "source": "digital_economy.docx",
                    "category": "Business",
                    "author": "Business Analyst",
                    "date": "2024-01-20",
                    "tags": ["digital", "economy", "business"]
                }
            }
        ]
        
        for i in range(count):
            template = templates[i % len(templates)]
            doc = template.copy()
            doc["metadata"] = template["metadata"].copy()
            doc["metadata"]["id"] = f"doc_{i+1}"
            documents.append(doc)
        
        return documents
    
    @staticmethod
    def create_search_results(query: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Cria resultados de busca simulados.
        
        Args:
            query: Query de busca
            count: Número de resultados
            
        Returns:
            List[Dict[str, Any]]: Resultados de busca simulados
        """
        results = []
        
        for i in range(count):
            score = 0.9 - (i * 0.1)  # Scores decrescentes
            distance = 0.1 + (i * 0.1)  # Distâncias crescentes
            
            result = {
                "content": f"Resultado {i+1} para a query '{query}'. "
                          f"Este conteúdo é relevante e contém informações úteis.",
                "metadata": {
                    "source": f"result_doc_{i+1}.txt",
                    "score": score,
                    "rank": i + 1,
                    "chunk_id": f"chunk_{i+1}"
                },
                "distance": distance,
                "score": score
            }
            results.append(result)
        
        return results


class MockFactory:
    """
    Factory para criação de mocks configurados.
    
    Fornece mocks pré-configurados para:
    - Componentes do RAG pipeline
    - APIs externas
    - Serviços de embedding
    - Stores de dados
    """
    
    @staticmethod
    def create_chunker_mock() -> Mock:
        """Cria mock configurado para chunkers."""
        mock = Mock()
        
        # Simular chunks de teste
        from src.chunking.base_chunker import Chunk
        
        mock_chunks = [
            Chunk(
                content="Primeiro chunk de teste com conteúdo relevante",
                metadata={"source": "test.txt", "page": 1, "section": "intro"},
                chunk_id="chunk_1",
                document_id="doc_1",
                position=0
            ),
            Chunk(
                content="Segundo chunk com mais informações importantes",
                metadata={"source": "test.txt", "page": 1, "section": "body"},
                chunk_id="chunk_2",
                document_id="doc_1",
                position=1
            )
        ]
        
        mock.chunk.return_value = mock_chunks
        mock.chunk_text.return_value = mock_chunks
        mock.get_chunk_size.return_value = 1000
        mock.get_overlap.return_value = 200
        
        return mock
    
    @staticmethod
    def create_embedding_mock() -> Mock:
        """Cria mock configurado para serviço de embeddings."""
        mock = Mock()
        
        # Embeddings simulados (384 dimensões)
        mock_embedding = [0.1] * 384
        mock_embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        
        mock.embed_text.return_value = mock_embedding
        mock.embed_texts.return_value = mock_embeddings
        mock.embed_chunks.return_value = mock_embeddings
        mock.embed_query.return_value = mock_embedding
        mock.get_dimension.return_value = 384
        mock.get_model_name.return_value = "test-embedding-model"
        
        return mock
    
    @staticmethod
    def create_retriever_mock() -> Mock:
        """Cria mock configurado para retriever."""
        mock = Mock()
        
        # Resultados de busca padrão
        default_results = TestDataFactory.create_search_results("test query")
        
        mock.retrieve.return_value = default_results
        mock.add_documents.return_value = None
        mock.get_stats.return_value = {
            "total_documents": 100,
            "total_chunks": 500,
            "index_size": "10MB"
        }
        mock.clear.return_value = None
        
        return mock
    
    @staticmethod
    def create_model_router_mock() -> Mock:
        """Cria mock configurado para model router."""
        mock = Mock()
        
        mock.route_query.return_value = {
            "model": "gpt-3.5-turbo",
            "strategy": "direct",
            "confidence": 0.9,
            "reasoning": "Query adequada para modelo padrão"
        }
        
        mock.generate_response.return_value = {
            "answer": "Resposta gerada pelo modelo",
            "model_used": "gpt-3.5-turbo",
            "tokens_used": 150,
            "processing_time": 1.2
        }
        
        mock.get_available_models.return_value = [
            "gpt-3.5-turbo", "gpt-4", "claude-3"
        ]
        
        return mock
    
    @staticmethod
    def create_openai_mock() -> Mock:
        """Cria mock configurado para OpenAI API."""
        mock_client = Mock()
        
        # Mock da resposta
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Resposta do OpenAI"
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        return mock_client


class TestValidators:
    """
    Validadores para resultados de testes.
    
    Fornece métodos para validar:
    - Estrutura de respostas
    - Qualidade de resultados
    - Performance
    - Conformidade com especificações
    """
    
    @staticmethod
    def validate_query_response(response: Dict[str, Any]) -> bool:
        """
        Valida estrutura de resposta de query.
        
        Args:
            response: Resposta a validar
            
        Returns:
            bool: True se válida
        """
        required_fields = ["answer"]
        optional_fields = ["sources", "metadata", "processing_time", "model_used"]
        
        # Verificar campos obrigatórios
        for field in required_fields:
            if field not in response:
                return False
        
        # Verificar tipos
        if not isinstance(response["answer"], str):
            return False
        
        if "sources" in response and not isinstance(response["sources"], list):
            return False
        
        return True
    
    @staticmethod
    def validate_document_processing(result: Any, expected_chunks: int = None) -> bool:
        """
        Valida resultado de processamento de documentos.
        
        Args:
            result: Resultado do processamento
            expected_chunks: Número esperado de chunks
            
        Returns:
            bool: True se válido
        """
        # Para add_documents, resultado pode ser None (sucesso silencioso)
        if result is None:
            return True
        
        # Se retorna algo, deve ser uma estrutura válida
        if isinstance(result, dict):
            return "status" in result or "chunks_created" in result
        
        return False
    
    @staticmethod
    def validate_search_results(results: List[Dict[str, Any]]) -> bool:
        """
        Valida resultados de busca.
        
        Args:
            results: Lista de resultados
            
        Returns:
            bool: True se válidos
        """
        if not isinstance(results, list):
            return False
        
        for result in results:
            if not isinstance(result, dict):
                return False
            
            # Campos essenciais
            if "content" not in result:
                return False
            
            if not isinstance(result["content"], str):
                return False
        
        return True


class PerformanceTestHelper:
    """
    Helper para testes de performance.
    
    Fornece utilitários para:
    - Medição de tempo
    - Monitoramento de memória
    - Análise de throughput
    - Benchmarking
    """
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """
        Mede tempo de execução de uma função.
        
        Args:
            func: Função a executar
            *args: Argumentos posicionais
            **kwargs: Argumentos nomeados
            
        Returns:
            tuple: (resultado, tempo_em_segundos)
        """
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    def create_large_document(size_kb: int = 100) -> Dict[str, Any]:
        """
        Cria documento grande para testes de performance.
        
        Args:
            size_kb: Tamanho aproximado em KB
            
        Returns:
            Dict[str, Any]: Documento grande
        """
        # Aproximadamente 1KB de texto
        base_text = "Este é um texto de teste para simular documentos grandes. " * 20
        
        # Repetir para atingir o tamanho desejado
        repetitions = (size_kb * 1024) // len(base_text)
        large_content = base_text * repetitions
        
        return {
            "content": large_content,
            "metadata": {
                "source": f"large_document_{size_kb}kb.txt",
                "size_kb": size_kb,
                "type": "performance_test"
            }
        }
    
    @staticmethod
    def benchmark_queries(pipeline, queries: List[str], iterations: int = 3):
        """
        Executa benchmark de queries.
        
        Args:
            pipeline: Pipeline RAG a testar
            queries: Lista de queries para testar
            iterations: Número de iterações por query
            
        Returns:
            Dict: Estatísticas de performance
        """
        import statistics
        
        results = {
            "total_queries": len(queries) * iterations,
            "query_times": [],
            "average_time": 0,
            "median_time": 0,
            "min_time": 0,
            "max_time": 0
        }
        
        for query in queries:
            for _ in range(iterations):
                _, exec_time = PerformanceTestHelper.measure_execution_time(
                    pipeline.query, query
                )
                results["query_times"].append(exec_time)
        
        if results["query_times"]:
            results["average_time"] = statistics.mean(results["query_times"])
            results["median_time"] = statistics.median(results["query_times"])
            results["min_time"] = min(results["query_times"])
            results["max_time"] = max(results["query_times"])
        
        return results


# Fixtures globais para reutilização
@pytest.fixture(scope="session")
def test_data_factory():
    """Factory para dados de teste."""
    return TestDataFactory()


@pytest.fixture(scope="session")
def mock_factory():
    """Factory para mocks."""
    return MockFactory()


@pytest.fixture(scope="session")
def validators():
    """Validadores de teste."""
    return TestValidators()


@pytest.fixture(scope="session")
def performance_helper():
    """Helper para testes de performance."""
    return PerformanceTestHelper()