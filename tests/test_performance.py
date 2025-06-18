"""
Testes de performance para o sistema RAG
Monitora degradação de performance e estabelece baselines
"""

import pytest
import time
import psutil
import threading
import concurrent.futures
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.mark.performance
class TestRAGPerformance:
    """Testes de performance do sistema RAG"""
    
    @pytest.fixture
    def mock_pipeline_fast(self):
        """Mock otimizado do pipeline para testes de performance"""
        with patch('src.api.main.get_pipeline') as mock:
            pipeline = MagicMock()
            
            # Simular respostas rápidas
            pipeline.query.return_value = {
                "answer": "Resposta rápida de teste",
                "sources": [{"content": "fonte", "metadata": {}}],
                "model": "gpt-3.5-turbo"
            }
            
            pipeline.query_llm_only.return_value = {
                "answer": "LLM apenas",
                "model": "gpt-3.5-turbo"
            }
            
            mock.return_value = pipeline
            yield pipeline
    
    @pytest.mark.benchmark
    def test_query_response_time(self, mock_pipeline_fast, benchmark):
        """Testa tempo de resposta de queries individuais"""
        from src.api.main import app
        client = TestClient(app)
        
        def query_request():
            response = client.post("/query", json={
                "question": "O que é Python?",
                "k": 5
            })
            assert response.status_code == 200
            return response.json()
        
        # Benchmark da query
        result = benchmark(query_request)
        assert "answer" in result
        
        # Verificar se está dentro do baseline (< 2 segundos)
        assert benchmark.stats.mean < 2.0, f"Query muito lenta: {benchmark.stats.mean:.2f}s"
    
    @pytest.mark.benchmark
    def test_health_check_performance(self, mock_pipeline_fast, benchmark):
        """Testa performance do health check"""
        from src.api.main import app
        client = TestClient(app)
        
        def health_request():
            response = client.get("/health")
            assert response.status_code == 200
            return response.json()
        
        result = benchmark(health_request)
        assert result["status"] in ["healthy", "unhealthy"]
        
        # Health check deve ser muito rápido (< 500ms)
        assert benchmark.stats.mean < 0.5, f"Health check muito lento: {benchmark.stats.mean:.2f}s"
    
    def test_concurrent_queries_performance(self, mock_pipeline_fast):
        """Testa performance com queries concorrentes"""
        from src.api.main import app
        client = TestClient(app)
        
        def make_query(query_id):
            start_time = time.time()
            response = client.post("/query", json={
                "question": f"Test query {query_id}",
                "k": 3
            })
            duration = time.time() - start_time
            return {
                "query_id": query_id,
                "status_code": response.status_code,
                "duration": duration,
                "success": response.status_code == 200
            }
        
        # Testar 10 queries concorrentes
        num_queries = 10
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_query, i) for i in range(num_queries)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Verificar resultados
        successful_queries = [r for r in results if r["success"]]
        avg_duration = sum(r["duration"] for r in successful_queries) / len(successful_queries)
        
        assert len(successful_queries) >= num_queries * 0.8, "Muitas queries falharam"
        assert avg_duration < 3.0, f"Queries concorrentes muito lentas: {avg_duration:.2f}s"
        assert total_time < 10.0, f"Tempo total excessivo: {total_time:.2f}s"
    
    def test_memory_usage_during_queries(self, mock_pipeline_fast):
        """Testa uso de memória durante queries"""
        from src.api.main import app
        client = TestClient(app)
        
        # Medir memória inicial
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Fazer múltiplas queries para estressar o sistema
        for i in range(20):
            response = client.post("/query", json={
                "question": f"Memory test query {i}",
                "k": 5
            })
            assert response.status_code == 200
        
        # Medir memória final
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verificar se não há vazamento significativo de memória
        assert memory_increase < 100, f"Possível vazamento de memória: +{memory_increase:.1f}MB"
    
    def test_cpu_usage_during_load(self, mock_pipeline_fast):
        """Testa uso de CPU durante carga"""
        from src.api.main import app
        client = TestClient(app)
        
        # Monitorar CPU em thread separada
        cpu_samples = []
        stop_monitoring = threading.Event()
        
        def monitor_cpu():
            while not stop_monitoring.is_set():
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        try:
            # Fazer múltiplas queries
            for i in range(10):
                response = client.post("/query", json={
                    "question": f"CPU test query {i}",
                    "k": 3
                })
                assert response.status_code == 200
                time.sleep(0.1)  # Pequena pausa entre queries
                
        finally:
            stop_monitoring.set()
            monitor_thread.join()
        
        # Analisar uso de CPU
        if cpu_samples:
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            max_cpu = max(cpu_samples)
            
            # CPU não deve estar constantemente alta
            assert avg_cpu < 80, f"CPU média muito alta: {avg_cpu:.1f}%"
            assert max_cpu < 95, f"Pico de CPU muito alto: {max_cpu:.1f}%"


@pytest.mark.performance
@pytest.mark.slow
class TestSystemPerformance:
    """Testes de performance do sistema como um todo"""
    
    def test_system_startup_time(self):
        """Testa tempo de inicialização do sistema"""
        start_time = time.time()
        
        try:
            # Simular importação de módulos principais
            from src.rag_pipeline import RAGPipeline
            from src.api.main import app
            from src.utils.structured_logger import get_logger
            from src.utils.circuit_breaker import get_breaker_manager
            
            startup_time = time.time() - start_time
            
            # Sistema deve inicializar rapidamente
            assert startup_time < 10.0, f"Sistema muito lento para inicializar: {startup_time:.2f}s"
            
        except ImportError as e:
            pytest.fail(f"Falha na importação durante teste de performance: {e}")
    
    def test_large_text_processing(self, mock_pipeline_fast):
        """Testa processamento de textos grandes"""
        from src.api.main import app
        client = TestClient(app)
        
        # Criar texto grande (mas dentro dos limites)
        large_text = "Este é um teste de performance. " * 50  # ~1500 caracteres
        
        start_time = time.time()
        response = client.post("/query", json={
            "question": large_text,
            "k": 3
        })
        duration = time.time() - start_time
        
        assert response.status_code == 200
        # Mesmo com texto grande, deve processar em tempo razoável
        assert duration < 5.0, f"Processamento de texto grande muito lento: {duration:.2f}s"
    
    @pytest.mark.parametrize("k_value", [1, 5, 10, 20])
    def test_retrieval_scaling(self, mock_pipeline_fast, k_value):
        """Testa como performance escala com diferentes valores de k"""
        from src.api.main import app
        client = TestClient(app)
        
        start_time = time.time()
        response = client.post("/query", json={
            "question": "Test scaling query",
            "k": k_value
        })
        duration = time.time() - start_time
        
        assert response.status_code == 200
        
        # Performance deve escalar linearmente (aproximadamente)
        expected_max_time = 0.5 + (k_value * 0.1)  # Base + escalamento
        assert duration < expected_max_time, \
            f"Query com k={k_value} muito lenta: {duration:.2f}s (máximo: {expected_max_time:.2f}s)"


@pytest.mark.performance
class TestPerformanceRegression:
    """Testes para detectar regressão de performance"""
    
    PERFORMANCE_BASELINES = {
        "query_response_time": 2.0,  # segundos
        "health_check_time": 0.5,   # segundos
        "concurrent_queries_avg": 3.0,  # segundos
        "memory_increase_limit": 100,  # MB
        "cpu_average_limit": 80,    # %
    }
    
    def test_performance_baselines(self, mock_pipeline_fast):
        """Testa se performance está dentro dos baselines estabelecidos"""
        from src.api.main import app
        client = TestClient(app)
        
        results = {}
        
        # Teste 1: Query response time
        start_time = time.time()
        response = client.post("/query", json={"question": "Baseline test", "k": 5})
        results["query_response_time"] = time.time() - start_time
        assert response.status_code == 200
        
        # Teste 2: Health check time
        start_time = time.time()
        response = client.get("/health")
        results["health_check_time"] = time.time() - start_time
        assert response.status_code == 200
        
        # Teste 3: Memory baseline
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Fazer algumas queries
        for i in range(5):
            client.post("/query", json={"question": f"Memory baseline {i}", "k": 3})
        
        final_memory = process.memory_info().rss / 1024 / 1024
        results["memory_increase"] = final_memory - initial_memory
        
        # Verificar todos os baselines
        failures = []
        for metric, value in results.items():
            baseline = self.PERFORMANCE_BASELINES.get(metric)
            if baseline and value > baseline:
                failures.append(f"{metric}: {value:.2f} > {baseline}")
        
        assert not failures, f"Performance regression detectada: {', '.join(failures)}"
    
    def test_performance_trends(self, mock_pipeline_fast):
        """Testa tendências de performance ao longo do tempo"""
        from src.api.main import app
        client = TestClient(app)
        
        # Simular múltiplas execuções para detectar degradação
        durations = []
        
        for i in range(10):
            start_time = time.time()
            response = client.post("/query", json={
                "question": f"Trend test {i}",
                "k": 5
            })
            duration = time.time() - start_time
            durations.append(duration)
            assert response.status_code == 200
        
        # Verificar se não há degradação significativa
        first_half_avg = sum(durations[:5]) / 5
        second_half_avg = sum(durations[5:]) / 5
        
        degradation = (second_half_avg - first_half_avg) / first_half_avg * 100
        
        # Não deve haver degradação > 50%
        assert degradation < 50, f"Degradação de performance detectada: {degradation:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"]) 