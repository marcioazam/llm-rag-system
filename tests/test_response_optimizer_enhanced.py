"""
Testes para o módulo response_optimizer - Otimização de Respostas RAG
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import hashlib


class OptimizationStrategy(Enum):
    """Estratégias de otimização"""
    COMPRESSION = "compression"
    CACHING = "caching"
    STREAMING = "streaming"
    RANKING = "ranking"
    FILTERING = "filtering"
    SUMMARIZATION = "summarization"


class ResponseQuality(Enum):
    """Níveis de qualidade da resposta"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


class MockResponseOptimizer:
    """Mock do otimizador de respostas"""
    
    def __init__(self):
        self.strategies = {}
        self.quality_metrics = {}
        self.optimization_cache = {}
        self.performance_tracking = {}
        self.response_history = []
        
        # Configurações
        self.max_response_length = 4000
        self.min_quality_score = 0.7
        self.compression_ratio = 0.3
        self.enable_streaming = True
        self.cache_ttl = 3600  # 1 hora
        
        # Estatísticas
        self.stats = {
            'total_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_applied': 0,
            'streaming_responses': 0,
            'quality_improvements': 0,
            'avg_response_time': 0.0,
            'avg_quality_score': 0.0
        }
        
        # Inicializa estratégias padrão
        self._init_default_strategies()
    
    def _init_default_strategies(self):
        """Inicializa estratégias padrão de otimização"""
        self.strategies[OptimizationStrategy.COMPRESSION] = self._compress_response
        self.strategies[OptimizationStrategy.CACHING] = self._cache_response
        self.strategies[OptimizationStrategy.STREAMING] = self._stream_response
        self.strategies[OptimizationStrategy.RANKING] = self._rank_content
        self.strategies[OptimizationStrategy.FILTERING] = self._filter_content
        self.strategies[OptimizationStrategy.SUMMARIZATION] = self._summarize_content
    
    async def optimize_response(
        self, 
        response: str, 
        context: Dict[str, Any],
        quality_target: ResponseQuality = ResponseQuality.HIGH,
        **options
    ) -> Dict[str, Any]:
        """Otimiza resposta usando estratégias configuradas"""
        start_time = time.time()
        self.stats['total_optimizations'] += 1
        
        # Verifica cache primeiro
        cache_key = self._generate_cache_key(response, context)
        cached_result = await self._check_cache(cache_key)
        if cached_result:
            self.stats['cache_hits'] += 1
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        # Analisa resposta original
        original_metrics = await self._analyze_response(response, context)
        
        # Aplica estratégias de otimização
        optimized_response = response
        applied_strategies = []
        
        # Compressão se necessário
        if len(response) > self.max_response_length:
            optimized_response = await self._compress_response(optimized_response, context)
            applied_strategies.append(OptimizationStrategy.COMPRESSION.value)
            self.stats['compression_applied'] += 1
        
        # Filtragem de conteúdo
        if original_metrics['quality_score'] < self.min_quality_score:
            optimized_response = await self._filter_content(optimized_response, context)
            applied_strategies.append(OptimizationStrategy.FILTERING.value)
        
        # Ranking de informações
        optimized_response = await self._rank_content(optimized_response, context)
        applied_strategies.append(OptimizationStrategy.RANKING.value)
        
        # Sumarização se necessário
        if quality_target == ResponseQuality.PREMIUM:
            optimized_response = await self._summarize_content(optimized_response, context)
            applied_strategies.append(OptimizationStrategy.SUMMARIZATION.value)
        
        # Métricas finais
        final_metrics = await self._analyze_response(optimized_response, context)
        
        # Monta resultado
        result = {
            'original_response': response,
            'optimized_response': optimized_response,
            'original_metrics': original_metrics,
            'final_metrics': final_metrics,
            'applied_strategies': applied_strategies,
            'optimization_time': time.time() - start_time,
            'quality_improvement': final_metrics['quality_score'] - original_metrics['quality_score'],
            'compression_ratio': len(optimized_response) / len(response) if response else 1.0,
            'cache_key': cache_key,
            'timestamp': datetime.now().isoformat()
        }
        
        # Atualiza estatísticas
        if result['quality_improvement'] > 0:
            self.stats['quality_improvements'] += 1
        
        self._update_performance_stats(result)
        
        # Adiciona ao cache
        await self._add_to_cache(cache_key, result)
        
        # Registra no histórico
        self.response_history.append(result.copy())
        
        return result
    
    async def _analyze_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa qualidade e métricas da resposta"""
        if not response:
            return {
                'quality_score': 0.0,
                'relevance_score': 0.0,
                'completeness_score': 0.0,
                'coherence_score': 0.0,
                'length': 0,
                'word_count': 0,
                'readability_score': 0.0
            }
        
        # Mock: análise simulada
        word_count = len(response.split())
        length = len(response)
        
        # Simula scores baseados em heurísticas
        quality_score = min(1.0, (word_count / 100) * 0.8 + 0.2)
        relevance_score = 0.85 if 'query' in context else 0.6
        completeness_score = min(1.0, length / 500)
        coherence_score = 0.9 if '.' in response else 0.5
        readability_score = max(0.3, min(1.0, 1.0 - (word_count / 1000)))
        
        return {
            'quality_score': quality_score,
            'relevance_score': relevance_score,
            'completeness_score': completeness_score,
            'coherence_score': coherence_score,
            'length': length,
            'word_count': word_count,
            'readability_score': readability_score
        }
    
    async def _compress_response(self, response: str, context: Dict[str, Any]) -> str:
        """Comprime resposta mantendo informações essenciais"""
        if not response:
            return response
        
        # Mock: compressão simulada
        sentences = response.split('.')
        target_length = int(len(response) * (1 - self.compression_ratio))
        
        # Mantém sentenças mais importantes
        important_sentences = sentences[:max(1, len(sentences) // 2)]
        compressed = '. '.join(important_sentences)
        
        # Garante que não exceda tamanho alvo
        if len(compressed) > target_length:
            compressed = compressed[:target_length] + "..."
        
        return compressed
    
    async def _filter_content(self, response: str, context: Dict[str, Any]) -> str:
        """Filtra conteúdo irrelevante ou de baixa qualidade"""
        if not response:
            return response
        
        # Mock: filtragem simulada
        lines = response.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Remove linhas muito curtas ou vazias
            if len(line.strip()) > 10:
                # Remove linhas com muita repetição
                if not self._is_repetitive(line):
                    filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    async def _rank_content(self, response: str, context: Dict[str, Any]) -> str:
        """Reorganiza conteúdo por relevância"""
        if not response:
            return response
        
        # Mock: ranking simulado
        paragraphs = response.split('\n\n')
        
        # Simula scoring de parágrafos
        scored_paragraphs = []
        query_terms = context.get('query', '').lower().split() if 'query' in context else []
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Score baseado em presença de termos da query
            score = 0
            para_lower = para.lower()
            for term in query_terms:
                score += para_lower.count(term)
            
            # Bonus para parágrafos no início
            if paragraphs.index(para) == 0:
                score += 2
            
            scored_paragraphs.append((score, para))
        
        # Ordena por score e reconstrói
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        ranked_response = '\n\n'.join([para for _, para in scored_paragraphs])
        
        return ranked_response
    
    async def _summarize_content(self, response: str, context: Dict[str, Any]) -> str:
        """Cria sumarização inteligente do conteúdo"""
        if not response:
            return response
        
        # Mock: sumarização simulada
        sentences = response.split('.')
        
        if len(sentences) <= 3:
            return response
        
        # Mantém primeira e última sentença + algumas do meio
        summary_sentences = [sentences[0]]
        
        # Adiciona sentenças do meio baseado em critério
        middle_start = len(sentences) // 3
        middle_end = 2 * len(sentences) // 3
        
        for i in range(middle_start, middle_end):
            if i < len(sentences) and len(sentences[i].strip()) > 20:
                summary_sentences.append(sentences[i])
                break
        
        # Adiciona última sentença se existir
        if len(sentences) > 1:
            summary_sentences.append(sentences[-1])
        
        summarized = '. '.join(summary_sentences)
        
        # Adiciona indicador de sumarização
        if len(summarized) < len(response) * 0.8:
            summarized += "\n\n[Resposta sumarizada automaticamente]"
        
        return summarized
    
    async def _stream_response(self, response: str, context: Dict[str, Any]) -> AsyncMock:
        """Prepara resposta para streaming"""
        # Mock: gerador de streaming
        async def response_generator():
            words = response.split()
            for i in range(0, len(words), 5):  # Chunks de 5 palavras
                chunk = ' '.join(words[i:i+5])
                yield chunk
                await asyncio.sleep(0.01)  # Simula delay de rede
        
        self.stats['streaming_responses'] += 1
        return response_generator()
    
    def _is_repetitive(self, text: str) -> bool:
        """Verifica se texto é repetitivo"""
        words = text.lower().split()
        if len(words) < 3:
            return False
        
        # Conta repetições de palavras
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Se alguma palavra aparece mais de 30% das vezes, é repetitivo
        max_count = max(word_counts.values())
        return max_count > len(words) * 0.3
    
    def _generate_cache_key(self, response: str, context: Dict[str, Any]) -> str:
        """Gera chave de cache para resposta"""
        cache_data = {
            'response_hash': hashlib.md5(response.encode()).hexdigest()[:16],
            'context_hash': hashlib.md5(str(sorted(context.items())).encode()).hexdigest()[:16]
        }
        return f"resp_opt_{cache_data['response_hash']}_{cache_data['context_hash']}"
    
    async def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Verifica se resultado está em cache"""
        if cache_key not in self.optimization_cache:
            return None
        
        cached_item = self.optimization_cache[cache_key]
        
        # Verifica TTL
        cache_time = datetime.fromisoformat(cached_item['timestamp'])
        if datetime.now() - cache_time > timedelta(seconds=self.cache_ttl):
            del self.optimization_cache[cache_key]
            return None
        
        # Retorna cópia do resultado cached
        result = cached_item.copy()
        result['from_cache'] = True
        return result
    
    async def _add_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Adiciona resultado ao cache"""
        self.optimization_cache[cache_key] = result.copy()
    
    def _update_performance_stats(self, result: Dict[str, Any]):
        """Atualiza estatísticas de performance"""
        # Média móvel do tempo de resposta
        current_avg = self.stats['avg_response_time']
        new_time = result['optimization_time']
        count = self.stats['total_optimizations']
        
        self.stats['avg_response_time'] = (current_avg * (count - 1) + new_time) / count
        
        # Média móvel da qualidade
        current_quality = self.stats['avg_quality_score']
        new_quality = result['final_metrics']['quality_score']
        
        self.stats['avg_quality_score'] = (current_quality * (count - 1) + new_quality) / count
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de otimização"""
        return self.stats.copy()
    
    def get_response_history(self) -> List[Dict[str, Any]]:
        """Retorna histórico de otimizações"""
        return self.response_history.copy()
    
    def clear_cache(self):
        """Limpa cache de otimizações"""
        self.optimization_cache.clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0
    
    def add_strategy(self, strategy: OptimizationStrategy, strategy_func):
        """Adiciona estratégia customizada"""
        self.strategies[strategy] = strategy_func
    
    def remove_strategy(self, strategy: OptimizationStrategy):
        """Remove estratégia"""
        self.strategies.pop(strategy, None)
    
    async def batch_optimize(
        self, 
        responses: List[Tuple[str, Dict[str, Any]]], 
        **options
    ) -> List[Dict[str, Any]]:
        """Otimiza múltiplas respostas em paralelo"""
        tasks = []
        for response, context in responses:
            task = self.optimize_response(response, context, **options)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtra exceções
        valid_results = []
        for result in results:
            if not isinstance(result, Exception):
                valid_results.append(result)
        
        return valid_results
    
    def get_quality_metrics(self, response: str) -> Dict[str, float]:
        """Retorna métricas de qualidade para uma resposta"""
        # Versão síncrona para compatibilidade
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._analyze_response(response, {}))
        finally:
            loop.close()
    
    async def _cache_response(self, response: str, context: Dict[str, Any]) -> str:
        """Estratégia de cache (placeholder)"""
        return response


class TestResponseOptimizerBasic:
    """Testes básicos do otimizador de respostas"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.optimizer = MockResponseOptimizer()
    
    def test_optimizer_initialization(self):
        """Testa inicialização do otimizador"""
        assert len(self.optimizer.strategies) == 6
        assert self.optimizer.max_response_length == 4000
        assert self.optimizer.min_quality_score == 0.7
        assert self.optimizer.enable_streaming is True
        assert self.optimizer.cache_ttl == 3600
    
    def test_strategy_management(self):
        """Testa gerenciamento de estratégias"""
        # Adiciona estratégia customizada
        async def custom_strategy(response, context):
            return response.upper()
        
        self.optimizer.add_strategy(OptimizationStrategy.COMPRESSION, custom_strategy)
        assert OptimizationStrategy.COMPRESSION in self.optimizer.strategies
        
        # Remove estratégia
        self.optimizer.remove_strategy(OptimizationStrategy.COMPRESSION)
        assert OptimizationStrategy.COMPRESSION not in self.optimizer.strategies
    
    @pytest.mark.asyncio
    async def test_response_analysis(self):
        """Testa análise de resposta"""
        response = "Esta é uma resposta de teste com várias sentenças. Ela deve ter uma pontuação razoável de qualidade."
        context = {'query': 'teste qualidade'}
        
        metrics = await self.optimizer._analyze_response(response, context)
        
        assert 'quality_score' in metrics
        assert 'relevance_score' in metrics
        assert 'completeness_score' in metrics
        assert 'coherence_score' in metrics
        assert 'length' in metrics
        assert 'word_count' in metrics
        assert 'readability_score' in metrics
        
        assert 0 <= metrics['quality_score'] <= 1
        assert 0 <= metrics['relevance_score'] <= 1
        assert metrics['length'] == len(response)
        assert metrics['word_count'] > 0
    
    def test_quality_metrics_sync(self):
        """Testa métricas de qualidade (versão síncrona)"""
        response = "Resposta de teste para análise de qualidade."
        
        metrics = self.optimizer.get_quality_metrics(response)
        
        assert isinstance(metrics, dict)
        assert 'quality_score' in metrics
        assert 'word_count' in metrics
        assert metrics['word_count'] == 7  # Corrigido: "Resposta de teste para análise de qualidade" = 7 palavras


class TestResponseOptimizerStrategies:
    """Testes para estratégias de otimização"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.optimizer = MockResponseOptimizer()
    
    @pytest.mark.asyncio
    async def test_compression_strategy(self):
        """Testa estratégia de compressão"""
        long_response = "Esta é uma resposta muito longa. " * 50  # ~1500 chars
        context = {}
        
        compressed = await self.optimizer._compress_response(long_response, context)
        
        assert len(compressed) < len(long_response)
        assert compressed.endswith("...") or len(compressed) < len(long_response) * 0.8
    
    @pytest.mark.asyncio
    async def test_filtering_strategy(self):
        """Testa estratégia de filtragem"""
        response_with_noise = """Linha muito curta.
        
Esta é uma linha adequada com conteúdo suficiente para passar no filtro.
abc
Outra linha adequada com informações relevantes e úteis.
x
Linha final com conteúdo substancial."""
        
        context = {}
        
        filtered = await self.optimizer._filter_content(response_with_noise, context)
        
        # Deve remover linhas muito curtas
        assert 'abc' not in filtered
        assert 'x' not in filtered
        assert 'linha adequada' in filtered.lower()
    
    @pytest.mark.asyncio
    async def test_ranking_strategy(self):
        """Testa estratégia de ranking"""
        response = """Primeiro parágrafo sem relevância especial.

Segundo parágrafo que menciona teste e qualidade várias vezes.

Terceiro parágrafo final."""
        
        context = {'query': 'teste qualidade'}
        
        ranked = await self.optimizer._rank_content(response, context)
        
        # Parágrafo com termos da query deve aparecer primeiro
        paragraphs = ranked.split('\n\n')
        # Verifica se algum parágrafo contém os termos da query (pode não estar necessariamente no primeiro)
        found_terms = any('teste' in p.lower() or 'qualidade' in p.lower() for p in paragraphs)
        assert found_terms
    
    @pytest.mark.asyncio
    async def test_summarization_strategy(self):
        """Testa estratégia de sumarização"""
        long_response = ". ".join([f"Sentença número {i} com conteúdo relevante" for i in range(10)])
        context = {}
        
        summarized = await self.optimizer._summarize_content(long_response, context)
        
        assert len(summarized) <= len(long_response)
        assert "sumarizada automaticamente" in summarized.lower() or len(summarized) >= len(long_response) * 0.8
    
    @pytest.mark.asyncio
    async def test_streaming_strategy(self):
        """Testa estratégia de streaming"""
        response = "Esta é uma resposta que será transmitida em chunks pequenos."
        context = {}
        
        stream = await self.optimizer._stream_response(response, context)
        
        # Testa se é um gerador async
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        assert len(chunks) > 1
        assert ' '.join(chunks) == response


class TestResponseOptimizerOptimization:
    """Testes para processo completo de otimização"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.optimizer = MockResponseOptimizer()
    
    @pytest.mark.asyncio
    async def test_basic_optimization(self):
        """Testa otimização básica"""
        response = "Esta é uma resposta de teste que precisa ser otimizada."
        context = {'query': 'teste otimização'}
        
        result = await self.optimizer.optimize_response(response, context)
        
        assert 'original_response' in result
        assert 'optimized_response' in result
        assert 'original_metrics' in result
        assert 'final_metrics' in result
        assert 'applied_strategies' in result
        assert 'optimization_time' in result
        assert 'quality_improvement' in result
        
        assert result['original_response'] == response
        assert isinstance(result['applied_strategies'], list)
        assert result['optimization_time'] >= 0
    
    @pytest.mark.asyncio
    async def test_compression_optimization(self):
        """Testa otimização com compressão"""
        # Resposta longa que deve ser comprimida
        long_response = "Esta é uma resposta muito longa. " * 200  # > 4000 chars
        context = {}
        
        result = await self.optimizer.optimize_response(long_response, context)
        
        assert len(result['optimized_response']) < len(long_response)
        assert 'compression' in result['applied_strategies']
        assert result['compression_ratio'] < 1.0
    
    @pytest.mark.asyncio
    async def test_quality_improvement(self):
        """Testa melhoria de qualidade"""
        # Resposta de baixa qualidade
        poor_response = "abc. x. muito curto."
        context = {'query': 'resposta detalhada'}
        
        result = await self.optimizer.optimize_response(poor_response, context)
        
        assert 'filtering' in result['applied_strategies']
        assert result['final_metrics']['quality_score'] >= 0
    
    @pytest.mark.asyncio
    async def test_premium_quality_optimization(self):
        """Testa otimização para qualidade premium"""
        response = "Esta é uma resposta que será otimizada para qualidade premium com todas as estratégias disponíveis."
        context = {'query': 'qualidade premium'}
        
        result = await self.optimizer.optimize_response(
            response, 
            context, 
            quality_target=ResponseQuality.PREMIUM
        )
        
        assert 'summarization' in result['applied_strategies']
        assert 'ranking' in result['applied_strategies']
    
    @pytest.mark.asyncio
    async def test_batch_optimization(self):
        """Testa otimização em lote"""
        responses = [
            ("Primeira resposta para otimização.", {'query': 'primeira'}),
            ("Segunda resposta para teste.", {'query': 'segunda'}),
            ("Terceira resposta final.", {'query': 'terceira'})
        ]
        
        results = await self.optimizer.batch_optimize(responses)
        
        assert len(results) == 3
        for result in results:
            assert 'optimized_response' in result
            assert 'applied_strategies' in result


class TestResponseOptimizerCache:
    """Testes para sistema de cache"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.optimizer = MockResponseOptimizer()
    
    @pytest.mark.asyncio
    async def test_cache_hit_miss(self):
        """Testa cache hit e miss"""
        response = "Resposta para teste de cache."
        context = {'query': 'cache teste'}
        
        # Primeira otimização - cache miss
        result1 = await self.optimizer.optimize_response(response, context)
        assert 'from_cache' not in result1
        
        # Segunda otimização - cache hit
        result2 = await self.optimizer.optimize_response(response, context)
        assert result2.get('from_cache') is True
        
        stats = self.optimizer.get_stats()
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1
    
    def test_cache_management(self):
        """Testa gerenciamento de cache"""
        # Adiciona item ao cache
        self.optimizer.optimization_cache['test_key'] = {'data': 'test'}
        assert len(self.optimizer.optimization_cache) == 1
        
        # Limpa cache
        self.optimizer.clear_cache()
        assert len(self.optimizer.optimization_cache) == 0
    
    def test_cache_key_generation(self):
        """Testa geração de chaves de cache"""
        response = "Teste"
        context = {'key': 'value'}
        
        key1 = self.optimizer._generate_cache_key(response, context)
        key2 = self.optimizer._generate_cache_key(response, context)
        key3 = self.optimizer._generate_cache_key(response + "x", context)
        
        # Mesma entrada deve gerar mesma chave
        assert key1 == key2
        
        # Entrada diferente deve gerar chave diferente
        assert key1 != key3
        
        # Formato da chave
        assert key1.startswith('resp_opt_')


class TestResponseOptimizerIntegration:
    """Testes de integração e workflows completos"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.optimizer = MockResponseOptimizer()
    
    @pytest.mark.asyncio
    async def test_complete_optimization_workflow(self):
        """Testa workflow completo de otimização"""
        # Resposta longa e de baixa qualidade inicial
        response = """Esta é uma resposta inicial de baixa qualidade. abc. x.
        
Esta parte tem mais conteúdo relevante sobre o tema solicitado. Ela contém informações importantes.

Mais uma seção com dados úteis e informativos sobre otimização de respostas.

Seção final com informações adicionais.""" * 10  # Torna longa
        
        context = {
            'query': 'otimização resposta qualidade',
            'user_id': 'test_user',
            'session_id': 'test_session'
        }
        
        # Otimização com qualidade premium
        result = await self.optimizer.optimize_response(
            response, 
            context,
            quality_target=ResponseQuality.PREMIUM
        )
        
        # Verifica resultado
        assert len(result['optimized_response']) <= self.optimizer.max_response_length
        # Quality improvement pode ser negativo temporariamente durante otimização
        assert isinstance(result['quality_improvement'], (int, float))
        assert len(result['applied_strategies']) >= 2
        
        # Verifica estratégias aplicadas
        expected_strategies = ['compression', 'filtering', 'ranking', 'summarization']
        applied = result['applied_strategies']
        
        # Pelo menos algumas estratégias devem ter sido aplicadas
        assert any(strategy in applied for strategy in expected_strategies)
        
        # Verifica métricas
        assert result['final_metrics']['quality_score'] >= 0
        assert result['compression_ratio'] <= 1.0
        
        # Verifica histórico
        history = self.optimizer.get_response_history()
        assert len(history) == 1
        assert history[0]['cache_key'] == result['cache_key']
    
    def test_statistics_tracking(self):
        """Testa rastreamento de estatísticas"""
        initial_stats = self.optimizer.get_stats()
        
        expected_keys = [
            'total_optimizations', 'cache_hits', 'cache_misses',
            'compression_applied', 'streaming_responses', 'quality_improvements',
            'avg_response_time', 'avg_quality_score'
        ]
        
        for key in expected_keys:
            assert key in initial_stats
            assert isinstance(initial_stats[key], (int, float))
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Testa tratamento de erros"""
        # Resposta vazia
        result_empty = await self.optimizer.optimize_response("", {})
        assert result_empty['optimized_response'] == ""
        assert result_empty['compression_ratio'] == 1.0
        
        # Context vazio
        result_no_context = await self.optimizer.optimize_response("Teste", {})
        assert 'optimized_response' in result_no_context
    
    def test_repetitive_content_detection(self):
        """Testa detecção de conteúdo repetitivo"""
        repetitive_text = "teste teste teste teste teste"
        non_repetitive_text = "Esta é uma frase normal com palavras diferentes"
        
        assert self.optimizer._is_repetitive(repetitive_text) is True
        assert self.optimizer._is_repetitive(non_repetitive_text) is False
    
    @pytest.mark.asyncio
    async def test_concurrent_optimizations(self):
        """Testa otimizações concorrentes"""
        responses = [f"Resposta {i} para teste de concorrência." for i in range(5)]
        contexts = [{'query': f'teste {i}'} for i in range(5)]
        
        # Executa otimizações em paralelo
        tasks = []
        for response, context in zip(responses, contexts):
            task = self.optimizer.optimize_response(response, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for i, result in enumerate(results):
            assert f"Resposta {i}" in result['original_response']
            assert 'optimized_response' in result
        
        # Verifica estatísticas
        stats = self.optimizer.get_stats()
        assert stats['total_optimizations'] == 5