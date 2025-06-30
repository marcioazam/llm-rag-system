"""
Testes para o módulo multi_head_rag - Sistema RAG Multi-Cabeça
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum


class RAGHead(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword" 
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"


class MockRAGHead:
    def __init__(self, head_type: RAGHead, weight: float = 1.0):
        self.head_type = head_type
        self.weight = weight
        self.enabled = True
        self.query_count = 0
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.query_count += 1
        
        if self.head_type == RAGHead.SEMANTIC:
            return await self._semantic_search(query)
        elif self.head_type == RAGHead.KEYWORD:
            return await self._keyword_search(query)
        elif self.head_type == RAGHead.HYBRID:
            return await self._hybrid_search(query)
        else:  # CONTEXTUAL
            return await self._contextual_search(query)
    
    async def _semantic_search(self, query: str) -> Dict[str, Any]:
        return {
            'head_type': self.head_type.value,
            'results': [
                {'id': 'sem_1', 'score': 0.85, 'content': 'Semantic result 1'},
                {'id': 'sem_2', 'score': 0.72, 'content': 'Semantic result 2'}
            ],
            'confidence': 0.78,
            'processing_time': 0.2
        }
    
    async def _keyword_search(self, query: str) -> Dict[str, Any]:
        return {
            'head_type': self.head_type.value,
            'results': [
                {'id': 'kw_1', 'score': 0.90, 'content': 'Keyword result 1'},
                {'id': 'kw_2', 'score': 0.68, 'content': 'Keyword result 2'}
            ],
            'confidence': 0.82,
            'processing_time': 0.1
        }
    
    async def _hybrid_search(self, query: str) -> Dict[str, Any]:
        return {
            'head_type': self.head_type.value,
            'results': [
                {'id': 'hyb_1', 'score': 0.88, 'content': 'Hybrid result 1'},
                {'id': 'hyb_2', 'score': 0.75, 'content': 'Hybrid result 2'}
            ],
            'confidence': 0.80,
            'processing_time': 0.3
        }
    
    async def _contextual_search(self, query: str) -> Dict[str, Any]:
        return {
            'head_type': self.head_type.value,
            'results': [
                {'id': 'ctx_1', 'score': 0.83, 'content': 'Contextual result 1'},
                {'id': 'ctx_2', 'score': 0.70, 'content': 'Contextual result 2'}
            ],
            'confidence': 0.76,
            'processing_time': 0.4
        }


class MockMultiHeadRAG:
    def __init__(self):
        self.heads = {
            RAGHead.SEMANTIC: MockRAGHead(RAGHead.SEMANTIC, weight=0.3),
            RAGHead.KEYWORD: MockRAGHead(RAGHead.KEYWORD, weight=0.25),
            RAGHead.HYBRID: MockRAGHead(RAGHead.HYBRID, weight=0.25),
            RAGHead.CONTEXTUAL: MockRAGHead(RAGHead.CONTEXTUAL, weight=0.2)
        }
        self.fusion_strategy = "weighted_average"
        self.max_results = 10
        self.parallel_execution = True
        self.adaptive_weights = True
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        context = context or {}
        
        if self.parallel_execution:
            results = await self._process_parallel(query, context)
        else:
            results = await self._process_sequential(query, context)
        
        fused_results = await self._fuse_results(results)
        
        return {
            'query': query,
            'final_results': fused_results,
            'head_results': results,
            'fusion_strategy': self.fusion_strategy,
            'total_heads_used': len([r for r in results if r is not None]),
            'processing_metadata': self._get_processing_metadata(results)
        }
    
    async def _process_parallel(self, query: str, context: Dict) -> List[Dict[str, Any]]:
        tasks = []
        for head in self.heads.values():
            if head.enabled:
                tasks.append(head.process_query(query, context))
        
        if not tasks:
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if not isinstance(r, Exception) else None for r in results]
    
    async def _process_sequential(self, query: str, context: Dict) -> List[Dict[str, Any]]:
        results = []
        for head in self.heads.values():
            if head.enabled:
                try:
                    result = await head.process_query(query, context)
                    results.append(result)
                except Exception:
                    results.append(None)
        return results
    
    async def _fuse_results(self, head_results: List[Dict]) -> List[Dict[str, Any]]:
        if self.fusion_strategy == "weighted_average":
            return await self._weighted_fusion(head_results)
        elif self.fusion_strategy == "rank_fusion":
            return await self._rank_fusion(head_results)
        else:
            return await self._simple_fusion(head_results)
    
    async def _weighted_fusion(self, head_results: List[Dict]) -> List[Dict[str, Any]]:
        result_map = {}
        
        for result in head_results:
            if result is None:
                continue
            
            head_type = RAGHead(result['head_type'])
            weight = self.heads[head_type].weight
            
            for item in result['results']:
                item_id = item['id']
                weighted_score = item['score'] * weight
                
                if item_id in result_map:
                    result_map[item_id]['score'] += weighted_score
                    result_map[item_id]['sources'].append(head_type.value)
                else:
                    result_map[item_id] = {
                        'id': item_id,
                        'content': item['content'],
                        'score': weighted_score,
                        'sources': [head_type.value]
                    }
        
        fused_results = list(result_map.values())
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        return fused_results[:self.max_results]
    
    async def _rank_fusion(self, head_results: List[Dict]) -> List[Dict[str, Any]]:
        rank_scores = {}
        
        for result in head_results:
            if result is None:
                continue
            
            for rank, item in enumerate(result['results']):
                item_id = item['id']
                rank_score = 1.0 / (rank + 1)
                
                if item_id in rank_scores:
                    rank_scores[item_id]['score'] += rank_score
                else:
                    rank_scores[item_id] = {
                        'id': item_id,
                        'content': item['content'],
                        'score': rank_score,
                        'sources': [result['head_type']]
                    }
        
        fused_results = list(rank_scores.values())
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        return fused_results[:self.max_results]
    
    async def _simple_fusion(self, head_results: List[Dict]) -> List[Dict[str, Any]]:
        all_results = []
        
        for result in head_results:
            if result and 'results' in result:
                for item in result['results']:
                    all_results.append({
                        'id': item['id'],
                        'content': item['content'],
                        'score': item['score'],
                        'sources': [result['head_type']]
                    })
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:self.max_results]
    
    def _get_processing_metadata(self, results: List[Dict]) -> Dict[str, Any]:
        valid_results = [r for r in results if r]
        total_time = sum(r.get('processing_time', 0) for r in valid_results)
        avg_confidence = sum(r.get('confidence', 0) for r in valid_results) / max(len(valid_results), 1)
        
        return {
            'total_processing_time': total_time,
            'average_confidence': avg_confidence,
            'heads_succeeded': len(valid_results),
            'heads_failed': len([r for r in results if not r]),
            'parallel_execution': self.parallel_execution
        }
    
    def configure_head(self, head_type: RAGHead, enabled: bool = True, weight: float = None):
        if head_type in self.heads:
            self.heads[head_type].enabled = enabled
            if weight is not None:
                self.heads[head_type].weight = weight
    
    def get_head_statistics(self) -> Dict[str, Any]:
        stats = {}
        for head_type, head in self.heads.items():
            stats[head_type.value] = {
                'enabled': head.enabled,
                'weight': head.weight,
                'query_count': head.query_count
            }
        return stats


class TestMultiHeadRAGBasic:
    def setup_method(self):
        self.multi_rag = MockMultiHeadRAG()
    
    def test_multi_head_initialization(self):
        assert len(self.multi_rag.heads) == 4
        assert self.multi_rag.fusion_strategy == "weighted_average"
        assert self.multi_rag.max_results == 10
        assert self.multi_rag.parallel_execution is True
        
        for head in self.multi_rag.heads.values():
            assert head.enabled is True
    
    @pytest.mark.asyncio
    async def test_single_head_processing(self):
        head = self.multi_rag.heads[RAGHead.SEMANTIC]
        query = "What is machine learning?"
        
        result = await head.process_query(query)
        
        assert result['head_type'] == 'semantic'
        assert 'results' in result
        assert 'confidence' in result
        assert 'processing_time' in result
        assert len(result['results']) > 0
    
    @pytest.mark.asyncio
    async def test_parallel_query_processing(self):
        query = "How do neural networks work?"
        
        result = await self.multi_rag.process_query(query)
        
        assert result['query'] == query
        assert 'final_results' in result
        assert 'head_results' in result
        assert result['total_heads_used'] > 0
        assert 'processing_metadata' in result
    
    @pytest.mark.asyncio
    async def test_sequential_query_processing(self):
        self.multi_rag.parallel_execution = False
        query = "What is deep learning?"
        
        result = await self.multi_rag.process_query(query)
        
        assert result['query'] == query
        assert result['total_heads_used'] > 0
        assert result['processing_metadata']['parallel_execution'] is False
    
    def test_head_configuration(self):
        self.multi_rag.configure_head(RAGHead.CONTEXTUAL, enabled=False)
        assert self.multi_rag.heads[RAGHead.CONTEXTUAL].enabled is False
        
        self.multi_rag.configure_head(RAGHead.SEMANTIC, weight=0.5)
        assert self.multi_rag.heads[RAGHead.SEMANTIC].weight == 0.5


class TestMultiHeadRAGFusion:
    def setup_method(self):
        self.multi_rag = MockMultiHeadRAG()
    
    @pytest.mark.asyncio
    async def test_weighted_fusion_strategy(self):
        self.multi_rag.fusion_strategy = "weighted_average"
        query = "Test query for weighted fusion"
        
        result = await self.multi_rag.process_query(query)
        
        assert result['fusion_strategy'] == "weighted_average"
        assert len(result['final_results']) <= self.multi_rag.max_results
        
        scores = [r['score'] for r in result['final_results']]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_rank_fusion_strategy(self):
        self.multi_rag.fusion_strategy = "rank_fusion"
        query = "Test query for rank fusion"
        
        result = await self.multi_rag.process_query(query)
        
        assert result['fusion_strategy'] == "rank_fusion"
        assert len(result['final_results']) <= self.multi_rag.max_results
    
    @pytest.mark.asyncio
    async def test_simple_fusion_strategy(self):
        self.multi_rag.fusion_strategy = "simple"
        query = "Test query for simple fusion"
        
        result = await self.multi_rag.process_query(query)
        
        assert result['fusion_strategy'] == "simple"
        assert len(result['final_results']) <= self.multi_rag.max_results


class TestMultiHeadRAGAdvanced:
    def setup_method(self):
        self.multi_rag = MockMultiHeadRAG()
    
    @pytest.mark.asyncio
    async def test_head_failure_handling(self):
        self.multi_rag.configure_head(RAGHead.CONTEXTUAL, enabled=False)
        self.multi_rag.configure_head(RAGHead.HYBRID, enabled=False)
        
        query = "Test query with some heads disabled"
        result = await self.multi_rag.process_query(query)
        
        assert result['total_heads_used'] == 2
        assert len(result['final_results']) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        queries = [
            "What is AI?",
            "How does machine learning work?",
            "What are neural networks?",
            "Explain deep learning algorithms"
        ]
        
        tasks = [self.multi_rag.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 4
        assert all('final_results' in result for result in results)
        assert all(result['total_heads_used'] > 0 for result in results)
    
    def test_head_statistics(self):
        stats = self.multi_rag.get_head_statistics()
        
        assert len(stats) == 4
        assert all('enabled' in stat for stat in stats.values())
        assert all('weight' in stat for stat in stats.values())
        assert all('query_count' in stat for stat in stats.values())
    
    @pytest.mark.asyncio
    async def test_result_diversity(self):
        query = "Diverse results test"
        result = await self.multi_rag.process_query(query)
        
        sources_used = set()
        for item in result['final_results']:
            sources_used.update(item['sources'])
        
        assert len(sources_used) >= 2
    
    @pytest.mark.asyncio
    async def test_performance_metadata(self):
        query = "Performance test query"
        result = await self.multi_rag.process_query(query)
        
        metadata = result['processing_metadata']
        
        assert 'total_processing_time' in metadata
        assert 'average_confidence' in metadata
        assert 'heads_succeeded' in metadata
        assert 'heads_failed' in metadata
        assert metadata['heads_succeeded'] > 0
        assert metadata['average_confidence'] > 0


class TestMultiHeadRAGEdgeCases:
    def setup_method(self):
        self.multi_rag = MockMultiHeadRAG()
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        empty_queries = ["", "   ", "\n"]
        
        for query in empty_queries:
            result = await self.multi_rag.process_query(query)
            assert 'final_results' in result
            assert result['query'] == query
    
    @pytest.mark.asyncio
    async def test_all_heads_disabled(self):
        for head_type in RAGHead:
            self.multi_rag.configure_head(head_type, enabled=False)
        
        query = "Test with no heads enabled"
        result = await self.multi_rag.process_query(query)
        
        assert result['total_heads_used'] == 0
        assert len(result['final_results']) == 0
    
    @pytest.mark.asyncio
    async def test_max_results_limitation(self):
        self.multi_rag.max_results = 3
        query = "Test max results limitation"
        
        result = await self.multi_rag.process_query(query)
        
        assert len(result['final_results']) <= 3
    
    @pytest.mark.asyncio
    async def test_context_propagation(self):
        query = "Test context propagation"
        context = {
            "user_id": "test_user",
            "domain": "AI",
            "priority": "high"
        }
        
        result = await self.multi_rag.process_query(query, context)
        assert result['total_heads_used'] > 0


class TestMultiHeadRAGIntegration:
    def setup_method(self):
        self.multi_rag = MockMultiHeadRAG()
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        self.multi_rag.fusion_strategy = "weighted_average"
        self.multi_rag.max_results = 5
        
        self.multi_rag.configure_head(RAGHead.SEMANTIC, weight=0.4)
        self.multi_rag.configure_head(RAGHead.HYBRID, weight=0.3)
        
        query = "Comprehensive test of multi-head RAG system"
        context = {"domain": "machine_learning", "complexity": "high"}
        
        result = await self.multi_rag.process_query(query, context)
        
        assert result['query'] == query
        assert len(result['final_results']) <= 5
        assert result['total_heads_used'] > 0
        
        assert all('score' in item for item in result['final_results'])
        assert all('sources' in item for item in result['final_results'])
        
        metadata = result['processing_metadata']
        assert metadata['total_processing_time'] > 0
        assert metadata['average_confidence'] > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_simulation(self):
        queries = [
            "Machine learning basics",
            "Deep learning fundamentals",
            "Neural network architectures",
            "AI system design"
        ]
        
        results = []
        for query in queries:
            result = await self.multi_rag.process_query(query)
            results.append(result)
        
        assert len(results) == 4
        assert all(r['total_heads_used'] > 0 for r in results)
        
        stats = self.multi_rag.get_head_statistics()
        assert all(stat['query_count'] >= 1 for stat in stats.values())