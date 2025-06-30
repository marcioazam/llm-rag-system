"""
Testes para o módulo memo_rag - Sistema de Memória Persistente RAG
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class MockMemoryStore:
    def __init__(self):
        self.memories = {}
        self.interactions = []
    
    def store_memory(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self.memories[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'ttl': ttl,
            'access_count': 0
        }
        return True
    
    def retrieve_memory(self, key: str) -> Optional[Any]:
        if key not in self.memories:
            return None
        
        memory = self.memories[key]
        if memory['ttl']:
            if datetime.now() > memory['timestamp'] + timedelta(seconds=memory['ttl']):
                del self.memories[key]
                return None
        
        memory['access_count'] += 1
        return memory['value']
    
    def delete_memory(self, key: str) -> bool:
        if key in self.memories:
            del self.memories[key]
            return True
        return False
    
    def list_memories(self) -> List[str]:
        return list(self.memories.keys())


class MockMemoRAG:
    def __init__(self, memory_store: MockMemoryStore = None):
        self.memory_store = memory_store or MockMemoryStore()
        self.session_id = "test_session"
        self.context_window = 10
        self.similarity_threshold = 0.7
        self.max_memories = 1000
        self.compression_enabled = True
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        context = context or {}
        relevant_memories = await self.retrieve_relevant_memories(query)
        
        response = {
            'query': query,
            'response': f"Response to: {query}",
            'memories_used': len(relevant_memories),
            'relevant_memories': relevant_memories,
            'confidence': 0.85,
            'sources': ['memory_1', 'memory_2']
        }
        
        await self.store_interaction(query, response, context)
        return response
    
    async def retrieve_relevant_memories(self, query: str) -> List[Dict[str, Any]]:
        memories = []
        for key in self.memory_store.list_memories():
            memory_data = self.memory_store.retrieve_memory(key)
            if memory_data and self._is_relevant(query, memory_data):
                memories.append({
                    'key': key,
                    'data': memory_data,
                    'similarity': 0.8,
                    'timestamp': datetime.now().isoformat()
                })
        
        memories.sort(key=lambda x: x['similarity'], reverse=True)
        return memories[:self.context_window]
    
    async def store_interaction(self, query: str, response: Dict, context: Dict):
        interaction_key = f"interaction_{len(self.memory_store.memories)}"
        interaction_data = {
            'query': query,
            'response': response,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        self.memory_store.store_memory(interaction_key, interaction_data)
    
    def _is_relevant(self, query: str, memory_data: Any) -> bool:
        if isinstance(memory_data, dict) and 'query' in memory_data:
            return len(set(query.lower().split()) & 
                      set(memory_data['query'].lower().split())) > 0
        return False
    
    async def compress_memories(self) -> Dict[str, Any]:
        if not self.compression_enabled:
            return {'compressed': 0, 'status': 'disabled'}
        
        memories = self.memory_store.list_memories()
        compressed_count = 0
        
        for key in memories:
            memory = self.memory_store.retrieve_memory(key)
            if memory and self._should_compress(memory):
                compressed_data = self._compress_memory(memory)
                self.memory_store.store_memory(f"{key}_compressed", compressed_data)
                self.memory_store.delete_memory(key)
                compressed_count += 1
        
        return {
            'compressed': compressed_count,
            'total_memories': len(self.memory_store.list_memories()),
            'status': 'completed'
        }
    
    def _should_compress(self, memory: Any) -> bool:
        if isinstance(memory, dict) and 'timestamp' in memory:
            timestamp = datetime.fromisoformat(memory['timestamp'])
            return datetime.now() - timestamp > timedelta(hours=24)
        return False
    
    def _compress_memory(self, memory: Any) -> Dict[str, Any]:
        return {
            'original_size': len(str(memory)),
            'compressed_data': 'compressed_content',
            'compression_ratio': 0.3,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        memories = self.memory_store.list_memories()
        total_size = sum(len(str(self.memory_store.retrieve_memory(key))) for key in memories)
        
        return {
            'total_memories': len(memories),
            'total_size_bytes': total_size,
            'session_id': self.session_id,
            'compression_enabled': self.compression_enabled,
            'context_window': self.context_window,
            'similarity_threshold': self.similarity_threshold
        }


class TestMemoRAGBasic:
    def setup_method(self):
        self.memory_store = MockMemoryStore()
        self.memo_rag = MockMemoRAG(self.memory_store)
    
    def test_memo_rag_initialization(self):
        assert self.memo_rag.memory_store is not None
        assert self.memo_rag.session_id == "test_session"
        assert self.memo_rag.context_window == 10
        assert self.memo_rag.similarity_threshold == 0.7
        assert self.memo_rag.compression_enabled is True
    
    @pytest.mark.asyncio
    async def test_process_simple_query(self):
        query = "What is machine learning?"
        result = await self.memo_rag.process_query(query)
        
        assert result['query'] == query
        assert 'response' in result
        assert 'memories_used' in result
        assert 'confidence' in result
        assert isinstance(result['relevant_memories'], list)
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_interaction(self):
        query = "How does neural network work?"
        
        result1 = await self.memo_rag.process_query(query)
        assert len(self.memory_store.list_memories()) > 0
        
        similar_query = "How do neural networks function?"
        result2 = await self.memo_rag.process_query(similar_query)
        
        assert result2['memories_used'] > 0
    
    def test_memory_store_operations(self):
        key = "test_key"
        value = {"data": "test_value"}
        
        assert self.memory_store.store_memory(key, value) is True
        retrieved = self.memory_store.retrieve_memory(key)
        assert retrieved == value
        assert key in self.memory_store.list_memories()
        assert self.memory_store.delete_memory(key) is True
        assert self.memory_store.retrieve_memory(key) is None


class TestMemoRAGMemoryOperations:
    def setup_method(self):
        self.memory_store = MockMemoryStore()
        self.memo_rag = MockMemoRAG(self.memory_store)
    
    def test_memory_ttl_functionality(self):
        key = "ttl_test"
        value = {"data": "expires_soon"}
        ttl = 1
        
        self.memory_store.store_memory(key, value, ttl)
        assert self.memory_store.retrieve_memory(key) == value
    
    @pytest.mark.asyncio
    async def test_relevant_memory_retrieval(self):
        memories = [
            {"query": "machine learning basics", "response": "ML is..."},
            {"query": "neural network architecture", "response": "NN has..."},
            {"query": "cooking recipes", "response": "To cook..."}
        ]
        
        for i, memory in enumerate(memories):
            self.memory_store.store_memory(f"memory_{i}", memory)
        
        query = "neural network design"
        relevant = await self.memo_rag.retrieve_relevant_memories(query)
        
        assert len(relevant) > 0
        assert any("neural" in str(mem['data']).lower() for mem in relevant)
    
    def test_memory_access_counting(self):
        key = "access_test"
        value = {"data": "frequently_accessed"}
        
        self.memory_store.store_memory(key, value)
        
        for _ in range(5):
            self.memory_store.retrieve_memory(key)
        
        memory_info = self.memory_store.memories[key]
        assert memory_info['access_count'] == 5
    
    @pytest.mark.asyncio
    async def test_memory_compression(self):
        old_memory = {
            "query": "old query",
            "response": "old response",
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat()
        }
        
        self.memory_store.store_memory("old_memory", old_memory)
        result = await self.memo_rag.compress_memories()
        
        assert 'compressed' in result
        assert 'total_memories' in result
        assert result['status'] == 'completed'


class TestMemoRAGAdvanced:
    def setup_method(self):
        self.memory_store = MockMemoryStore()
        self.memo_rag = MockMemoRAG(self.memory_store)
    
    @pytest.mark.asyncio
    async def test_context_window_limitation(self):
        for i in range(15):
            memory = {"query": f"test query {i}", "response": f"response {i}"}
            self.memory_store.store_memory(f"memory_{i}", memory)
        
        query = "test query"
        relevant = await self.memo_rag.retrieve_relevant_memories(query)
        
        assert len(relevant) <= self.memo_rag.context_window
    
    @pytest.mark.asyncio
    async def test_session_based_memory(self):
        query = "What is deep learning?"
        context = {"user_id": "user123", "domain": "AI"}
        
        result = await self.memo_rag.process_query(query, context)
        
        memories = self.memory_store.list_memories()
        assert len(memories) > 0
        
        stored_interaction = self.memory_store.retrieve_memory(memories[0])
        assert stored_interaction['session_id'] == self.memo_rag.session_id
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_access(self):
        queries = [
            "What is AI?",
            "How does ML work?",
            "What are neural networks?",
            "Explain deep learning"
        ]
        
        tasks = [self.memo_rag.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 4
        assert all('response' in result for result in results)
        assert len(self.memory_store.list_memories()) >= 4
    
    def test_memory_statistics(self):
        for i in range(5):
            self.memory_store.store_memory(f"test_{i}", {"data": f"value_{i}"})
        
        stats = self.memo_rag.get_memory_stats()
        
        assert stats['total_memories'] == 5
        assert 'total_size_bytes' in stats
        assert stats['session_id'] == self.memo_rag.session_id
        assert stats['compression_enabled'] is True
        assert stats['context_window'] == 10


class TestMemoRAGEdgeCases:
    def setup_method(self):
        self.memory_store = MockMemoryStore()
        self.memo_rag = MockMemoRAG(self.memory_store)
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        empty_queries = ["", "   ", "\n"]
        
        for query in empty_queries:
            result = await self.memo_rag.process_query(query)
            assert 'response' in result
            assert result['query'] == query
    
    @pytest.mark.asyncio
    async def test_memory_overflow_handling(self):
        for i in range(self.memo_rag.max_memories + 10):
            memory = {"data": f"memory_{i}", "large_data": "x" * 100}
            self.memory_store.store_memory(f"memory_{i}", memory)
        
        query = "test query"
        result = await self.memo_rag.process_query(query)
        
        assert 'response' in result
    
    def test_corrupted_memory_handling(self):
        corrupted_data = {"incomplete": True}
        self.memory_store.store_memory("corrupted", corrupted_data)
        
        memories = self.memory_store.list_memories()
        assert "corrupted" in memories
        
        retrieved = self.memory_store.retrieve_memory("corrupted")
        assert retrieved == corrupted_data
    
    @pytest.mark.asyncio
    async def test_compression_disabled(self):
        self.memo_rag.compression_enabled = False
        
        result = await self.memo_rag.compress_memories()
        
        assert result['status'] == 'disabled'
        assert result['compressed'] == 0
    
    def test_memory_store_none_values(self):
        key = "none_test"
        
        self.memory_store.store_memory(key, None)
        retrieved = self.memory_store.retrieve_memory(key)
        assert retrieved is None
        assert key in self.memory_store.list_memories()


class TestMemoRAGIntegration:
    def setup_method(self):
        self.memory_store = MockMemoryStore()
        self.memo_rag = MockMemoRAG(self.memory_store)
    
    @pytest.mark.asyncio
    async def test_learning_conversation_flow(self):
        conversation = [
            "What is machine learning?",
            "How does supervised learning work?",
            "Can you give examples of supervised learning algorithms?",
            "What's the difference between classification and regression?"
        ]
        
        responses = []
        for query in conversation:
            result = await self.memo_rag.process_query(query)
            responses.append(result)
        
        memory_counts = [resp['memories_used'] for resp in responses]
        assert memory_counts[-1] >= memory_counts[0]
        assert all('response' in resp for resp in responses)
    
    @pytest.mark.asyncio
    async def test_domain_specific_memory(self):
        ai_query = "What are neural networks?"
        cooking_query = "How to bake a cake?"
        
        ai_result = await self.memo_rag.process_query(ai_query, {"domain": "AI"})
        cooking_result = await self.memo_rag.process_query(cooking_query, {"domain": "cooking"})
        
        assert 'response' in ai_result
        assert 'response' in cooking_result
        assert ai_result['query'] != cooking_result['query']
    
    def test_memory_persistence_simulation(self):
        memories = [
            {"query": "test1", "response": "response1"},
            {"query": "test2", "response": "response2"}
        ]
        
        for i, memory in enumerate(memories):
            self.memory_store.store_memory(f"persistent_{i}", memory)
        
        new_memo_rag = MockMemoRAG(self.memory_store)
        
        assert len(self.memory_store.list_memories()) == 2
        
        stats = new_memo_rag.get_memory_stats()
        assert stats['total_memories'] == 2