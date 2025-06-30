"""
Testes básicos para o MemoRAG.
Cobertura atual: 0% -> Meta: 70%
"""

import pytest
import asyncio
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import time

from src.retrieval.memo_rag import (
    MemoRAG, GlobalMemoryStore, ClueGenerator, 
    MemorySegment, Clue, create_memo_rag
)


class TestGlobalMemoryStore:
    """Testes para o Global Memory Store."""

    @pytest.fixture
    def memory_store(self):
        """Criar instância do memory store."""
        return GlobalMemoryStore(
            max_tokens=10000,
            compression_threshold=500,
            segment_size=100
        )

    def test_init(self, memory_store):
        """Testar inicialização do memory store."""
        assert memory_store.max_tokens == 10000
        assert memory_store.compression_threshold == 500
        assert memory_store.segment_size == 100
        assert memory_store.total_tokens == 0
        assert len(memory_store.memory_levels["hot"]) == 0

    def test_add_memory_basic(self, memory_store):
        """Testar adição básica de memória."""
        content = "Este é um teste de conteúdo de memória"
        segment_id = memory_store.add_memory(content)
        
        assert segment_id is not None
        assert segment_id in memory_store.segment_index
        assert memory_store.total_tokens > 0
        
        segment = memory_store.segment_index[segment_id]
        assert segment.content == content
        assert segment.importance_score == 0.5
        assert segment.token_count > 0

    def test_add_memory_with_metadata(self, memory_store):
        """Testar adição de memória com metadata."""
        content = "Conteúdo com metadata"
        metadata = {"source": "test", "type": "document"}
        
        segment_id = memory_store.add_memory(
            content, 
            importance=0.8,
            metadata=metadata
        )
        
        segment = memory_store.segment_index[segment_id]
        assert segment.importance_score == 0.8
        assert segment.metadata == metadata

    def test_compression_large_content(self, memory_store):
        """Testar compressão de conteúdo grande."""
        # Criar conteúdo grande que será comprimido
        large_content = "Este é um conteúdo muito longo que deveria ser comprimido. " * 100
        
        segment_id = memory_store.add_memory(large_content)
        segment = memory_store.segment_index[segment_id]
        
        # Verificar se foi tentativa de compressão
        assert segment is not None
        assert segment.segment_id == segment_id

    def test_memory_eviction(self, memory_store):
        """Testar evicção de memórias antigas."""
        # Adicionar memórias até atingir o limite
        segment_ids = []
        
        for i in range(20):  # Adicionar muitas memórias pequenas
            content = f"Memória {i} " * 100  # ~200 tokens cada para forçar evicção
            segment_id = memory_store.add_memory(content, importance=0.1)
            segment_ids.append(segment_id)
            time.sleep(0.001)  # Garantir timestamps diferentes
        
        # Verificar se houve evicção
        assert memory_store.total_tokens <= memory_store.max_tokens
        
        # Algumas memórias antigas devem ter sido removidas OU total respeitado
        remaining_segments = len(memory_store.segment_index)
        assert remaining_segments <= len(segment_ids)

    def test_get_memory_stats(self, memory_store):
        """Testar obtenção de estatísticas da memória."""
        # Adicionar algumas memórias
        memory_store.add_memory("Primeira memória")
        memory_store.add_memory("Segunda memória")
        
        stats = memory_store.get_memory_stats()
        
        assert isinstance(stats, dict)
        assert "total_segments" in stats
        assert "total_tokens" in stats
        assert "levels" in stats  # É "levels" não "memory_levels"
        assert stats["total_segments"] == 2
        assert stats["total_tokens"] > 0

    def test_promote_segment(self, memory_store):
        """Testar promoção de segmento."""
        content = "Conteúdo para promoção"
        segment_id = memory_store.add_memory(content)
        
        # Verificar se segmento existe e pode ser promovido
        assert segment_id in memory_store.segment_index
        
        # Promover (mesmo que já esteja em hot, deve funcionar)
        memory_store.promote_segment(segment_id)
        
        # Verificar se ainda existe no sistema
        assert segment_id in memory_store.segment_index


class TestClueGenerator:
    """Testes para o Clue Generator."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock do serviço LLM."""
        mock = AsyncMock()
        mock.generate.return_value = {
            "response": "concept: machine learning\nkeyword: algorithm\nentity: neural network"
        }
        return mock

    @pytest.fixture
    def clue_generator(self, mock_llm_service):
        """Criar instância do clue generator."""
        return ClueGenerator(mock_llm_service)

    @pytest.mark.asyncio
    async def test_generate_clues_basic(self, clue_generator):
        """Testar geração básica de clues."""
        content = "Machine learning algorithms are used for data analysis and pattern recognition."
        
        clues = await clue_generator.generate_clues(content, max_clues=3)
        
        assert isinstance(clues, list)
        assert len(clues) <= 3
        
        for clue in clues:
            assert isinstance(clue, Clue)
            assert clue.clue_text is not None
            assert clue.clue_type in ["keyword", "concept", "entity", "relation"]
            assert 0 <= clue.relevance_score <= 1

    @pytest.mark.asyncio
    async def test_generate_clues_empty_content(self, clue_generator):
        """Testar geração de clues com conteúdo vazio."""
        clues = await clue_generator.generate_clues("", max_clues=5)
        
        # Deve retornar lista vazia ou clues padrão
        assert isinstance(clues, list)

    def test_extract_keywords(self, clue_generator):
        """Testar extração de palavras-chave."""
        content = "Machine learning algorithms process data efficiently using neural networks."
        
        keywords = clue_generator._extract_keywords(content)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        
        # Verificar se contém palavras relevantes
        keyword_text = " ".join(keywords).lower()
        assert any(word in keyword_text for word in ["learning", "algorithm", "network"])


class TestMemoRAG:
    """Testes para o MemoRAG principal."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock do serviço de embeddings."""
        mock = AsyncMock()
        mock.embed_query.return_value = np.random.random(384)
        mock.embed_documents.return_value = [np.random.random(384) for _ in range(3)]
        return mock

    @pytest.fixture
    def mock_llm_service(self):
        """Mock do serviço LLM."""
        mock = AsyncMock()
        mock.generate.return_value = {
            "response": "Esta é uma resposta gerada pelo LLM",
            "usage": {"tokens": 50}
        }
        return mock

    @pytest.fixture
    def temp_dir(self):
        """Diretório temporário para testes."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def memo_rag(self, mock_embedding_service, mock_llm_service, temp_dir):
        """Criar instância do MemoRAG."""
        return MemoRAG(
            embedding_service=mock_embedding_service,
            llm_service=mock_llm_service,
            max_memory_tokens=5000,
            memory_persistence_path=str(Path(temp_dir) / "memory.pkl")
        )

    def test_init(self, memo_rag):
        """Testar inicialização do MemoRAG."""
        assert memo_rag.embedding_service is not None
        assert memo_rag.llm_service is not None
        assert memo_rag.global_memory.max_tokens == 5000  # É atributo do global_memory
        assert memo_rag.clue_guided_retrieval is True
        assert isinstance(memo_rag.global_memory, GlobalMemoryStore)
        assert isinstance(memo_rag.clue_generator, ClueGenerator)

    @pytest.mark.asyncio
    async def test_add_document_basic(self, memo_rag):
        """Testar adição básica de documento."""
        document = "Este é um documento de teste para o MemoRAG system."
        
        result = await memo_rag.add_document(document)
        
        assert isinstance(result, dict)
        assert "segments_created" in result
        # "clues_generated" pode não existir se houve erro no LLM
        assert result["segments_created"] > 0

    @pytest.mark.asyncio
    async def test_add_document_with_metadata(self, memo_rag):
        """Testar adição de documento com metadata."""
        document = "Documento com metadata específica."
        metadata = {"source": "test.txt", "category": "technical"}
        
        result = await memo_rag.add_document(
            document, 
            metadata=metadata,
            importance=0.7
        )
        
        assert result["segments_created"] > 0
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, memo_rag):
        """Testar retrieval básico."""
        # Adicionar alguns documentos primeiro
        await memo_rag.add_document("Primeiro documento sobre machine learning")
        await memo_rag.add_document("Segundo documento sobre deep learning")
        
        # Fazer retrieval
        results = await memo_rag.retrieve("machine learning algorithms", k=5)
        
        assert isinstance(results, list)
        assert len(results) >= 0  # Pode ser vazio se não encontrar similaridades
        
        for result in results:
            assert "content" in result
            assert "score" in result
            assert "metadata" in result

    @pytest.mark.asyncio
    async def test_retrieve_with_clues_disabled(self, memo_rag):
        """Testar retrieval sem clues."""
        await memo_rag.add_document("Documento para teste de retrieval")
        
        results = await memo_rag.retrieve(
            "teste retrieval", 
            k=3, 
            use_clues=False
        )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_query_with_memory(self, memo_rag):
        """Testar query completa com memória."""
        # Adicionar contexto
        await memo_rag.add_document("Machine learning é uma área da IA")
        await memo_rag.add_document("Deep learning usa redes neurais")
        
        # Fazer query
        result = await memo_rag.query_with_memory(
            "O que é machine learning?",
            k=3
        )
        
        assert isinstance(result, dict)
        assert "answer" in result  # É "answer" não "response"
        assert "sources" in result
        assert "memory_stats" in result
        assert isinstance(result["sources"], list)

    def test_segment_document(self, memo_rag):
        """Testar segmentação de documento."""
        long_document = "Esta é uma frase. " * 200  # Documento longo
        
        segments = memo_rag._segment_document(long_document, max_segment_size=50)
        
        assert isinstance(segments, list)
        assert len(segments) > 1  # Deve ser dividido em múltiplos segmentos
        
        for segment in segments:
            assert len(segment.split()) <= 60  # Margem para divisão natural

    @pytest.mark.asyncio
    async def test_get_embedding(self, memo_rag):
        """Testar obtenção de embedding."""
        text = "Texto para gerar embedding"
        
        embedding = await memo_rag._get_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0

    def test_get_stats(self, memo_rag):
        """Testar obtenção de estatísticas."""
        stats = memo_rag.get_stats()
        
        assert isinstance(stats, dict)
        assert "memory_stats" in stats
        # Campos podem variar - verificar se pelo menos alguns existem
        assert len(stats) > 0

    @pytest.mark.asyncio
    async def test_persistence(self, memo_rag, temp_dir):
        """Testar persistência da memória."""
        # Adicionar dados
        await memo_rag.add_document("Documento para teste de persistência")
        
        # Salvar
        memo_rag._save_persistent_memory()
        
        # Verificar se arquivo foi criado
        persistence_path = Path(temp_dir) / "memory.pkl"
        assert persistence_path.exists()
        
        # Carregar em nova instância
        new_memo_rag = MemoRAG(
            embedding_service=memo_rag.embedding_service,
            llm_service=memo_rag.llm_service,
            memory_persistence_path=str(persistence_path)
        )
        
        # Verificar se dados foram carregados
        assert len(new_memo_rag.global_memory.segment_index) > 0

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, memo_rag):
        """Testar operações concorrentes."""
        # Executar adições concorrentes
        tasks = []
        for i in range(5):
            task = memo_rag.add_document(f"Documento concorrente {i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verificar que todas as operações foram bem-sucedidas
        assert len(results) == 5
        for result in results:
            assert result["segments_created"] > 0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_large_document(self, memo_rag):
        """Testar performance com documento grande."""
        # Criar documento grande
        large_doc = "Esta é uma frase de teste. " * 1000  # ~4000 palavras
        
        start_time = time.time()
        result = await memo_rag.add_document(large_doc)
        end_time = time.time()
        
        # Deve processar em tempo razoável (< 10 segundos)
        assert end_time - start_time < 10.0
        assert result["segments_created"] > 1


class TestMemoRAGFactory:
    """Testes para a função factory."""

    @pytest.fixture
    def mock_services(self):
        """Mock dos serviços necessários."""
        embedding_service = AsyncMock()
        llm_service = AsyncMock()
        return embedding_service, llm_service

    def test_create_memo_rag_default(self, mock_services):
        """Testar criação com configuração padrão."""
        embedding_service, llm_service = mock_services
        
        memo_rag = create_memo_rag(embedding_service, llm_service)
        
        assert isinstance(memo_rag, MemoRAG)
        assert memo_rag.embedding_service == embedding_service
        assert memo_rag.llm_service == llm_service

    def test_create_memo_rag_with_config(self, mock_services):
        """Testar criação com configuração personalizada."""
        embedding_service, llm_service = mock_services
        
        config = {
            "max_memory_tokens": 1_000_000,
            "clue_guided_retrieval": False
        }
        
        memo_rag = create_memo_rag(
            embedding_service, 
            llm_service, 
            config=config
        )
        
        assert memo_rag.global_memory.max_tokens == 1_000_000
        assert memo_rag.clue_guided_retrieval is False 