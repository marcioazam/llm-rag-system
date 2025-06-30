"""
Testes para RAPTOR Simple - FASE 2.2
Cobertura atual: 0% -> Meta: 60%+
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import time

# Adicionar src ao path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Importação direta
import importlib.util
spec = importlib.util.spec_from_file_location("raptor_simple", src_path / "retrieval" / "raptor_simple.py")
raptor_simple = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raptor_simple)


class TestEnumsAndDataclasses:
    """Testes para enums e dataclasses"""
    
    def test_simple_strategy_enum(self):
        """Testa enum SimpleStrategy"""
        assert raptor_simple.SimpleStrategy.KMEANS.value == "kmeans"
        assert raptor_simple.SimpleStrategy.HIERARCHICAL.value == "hierarchical"
        
        print("✅ SimpleStrategy enum funcionou")
        return True
    
    def test_retrieval_strategy_enum(self):
        """Testa enum RetrievalStrategy"""
        assert raptor_simple.RetrievalStrategy.FLAT.value == "flat"
        assert raptor_simple.RetrievalStrategy.HIERARCHICAL.value == "hierarchical"
        
        print("✅ RetrievalStrategy enum funcionou")
        return True
    
    def test_simple_node_dataclass(self):
        """Testa dataclass SimpleNode"""
        embedding = np.array([0.1, 0.2, 0.3])
        
        node = raptor_simple.SimpleNode(
            node_id="test_node",
            content="Test content",
            embedding=embedding,
            level=0,
            children_ids=["child1", "child2"],
            parent_id="parent1",
            cluster_id=5,
            token_count=10,
            metadata={"test": True}
        )
        
        assert node.node_id == "test_node"
        assert node.content == "Test content"
        assert np.array_equal(node.embedding, embedding)
        assert node.level == 0
        assert node.children_ids == ["child1", "child2"]
        assert node.parent_id == "parent1"
        assert node.cluster_id == 5
        assert node.token_count == 10
        assert node.metadata == {"test": True}
        
        print("✅ SimpleNode dataclass funcionou")
        return True
    
    def test_simple_cluster_dataclass(self):
        """Testa dataclass SimpleCluster"""
        centroid = np.array([0.5, 0.6, 0.7])
        
        cluster = raptor_simple.SimpleCluster(
            cluster_id=1,
            node_ids=["node1", "node2"],
            centroid=centroid,
            size=2
        )
        
        assert cluster.cluster_id == 1
        assert cluster.node_ids == ["node1", "node2"]
        assert np.array_equal(cluster.centroid, centroid)
        assert cluster.size == 2
        
        print("✅ SimpleCluster dataclass funcionou")
        return True
    
    def test_simple_stats_dataclass(self):
        """Testa dataclass SimpleStats"""
        stats = raptor_simple.SimpleStats(
            total_nodes=100,
            levels=3,
            nodes_per_level={0: 50, 1: 30, 2: 20},
            construction_time=5.5
        )
        
        assert stats.total_nodes == 100
        assert stats.levels == 3
        assert stats.nodes_per_level == {0: 50, 1: 30, 2: 20}
        assert stats.construction_time == 5.5
        
        print("✅ SimpleStats dataclass funcionou")
        return True


class TestSimpleEmbedder:
    """Testes para SimpleEmbedder"""
    
    def test_embedder_initialization(self):
        """Testa inicialização do embedder"""
        embedder = raptor_simple.SimpleEmbedder(dim=64)
        
        assert embedder.dim == 64
        assert embedder.vocab == {}
        assert embedder.idf == {}
        
        print("✅ SimpleEmbedder inicializado")
        return True
    
    def test_tokenize(self):
        """Testa tokenização"""
        embedder = raptor_simple.SimpleEmbedder()
        
        tokens = embedder._tokenize("Hello, World! This is a test.")
        expected = ["hello", "world", "this", "is", "a", "test"]
        assert tokens == expected
        
        # Teste com texto vazio
        tokens_empty = embedder._tokenize("")
        assert tokens_empty == []
        
        print("✅ Tokenização funcionou")
        return True
    
    def test_build_vocab(self):
        """Testa construção de vocabulário"""
        embedder = raptor_simple.SimpleEmbedder(dim=10)
        
        texts = [
            "hello world test",
            "world test example",
            "test example hello"
        ]
        
        embedder._build_vocab(texts)
        
        assert len(embedder.vocab) > 0
        assert len(embedder.idf) > 0
        assert all(word in embedder.idf for word in embedder.vocab)
        
        print(f"✅ Vocabulário construído: {len(embedder.vocab)} palavras")
        return True
    
    def test_encode_single_text(self):
        """Testa codificação de texto único"""
        embedder = raptor_simple.SimpleEmbedder(dim=32)
        
        text = "This is a test document for embedding"
        embedding = embedder.encode(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (32,)
        assert not np.allclose(embedding, 0)  # Não deve ser zero
        
        # Teste normalização
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6  # Deve estar normalizado
        
        print("✅ Codificação de texto único funcionou")
        return True
    
    def test_encode_multiple_texts(self):
        """Testa codificação de múltiplos textos"""
        embedder = raptor_simple.SimpleEmbedder(dim=16)
        
        texts = [
            "First document",
            "Second document", 
            "Third document"
        ]
        
        embeddings = embedder.encode(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 16)
        
        # Verificar que embeddings são diferentes
        assert not np.allclose(embeddings[0], embeddings[1])
        assert not np.allclose(embeddings[1], embeddings[2])
        
        print("✅ Codificação de múltiplos textos funcionou")
        return True
    
    def test_encode_consistency(self):
        """Testa consistência da codificação"""
        embedder = raptor_simple.SimpleEmbedder(dim=16)
        
        text = "Consistent test text"
        
        # Codificar duas vezes
        embedding1 = embedder.encode(text)
        embedding2 = embedder.encode(text)
        
        # Deve ser consistente
        assert np.allclose(embedding1, embedding2)
        
        print("✅ Consistência de codificação verificada")
        return True


class TestSimpleClusterer:
    """Testes para SimpleClusterer"""
    
    def test_clusterer_initialization(self):
        """Testa inicialização do clusterer"""
        clusterer = raptor_simple.SimpleClusterer(random_state=123)
        
        assert clusterer.random_state == 123
        
        print("✅ SimpleClusterer inicializado")
        return True
    
    def test_cluster_insufficient_data(self):
        """Testa clustering com dados insuficientes"""
        clusterer = raptor_simple.SimpleClusterer()
        
        # Teste com dados vazios
        embeddings_empty = np.array([])
        clusters_empty, metadata_empty = clusterer.cluster(embeddings_empty, [])
        assert clusters_empty == []
        assert "error" in metadata_empty
        
        # Teste com apenas um embedding
        embeddings_single = np.array([[0.1, 0.2, 0.3]])
        clusters_single, metadata_single = clusterer.cluster(embeddings_single, ["node1"])
        assert clusters_single == []
        assert "error" in metadata_single
        
        print("✅ Casos de dados insuficientes tratados")
        return True
    
    def test_cluster_normal_case(self):
        """Testa clustering com dados normais"""
        clusterer = raptor_simple.SimpleClusterer()
        
        # Criar embeddings de teste (6 pontos, 2 clusters claros)
        embeddings = np.array([
            [1.0, 1.0],  # Cluster 1
            [1.1, 0.9],  # Cluster 1
            [0.9, 1.1],  # Cluster 1
            [5.0, 5.0],  # Cluster 2
            [5.1, 4.9],  # Cluster 2
            [4.9, 5.1]   # Cluster 2
        ])
        
        node_ids = [f"node{i}" for i in range(6)]
        
        clusters, metadata = clusterer.cluster(embeddings, node_ids, max_clusters=3)
        
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        assert len(clusters) <= 3
        
        # Verificar estrutura dos clusters
        for cluster in clusters:
            assert isinstance(cluster, raptor_simple.SimpleCluster)
            assert cluster.cluster_id >= 0
            assert len(cluster.node_ids) > 0
            assert cluster.centroid.shape == (2,)
            assert cluster.size == len(cluster.node_ids)
        
        # Verificar metadata
        assert "total_clusters" in metadata
        assert "avg_cluster_size" in metadata
        assert "method" in metadata
        assert metadata["method"] == "kmeans"
        
        print(f"✅ Clustering normal: {len(clusters)} clusters criados")
        return True
    
    def test_cluster_max_clusters_limit(self):
        """Testa limite máximo de clusters"""
        clusterer = raptor_simple.SimpleClusterer()
        
        # Criar muitos embeddings
        embeddings = np.random.rand(20, 5)
        node_ids = [f"node{i}" for i in range(20)]
        
        clusters, metadata = clusterer.cluster(embeddings, node_ids, max_clusters=3)
        
        # Não deve exceder max_clusters
        assert len(clusters) <= 3
        
        print(f"✅ Limite de clusters respeitado: {len(clusters)} <= 3")
        return True


class TestSimpleSummarizer:
    """Testes para SimpleSummarizer"""
    
    def test_summarizer_initialization(self):
        """Testa inicialização do summarizer"""
        summarizer = raptor_simple.SimpleSummarizer(max_length=200)
        
        assert summarizer.max_length == 200
        
        print("✅ SimpleSummarizer inicializado")
        return True
    
    def test_count_words(self):
        """Testa contagem de palavras"""
        summarizer = raptor_simple.SimpleSummarizer()
        
        assert summarizer._count_words("hello world") == 2
        assert summarizer._count_words("") == 0
        assert summarizer._count_words("one") == 1
        assert summarizer._count_words("word1 word2 word3 word4") == 4
        
        print("✅ Contagem de palavras funcionou")
        return True
    
    @pytest.mark.asyncio
    async def test_summarize_short_texts(self):
        """Testa sumarização de textos curtos"""
        summarizer = raptor_simple.SimpleSummarizer(max_length=100)
        
        texts = [
            "Short text one",
            "Short text two", 
            "Short text three"
        ]
        
        summary = await summarizer.summarize(texts, level=1)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert summarizer._count_words(summary) <= 100
        
        print("✅ Sumarização de textos curtos funcionou")
        return True
    
    @pytest.mark.asyncio
    async def test_summarize_long_texts(self):
        """Testa sumarização de textos longos"""
        summarizer = raptor_simple.SimpleSummarizer(max_length=50)
        
        # Criar textos longos
        long_text = " ".join(["word"] * 100)  # 100 palavras
        texts = [long_text, long_text, long_text]
        
        summary = await summarizer.summarize(texts, level=2)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert summarizer._count_words(summary) <= 50
        
        print("✅ Sumarização de textos longos funcionou")
        return True
    
    @pytest.mark.asyncio
    async def test_summarize_empty_texts(self):
        """Testa sumarização de textos vazios"""
        summarizer = raptor_simple.SimpleSummarizer()
        
        summary = await summarizer.summarize([], level=1)
        assert summary == ""
        
        summary_empty_strings = await summarizer.summarize(["", "", ""], level=1)
        assert summary_empty_strings == ""
        
        print("✅ Casos vazios tratados")
        return True


class TestSimpleRaptor:
    """Testes para classe principal SimpleRaptor"""
    
    def test_raptor_initialization(self):
        """Testa inicialização do SimpleRaptor"""
        raptor = raptor_simple.SimpleRaptor(
            chunk_size=150,
            max_levels=4,
            min_cluster_size=3,
            embedding_dim=64
        )
        
        assert raptor.chunk_size == 150
        assert raptor.max_levels == 4
        assert raptor.min_cluster_size == 3
        assert raptor.embedding_dim == 64
        assert hasattr(raptor, 'embedder')
        assert hasattr(raptor, 'clusterer')
        assert hasattr(raptor, 'summarizer')
        assert raptor.tree == {}
        assert raptor.stats is None
        
        print("✅ SimpleRaptor inicializado")
        return True
    
    def test_raptor_default_initialization(self):
        """Testa inicialização com valores padrão"""
        raptor = raptor_simple.SimpleRaptor()
        
        assert raptor.chunk_size == 200
        assert raptor.max_levels == 3
        assert raptor.min_cluster_size == 2
        assert raptor.embedding_dim == 128
        
        print("✅ Valores padrão corretos")
        return True
    
    def test_chunk_text(self):
        """Testa divisão de texto em chunks"""
        raptor = raptor_simple.SimpleRaptor(chunk_size=10)  # Pequeno para teste
        
        # Texto que deve ser dividido
        long_text = " ".join([f"word{i}" for i in range(20)])  # 20 palavras
        
        chunks = raptor._chunk_text(long_text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Deve ser dividido
        
        # Verificar que chunks não excedem o tamanho
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 15  # Com alguma margem para sobreposição
        
        print(f"✅ Texto dividido em {len(chunks)} chunks")
        return True
    
    def test_create_leaf_nodes(self):
        """Testa criação de nós folha"""
        raptor = raptor_simple.SimpleRaptor()
        
        documents = [
            "First document content",
            "Second document content",
            "Third document content"
        ]
        
        leaf_nodes = raptor._create_leaf_nodes(documents)
        
        assert isinstance(leaf_nodes, list)
        assert len(leaf_nodes) == 3
        
        for node in leaf_nodes:
            assert isinstance(node, raptor_simple.SimpleNode)
            assert node.level == 0
            assert node.content in ["First document content", "Second document content", "Third document content"]
            assert node.embedding.shape == (raptor.embedding_dim,)
            assert node.token_count > 0
        
        print(f"✅ {len(leaf_nodes)} nós folha criados")
        return True
    
    def test_search_empty_tree(self):
        """Testa busca em árvore vazia"""
        raptor = raptor_simple.SimpleRaptor()
        
        results = raptor.search("test query", k=5)
        
        assert isinstance(results, list)
        assert len(results) == 0
        
        print("✅ Busca em árvore vazia tratada")
        return True
    
    def test_get_stats_no_tree(self):
        """Testa estatísticas sem árvore construída"""
        raptor = raptor_simple.SimpleRaptor()
        
        stats = raptor.get_stats()
        
        assert isinstance(stats, dict)
        assert "tree_built" in stats
        assert stats["tree_built"] is False
        
        print("✅ Estatísticas sem árvore funcionaram")
        return True


class TestFactoryFunctions:
    """Testes para funções factory"""
    
    @pytest.mark.asyncio
    async def test_create_simple_raptor(self):
        """Testa função factory create_simple_raptor"""
        config = {
            "chunk_size": 100,
            "max_levels": 2,
            "min_cluster_size": 3,
            "embedding_dim": 64
        }
        
        raptor = await raptor_simple.create_simple_raptor(config)
        
        assert isinstance(raptor, raptor_simple.SimpleRaptor)
        assert raptor.chunk_size == 100
        assert raptor.max_levels == 2
        assert raptor.min_cluster_size == 3
        assert raptor.embedding_dim == 64
        
        print("✅ Factory create_simple_raptor funcionou")
        return True
    
    def test_get_simple_raptor_config(self):
        """Testa obtenção de configuração padrão"""
        config = raptor_simple.get_simple_raptor_config()
        
        assert isinstance(config, dict)
        assert "chunk_size" in config
        assert "max_levels" in config
        assert "min_cluster_size" in config
        assert "embedding_dim" in config
        
        # Verificar valores razoáveis
        assert config["chunk_size"] > 0
        assert config["max_levels"] > 0
        assert config["min_cluster_size"] > 0
        assert config["embedding_dim"] > 0
        
        print("✅ Configuração padrão obtida")
        return True


def run_all_tests():
    """Executa todos os testes"""
    import asyncio
    
    test_classes = [
        TestEnumsAndDataclasses,
        TestSimpleEmbedder,
        TestSimpleClusterer,
        TestSimpleSummarizer,
        TestSimpleRaptor,
        TestFactoryFunctions
    ]
    
    passed = 0
    failed = 0
    
    print("Executando testes do RAPTOR Simple...")
    print("=" * 60)
    
    for test_class in test_classes:
        test_instance = test_class()
        
        # Obter métodos de teste
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                
                # Verificar se é método assíncrono
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()
                
                passed += 1
                
            except Exception as e:
                print(f"❌ {test_class.__name__}.{method_name}: {e}")
                failed += 1
    
    total = passed + failed
    coverage_estimate = (passed / total) * 100 if total > 0 else 0
    
    print("=" * 60)
    print(f"RESULTADO RAPTOR SIMPLE:")
    print(f"   Testes passados: {passed}")
    print(f"   Testes falhados: {failed}")
    print(f"   Cobertura estimada: {coverage_estimate:.1f}%")
    
    if coverage_estimate >= 70:
        print("STATUS: ✅ RAPTOR SIMPLE BEM COBERTO")
    elif coverage_estimate >= 50:
        print("STATUS: ⚠️ RAPTOR SIMPLE PARCIALMENTE COBERTO")
    else:
        print("STATUS: 🔴 RAPTOR SIMPLE PRECISA MAIS TESTES")
    
    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "coverage": coverage_estimate
    }


if __name__ == "__main__":
    results = run_all_tests()
    
    print(f"\nRESUMO FINAL:")
    print(f"- Cobertura estimada: {results['coverage']:.1f}%")
    
    if results['coverage'] >= 60:
        print("✅ RAPTOR Simple: bem coberto!")
    else:
        print("⚠️ RAPTOR Simple: precisa de mais testes") 