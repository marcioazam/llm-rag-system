"""
Teste completo da implementação RAPTOR (Recursive Abstraction Processing for Tree-Organized Retrieval)

Este teste demonstra:
1. Clustering recursivo com UMAP + GMM
2. Summarização hierárquica multi-nível
3. Retrieval em múltiplas abstrações
4. Tree traversal e collapsed tree

Baseado no paper: https://arxiv.org/abs/2401.18059
"""

import asyncio
import logging
import time
from typing import List, Dict
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.retrieval.raptor_retriever import (
    RaptorRetriever, 
    create_raptor_retriever, 
    get_default_raptor_config,
    ClusteringStrategy,
    RetrievalStrategy
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Documentos de exemplo para teste
SAMPLE_DOCUMENTS = [
    # Documentos sobre Python
    """
    Python é uma linguagem de programação interpretada, orientada a objetos e de alto nível.
    Foi criada por Guido van Rossum em 1991. Python é conhecida por sua sintaxe simples e legível,
    o que a torna ideal para iniciantes. A filosofia do Python enfatiza a legibilidade do código
    e a produtividade do programador. Python suporta múltiplos paradigmas de programação,
    incluindo programação orientada a objetos, programação funcional e programação procedural.
    """,
    
    """
    As principais características do Python incluem tipagem dinâmica, gerenciamento automático
    de memória e uma vasta biblioteca padrão. Python é amplamente usado em desenvolvimento web,
    ciência de dados, inteligência artificial, automação e scripting. Frameworks populares
    como Django e Flask facilitam o desenvolvimento web em Python. Para ciência de dados,
    bibliotecas como NumPy, Pandas e Matplotlib são fundamentais.
    """,
    
    """
    Python Virtual Environments (venv) são ambientes isolados que permitem instalar pacotes
    Python específicos para cada projeto sem conflitos. O pip é o gerenciador de pacotes
    padrão do Python, usado para instalar e gerenciar bibliotecas de terceiros. Python
    suporta diferentes versões simultaneamente no sistema, e ferramentas como pyenv
    facilitam o gerenciamento de múltiplas versões.
    """,
    
    # Documentos sobre Machine Learning
    """
    Machine Learning é um subcampo da inteligência artificial que permite aos computadores
    aprender e melhorar automaticamente através da experiência sem serem explicitamente
    programados. Os algoritmos de machine learning constroem modelos baseados em dados
    de treinamento para fazer previsões ou decisões. Existem três tipos principais:
    aprendizado supervisionado, não supervisionado e por reforço.
    """,
    
    """
    O aprendizado supervisionado usa dados rotulados para treinar modelos que podem
    fazer previsões sobre novos dados. Exemplos incluem classificação (prever categorias)
    e regressão (prever valores contínuos). Algoritmos populares incluem regressão linear,
    árvores de decisão, random forests e redes neurais. O aprendizado não supervisionado
    trabalha com dados não rotulados para encontrar padrões ocultos, como clustering.
    """,
    
    """
    Deep Learning é um subconjunto do machine learning baseado em redes neurais artificiais
    com múltiplas camadas. É especialmente eficaz para problemas como reconhecimento de
    imagens, processamento de linguagem natural e reconhecimento de fala. Frameworks
    como TensorFlow, PyTorch e Keras facilitam o desenvolvimento de modelos de deep learning.
    GPUs são frequentemente usadas para acelerar o treinamento de modelos profundos.
    """,
    
    # Documentos sobre RAG
    """
    Retrieval-Augmented Generation (RAG) é uma arquitetura que combina modelos de linguagem
    com sistemas de recuperação de informações. RAG permite que modelos de linguagem
    acessem informações externas durante a geração, melhorando a precisão e reduzindo
    alucinações. O processo típico envolve: codificar a query, buscar documentos relevantes,
    e usar tanto a query quanto os documentos recuperados para gerar a resposta.
    """,
    
    """
    Os componentes principais de um sistema RAG incluem: um corpus de documentos,
    um modelo de embeddings para codificar textos, um banco de dados vetorial para
    armazenamento e busca eficiente, e um modelo de linguagem para geração. Técnicas
    avançadas incluem re-ranking de documentos, chunking inteligente, e métodos de
    fusão de múltiplas queries. RAG é especialmente útil para Q&A, resumos e assistentes.
    """,
    
    """
    RAPTOR (Recursive Abstraction Processing for Tree-Organized Retrieval) é uma técnica
    avançada de RAG que constrói uma árvore hierárquica de resumos. RAPTOR usa clustering
    recursivo para agrupar documentos similares e criar resumos em múltiplos níveis
    de abstração. Durante o retrieval, RAPTOR pode buscar em diferentes níveis da árvore,
    permitindo capturar tanto detalhes específicos quanto visões de alto nível.
    """,
    
    # Documentos sobre Cloud Computing
    """
    Cloud Computing é o fornecimento de serviços de computação através da internet,
    incluindo servidores, armazenamento, bancos de dados, redes, software e analytics.
    Os principais modelos de serviço são: Infrastructure as a Service (IaaS),
    Platform as a Service (PaaS) e Software as a Service (SaaS). Vantagens incluem
    escalabilidade, redução de custos e acesso remoto a recursos computacionais.
    """,
    
    """
    Amazon Web Services (AWS), Microsoft Azure e Google Cloud Platform são os principais
    provedores de cloud. AWS oferece serviços como EC2 para computação, S3 para armazenamento
    e RDS para bancos de dados. Containers e orquestração com Docker e Kubernetes revolucionaram
    o deployment de aplicações na cloud. Serverless computing permite executar código
    sem gerenciar servidores, com serviços como AWS Lambda e Azure Functions.
    """,
    
    """
    DevOps integra desenvolvimento e operações para melhorar a colaboração e produtividade.
    Práticas DevOps incluem integração contínua (CI), entrega contínua (CD), infrastructure
    as code (IaC) e monitoramento. Ferramentas populares incluem Jenkins, GitLab CI,
    Terraform, Ansible e Prometheus. A cultura DevOps enfatiza automação, colaboração
    e feedback rápido para acelerar o desenvolvimento e deployment de software.
    """
]

class RaptorTester:
    """Tester para RAPTOR retriever"""
    
    def __init__(self):
        self.raptor: RaptorRetriever = None
        
    async def test_raptor_complete_workflow(self):
        """Teste completo do workflow RAPTOR"""
        
        print("🚀 TESTE COMPLETO RAPTOR - Recursive Abstraction Processing")
        print("=" * 70)
        
        # 1. Configuração
        await self._test_configuration()
        
        # 2. Construção da árvore
        await self._test_tree_construction()
        
        # 3. Análise da árvore
        await self._test_tree_analysis()
        
        # 4. Testes de retrieval
        await self._test_retrieval_strategies()
        
        # 5. Comparação de performance
        await self._test_performance_comparison()
        
        print("\n✅ TESTE RAPTOR CONCLUÍDO COM SUCESSO!")
        
    async def _test_configuration(self):
        """Teste configuração e inicialização"""
        
        print("\n📋 1. CONFIGURAÇÃO E INICIALIZAÇÃO")
        print("-" * 50)
        
        # Configuração customizada
        config = get_default_raptor_config()
        config.update({
            "chunk_size": 200,
            "chunk_overlap": 40,
            "clustering_strategy": "global_local",
            "retrieval_strategy": "collapsed_tree",
            "max_levels": 4,
            "min_cluster_size": 2,
            "max_cluster_size": 50
        })
        
        print(f"✓ Configuração: {config['clustering_strategy']} + {config['retrieval_strategy']}")
        print(f"✓ Chunks: {config['chunk_size']} tokens, overlap {config['chunk_overlap']}")
        print(f"✓ Max níveis: {config['max_levels']}, cluster size: {config['min_cluster_size']}-{config['max_cluster_size']}")
        
        # Criar RAPTOR retriever
        start_time = time.time()
        self.raptor = await create_raptor_retriever(config)
        init_time = time.time() - start_time
        
        print(f"✓ RAPTOR inicializado em {init_time:.2f}s")
        print(f"✓ Modelo embedding: {self.raptor.embedding_model_name}")
        
    async def _test_tree_construction(self):
        """Teste construção da árvore hierárquica"""
        
        print("\n🌳 2. CONSTRUÇÃO DA ÁRVORE HIERÁRQUICA")
        print("-" * 50)
        
        print(f"📄 Documentos de entrada: {len(SAMPLE_DOCUMENTS)}")
        for i, doc in enumerate(SAMPLE_DOCUMENTS):
            preview = doc.strip()[:100] + "..." if len(doc.strip()) > 100 else doc.strip()
            print(f"   Doc {i+1}: {preview}")
        
        # Construir árvore
        print("\n🔨 Construindo árvore RAPTOR...")
        start_time = time.time()
        
        stats = await self.raptor.build_tree(SAMPLE_DOCUMENTS)
        
        construction_time = time.time() - start_time
        
        print(f"\n✅ Árvore construída em {construction_time:.2f}s")
        print(f"📊 Estatísticas da árvore:")
        print(f"   • Total de nós: {stats.total_nodes}")
        print(f"   • Níveis: {stats.levels}")
        print(f"   • Compressão: {stats.compression_ratio:.2f}x")
        print(f"   • Memória: {stats.memory_usage_mb:.2f}MB")
        
        print(f"\n📈 Distribuição por nível:")
        for level, count in stats.nodes_per_level.items():
            print(f"   Nível {level}: {count} nós")
            
        if stats.clusters_per_level:
            print(f"\n🔗 Clusters por nível:")
            for level, count in stats.clusters_per_level.items():
                print(f"   Nível {level}: {count} clusters")
    
    async def _test_tree_analysis(self):
        """Teste análise detalhada da árvore"""
        
        print("\n🔍 3. ANÁLISE DETALHADA DA ÁRVORE")
        print("-" * 50)
        
        # Obter resumo da árvore
        tree_summary = self.raptor.get_tree_summary()
        
        print(f"📋 Status: {tree_summary['status']}")
        print(f"🌲 Nós raiz: {tree_summary['root_nodes']}")
        print(f"🍃 Nós folha: {tree_summary['leaf_nodes']}")
        
        if 'stats' in tree_summary:
            stats = tree_summary['stats']
            print(f"\n📊 Métricas detalhadas:")
            print(f"   • Tempo de construção: {stats['construction_time']:.2f}s")
            print(f"   • Uso de memória: {stats['memory_usage_mb']:.2f}MB")
            print(f"   • Taxa de compressão: {stats['compression_ratio']:.2f}x")
        
        # Mostrar exemplos de nós em diferentes níveis
        print(f"\n📝 Exemplo de conteúdo por nível:")
        for level in range(min(3, tree_summary['stats']['levels'] + 1)):
            level_nodes = [nid for nid in self.raptor.levels.get(level, [])]
            if level_nodes:
                example_node = self.raptor.tree[level_nodes[0]]
                content_preview = example_node.content[:150] + "..." if len(example_node.content) > 150 else example_node.content
                print(f"\n   Nível {level} (exemplo):")
                print(f"   {content_preview}")
                print(f"   [Tokens: {example_node.token_count}, Filhos: {len(example_node.children_ids)}]")
    
    async def _test_retrieval_strategies(self):
        """Teste diferentes estratégias de retrieval"""
        
        print("\n🎯 4. TESTE DE ESTRATÉGIAS DE RETRIEVAL")
        print("-" * 50)
        
        test_queries = [
            "O que é Python e quais são suas principais características?",
            "Como funciona machine learning e quais são os tipos principais?",
            "Explique sobre RAG e RAPTOR em detalhes",
            "Quais são as vantagens do cloud computing?",
            "Como Docker e Kubernetes se relacionam com DevOps?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            print("─" * 60)
            
            # Teste retrieval collapsed tree
            start_time = time.time()
            results = self.raptor.search(
                query=query,
                k=5,
                max_tokens=1500,
                strategy=RetrievalStrategy.COLLAPSED_TREE
            )
            retrieval_time = time.time() - start_time
            
            print(f"⚡ Collapsed Tree: {len(results)} resultados em {retrieval_time:.3f}s")
            
            # Mostrar resultados
            total_tokens = 0
            for j, result in enumerate(results[:3]):
                level = result['metadata']['level']
                score = result['score']
                tokens = result['metadata']['token_count']
                total_tokens += tokens
                
                content_preview = result['content'][:120] + "..." if len(result['content']) > 120 else result['content']
                print(f"   {j+1}. [Nível {level}, Score: {score:.3f}, {tokens} tokens]")
                print(f"      {content_preview}")
            
            print(f"   📊 Total: {total_tokens} tokens usados")
            
            # Análise de distribuição por nível
            level_dist = {}
            for result in results:
                level = result['metadata']['level']
                level_dist[level] = level_dist.get(level, 0) + 1
            
            print(f"   📈 Distribuição: {dict(sorted(level_dist.items()))}")
    
    async def _test_performance_comparison(self):
        """Teste comparação de performance"""
        
        print("\n⚡ 5. COMPARAÇÃO DE PERFORMANCE")
        print("-" * 50)
        
        test_query = "Explique machine learning, Python e suas aplicações em cloud computing"
        
        # Teste com diferentes valores de k
        k_values = [3, 5, 10, 15]
        
        print(f"🎯 Query: {test_query}")
        print(f"📊 Testando diferentes valores de k:")
        
        for k in k_values:
            start_time = time.time()
            results = self.raptor.search(query=test_query, k=k, max_tokens=2000)
            search_time = time.time() - start_time
            
            total_tokens = sum(r['metadata']['token_count'] for r in results)
            avg_score = sum(r['score'] for r in results) / len(results) if results else 0
            
            # Distribuição por nível
            level_counts = {}
            for result in results:
                level = result['metadata']['level']
                level_counts[level] = level_counts.get(level, 0) + 1
            
            print(f"   k={k:2d}: {len(results):2d} docs, {search_time:.3f}s, "
                  f"{total_tokens:4d} tokens, score avg: {avg_score:.3f}")
            print(f"         Níveis: {dict(sorted(level_counts.items()))}")
        
        # Teste com diferentes limites de tokens
        print(f"\n📋 Testando diferentes limites de tokens:")
        token_limits = [500, 1000, 1500, 2000]
        
        for max_tokens in token_limits:
            start_time = time.time()
            results = self.raptor.search(query=test_query, k=10, max_tokens=max_tokens)
            search_time = time.time() - start_time
            
            total_tokens = sum(r['metadata']['token_count'] for r in results)
            
            print(f"   {max_tokens:4d} tokens max: {len(results):2d} docs, "
                  f"{search_time:.3f}s, {total_tokens:4d} tokens usados")
    
    async def demonstrate_hierarchical_nature(self):
        """Demonstra a natureza hierárquica do RAPTOR"""
        
        print("\n🌳 DEMONSTRAÇÃO: NATUREZA HIERÁRQUICA")
        print("=" * 60)
        
        query = "Python para machine learning"
        
        # Buscar em todos os níveis
        print(f"🔍 Query: {query}")
        print(f"📊 Analisando resultados por nível:")
        
        results = self.raptor.search(query=query, k=15, max_tokens=3000)
        
        # Agrupar por nível
        by_level = {}
        for result in results:
            level = result['metadata']['level']
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(result)
        
        # Mostrar resultados por nível
        for level in sorted(by_level.keys()):
            level_results = by_level[level]
            print(f"\n📋 NÍVEL {level} ({len(level_results)} resultados):")
            
            for i, result in enumerate(level_results[:2]):  # Mostrar apenas 2 por nível
                score = result['score']
                tokens = result['metadata']['token_count']
                content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                
                print(f"   {i+1}. Score: {score:.3f}, {tokens} tokens")
                print(f"      {content}")
                print()
        
        print(f"💡 Observação: Níveis mais altos contêm resumos mais abstratos,")
        print(f"   enquanto níveis mais baixos têm detalhes específicos.")

async def main():
    """Função principal de teste"""
    
    try:
        print("🧪 TESTE RAPTOR - RECURSIVE ABSTRACTION PROCESSING")
        print("📚 Paper: https://arxiv.org/abs/2401.18059")
        print("🏗️  Implementação: Clustering UMAP+GMM + Summarização Hierárquica")
        print("=" * 80)
        
        tester = RaptorTester()
        
        # Executar teste completo
        await tester.test_raptor_complete_workflow()
        
        # Demonstração adicional
        await tester.demonstrate_hierarchical_nature()
        
        print("\n🎉 TESTE RAPTOR FINALIZADO!")
        print("✨ RAPTOR demonstrou capacidades de:")
        print("   • ✓ Clustering recursivo hierárquico")
        print("   • ✓ Summarização multi-nível")
        print("   • ✓ Retrieval em múltiplas abstrações")
        print("   • ✓ Balanceamento entre detalhes e visão geral")
        
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())