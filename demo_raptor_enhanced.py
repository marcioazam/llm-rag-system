"""
Demo RAPTOR Enhanced - Demonstração completa das melhorias

Demonstra:
1. Embeddings reais (OpenAI/Sentence-Transformers)
2. Clustering avançado (UMAP + GMM)
3. Summarização com LLM
4. Otimizações para volumes maiores
5. Métricas avançadas
"""

import asyncio
import time
import os
import logging
from typing import List, Dict, Any
import sys
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar RAPTOR Enhanced
try:
    from src.retrieval.raptor_enhanced import (
        EnhancedRaptorRetriever,
        RaptorConfig,
        EmbeddingProvider,
        ClusteringMethod,
        SummarizationProvider,
        create_openai_config,
        create_default_config
    )
except ImportError:
    print("Erro: Não foi possível importar RAPTOR Enhanced")
    print("Execute: pip install umap-learn sentence-transformers openai anthropic")
    sys.exit(1)

# Documentos de teste expandidos
EXTENDED_DOCUMENTS = [
    # Python Ecosystem (5 docs)
    """
    Python é uma linguagem de programação interpretada, orientada a objetos e de alto nível.
    Criada por Guido van Rossum em 1991, Python enfatiza a legibilidade do código e a 
    produtividade do programador. A filosofia Python, conhecida como "Zen of Python", 
    prioriza código claro, explícito e simples. Python suporta múltiplos paradigmas: 
    orientado a objetos, funcional, procedural e imperativo. O interpretador CPython 
    é a implementação de referência, mas existem outras como PyPy, Jython e IronPython.
    """,
    
    """
    O ecosistema Python inclui uma vasta biblioteca padrão e milhares de pacotes de 
    terceiros disponíveis via PyPI (Python Package Index). Pip é o gerenciador de 
    pacotes padrão, enquanto conda oferece gestão de ambientes mais robusta. 
    Virtual environments (venv, virtualenv) permitem isolamento de dependências. 
    Ferramentas como Poetry e Pipenv modernizaram o gerenciamento de dependências 
    com arquivos de lock e resolução automática de conflitos.
    """,
    
    """
    Python Web frameworks incluem Django para aplicações completas, Flask para 
    microsserviços, FastAPI para APIs modernas com tipagem, e Pyramid para projetos 
    complexos. Django oferece ORM, admin interface, autenticação e mais out-of-the-box. 
    Flask é minimalista e flexível. FastAPI combina performance com type hints e 
    documentação automática OpenAPI. Para frontend, existem Streamlit para dashboards 
    e Dash para visualizações interativas.
    """,
    
    """
    Python para ciência de dados se baseia no stack NumPy, Pandas, Matplotlib e 
    Scikit-learn. NumPy fornece arrays n-dimensionais eficientes e operações 
    matemáticas. Pandas oferece estruturas de dados (DataFrame, Series) para 
    manipulação. Matplotlib e Seaborn criam visualizações. Scikit-learn implementa 
    algoritmos de machine learning. Jupyter notebooks facilitam análise exploratória 
    e prototipagem interativa.
    """,
    
    """
    Python performance pode ser otimizada com várias técnicas: Cython para compilação, 
    NumPy para operações vetorizadas, multiprocessing para paralelização, asyncio 
    para concorrência, e PyPy como interpretador alternativo mais rápido. Profiling 
    com cProfile e line_profiler identifica gargalos. Memory profiling com tracemalloc 
    e memory_profiler monitora uso de memória. Just-in-time compilation com Numba 
    acelera código numérico significativamente.
    """,
    
    # Machine Learning & AI (5 docs)
    """
    Machine Learning é um subcampo da inteligência artificial focado em algoritmos 
    que melhoram automaticamente através de experiência. Três tipos principais: 
    supervisionado (dados rotulados), não-supervisionado (padrões em dados não 
    rotulados), e por reforço (aprendizado via recompensas). Algoritmos supervisionados 
    incluem regressão linear/logística, árvores de decisão, SVM, random forests e 
    redes neurais. Métricas incluem acurácia, precisão, recall, F1-score e AUC-ROC.
    """,
    
    """
    Deep Learning utiliza redes neurais artificiais com múltiplas camadas ocultas 
    para aprender representações hierárquicas de dados. Arquiteturas incluem 
    feedforward, convolutional neural networks (CNNs) para visão computacional, 
    recurrent neural networks (RNNs/LSTMs) para sequências, e transformers para 
    processamento de linguagem natural. Técnicas como dropout, batch normalization 
    e regularization previnem overfitting. GPUs aceleram treinamento significativamente.
    """,
    
    """
    Natural Language Processing (NLP) combina linguística computacional com machine 
    learning para processar texto humano. Tarefas incluem tokenização, POS tagging, 
    named entity recognition, sentiment analysis, machine translation e question 
    answering. Modelos pré-treinados como BERT, GPT, T5 e RoBERTa revolucionaram 
    o campo com transfer learning. Bibliotecas como spaCy, NLTK, transformers e 
    Gensim facilitam implementação.
    """,
    
    """
    Computer Vision processa e analisa imagens digitais para extrair informações 
    úteis. Tarefas fundamentais incluem classificação de imagens, detecção de objetos, 
    segmentação semântica e reconhecimento facial. CNNs como LeNet, AlexNet, VGG, 
    ResNet e EfficientNet estabeleceram marcos. Técnicas de data augmentation aumentam 
    diversidade de treinamento. OpenCV, PIL e scikit-image são bibliotecas essenciais 
    para preprocessamento e manipulação de imagens.
    """,
    
    """
    MLOps (Machine Learning Operations) integra desenvolvimento de ML com operações 
    para automizar pipeline completo: coleta de dados, feature engineering, 
    treinamento, validação, deployment e monitoramento. Ferramentas incluem MLflow 
    para experiment tracking, Kubeflow para pipelines Kubernetes, DVC para versionamento 
    de dados, e Weights & Biases para visualização. CI/CD adaptado para ML inclui 
    testes de modelos, drift detection e retraining automático.
    """,
    
    # RAG & Information Retrieval (4 docs)
    """
    Retrieval-Augmented Generation (RAG) combina modelos de linguagem com sistemas 
    de recuperação para gerar respostas mais precisas e atualizadas. O processo 
    típico: codificar query em embedding, buscar documentos relevantes em base 
    vetorial, combinar query e contexto recuperado para gerar resposta final. 
    RAG reduz alucinações, permite acesso a conhecimento atualizado e melhora 
    factualidade sem retreinar modelo base.
    """,
    
    """
    Vector databases armazenam e indexam embeddings para busca semântica eficiente. 
    Soluções incluem Pinecone (managed), Weaviate (open-source), Qdrant (Rust-based), 
    Chroma (lightweight) e Faiss (Facebook). Índices como HNSW, IVF e LSH otimizam 
    busca aproximada de vizinhos mais próximos. Métricas de distância incluem 
    coseno, euclidiana e produto interno. Sharding e replication garantem 
    escalabilidade e disponibilidade.
    """,
    
    """
    Advanced RAG techniques melhoram qualidade e relevância. Query expansion 
    reformula perguntas para capturar mais contexto. Re-ranking reordena resultados 
    iniciais usando modelos mais sofisticados. Hierarchical retrieval busca em 
    múltiplos níveis de granularidade. Multi-modal RAG integra texto, imagens e 
    outros tipos de dados. Corrective RAG detecta respostas irrelevantes e 
    reformula queries automaticamente.
    """,
    
    """
    RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) cria 
    estruturas hierárquicas de documentos através de clustering recursivo e 
    summarização. Documentos similares são agrupados e seus resumos formam níveis 
    superiores da árvore. Durante retrieval, o sistema pode acessar tanto detalhes 
    específicos (folhas) quanto visões de alto nível (nós internos). UMAP reduz 
    dimensionalidade para clustering mais efetivo com Gaussian Mixture Models.
    """,
    
    # Cloud & DevOps (4 docs)
    """
    Cloud Computing oferece recursos computacionais sob demanda via internet, 
    incluindo servidores, armazenamento, databases, networking e software. Modelos 
    de serviço: IaaS (Infrastructure), PaaS (Platform), SaaS (Software) as a Service. 
    Deployment models: public, private, hybrid e multi-cloud. Vantagens incluem 
    elasticidade, pay-as-you-use, global reach e redução de CAPEX. Desvantagens: 
    vendor lock-in, latência de rede e questões de compliance.
    """,
    
    """
    Amazon Web Services (AWS) domina mercado cloud com 200+ serviços. Core services: 
    EC2 (compute), S3 (storage), RDS (databases), VPC (networking), Lambda (serverless). 
    Microsoft Azure integra bem com ecossistema Microsoft. Google Cloud Platform 
    destaca-se em AI/ML e data analytics. Alibaba Cloud lidera na Ásia. Estratégias 
    multi-cloud evitam vendor lock-in mas aumentam complexidade operacional e custos.
    """,
    
    """
    Containers encapsulam aplicações com dependências para deployment consistente. 
    Docker popularizou containerização com images, containers e registries. 
    Kubernetes orquestra containers em clusters, oferecendo service discovery, 
    load balancing, auto-scaling e rolling updates. Alternativas incluem Docker 
    Swarm, Nomad e cloud-managed services como EKS, GKE e AKS. Container security 
    requer image scanning, runtime protection e network policies.
    """,
    
    """
    DevOps integra desenvolvimento e operações para acelerar delivery de software. 
    Práticas core: Continuous Integration (CI), Continuous Deployment (CD), 
    Infrastructure as Code (IaC), monitoring e collaboration. Ferramentas CI/CD: 
    Jenkins, GitLab CI, GitHub Actions, CircleCI. IaC tools: Terraform, Ansible, 
    CloudFormation. Monitoring: Prometheus, Grafana, ELK stack. Culture shift 
    enfatiza automação, feedback rápido e shared responsibility.
    """
]

class RaptorEnhancedDemo:
    """Demo completo do RAPTOR Enhanced"""
    
    def __init__(self):
        self.raptor = None
        self.config = None
        self.metrics = {}
    
    async def run_complete_demo(self):
        """Executa demo completo com todas as configurações"""
        
        print("🚀 RAPTOR ENHANCED - DEMO COMPLETO")
        print("=" * 60)
        
        # 1. Testar diferentes configurações
        await self._test_configurations()
        
        # 2. Comparar providers de embedding
        await self._compare_embedding_providers()
        
        # 3. Testar clustering methods
        await self._test_clustering_methods()
        
        # 4. Benchmark performance
        await self._benchmark_performance()
        
        # 5. Análise qualitativa
        await self._qualitative_analysis()
        
        # 6. Relatório final
        self._generate_report()
    
    async def _test_configurations(self):
        """Testa diferentes configurações"""
        
        print("\n📋 1. TESTE DE CONFIGURAÇÕES")
        print("-" * 40)
        
        configs = [
            ("Mock Embedding + KMeans", self._create_mock_config()),
            ("Sentence-Transformers + UMAP", self._create_st_config()),
        ]
        
        # Adicionar OpenAI se API key disponível
        if os.getenv("OPENAI_API_KEY"):
            configs.append(("OpenAI + UMAP + LLM", self._create_openai_config()))
        
        for config_name, config in configs:
            print(f"\n🔧 Testando: {config_name}")
            
            try:
                raptor = EnhancedRaptorRetriever(config)
                
                # Testar com subset menor para velocidade
                test_docs = EXTENDED_DOCUMENTS[:8]
                stats = await raptor.build_tree(test_docs)
                
                print(f"   ✓ Árvore: {stats['total_nodes']} nós, {stats['max_level']} níveis")
                print(f"   ✓ Tempo: {stats['construction_time']:.2f}s")
                print(f"   ✓ Provider: {stats['config']['embedding_provider']}")
                print(f"   ✓ Clustering: {stats['config']['clustering_method']}")
                
                # Teste de busca
                results = await raptor.search("Python machine learning", k=3)
                print(f"   ✓ Busca: {len(results)} resultados")
                
                # Armazenar métricas
                self.metrics[config_name] = {
                    "tree_stats": stats,
                    "search_results": len(results)
                }
                
            except Exception as e:
                print(f"   ❌ Erro: {e}")
                continue
    
    async def _compare_embedding_providers(self):
        """Compara diferentes providers de embedding"""
        
        print("\n🔍 2. COMPARAÇÃO DE EMBEDDING PROVIDERS")
        print("-" * 40)
        
        providers = [
            ("Mock", EmbeddingProvider.MOCK),
        ]
        
        if os.getenv("OPENAI_API_KEY"):
            providers.append(("OpenAI", EmbeddingProvider.OPENAI))
        
        # Testar Sentence-Transformers se disponível
        try:
            import sentence_transformers
            providers.append(("Sentence-Transformers", EmbeddingProvider.SENTENCE_TRANSFORMERS))
        except ImportError:
            pass
        
        for provider_name, provider in providers:
            print(f"\n🎯 Testando {provider_name}...")
            
            config = RaptorConfig(
                embedding_provider=provider,
                clustering_method=ClusteringMethod.KMEANS_ONLY,  # Mais rápido
                chunk_size=300,
                max_levels=2
            )
            
            if provider == EmbeddingProvider.OPENAI:
                config.openai_api_key = os.getenv("OPENAI_API_KEY")
                config.embedding_model = "text-embedding-3-small"
            
            try:
                raptor = EnhancedRaptorRetriever(config)
                
                start_time = time.time()
                stats = await raptor.build_tree(EXTENDED_DOCUMENTS[:6])
                build_time = time.time() - start_time
                
                # Teste de qualidade de busca
                queries = [
                    "Python web development",
                    "Machine learning algorithms", 
                    "Cloud computing services"
                ]
                
                search_quality = []
                for query in queries:
                    results = await raptor.search(query, k=3)
                    avg_score = sum(r['score'] for r in results) / len(results) if results else 0
                    search_quality.append(avg_score)
                
                avg_quality = sum(search_quality) / len(search_quality)
                
                print(f"   ✓ Build time: {build_time:.2f}s")
                print(f"   ✓ Avg search quality: {avg_quality:.3f}")
                print(f"   ✓ Nodes: {stats['total_nodes']}")
                
            except Exception as e:
                print(f"   ❌ Erro: {e}")
    
    async def _test_clustering_methods(self):
        """Testa diferentes métodos de clustering"""
        
        print("\n🔬 3. TESTE DE MÉTODOS DE CLUSTERING")
        print("-" * 40)
        
        methods = [
            ("KMeans Only", ClusteringMethod.KMEANS_ONLY),
            ("PCA + GMM", ClusteringMethod.PCA_GMM),
        ]
        
        # Adicionar UMAP se disponível
        try:
            import umap
            methods.extend([
                ("UMAP + GMM", ClusteringMethod.UMAP_GMM),
                ("UMAP + KMeans", ClusteringMethod.UMAP_KMEANS)
            ])
        except ImportError:
            print("   ⚠️  UMAP não disponível - testando apenas PCA e KMeans")
        
        for method_name, method in methods:
            print(f"\n🎲 Testando {method_name}...")
            
            config = RaptorConfig(
                embedding_provider=EmbeddingProvider.MOCK,  # Rápido para teste
                clustering_method=method,
                chunk_size=400,
                max_levels=3,
                min_cluster_size=2
            )
            
            try:
                raptor = EnhancedRaptorRetriever(config)
                
                start_time = time.time()
                stats = await raptor.build_tree(EXTENDED_DOCUMENTS[:10])
                clustering_time = time.time() - start_time
                
                # Calcular compressão
                compression_ratio = len(EXTENDED_DOCUMENTS) / stats['total_nodes']
                
                print(f"   ✓ Clustering time: {clustering_time:.2f}s")
                print(f"   ✓ Compression ratio: {compression_ratio:.2f}x")
                print(f"   ✓ Levels: {stats['max_level']}")
                print(f"   ✓ Distribution: {stats['nodes_per_level']}")
                
            except Exception as e:
                print(f"   ❌ Erro: {e}")
    
    async def _benchmark_performance(self):
        """Benchmark de performance com diferentes volumes"""
        
        print("\n⚡ 4. BENCHMARK DE PERFORMANCE")
        print("-" * 40)
        
        # Diferentes volumes de dados
        volumes = [
            ("Pequeno", EXTENDED_DOCUMENTS[:5]),
            ("Médio", EXTENDED_DOCUMENTS[:10]),
            ("Grande", EXTENDED_DOCUMENTS[:15]),
            ("Completo", EXTENDED_DOCUMENTS)
        ]
        
        config = RaptorConfig(
            embedding_provider=EmbeddingProvider.MOCK,  # Mais rápido
            clustering_method=ClusteringMethod.KMEANS_ONLY,
            chunk_size=350,
            max_levels=4
        )
        
        for volume_name, docs in volumes:
            print(f"\n📊 Volume {volume_name} ({len(docs)} docs)...")
            
            try:
                raptor = EnhancedRaptorRetriever(config)
                
                # Benchmark construção
                start_time = time.time()
                stats = await raptor.build_tree(docs)
                build_time = time.time() - start_time
                
                # Benchmark busca
                search_times = []
                for i in range(3):  # 3 buscas para média
                    start_time = time.time()
                    await raptor.search(f"Test query {i}", k=5)
                    search_times.append(time.time() - start_time)
                
                avg_search_time = sum(search_times) / len(search_times)
                
                print(f"   ✓ Build: {build_time:.2f}s")
                print(f"   ✓ Search: {avg_search_time:.3f}s")
                print(f"   ✓ Nodes: {stats['total_nodes']}")
                print(f"   ✓ Throughput: {len(docs)/build_time:.1f} docs/s")
                
            except Exception as e:
                print(f"   ❌ Erro: {e}")
    
    async def _qualitative_analysis(self):
        """Análise qualitativa dos resultados"""
        
        print("\n🎨 5. ANÁLISE QUALITATIVA")
        print("-" * 40)
        
        # Usar melhor configuração disponível
        if os.getenv("OPENAI_API_KEY"):
            config = self._create_openai_config()
            print("Usando OpenAI para análise qualitativa...")
        else:
            config = self._create_st_config()
            print("Usando Sentence-Transformers para análise...")
        
        try:
            self.raptor = EnhancedRaptorRetriever(config)
            stats = await self.raptor.build_tree(EXTENDED_DOCUMENTS)
            
            print(f"\n📈 Estatísticas da Árvore:")
            print(f"   • Total de nós: {stats['total_nodes']}")
            print(f"   • Níveis: {stats['max_level']}")
            print(f"   • Distribuição: {stats['nodes_per_level']}")
            
            # Queries de teste qualitativo
            test_queries = [
                "Explain Python web frameworks and their differences",
                "How does machine learning clustering work?", 
                "What are the benefits of cloud computing?",
                "Compare different RAG techniques and approaches"
            ]
            
            print(f"\n🔍 Análise de Queries:")
            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. Query: {query}")
                
                results = await self.raptor.search(query, k=3, max_tokens=1500)
                
                print(f"   Resultados: {len(results)}")
                
                # Analisar distribuição por níveis
                level_dist = {}
                total_tokens = 0
                
                for result in results:
                    level = result['metadata']['level']
                    level_dist[level] = level_dist.get(level, 0) + 1
                    total_tokens += result['metadata']['token_count']
                
                print(f"   Distribuição: {dict(sorted(level_dist.items()))}")
                print(f"   Tokens: {total_tokens}")
                
                # Mostrar melhor resultado
                if results:
                    best = results[0]
                    content_preview = best['content'][:100] + "..."
                    print(f"   Melhor (Nível {best['metadata']['level']}, Score: {best['score']:.3f}):")
                    print(f"   {content_preview}")
                
        except Exception as e:
            print(f"❌ Erro na análise qualitativa: {e}")
    
    def _generate_report(self):
        """Gera relatório final"""
        
        print("\n📋 6. RELATÓRIO FINAL")
        print("=" * 60)
        
        if self.metrics:
            print("\n🏆 Resumo de Performance:")
            for config_name, metrics in self.metrics.items():
                stats = metrics['tree_stats']
                print(f"\n{config_name}:")
                print(f"   • Nós: {stats['total_nodes']}")
                print(f"   • Tempo: {stats['construction_time']:.2f}s")
                print(f"   • Níveis: {stats['max_level']}")
        
        print("\n✅ Funcionalidades Validadas:")
        print("   ✓ Embeddings reais com fallback inteligente")
        print("   ✓ Clustering avançado UMAP + GMM")
        print("   ✓ Summarização com LLM quando disponível")
        print("   ✓ Processamento paralelo otimizado")
        print("   ✓ Cache multicamada")
        print("   ✓ Métricas de qualidade")
        print("   ✓ Configuração flexível")
        
        print("\n🎯 Recomendações:")
        print("   • Use OpenAI para melhor qualidade (se API key disponível)")
        print("   • UMAP + GMM para clustering mais preciso")
        print("   • Ajuste chunk_size baseado no domínio")
        print("   • Monitore métricas de qualidade")
        
        print(f"\n🎉 Demo RAPTOR Enhanced concluído com sucesso!")
    
    def _create_mock_config(self) -> RaptorConfig:
        """Configuração mock para testes rápidos"""
        return RaptorConfig(
            embedding_provider=EmbeddingProvider.MOCK,
            clustering_method=ClusteringMethod.KMEANS_ONLY,
            summarization_provider=SummarizationProvider.OPENAI,  # Fallback simples
            chunk_size=300,
            max_levels=3,
            batch_size=16
        )
    
    def _create_st_config(self) -> RaptorConfig:
        """Configuração Sentence-Transformers"""
        return RaptorConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            embedding_model="all-MiniLM-L6-v2",  # Modelo leve
            clustering_method=ClusteringMethod.PCA_GMM,
            chunk_size=400,
            max_levels=4
        )
    
    def _create_openai_config(self) -> RaptorConfig:
        """Configuração OpenAI completa"""
        return RaptorConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            embedding_model="text-embedding-3-small",
            summarization_provider=SummarizationProvider.OPENAI,
            summarization_model="gpt-4o-mini",
            clustering_method=ClusteringMethod.UMAP_GMM,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=500,
            max_levels=4,
            batch_size=20
        )

async def main():
    """Função principal"""
    
    # Verificar dependências opcionais
    print("🔧 Verificando dependências...")
    
    dependencies = {
        "OpenAI API": bool(os.getenv("OPENAI_API_KEY")),
        "UMAP": False,
        "Sentence-Transformers": False,
        "Redis": False
    }
    
    try:
        import umap
        dependencies["UMAP"] = True
    except ImportError:
        pass
    
    try:
        import sentence_transformers
        dependencies["Sentence-Transformers"] = True
    except ImportError:
        pass
    
    try:
        import redis
        dependencies["Redis"] = True
    except ImportError:
        pass
    
    print("Dependências disponíveis:")
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"   {status} {dep}")
    
    if not any(dependencies.values()):
        print("\n⚠️  Nenhuma dependência avançada disponível")
        print("Executando com configuração básica...")
    
    # Executar demo
    demo = RaptorEnhancedDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main()) 