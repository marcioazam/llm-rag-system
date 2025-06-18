#!/usr/bin/env python3
"""
Quick Start Script - Hybrid Search com Qdrant 1.8.0
Configura e executa sistema de busca híbrida em minutos
"""

import asyncio
import logging
import sys
from pathlib import Path
import os

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.hybrid_indexing_pipeline import HybridIndexingPipeline
from src.retrieval.hybrid_retriever import HybridRetriever

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridSearchQuickStart:
    """
    Quick start para sistema de busca híbrida
    """
    
    def __init__(self):
        self.indexing_pipeline = None
        self.retriever = None
        
    async def setup_system(self):
        """
        Configura sistema híbrido
        """
        print("🚀 Configurando Sistema de Busca Híbrida")
        print("=" * 50)
        
        # Verificar configurações
        await self._check_requirements()
        
        # Inicializar componentes
        print("📦 Inicializando componentes...")
        self.indexing_pipeline = HybridIndexingPipeline()
        self.retriever = HybridRetriever()
        
        print("✅ Sistema configurado!")
        
    async def _check_requirements(self):
        """
        Verifica requisitos do sistema
        """
        print("🔍 Verificando requisitos...")
        
        # Verificar API keys
        required_env_vars = ["OPENAI_API_KEY"]
        missing_vars = []
        
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print("❌ Variáveis de ambiente faltando:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\n💡 Configure as variáveis antes de continuar:")
            print("   export OPENAI_API_KEY='sua_key_aqui'")
            sys.exit(1)
        
        # Verificar diretórios
        config_dir = Path("config")
        if not config_dir.exists():
            config_dir.mkdir(parents=True)
            print("📁 Criado diretório config/")
        
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            print("📁 Criado diretório data/")
        
        print("✅ Requisitos verificados!")
    
    async def quick_demo(self):
        """
        Demonstração rápida do sistema
        """
        print("\n🎯 Executando Demonstração Rápida")
        print("=" * 40)
        
        # Criar documentos de exemplo
        await self._create_sample_docs()
        
        # Indexar documentos
        await self._quick_indexing()
        
        # Testar buscas
        await self._test_searches()
        
    async def _create_sample_docs(self):
        """
        Cria documentos de exemplo
        """
        print("📄 Criando documentos de exemplo...")
        
        sample_docs = {
            "rag_basics.txt": """
            RAG (Retrieval-Augmented Generation) é uma técnica que combina recuperação de informações
            com geração de texto usando LLMs. Os componentes principais são:
            
            1. Vector Database - armazena embeddings
            2. Retriever - busca documentos relevantes  
            3. Generator - LLM que gera respostas
            
            Vantagens: conhecimento atualizado, menos alucinações, fontes transparentes.
            """,
            
            "python_tips.txt": """
            Python é uma linguagem versátil para desenvolvimento. Dicas importantes:
            
            - Use list comprehensions para código mais limpo
            - Aproveite decorators para funcionalidade cross-cutting
            - Implemente context managers com 'with'
            - Use type hints para melhor documentação
            - Prefira f-strings para formatação de strings
            
            Exemplo: def greet(name: str) -> str: return f"Hello, {name}!"
            """,
            
            "hybrid_search.txt": """
            Hybrid search combina dense retrieval (embeddings semânticos) com sparse retrieval (BM25).
            
            Dense vectors capturam similaridade semântica.
            Sparse vectors fazem matching exato de keywords.
            
            Qdrant 1.8.0 trouxe 16x improvement em sparse vector search.
            RRF (Reciprocal Rank Fusion) combina os resultados eficientemente.
            """
        }
        
        # Salvar documentos
        data_dir = Path("data/quick_start")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, content in sample_docs.items():
            file_path = data_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
        
        print(f"✅ Criados {len(sample_docs)} documentos")
    
    async def _quick_indexing(self):
        """
        Indexação rápida dos documentos
        """
        print("🔄 Indexando documentos...")
        
        # Encontrar documentos
        data_dir = Path("data/quick_start")
        doc_paths = [str(p) for p in data_dir.glob("*.txt")]
        
        # Indexar
        stats = await self.indexing_pipeline.index_documents(doc_paths)
        
        print(f"✅ Indexação concluída!")
        print(f"   📊 {stats['documents_processed']} documentos")
        print(f"   📊 {stats['chunks_created']} chunks")
        print(f"   ⏱️  {stats['indexing_time']:.2f}s")
    
    async def _test_searches(self):
        """
        Testa diferentes tipos de busca
        """
        print("🔍 Testando buscas...")
        
        test_queries = [
            {
                "query": "O que é RAG?",
                "description": "Query semântica"
            },
            {
                "query": "Python função exemplo código",
                "description": "Query com keywords"
            },
            {
                "query": "Qdrant 1.8.0 sparse vectors performance",
                "description": "Query híbrida"
            }
        ]
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n--- Teste {i}: {test['description']} ---")
            print(f"Query: '{test['query']}'")
            
            # Executar busca
            results = await self.retriever.retrieve(
                query=test['query'],
                limit=2,
                strategy="auto"
            )
            
            if results:
                top_result = results[0]
                print(f"✅ Encontrado: {len(results)} resultados")
                print(f"   🎯 Score: {top_result.combined_score:.3f}")
                print(f"   📄 Trecho: {top_result.content[:100]}...")
            else:
                print("❌ Nenhum resultado encontrado")
    
    async def interactive_mode(self):
        """
        Modo interativo para testes
        """
        print("\n🎮 Modo Interativo")
        print("=" * 30)
        print("Digite suas queries (ou 'quit' para sair)")
        
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                # Executar busca
                results = await self.retriever.retrieve(
                    query=query,
                    limit=3,
                    strategy="auto"
                )
                
                print(f"\n📊 Resultados: {len(results)}")
                
                for i, result in enumerate(results, 1):
                    print(f"\n--- Resultado {i} ---")
                    print(f"Score: {result.combined_score:.3f}")
                    print(f"Método: {result.retrieval_method}")
                    print(f"Conteúdo: {result.content[:200]}...")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        print("\n👋 Saindo do modo interativo")
    
    def show_next_steps(self):
        """
        Mostra próximos passos
        """
        print("\n🎉 Quick Start Concluído!")
        print("=" * 40)
        print("\n📚 Próximos passos:")
        print("1. 📖 Leia o guia completo: HYBRID_SEARCH_IMPLEMENTATION_GUIDE.md")
        print("2. 🧪 Execute testes: python scripts/test_hybrid_performance.py")
        print("3. 🎯 Veja exemplo completo: python examples/hybrid_search_example.py")
        print("4. ⚙️  Configure para produção: config/hybrid_search_config.yaml")
        
        print("\n🔗 Recursos úteis:")
        print("- Documentação Qdrant: https://qdrant.tech/documentation/")
        print("- RAG Techniques: https://github.com/FareedKhan-dev/all-rag-techniques")
        print("- Qdrant 1.8.0: https://qdrant.tech/articles/qdrant-1.8.x/")

async def main():
    """
    Função principal do quick start
    """
    print("🚀 HYBRID SEARCH QUICK START")
    print("Qdrant 1.8.0 + Sparse Vectors + Dense Embeddings")
    print("=" * 60)
    
    quick_start = HybridSearchQuickStart()
    
    try:
        # Configurar sistema
        await quick_start.setup_system()
        
        # Executar demo
        await quick_start.quick_demo()
        
        # Modo interativo
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            await quick_start.interactive_mode()
        
        # Mostrar próximos passos
        quick_start.show_next_steps()
        
    except Exception as e:
        logger.error(f"Erro no quick start: {e}")
        print(f"\n❌ Erro: {e}")
        print("\n🔧 Soluções possíveis:")
        print("1. Verifique se OPENAI_API_KEY está configurada")
        print("2. Instale dependências: pip install -r requirements.txt")
        print("3. Verifique logs para mais detalhes")

if __name__ == "__main__":
    asyncio.run(main()) 