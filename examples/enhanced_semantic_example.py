"""
Exemplo de uso do Enhanced Semantic Chunker
Demonstra as melhorias incorporadas da proposta original
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking.semantic_chunker_enhanced import EnhancedSemanticChunker, create_semantic_chunker

def exemplo_basico():
    """Exemplo básico de uso do Enhanced Semantic Chunker"""
    
    # Texto de exemplo em português
    texto = """
    O processamento de linguagem natural é uma área fascinante da inteligência artificial. 
    Ela envolve o desenvolvimento de algoritmos capazes de compreender e processar textos humanos.
    
    Uma das aplicações mais importantes é o chunking semântico. Esta técnica divide documentos 
    em segmentos coerentes baseados no significado. Diferente do chunking simples por tamanho,
    o chunking semântico preserva a coesão temática.
    
    Os modelos de embedding como BERT e sentence-transformers revolucionaram esta área.
    Eles conseguem capturar relações semânticas complexas entre palavras e frases.
    Isso permite agrupar conteúdo relacionado de forma mais inteligente.
    
    As aplicações práticas incluem sistemas de busca, resumo automático e chatbots.
    Todos esses sistemas se beneficiam de uma segmentação mais precisa do texto.
    """
    
    print("🚀 Enhanced Semantic Chunker - Exemplo Básico\n")
    
    # Criar chunker aprimorado
    chunker = EnhancedSemanticChunker(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.6,
        max_chunk_size=300,
        language="portuguese",
        use_centroids=True
    )
    
    # Processar texto
    chunks = chunker.chunk(texto, {"document_id": "exemplo_1", "source": "manual"})
    
    print(f"📄 Texto original: {len(texto)} caracteres")
    print(f"🔨 Chunks gerados: {len(chunks)}")
    print("=" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📦 Chunk {i}:")
        print(f"   Tamanho: {len(chunk.content)} caracteres")
        print(f"   Sentenças: {chunk.metadata.get('sentence_count', 'N/A')}")
        print(f"   Conteúdo: {chunk.content[:100]}...")
        print(f"   ID: {chunk.chunk_id[:8]}...")

def exemplo_comparativo():
    """Compara o Enhanced com implementação original"""
    
    print("\n🔄 Comparação: Enhanced vs Original\n")
    
    texto = """
    Python é uma linguagem de programação versátil. É amplamente usada em ciência de dados.
    Machine learning tornou-se muito popular recentemente. Bibliotecas como scikit-learn facilitam o desenvolvimento.
    O processamento de texto é uma aplicação comum. NLTK e spaCy são ferramentas populares para isso.
    """
    
    # Enhanced Chunker
    enhanced = EnhancedSemanticChunker(
        similarity_threshold=0.5,
        max_chunk_size=150
    )
    
    chunks_enhanced = enhanced.chunk(texto, {"source": "teste"})
    
    print("🆕 Enhanced Semantic Chunker:")
    for i, chunk in enumerate(chunks_enhanced, 1):
        print(f"   Chunk {i}: {len(chunk.content)} chars - {chunk.content[:50]}...")
    
    # Usando método compatível com proposta original
    print("\n🔄 Método compatível com proposta original:")
    chunks_simples = enhanced.semantic_chunking(texto, max_chunk_size=150)
    
    for i, chunk in enumerate(chunks_simples, 1):
        print(f"   Chunk {i}: {len(chunk)} chars - {chunk[:50]}...")

def exemplo_configuracoes_avancadas():
    """Demonstra configurações avançadas"""
    
    print("\n⚙️ Configurações Avançadas\n")
    
    texto = """
    Artificial intelligence has revolutionized many industries. Machine learning algorithms can now process vast amounts of data.
    Natural language processing is particularly exciting. It enables computers to understand human language.
    Deep learning models like transformers have shown remarkable results. They can generate text that is almost indistinguishable from human writing.
    """
    
    # Configurações diferentes
    configs = [
        {"name": "Conservador", "threshold": 0.8, "centroids": True},
        {"name": "Balanceado", "threshold": 0.6, "centroids": True},
        {"name": "Agressivo", "threshold": 0.4, "centroids": False}
    ]
    
    for config in configs:
        print(f"\n📋 {config['name']} (threshold={config['threshold']}, centroids={config['centroids']}):")
        
        chunker = EnhancedSemanticChunker(
            similarity_threshold=config['threshold'],
            max_chunk_size=200,
            language="english",
            use_centroids=config['centroids']
        )
        
        chunks = chunker.chunk(texto, {"config": config['name']})
        
        print(f"   Chunks gerados: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"   Chunk {i}: {len(chunk.content)} chars")

def exemplo_compatibilidade():
    """Demonstra compatibilidade com interface proposta"""
    
    print("\n🔗 Compatibilidade com Interface Original\n")
    
    # Usando função de conveniência (como na proposta)
    chunker = create_semantic_chunker(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.6
    )
    
    texto = "Esta é uma sentença. Esta é outra sentença relacionada. Esta é uma sentença diferente sobre outro tópico."
    
    # Método compatível com a proposta original
    chunks = chunker.semantic_chunking(texto, max_chunk_size=512)
    
    print(f"📄 Chunks usando interface compatível:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   {i}: {chunk}")

if __name__ == "__main__":
    exemplo_basico()
    exemplo_comparativo()
    exemplo_configuracoes_avancadas()
    exemplo_compatibilidade()
    
    print("\n" + "="*60)
    print("✅ Enhanced Semantic Chunker oferece:")
    print("   • NLTK para divisão de sentenças mais precisa")
    print("   • Cálculo de centroides para melhor representação")
    print("   • Suporte nativo ao português")
    print("   • Cache LRU para performance")
    print("   • Metadados ricos e UUIDs")
    print("   • Compatibilidade com sistema existente")
    print("   • Interface compatível com proposta original") 