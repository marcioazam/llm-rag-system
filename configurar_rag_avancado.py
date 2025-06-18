#!/usr/bin/env python3
"""
Script de Configuração: Sistema RAG Avançado 100% API
Configura automaticamente o sistema para usar apenas APIs externas
"""

import os
import shutil
from pathlib import Path

def main():
    print("🚀 CONFIGURANDO SISTEMA RAG AVANÇADO 100% API")
    print("=" * 60)
    
    # 1. Criar arquivo .env se não existir
    env_file = Path(".env")
    env_example = Path("config/env_example.txt")
    
    if not env_file.exists():
        if env_example.exists():
            print("📋 Copiando template .env...")
            shutil.copy(env_example, env_file)
            print("✅ Arquivo .env criado!")
        else:
            print("❌ Template config/env_example.txt não encontrado!")
            return
    else:
        print("✅ Arquivo .env já existe")
    
    # 2. Verificar se pipeline_dependency.py está correto
    dependency_file = Path("src/api/pipeline_dependency.py")
    
    if dependency_file.exists():
        content = dependency_file.read_text(encoding='utf-8')
        if "AdvancedRAGPipeline" in content:
            print("✅ API configurada para usar AdvancedRAGPipeline")
        else:
            print("⚠️ API ainda não está usando AdvancedRAGPipeline")
    
    # 3. Verificar arquivos de configuração
    config_files = [
        "config/llm_providers_config.yaml",
        "config/env_example.txt"
    ]
    
    print("\n📁 Verificando arquivos de configuração:")
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file}")
    
    # 4. Instruções para o usuário
    print("\n" + "=" * 60)
    print("🎯 PRÓXIMOS PASSOS:")
    print("=" * 60)
    
    print("\n1️⃣ CONFIGURE API KEYS:")
    print("   📝 Edite o arquivo .env")
    print("   🔑 Configure pelo menos: OPENAI_API_KEY=sk-sua-key-aqui")
    print("   💡 Outras APIs opcionais: ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY")
    
    print("\n2️⃣ INICIE OS SERVIÇOS:")
    print("   🐳 Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print("   🕸️ Neo4j (opcional): docker run -p 7474:7474 -p 7687:7687 neo4j")
    
    print("\n3️⃣ TESTE O SISTEMA:")
    print("   🧪 Execute: python test_advanced_rag_system.py")
    print("   ✅ Verifique se todos os testes passam")
    
    print("\n4️⃣ INICIE A API:")
    print("   🚀 Execute: uvicorn src.api.main:app --reload")
    print("   🌐 Acesse: http://localhost:8000/docs")
    
    print("\n" + "=" * 60)
    print("📊 RECURSOS DISPONÍVEIS:")
    print("=" * 60)
    print("✅ Corrective RAG - Auto-correção de queries")
    print("✅ Multi-Query RAG - Múltiplas perspectivas")
    print("✅ Adaptive Retrieval - K dinâmico por tipo de query")
    print("✅ Enhanced GraphRAG - Enriquecimento com grafo")
    print("✅ Roteamento Inteligente - 4 provedores LLM")
    print("✅ Controle de Custos - Orçamento e monitoramento")
    print("✅ 100% APIs Externas - Zero modelos locais")
    
    print("\n🎉 CONFIGURAÇÃO CONCLUÍDA!")
    print("💡 Configure suas API keys no .env e teste o sistema!")

if __name__ == "__main__":
    main() 