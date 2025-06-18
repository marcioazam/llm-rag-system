"""
Setup Script para RAPTOR Enhanced

Instala e configura todas as dependências necessárias para
executar o RAPTOR Enhanced com todas as funcionalidades.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package: str, description: str = ""):
    """Instala um pacote via pip"""
    print(f"📦 Instalando {package}...")
    if description:
        print(f"   {description}")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"   ✅ {package} instalado com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erro ao instalar {package}: {e}")
        return False

def check_package(package: str) -> bool:
    """Verifica se um pacote está instalado"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def setup_environment():
    """Configura ambiente para RAPTOR Enhanced"""
    
    print("🚀 SETUP RAPTOR ENHANCED")
    print("=" * 50)
    
    # Dependências core (obrigatórias)
    core_deps = [
        ("numpy", "Computação numérica"),
        ("scikit-learn", "Machine learning básico"),
        ("asyncio", "Programação assíncrona")  # Built-in, mas verificar
    ]
    
    # Dependências avançadas (opcionais mas recomendadas)
    advanced_deps = [
        ("umap-learn", "UMAP para clustering avançado"),
        ("sentence-transformers", "Embeddings de alta qualidade"),
        ("openai", "Integração com OpenAI"),
        ("anthropic", "Integração com Claude"),
        ("redis", "Cache avançado"),
        ("tiktoken", "Tokenização OpenAI"),
        ("google-generativeai", "Integração com Gemini"),
        ("httpx", "Cliente HTTP assíncrono"),
        ("aiohttp", "Servidor HTTP assíncrono"),
        ("rich", "Output colorido"),
        ("tqdm", "Barras de progresso")
    ]
    
    print("\n📋 1. VERIFICANDO DEPENDÊNCIAS CORE")
    print("-" * 30)
    
    core_ok = True
    for package, desc in core_deps:
        if package == "asyncio":
            # asyncio é built-in no Python 3.7+
            if sys.version_info >= (3, 7):
                print(f"   ✅ {package} - {desc}")
            else:
                print(f"   ❌ Python 3.7+ necessário para asyncio")
                core_ok = False
        else:
            if check_package(package):
                print(f"   ✅ {package} - {desc}")
            else:
                print(f"   ❌ {package} - {desc}")
                if not install_package(package, desc):
                    core_ok = False
    
    if not core_ok:
        print("\n❌ Erro: Dependências core não puderam ser instaladas")
        return False
    
    print("\n📦 2. INSTALANDO DEPENDÊNCIAS AVANÇADAS")
    print("-" * 30)
    
    installed_advanced = []
    failed_advanced = []
    
    for package, desc in advanced_deps:
        if check_package(package.split("[")[0]):  # Remove extras se existirem
            print(f"   ✅ {package} já instalado - {desc}")
            installed_advanced.append(package)
        else:
            if install_package(package, desc):
                installed_advanced.append(package)
            else:
                failed_advanced.append((package, desc))
    
    print(f"\n📊 3. RESUMO DA INSTALAÇÃO")
    print("-" * 30)
    print(f"✅ Instalados: {len(installed_advanced)}")
    print(f"❌ Falhas: {len(failed_advanced)}")
    
    if failed_advanced:
        print("\n⚠️  Pacotes que falharam:")
        for package, desc in failed_advanced:
            print(f"   • {package} - {desc}")
        print("\nVocê pode tentar instalar manualmente:")
        for package, _ in failed_advanced:
            print(f"   pip install {package}")
    
    return True

def create_env_template():
    """Cria template do arquivo .env"""
    
    print("\n🔧 4. CONFIGURAÇÃO DE AMBIENTE")
    print("-" * 30)
    
    env_file = Path(".env")
    
    if env_file.exists():
        print("   ✅ Arquivo .env já existe")
        return
    
    env_template = """# RAPTOR Enhanced - Configuração de Ambiente

# APIs de LLM (pelo menos uma é recomendada)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Redis (opcional - para cache avançado)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=2
REDIS_PASSWORD=

# Configurações de Performance
MAX_WORKERS=4
BATCH_SIZE=32
CACHE_TTL=3600

# Logging
LOG_LEVEL=INFO
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_template)
        print("   ✅ Arquivo .env criado")
        print("   📝 Configure suas API keys no arquivo .env")
    except Exception as e:
        print(f"   ❌ Erro ao criar .env: {e}")

def test_installation():
    """Testa a instalação"""
    
    print("\n🧪 5. TESTE DE INSTALAÇÃO")
    print("-" * 30)
    
    # Teste básico de import
    tests = [
        ("numpy", "import numpy as np"),
        ("sklearn", "from sklearn.cluster import KMeans"),
        ("asyncio", "import asyncio"),
    ]
    
    # Testes opcionais
    optional_tests = [
        ("umap", "import umap"),
        ("sentence_transformers", "from sentence_transformers import SentenceTransformer"),
        ("openai", "import openai"),
        ("redis", "import redis"),
    ]
    
    all_passed = True
    
    print("Testes básicos:")
    for name, test_code in tests:
        try:
            exec(test_code)
            print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
            all_passed = False
    
    print("\nTestes opcionais:")
    optional_available = 0
    for name, test_code in optional_tests:
        try:
            exec(test_code)
            print(f"   ✅ {name}")
            optional_available += 1
        except Exception as e:
            print(f"   ⚠️  {name}: {e}")
    
    print(f"\n📈 Funcionalidades disponíveis: {optional_available}/{len(optional_tests)} opcionais")
    
    return all_passed, optional_available

def show_usage_examples():
    """Mostra exemplos de uso"""
    
    print("\n📚 6. EXEMPLOS DE USO")
    print("-" * 30)
    
    examples = [
        ("Demo básico", "python demo_raptor_enhanced.py"),
        ("Demo original funcional", "python demo_raptor_working.py"),
        ("Teste implementação", "python test_raptor_implementation.py"),
    ]
    
    print("Comandos para executar:")
    for desc, cmd in examples:
        print(f"   • {desc}:")
        print(f"     {cmd}")
    
    print(f"\n💡 Dicas:")
    print("   • Configure API keys no arquivo .env para melhor experiência")
    print("   • Use OpenAI para summarização de alta qualidade")
    print("   • UMAP + GMM oferece clustering mais preciso")
    print("   • Monitore logs para debug e otimização")

def main():
    """Função principal"""
    
    try:
        # 1. Setup ambiente
        if not setup_environment():
            print("\n❌ Setup falhou")
            return 1
        
        # 2. Criar template .env
        create_env_template()
        
        # 3. Testar instalação
        basic_ok, optional_count = test_installation()
        
        # 4. Mostrar exemplos
        show_usage_examples()
        
        # 5. Relatório final
        print("\n" + "=" * 50)
        if basic_ok:
            print("✅ SETUP CONCLUÍDO COM SUCESSO!")
            print(f"🎯 {optional_count} funcionalidades avançadas disponíveis")
            
            if optional_count >= 3:
                print("🚀 Sistema totalmente otimizado!")
            elif optional_count >= 1:
                print("⚡ Sistema funcionalmente completo")
            else:
                print("📝 Sistema básico - considere instalar dependências opcionais")
                
        else:
            print("⚠️  Setup parcial - algumas dependências core falharam")
            return 1
        
        print("\n🎉 Pronto para usar RAPTOR Enhanced!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelado pelo usuário")
        return 1
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 