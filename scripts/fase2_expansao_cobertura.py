#!/usr/bin/env python3
"""
Script para Fase 2: Expansão de Cobertura de Testes
Meta: Alcançar 25-35% de cobertura focando em Chunking + Pipelines + VectorDB
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Executa comando e mostra resultado"""
    print(f"\n🔄 {description}")
    print(f"Comando: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ SUCESSO")
            if result.stdout:
                # Mostrar apenas resumo
                lines = result.stdout.split('\n')
                summary_lines = [l for l in lines if 'passed' in l or 'failed' in l or 'TOTAL' in l or '=' in l][-3:]
                print('\n'.join(summary_lines))
        else:
            print("❌ ERRO")
            if result.stderr:
                print(f"Erro: {result.stderr[:300]}")
            # Tentar mostrar stdout para entender o erro
            if result.stdout:
                error_lines = result.stdout.split('\n')[-10:]
                print("Últimas linhas do output:")
                print('\n'.join(error_lines))
                
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - comando muito demorado")
        return False
    except Exception as e:
        print(f"❌ EXCEÇÃO: {e}")
        return False

def check_coverage():
    """Verifica cobertura atual"""
    print("\n📊 VERIFICANDO COBERTURA ATUAL...")
    result = subprocess.run("python -m coverage report", shell=True, capture_output=True, text=True)
    if result.stdout:
        lines = result.stdout.split('\n')
        total_line = [l for l in lines if 'TOTAL' in l]
        if total_line:
            print(f"📈 {total_line[0]}")

def fase2_chunking_advanced():
    """Fase 2A: Testes avançados de chunking"""
    print("\n🎯 FASE 2A: CHUNKING AVANÇADO")
    
    chunking_tests = [
        "tests/test_advanced_chunker.py",
        "tests/test_chunkers_comprehensive.py",
        "tests/test_language_aware_chunker_comprehensive.py",
        "tests/test_semantic_chunker_enhanced.py"
    ]
    
    for test in chunking_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Chunking avançado: {test}")
            if success:
                check_coverage()

def fase2_pipelines():
    """Fase 2B: RAG Pipelines"""
    print("\n🎯 FASE 2B: RAG PIPELINES")
    
    pipeline_tests = [
        "tests/test_rag_pipeline_base_complete.py",
        "tests/test_rag_pipeline_advanced_basic.py",
        "tests/test_rag_integration.py"
    ]
    
    for test in pipeline_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Pipeline: {test}")
            if success:
                check_coverage()

def fase2_embeddings():
    """Fase 2C: Embedding Services"""
    print("\n🎯 FASE 2C: EMBEDDING SERVICES")
    
    embedding_tests = [
        "tests/test_api_embedding_service_complete.py",
        "tests/test_embedding_service_comprehensive.py",
        "tests/test_embedding_service.py"
    ]
    
    for test in embedding_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Embeddings: {test}")
            if success:
                check_coverage()

def fase2_vectordb():
    """Fase 2D: Vector Databases"""
    print("\n🎯 FASE 2D: VECTOR DATABASES")
    
    vectordb_tests = [
        "tests/test_qdrant_store.py",
        "tests/test_chroma_store.py"
    ]
    
    for test in vectordb_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"VectorDB: {test}")
            if success:
                check_coverage()

def fase2_code_analysis():
    """Fase 2E: Code Analysis"""
    print("\n🎯 FASE 2E: CODE ANALYSIS")
    
    code_tests = [
        "tests/test_python_analyzer.py",
        "tests/test_dependency_analyzer.py",
        "tests/test_base_analyzer_enhanced.py"
    ]
    
    for test in code_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Code Analysis: {test}")
            if success:
                check_coverage()

def fase2_utilities():
    """Fase 2F: Utilities essenciais"""
    print("\n🎯 FASE 2F: UTILITIES ESSENCIAIS")
    
    util_tests = [
        "tests/test_intelligent_preprocessor.py",
        "tests/test_document_loader.py"
    ]
    
    for test in util_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Utilities: {test}")
            if success:
                check_coverage()

def generate_comprehensive_report():
    """Gera relatório abrangente da Fase 2"""
    print("\n📈 GERANDO RELATÓRIO ABRANGENTE DA FASE 2")
    
    # Relatório detalhado por categoria
    print("\n🔍 RELATÓRIO POR CATEGORIA:")
    
    categories = {
        "Chunking": ["chunking"],
        "Embeddings": ["embedding"],
        "CodeAnalysis": ["code_analysis"],
        "Storage": ["storage", "metadata"],
        "Pipelines": ["rag_pipeline"],
        "Utils": ["utils", "preprocessing"]
    }
    
    for category, patterns in categories.items():
        print(f"\n📊 {category.upper()}:")
        for pattern in patterns:
            cmd = f"python -m coverage report --include='*{pattern}*'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout and "TOTAL" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if pattern in line or "TOTAL" in line:
                        print(f"  {line}")
    
    run_command("python -m coverage report", "Relatório geral")
    run_command("python -m coverage html", "Relatório HTML")

def main():
    """Executa Fase 2 completa"""
    print("🚀 INICIANDO FASE 2: EXPANSÃO AVANÇADA DE COBERTURA")
    print("Meta: 25-35% de cobertura de testes")
    print("=" * 60)
    
    # Verificar estado inicial
    check_coverage()
    
    # Executar fases em sequência
    fase2_chunking_advanced()
    fase2_pipelines()
    fase2_embeddings()
    fase2_vectordb()
    fase2_code_analysis()
    fase2_utilities()
    
    # Relatório final
    generate_comprehensive_report()
    
    print("\n🎉 FASE 2 EXECUTADA!")
    print("➡️  Próximo passo: Executar Fase 3 (Retrieval + Models + Monitoring)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 