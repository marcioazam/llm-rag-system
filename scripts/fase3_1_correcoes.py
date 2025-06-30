#!/usr/bin/env python3
"""
Script para Fase 3.1: Correções Focadas
Meta: Corrigir problemas específicos e alcançar 25-28% de cobertura estável
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
                lines = result.stdout.split('\n')
                summary_lines = [l for l in lines if 'passed' in l or 'failed' in l or 'TOTAL' in l or '=' in l][-3:]
                print('\n'.join(summary_lines))
        else:
            print("❌ ERRO")
            if result.stderr:
                print(f"Erro: {result.stderr[:200]}")
            if result.stdout:
                error_lines = result.stdout.split('\n')[-5:]
                print("Últimas linhas:")
                print('\n'.join(error_lines))
                
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - comando muito demorado")
        return False
    except Exception as e:
        print(f"❌ EXCEÇÃO: {e}")
        return False

def check_coverage():
    """Verifica cobertura atual com detalhes"""
    result = subprocess.run("python -m coverage report", shell=True, capture_output=True, text=True)
    if result.stdout:
        lines = result.stdout.split('\n')
        total_line = [l for l in lines if 'TOTAL' in l]
        if total_line:
            print(f"📈 {total_line[0]}")

def fase3_1_modulos_funcionais():
    """Fase 3.1A: Expandir módulos que já funcionam bem"""
    print("\n🎯 FASE 3.1A: EXPANDINDO MÓDULOS FUNCIONAIS")
    
    # Módulos que funcionaram bem na Fase 3
    working_tests = [
        "tests/test_query_enhancer.py",  # 30 testes OK
        "tests/test_rag_pipeline_advanced.py",  # 27 testes OK
        "tests/test_template_renderer.py",  # Se existir
        "tests/test_prompt_selector.py"  # Se existir
    ]
    
    for test in working_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Módulo Funcional: {test}")
            if success:
                check_coverage()

def fase3_1_model_router_correcoes():
    """Fase 3.1B: Corrigir as 2 falhas específicas do Model Router"""
    print("\n🎯 FASE 3.1B: CORRIGINDO MODEL ROUTER")
    
    # Tentar executar apenas o model router que já funciona parcialmente
    test_file = "tests/test_model_router.py"
    if Path(test_file).exists():
        # Primeiro executar para ver o estado atual
        cmd = f"python -m coverage run --append --source=src -m pytest {test_file} -v --tb=short"
        success = run_command(cmd, f"Model Router (38 sucessos esperados): {test_file}")
        if success:
            check_coverage()

def fase3_1_utilities_basicos():
    """Fase 3.1C: Adicionar utilities básicos que provavelmente funcionam"""
    print("\n🎯 FASE 3.1C: UTILITIES BÁSICOS")
    
    utility_tests = [
        "tests/test_circuit_breaker.py",  # Já funcionou antes
        "tests/test_circuit_breaker_enhanced.py",
        "tests/test_base_analyzer.py",  # Já funcionou antes  
        "tests/test_base_analyzer_enhanced.py"
    ]
    
    for test in utility_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Utility: {test}")
            if success:
                check_coverage()

def fase3_1_storage_metadata():
    """Fase 3.1D: Sistemas de storage que já funcionam"""
    print("\n🎯 FASE 3.1D: STORAGE E METADATA")
    
    storage_tests = [
        "tests/test_qdrant_store.py",  # 100% cobertura
        "tests/test_embedding_service.py",  # Já funcionou
        "tests/test_embedding_service_comprehensive.py"
    ]
    
    for test in storage_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Storage: {test}")
            if success:
                check_coverage()

def fase3_1_code_analysis():
    """Fase 3.1E: Code analysis que já funcionou na Fase 2"""
    print("\n🎯 FASE 3.1E: CODE ANALYSIS")
    
    code_tests = [
        "tests/test_dependency_analyzer.py",  # 93% cobertura na Fase 2
        "tests/test_code_analyzer.py",
        "tests/test_python_analyzer.py"
    ]
    
    for test in code_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Code Analysis: {test}")
            if success:
                check_coverage()

def fase3_1_preprocessing():
    """Fase 3.1F: Preprocessing que funcionou na Fase 2"""
    print("\n🎯 FASE 3.1F: PREPROCESSING")
    
    preprocessing_tests = [
        "tests/test_intelligent_preprocessor.py",  # 83% cobertura na Fase 2
        "tests/test_preprocessing_comprehensive.py"
    ]
    
    for test in preprocessing_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Preprocessing: {test}")
            if success:
                check_coverage()

def fase3_1_chunking_funcionais():
    """Fase 3.1G: Chunking que já funcionam"""
    print("\n🎯 FASE 3.1G: CHUNKING FUNCIONAIS")
    
    chunking_tests = [
        "tests/test_base_chunker.py",  # 93% cobertura
        "tests/test_advanced_chunker.py"  # Se funcionar
    ]
    
    for test in chunking_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Chunking: {test}")
            if success:
                check_coverage()

def fase3_1_tentativas_simples():
    """Fase 3.1H: Tentativas em módulos simples que podem funcionar"""
    print("\n🎯 FASE 3.1H: TENTATIVAS SIMPLES")
    
    simple_tests = [
        "tests/test_context_injector.py",  # Pode ser simples
        "tests/test_devtools_comprehensive.py",  # Devtools podem funcionar
        "tests/test_generation_comprehensive.py"  # Generation pode funcionar
    ]
    
    for test in simple_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Simples: {test}")
            if success:
                check_coverage()

def generate_fase3_1_report():
    """Gera relatório detalhado da Fase 3.1"""
    print("\n📈 GERANDO RELATÓRIO DETALHADO DA FASE 3.1")
    
    # Módulos que esperamos ter melhorado
    key_modules = [
        "src/retrieval/query_enhancer.py",
        "src/rag_pipeline_advanced.py",
        "src/rag_pipeline_base.py", 
        "src/models/model_router.py",
        "src/template_renderer.py",
        "src/code_analysis/dependency_analyzer.py",
        "src/preprocessing/intelligent_preprocessor.py",
        "src/embedding/embedding_service.py",
        "src/vectordb/qdrant_store.py"
    ]
    
    print("\n🔍 COBERTURA DOS MÓDULOS CRÍTICOS:")
    for module in key_modules:
        cmd = f"python -m coverage report --include='{module}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                module_name = module.split('/')[-1].replace('.py', '')
                if module_name in line and '%' in line:
                    print(f"  📊 {line}")
    
    run_command("python -m coverage report", "Relatório Final")
    run_command("python -m coverage html", "Relatório HTML")

def main():
    """Executa Fase 3.1 completa com foco em correções"""
    print("🚀 INICIANDO FASE 3.1: CORREÇÕES FOCADAS")
    print("Meta: 25-28% de cobertura estável")
    print("Estratégia: Focar em módulos que já funcionam + correções simples")
    print("=" * 70)
    
    # Verificar estado inicial (deve estar em ~21%)
    check_coverage()
    
    # Executar fases estratégicas focadas
    print("\n🎯 ESTRATÉGIA: EXPANDIR SUCESSOS + CORRIGIR PROBLEMAS SIMPLES")
    
    # Começar com o que já funciona para garantir progresso
    fase3_1_modulos_funcionais()    # Query Enhancer + RAG Pipeline
    fase3_1_storage_metadata()      # Qdrant + Embedding (100% de sucesso)
    fase3_1_code_analysis()         # Dependency Analyzer (93% já)
    fase3_1_preprocessing()         # Intelligent Preprocessor (83% já)
    fase3_1_utilities_basicos()     # Circuit Breaker + Base Analyzer
    fase3_1_chunking_funcionais()   # Base Chunker (93% já)
    fase3_1_model_router_correcoes() # Tentar corrigir 2 falhas
    fase3_1_tentativas_simples()    # Módulos que podem funcionar
    
    # Relatório final detalhado
    generate_fase3_1_report()
    
    print("\n🎉 FASE 3.1 EXECUTADA!")
    print("🎯 Meta: 25-28% de cobertura estável")
    print("🔄 Estratégia: Consolidar sucessos antes de expandir")
    print("➡️  Próximo passo: Avaliar se prosseguimos para Fase 4 ou otimizamos Fase 3.2")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 