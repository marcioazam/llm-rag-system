#!/usr/bin/env python3
"""
Script para Fase 3: Expansão de Cobertura de Testes
Meta: Alcançar 30-40% de cobertura focando em Retrieval + Pipelines + Model Routing
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

def fase3_retrieval_core():
    """Fase 3A: Core Retrieval Components"""
    print("\n🎯 FASE 3A: CORE RETRIEVAL")
    
    retrieval_tests = [
        "tests/test_query_enhancer.py",
        "tests/test_adaptive_retriever.py", 
        "tests/test_hybrid_retriever.py",
        "tests/test_reranker.py"
    ]
    
    for test in retrieval_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Retrieval Core: {test}")
            if success:
                check_coverage()

def fase3_advanced_retrieval():
    """Fase 3B: Advanced Retrieval (que já têm alguma cobertura)"""
    print("\n🎯 FASE 3B: ADVANCED RETRIEVAL")
    
    advanced_tests = [
        "tests/test_corrective_rag.py",
        "tests/test_multi_query_rag.py",
        "tests/test_hyde_enhancer.py"
    ]
    
    for test in advanced_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Advanced Retrieval: {test}")
            if success:
                check_coverage()

def fase3_rag_pipelines():
    """Fase 3C: RAG Pipelines (focar nos funcionais)"""
    print("\n🎯 FASE 3C: RAG PIPELINES")
    
    pipeline_tests = [
        "tests/test_rag_pipeline_advanced.py",
        "tests/test_rag_pipeline_advanced_enhanced.py",
        "tests/test_rag_pipeline_base.py"
    ]
    
    for test in pipeline_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"RAG Pipeline: {test}")
            if success:
                check_coverage()

def fase3_model_routing():
    """Fase 3D: Model Routing"""
    print("\n🎯 FASE 3D: MODEL ROUTING")
    
    model_tests = [
        "tests/test_model_router.py",
        "tests/test_api_model_router.py",
        "tests/test_api_model_router_complete.py"
    ]
    
    for test in model_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Model Routing: {test}")
            if success:
                check_coverage()

def fase3_memo_multihead():
    """Fase 3E: Memo RAG e Multi-Head (módulos específicos)"""
    print("\n🎯 FASE 3E: MEMO & MULTI-HEAD RAG")
    
    advanced_rag_tests = [
        "tests/test_memo_rag.py",
        "tests/test_memo_rag_complete.py",
        "tests/test_multi_head_rag.py"
    ]
    
    for test in advanced_rag_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Advanced RAG: {test}")
            if success:
                check_coverage()

def fase3_utilities_missing():
    """Fase 3F: Utilities que ainda não foram testadas"""
    print("\n🎯 FASE 3F: UTILITIES MISSING")
    
    util_tests = [
        "tests/test_document_loader.py",
        "tests/test_smart_document_loader.py"
    ]
    
    for test in util_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"Utilities: {test}")
            if success:
                check_coverage()

def fase3_raptor_focus():
    """Fase 3G: RAPTOR (focar em módulos que já têm progresso)"""
    print("\n🎯 FASE 3G: RAPTOR SYSTEMS")
    
    raptor_tests = [
        "tests/test_raptor_simple.py",
        "tests/test_raptor_simple_comprehensive.py",
        "tests/test_raptor_enhanced.py"
    ]
    
    for test in raptor_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            success = run_command(cmd, f"RAPTOR: {test}")
            if success:
                check_coverage()

def generate_fase3_report():
    """Gera relatório detalhado da Fase 3"""
    print("\n📈 GERANDO RELATÓRIO DETALHADO DA FASE 3")
    
    # Relatório por módulos críticos
    print("\n🔍 MÓDULOS COM MAIOR IMPACTO:")
    
    key_modules = [
        "src/retrieval/query_enhancer.py",
        "src/retrieval/adaptive_retriever.py", 
        "src/retrieval/hybrid_retriever.py",
        "src/rag_pipeline_base.py",
        "src/rag_pipeline_advanced.py",
        "src/models/api_model_router.py",
        "src/models/model_router.py",
        "src/retrieval/memo_rag.py",
        "src/retrieval/corrective_rag.py"
    ]
    
    for module in key_modules:
        cmd = f"python -m coverage report --include='{module}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout and module.split('/')[-1] in result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if module.split('/')[-1].replace('.py', '') in line:
                    print(f"  📊 {line}")
    
    run_command("python -m coverage report", "Relatório geral")
    run_command("python -m coverage html", "Relatório HTML")

def main():
    """Executa Fase 3 completa"""
    print("🚀 INICIANDO FASE 3: RETRIEVAL + PIPELINES + MODEL ROUTING")
    print("Meta: 30-40% de cobertura de testes")
    print("=" * 60)
    
    # Verificar estado inicial (deve estar em ~19%)
    check_coverage()
    
    # Executar fases em sequência estratégica
    print("\n🎯 EXECUTANDO FASE 3 - FOCO EM MÓDULOS FUNCIONAIS")
    
    # Começar com módulos que têm maior chance de sucesso
    fase3_retrieval_core()
    fase3_model_routing()
    fase3_rag_pipelines()
    fase3_advanced_retrieval()
    fase3_memo_multihead()
    fase3_utilities_missing()
    fase3_raptor_focus()
    
    # Relatório final detalhado
    generate_fase3_report()
    
    print("\n🎉 FASE 3 EXECUTADA!")
    print("🎯 Meta: Alcançar 30-40% de cobertura")
    print("➡️  Próximo passo: Avaliar se continuamos para Fase 4 ou otimizamos")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 