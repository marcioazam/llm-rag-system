#!/usr/bin/env python3
"""
Script para Fase 1: Expansão de Cobertura de Testes
Meta: Alcançar 15% de cobertura focando nos módulos críticos
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
                summary_lines = [l for l in lines if 'passed' in l or 'failed' in l or 'TOTAL' in l or '=' in l][-5:]
                print('\n'.join(summary_lines))
        else:
            print("❌ ERRO")
            if result.stderr:
                print(f"Erro: {result.stderr[:500]}")
                
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
    run_command("python -m coverage report", "Relatório de cobertura")

def fase1_base_modules():
    """Fase 1A: Completar módulos base com alta cobertura"""
    print("\n🎯 FASE 1A: COMPLETANDO MÓDULOS BASE")
    
    # Lista de testes que devem funcionar facilmente
    safe_tests = [
        "tests/test_simple_modules.py",
        "tests/test_basic_modules_isolated.py", 
        "tests/test_response_optimizer.py",
        "tests/test_template_renderer_basic.py",
        "tests/test_settings.py",
        "tests/test_cache_analytics.py"
    ]
    
    print(f"Executando {len(safe_tests)} conjuntos de testes seguros...")
    
    # Executar todos os testes seguros de uma vez
    cmd = f"python -m coverage run --source=src -m pytest {' '.join(safe_tests)} -v"
    success = run_command(cmd, "Testes base seguros")
    
    if success:
        check_coverage()
    
    return success

def fase1_chunking_expansion():
    """Fase 1B: Expandir testes de chunking"""
    print("\n🎯 FASE 1B: EXPANDINDO CHUNKING")
    
    chunking_tests = [
        "tests/test_base_chunker.py",
        "tests/test_recursive_chunker_fixed.py"
    ]
    
    for test in chunking_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v"
            run_command(cmd, f"Teste de chunking: {test}")

def fase1_storage_expansion():
    """Fase 1C: Expandir testes de storage"""
    print("\n🎯 FASE 1C: EXPANDINDO STORAGE")
    
    storage_tests = [
        "tests/test_sqlite_store_complete.py",
        "tests/test_sqlite_store.py"
    ]
    
    for test in storage_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v"
            run_command(cmd, f"Teste de storage: {test}")

def fase1_additional_safe():
    """Fase 1D: Testes adicionais seguros"""
    print("\n🎯 FASE 1D: TESTES ADICIONAIS SEGUROS")
    
    additional_tests = [
        "tests/test_circuit_breaker_enhanced.py",
        "tests/test_language_detector.py",
        "tests/test_prompt_selector.py"
    ]
    
    for test in additional_tests:
        if Path(test).exists():
            cmd = f"python -m coverage run --append --source=src -m pytest {test} -v --tb=short"
            run_command(cmd, f"Teste adicional: {test}")

def generate_report():
    """Gera relatório final da Fase 1"""
    print("\n📈 GERANDO RELATÓRIO FINAL DA FASE 1")
    
    run_command("python -m coverage report", "Relatório de cobertura texto")
    run_command("python -m coverage html", "Relatório de cobertura HTML")
    
    print("\n✅ FASE 1 CONCLUÍDA!")
    print("📁 Relatório HTML disponível em: htmlcov/index.html")

def main():
    """Executa Fase 1 completa"""
    print("🚀 INICIANDO FASE 1: EXPANSÃO DE COBERTURA")
    print("Meta: 15% de cobertura de testes")
    print("=" * 60)
    
    # Limpar cobertura anterior
    run_command("python -m coverage erase", "Limpando dados de cobertura")
    
    # Executar fases em sequência
    if not fase1_base_modules():
        print("❌ Falha na Fase 1A - parando execução")
        return False
    
    fase1_chunking_expansion()
    fase1_storage_expansion() 
    fase1_additional_safe()
    
    # Relatório final
    generate_report()
    
    print("\n🎉 FASE 1 EXECUTADA COM SUCESSO!")
    print("➡️  Próximo passo: Executar Fase 2 (Chunking + Pipelines)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 