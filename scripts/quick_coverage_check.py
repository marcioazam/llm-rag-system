#!/usr/bin/env python3
"""
Quick Coverage Check - Análise Rápida de Cobertura
Identifica problemas urgentes e módulos prioritários para correção
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile

class QuickCoverageAnalyzer:
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        
        # Módulos críticos com 0% cobertura
        self.critical_zero_coverage = [
            "template_renderer.py",
            "language_aware_chunker.py", 
            "adaptive_rag_router.py",
            "memo_rag.py",
            "multi_head_rag.py",
            "raptor_simple.py",
            "colbert_reranker.py"
        ]
        
        # Módulos com baixa cobertura prioritários
        self.low_coverage_priority = [
            "rag_pipeline_advanced.py",
            "api_model_router.py",
            "model_router.py",
            "qdrant_store.py"
        ]

    def test_basic_imports(self) -> Dict[str, bool]:
        """Testa se módulos básicos podem ser importados"""
        print("🔍 Testando importações básicas...")
        
        results = {}
        basic_modules = [
            ("src.settings", "Settings básico"),
            ("src.template_renderer", "Template Renderer"),
            ("src.prompt_selector", "Prompt Selector")
        ]
        
        for module, description in basic_modules:
            try:
                # Teste de importação isolado
                test_code = f"import sys; sys.path.insert(0, 'src'); import {module.replace('src.', '')}; print('✅ OK')"
                result = subprocess.run([
                    sys.executable, "-c", test_code
                ], capture_output=True, text=True, cwd=self.project_root, timeout=10)
                
                success = result.returncode == 0
                results[module] = success
                
                status = "✅ OK" if success else "❌ FALHA"
                print(f"  {status} {description}")
                
                if not success:
                    print(f"    Erro: {result.stderr.strip()}")
                    
            except Exception as e:
                results[module] = False
                print(f"  ❌ FALHA {description}: {e}")
        
        return results

    def identify_problematic_modules(self) -> List[Tuple[str, str, str]]:
        """Identifica módulos problemáticos por dependência"""
        print("\n🔍 Identificando módulos problemáticos...")
        
        problematic = []
        
        # Verificar módulos que dependem de sentence-transformers
        sentence_transformer_modules = [
            ("src/chunking/language_aware_chunker.py", "sentence-transformers", "CRÍTICO"),
            ("src/chunking/semantic_chunker.py", "sentence-transformers", "MÉDIO"),
            ("src/chunking/semantic_chunker_enhanced.py", "sentence-transformers", "MÉDIO")
        ]
        
        # Verificar módulos que dependem de qdrant
        qdrant_modules = [
            ("src/vectordb/qdrant_store.py", "qdrant-client", "CRÍTICO"),
            ("src/vectordb/hybrid_qdrant_store.py", "qdrant-client", "CRÍTICO")
        ]
        
        # Verificar módulos que dependem de openai
        openai_modules = [
            ("src/models/api_model_router.py", "openai", "CRÍTICO"),
            ("src/embeddings/api_embedding_service.py", "openai", "CRÍTICO")
        ]
        
        all_modules = sentence_transformer_modules + qdrant_modules + openai_modules
        
        for module_path, dependency, priority in all_modules:
            full_path = self.project_root / module_path
            if full_path.exists():
                problematic.append((module_path, dependency, priority))
                print(f"  🔴 {priority}: {module_path} (dep: {dependency})")
            
        return problematic

    def check_test_files_status(self) -> Dict[str, str]:
        """Verifica status dos arquivos de teste"""
        print("\n📝 Verificando status dos arquivos de teste...")
        
        test_status = {}
        
        # Verificar testes para módulos críticos
        for module in self.critical_zero_coverage:
            test_name = f"test_{module.replace('.py', '')}"
            
            # Procurar por arquivos de teste
            test_files = list(self.tests_dir.glob(f"{test_name}*.py"))
            
            if test_files:
                status = f"✅ Existe ({len(test_files)} arquivos)"
                # Verificar se o teste é funcional
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", str(test_files[0]), "--collect-only"
                    ], capture_output=True, text=True, cwd=self.project_root, timeout=10)
                    
                    if result.returncode != 0:
                        status = f"⚠️ Existe mas problemático"
                        
                except Exception:
                    status = f"❌ Existe mas não executa"
            else:
                status = "❌ Não existe"
            
            test_status[module] = status
            print(f"  {module}: {status}")
        
        return test_status

    def suggest_immediate_actions(self, import_results: Dict, problematic: List, test_status: Dict):
        """Sugere ações imediatas baseadas na análise"""
        print("\n🎯 AÇÕES IMEDIATAS RECOMENDADAS:")
        
        # 1. Problemas de conftest.py
        if not all(import_results.values()):
            print("\n1️⃣ URGENTE: Corrigir conftest.py")
            print("   - Problemas de importação detectados")
            print("   - Implementar mocks para dependências pesadas")
            print("   - Comando: cp tests/conftest.py tests/conftest.py.backup")
            
        # 2. Testes inexistentes
        missing_tests = [module for module, status in test_status.items() if "❌ Não existe" in status]
        if missing_tests:
            print(f"\n2️⃣ ALTA PRIORIDADE: Criar testes básicos ({len(missing_tests)} módulos)")
            for module in missing_tests[:3]:  # Top 3
                print(f"   - {module} (0% cobertura)")
            
        # 3. Dependências problemáticas
        critical_deps = [p for p in problematic if p[2] == "CRÍTICO"]
        if critical_deps:
            print(f"\n3️⃣ MÉDIO PRAZO: Resolver dependências críticas ({len(critical_deps)} módulos)")
            deps = set(p[1] for p in critical_deps)
            for dep in deps:
                print(f"   - Mock para {dep}")
        
        # 4. Próximos passos
        print("\n4️⃣ PRÓXIMOS PASSOS:")
        print("   1. python scripts/boost_test_coverage.py --phase=1")
        print("   2. Criar teste básico para template_renderer.py (mais simples)")
        print("   3. python -m pytest tests/test_template_renderer.py --cov=src.template_renderer")
        print("   4. Expandir para language_aware_chunker.py")

    def run_quick_test(self) -> bool:
        """Executa teste rápido em módulos funcionais"""
        print("\n⚡ Executando teste rápido em módulos funcionais...")
        
        try:
            # Testar apenas módulos que sabemos que funcionam
            functional_tests = [
                "tests/test_settings.py",
                "tests/test_prompt_selector.py"
            ]
            
            existing_tests = [t for t in functional_tests if (self.project_root / t).exists()]
            
            if existing_tests:
                result = subprocess.run([
                    sys.executable, "-m", "pytest"
                ] + existing_tests + [
                    "--cov=src", "--cov-report=term-missing", "-v", "--tb=short"
                ], capture_output=True, text=True, cwd=self.project_root, timeout=60)
                
                success = result.returncode == 0
                print(f"{'✅' if success else '❌'} Teste rápido: {'PASSOU' if success else 'FALHOU'}")
                
                if success:
                    # Extrair informação de cobertura
                    if "TOTAL" in result.stdout:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if "TOTAL" in line and "%" in line:
                                print(f"  📊 {line.strip()}")
                
                return success
            else:
                print("⚠️ Nenhum teste funcional encontrado")
                return False
                
        except Exception as e:
            print(f"❌ Erro no teste rápido: {e}")
            return False

    def create_emergency_conftest(self):
        """Cria conftest.py de emergência com mocks básicos"""
        print("\n🚑 Criando conftest.py de emergência...")
        
        emergency_conftest = '''"""
Conftest de emergência com mocks básicos
Criado automaticamente pelo QuickCoverageAnalyzer
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session", autouse=True)
def mock_heavy_dependencies():
    """Mock automático para dependências pesadas"""
    mocks = {}
    
    # Mock sentence-transformers
    try:
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_st.return_value.encode.return_value = [[0.1] * 384]
            mocks['sentence_transformer'] = mock_st
            yield mocks
    except:
        pass
    
    # Mock qdrant-client
    try:
        with patch('qdrant_client.QdrantClient') as mock_qc:
            mock_qc.return_value.search.return_value = []
            mocks['qdrant_client'] = mock_qc
    except:
        pass
    
    # Mock openai
    try:
        with patch('openai.OpenAI') as mock_openai:
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_openai.return_value.embeddings.create.return_value = mock_response
            mocks['openai'] = mock_openai
    except:
        pass

@pytest.fixture
def sample_text():
    """Texto de exemplo para testes"""
    return "Este é um texto de exemplo para testes de RAG."

@pytest.fixture
def sample_embedding():
    """Embedding de exemplo"""
    return [0.1] * 384

@pytest.fixture
def sample_query():
    """Query de exemplo"""
    return "Como funciona o sistema RAG?"
'''
        
        conftest_path = self.tests_dir / "conftest_emergency.py"
        with open(conftest_path, 'w', encoding='utf-8') as f:
            f.write(emergency_conftest)
        
        print(f"✅ Conftest de emergência criado: {conftest_path}")
        print("   Para usar: mv tests/conftest.py tests/conftest_old.py && mv tests/conftest_emergency.py tests/conftest.py")

    def main(self):
        """Executa análise completa"""
        print("🚀 Quick Coverage Check - Análise Rápida\n" + "="*50)
        
        # 1. Testar importações básicas
        import_results = self.test_basic_imports()
        
        # 2. Identificar módulos problemáticos
        problematic = self.identify_problematic_modules()
        
        # 3. Verificar arquivos de teste
        test_status = self.check_test_files_status()
        
        # 4. Executar teste rápido
        quick_test_passed = self.run_quick_test()
        
        # 5. Criar conftest de emergência
        if not all(import_results.values()):
            self.create_emergency_conftest()
        
        # 6. Sugerir ações
        self.suggest_immediate_actions(import_results, problematic, test_status)
        
        # 7. Resumo final
        print(f"\n📊 RESUMO:")
        print(f"   ✅ Importações funcionais: {sum(import_results.values())}/{len(import_results)}")
        print(f"   🔴 Módulos problemáticos: {len(problematic)}")
        print(f"   📝 Testes existentes: {sum(1 for s in test_status.values() if '✅' in s)}/{len(test_status)}")
        print(f"   ⚡ Teste rápido: {'PASSOU' if quick_test_passed else 'FALHOU'}")
        
        # Status geral
        if all(import_results.values()) and quick_test_passed:
            print(f"\n🎯 STATUS: ✅ PRONTO PARA EXPANSÃO DE TESTES")
        elif any(import_results.values()):
            print(f"\n🎯 STATUS: ⚠️ PARCIALMENTE FUNCIONAL - CORRIGIR DEPENDÊNCIAS")
        else:
            print(f"\n🎯 STATUS: 🔴 CRÍTICO - RESOLVER CONFTEST.PY PRIMEIRO")

if __name__ == "__main__":
    analyzer = QuickCoverageAnalyzer()
    analyzer.main() 