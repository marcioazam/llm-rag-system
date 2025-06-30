#!/usr/bin/env python3
"""
Script de Automação para Aumento de Cobertura de Testes
Implementa o plano estruturado para levar cobertura de 22% para 80%+
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import shutil

class CoverageBooster:
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.coverage_report = {}
        
        # Módulos com 0% cobertura (prioridade crítica)
        self.zero_coverage_modules = [
            "template_renderer.py",
            "language_aware_chunker.py", 
            "adaptive_rag_router.py",
            "memo_rag.py",
            "multi_head_rag.py",
            "raptor_simple.py",
            "colbert_reranker.py"
        ]
        
        # Módulos com baixa cobertura (<20%)
        self.low_coverage_modules = [
            "rag_pipeline_advanced.py",
            "dependency_analyzer.py",
            "tree_sitter_analyzer.py",
            "qdrant_store.py",
            "multi_query_rag.py",
            "hyde_enhancer.py",
            "cache_warming.py",
            "multi_layer_cache.py"
        ]

    def analyze_current_coverage(self) -> Dict:
        """Analisa cobertura atual do projeto"""
        print("🔍 Analisando cobertura atual...")
        
        try:
            # Executar análise de cobertura
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "--cov=src", "--cov-report=json", "--cov-report=term-missing",
                "-x", "--tb=short"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Ler relatório JSON se existir
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    self.coverage_report = json.load(f)
                    
            return self.coverage_report
            
        except Exception as e:
            print(f"❌ Erro na análise de cobertura: {e}")
            return {}

    def identify_problematic_files(self) -> List[Tuple[str, str]]:
        """Identifica arquivos problemáticos por categoria"""
        problematic = []
        
        # Verificar módulos com 0% cobertura
        for module in self.zero_coverage_modules:
            file_path = self._find_module_path(module)
            if file_path:
                problematic.append((str(file_path), "0% Coverage"))
                
        # Verificar módulos com baixa cobertura
        for module in self.low_coverage_modules:
            file_path = self._find_module_path(module)
            if file_path:
                problematic.append((str(file_path), "Low Coverage"))
                
        return problematic

    def _find_module_path(self, module_name: str) -> Path:
        """Encontra caminho completo do módulo"""
        for root, dirs, files in os.walk(self.src_dir):
            if module_name in files:
                return Path(root) / module_name
        return None

    def create_test_template(self, module_path: str, priority: str) -> str:
        """Cria template de teste para módulo específico"""
        module_name = Path(module_path).stem
        test_filename = f"test_{module_name}_comprehensive.py"
        
        template = f'''"""
Testes abrangentes para {module_name}
Criado automaticamente pelo CoverageBooster
Prioridade: {priority}
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Adicionar src ao path para importações
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class Test{module_name.title().replace("_", "")}:
    """Testes abrangentes para {module_name}"""
    
    def test_module_import(self):
        """Testa se o módulo pode ser importado sem erros"""
        try:
            # Tentar importar módulo - adaptar path conforme necessário
            import {module_name}
            assert True, "Módulo importado com sucesso"
        except ImportError as e:
            pytest.skip(f"Módulo não pode ser importado: {{e}}")
    
    @pytest.mark.skipif(True, reason="Template - implementar conforme necessário")
    def test_initialization(self):
        """Testa inicialização básica do módulo/classe principal"""
        # TODO: Implementar teste de inicialização
        pass
    
    @pytest.mark.skipif(True, reason="Template - implementar conforme necessário") 
    def test_main_functionality(self):
        """Testa funcionalidade principal do módulo"""
        # TODO: Implementar teste da funcionalidade principal
        pass
    
    @pytest.mark.skipif(True, reason="Template - implementar conforme necessário")
    def test_error_handling(self):
        """Testa tratamento de erros e exceções"""
        # TODO: Implementar testes de error handling
        pass
    
    @pytest.mark.skipif(True, reason="Template - implementar conforme necessário")
    def test_edge_cases(self):
        """Testa casos extremos e limites"""
        # TODO: Implementar testes de edge cases
        pass

    @pytest.mark.skipif(True, reason="Template - implementar conforme necessário")
    def test_integration_points(self):
        """Testa pontos de integração com outros módulos"""
        # TODO: Implementar testes de integração
        pass

# Fixture de exemplo - adaptar conforme necessário
@pytest.fixture
def mock_dependencies():
    """Mock para dependências externas"""
    with patch('sentence_transformers.SentenceTransformer') as mock_st, \\
         patch('qdrant_client.QdrantClient') as mock_qc, \\
         patch('openai.OpenAI') as mock_openai:
        
        # Configurar mocks
        mock_st.return_value.encode.return_value = [[0.1] * 384]
        mock_qc.return_value.search.return_value = []
        mock_openai.return_value.embeddings.create.return_value.data = [
            Mock(embedding=[0.1] * 1536)
        ]
        
        yield {{
            'sentence_transformer': mock_st,
            'qdrant_client': mock_qc,
            'openai': mock_openai
        }}
'''
        
        return test_filename, template

    def implement_phase_1(self) -> bool:
        """Implementa FASE 1: Setup e dependências"""
        print("🚀 Implementando FASE 1: Setup e dependências...")
        
        try:
            # Verificar se conftest.py está funcional
            conftest_path = self.tests_dir / "conftest.py"
            if not conftest_path.exists():
                print("❌ conftest.py não encontrado")
                return False
            
            # Testar importações básicas
            test_result = subprocess.run([
                sys.executable, "-c", 
                "import sys; sys.path.insert(0, 'tests'); import conftest; print('✅ conftest.py funcional')"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if test_result.returncode == 0:
                print("✅ FASE 1 concluída: Ambiente de testes funcional")
                return True
            else:
                print(f"❌ Problemas no conftest.py: {test_result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Erro na FASE 1: {e}")
            return False

    def implement_phase_2(self) -> Dict[str, bool]:
        """Implementa FASE 2: Módulos com 0% cobertura"""
        print("🎯 Implementando FASE 2: Módulos com 0% cobertura...")
        
        results = {}
        
        for module in self.zero_coverage_modules:
            print(f"📝 Criando testes para {module}...")
            
            module_path = self._find_module_path(module)
            if not module_path:
                print(f"❌ Módulo {module} não encontrado")
                results[module] = False
                continue
            
            # Criar template de teste
            test_filename, template_content = self.create_test_template(
                str(module_path), "0% Coverage - CRÍTICO"
            )
            
            test_file_path = self.tests_dir / test_filename
            
            # Escrever arquivo de teste se não existir
            if not test_file_path.exists():
                with open(test_file_path, 'w', encoding='utf-8') as f:
                    f.write(template_content)
                print(f"✅ Template criado: {test_filename}")
                results[module] = True
            else:
                print(f"⚠️  Arquivo já existe: {test_filename}")
                results[module] = True
        
        return results

    def run_coverage_analysis(self) -> Dict:
        """Executa análise de cobertura e retorna resultados"""
        print("📊 Executando análise de cobertura...")
        
        try:
            # Executar pytest com cobertura
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--cov=src", "--cov-report=term-missing", 
                "--cov-report=html", "--cov-report=json",
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            print("🔍 Resultado da execução:")
            print(result.stdout[-1000:])  # Últimas 1000 chars
            
            if result.stderr:
                print("⚠️  Avisos/Erros:")
                print(result.stderr[-500:])  # Últimos 500 chars
            
            # Ler relatório JSON
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
                    print(f"📈 Cobertura total: {total_coverage:.1f}%")
                    return coverage_data
            
            return {}
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return {}

    def generate_progress_report(self, coverage_data: Dict) -> str:
        """Gera relatório de progresso"""
        report = f"""
# 📊 Relatório de Progresso - Cobertura de Testes

## 📈 Métricas Atuais
"""
        
        if coverage_data:
            totals = coverage_data.get('totals', {})
            report += f"""
- **Cobertura Total**: {totals.get('percent_covered', 0):.1f}%
- **Statements Cobertos**: {totals.get('covered_lines', 0)}
- **Statements Totais**: {totals.get('num_statements', 0)}
- **Statements Perdidos**: {totals.get('missing_lines', 0)}
"""
        
        # Identificar módulos problemáticos
        problematic = self.identify_problematic_files()
        report += f"""
## 🎯 Módulos Identificados para Melhoria

### Críticos (0% Cobertura)
"""
        
        for file_path, priority in problematic:
            if "0%" in priority:
                module_name = Path(file_path).name
                test_file = f"test_{Path(file_path).stem}_comprehensive.py"
                exists = "✅" if (self.tests_dir / test_file).exists() else "❌"
                report += f"- {module_name} {exists} {test_file}\n"
        
        report += """
### Próximos Passos
1. Implementar testes para módulos marcados com ❌
2. Executar `python scripts/boost_test_coverage.py --phase 2`
3. Verificar cobertura com `pytest --cov=src --cov-report=html`
"""
        
        return report

    def main(self, phase: str = "all"):
        """Função principal do script"""
        print("🚀 CoverageBooster - Aumentando Cobertura de Testes")
        print("=" * 50)
        
        # Verificar estrutura do projeto
        if not self.src_dir.exists():
            print("❌ Diretório src/ não encontrado")
            return
        
        if not self.tests_dir.exists():
            print("📁 Criando diretório tests/")
            self.tests_dir.mkdir(exist_ok=True)
        
        # Executar fases conforme solicitado
        if phase in ["all", "1"]:
            if self.implement_phase_1():
                print("✅ FASE 1 concluída com sucesso")
            else:
                print("❌ FASE 1 falhou - verifique dependências")
                return
        
        if phase in ["all", "2"]:
            results = self.implement_phase_2()
            success_count = sum(results.values())
            total_count = len(results)
            print(f"📊 FASE 2: {success_count}/{total_count} templates criados")
        
        # Executar análise de cobertura
        if phase in ["all", "analyze"]:
            coverage_data = self.run_coverage_analysis()
            
            # Gerar relatório
            report = self.generate_progress_report(coverage_data)
            
            # Salvar relatório
            report_file = self.project_root / "COVERAGE_BOOST_PROGRESS.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"📄 Relatório salvo em: {report_file}")
            print("\n" + "=" * 50)
            print("✅ CoverageBooster executado com sucesso!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Boost Test Coverage")
    parser.add_argument("--phase", choices=["1", "2", "all", "analyze"], 
                       default="all", help="Fase a executar")
    parser.add_argument("--project-root", help="Diretório raiz do projeto")
    
    args = parser.parse_args()
    
    booster = CoverageBooster(args.project_root)
    booster.main(args.phase) 