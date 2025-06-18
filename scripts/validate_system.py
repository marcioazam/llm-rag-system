#!/usr/bin/env python3
"""
Script de Validação do Sistema RAG
Valida todas as correções implementadas nas fases de melhoria
"""

import os
import sys
import time
import yaml
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Adicionar diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))


class SystemValidator:
    """Validador completo do sistema RAG"""
    
    def __init__(self):
        self.results = {
            'security': [],
            'stability': [],
            'optimization': [],
            'advanced': [],
            'overall_score': 0
        }
        self.total_checks = 0
        self.passed_checks = 0
    
    def validate_all(self) -> Dict[str, Any]:
        """Executa todas as validações"""
        print("🔍 Iniciando validação completa do sistema RAG...")
        print("=" * 60)
        
        # Fase 1 - Segurança
        print("\n🚨 FASE 1 - VALIDAÇÃO DE SEGURANÇA")
        self._validate_security()
        
        # Fase 2 - Estabilidade
        print("\n🔧 FASE 2 - VALIDAÇÃO DE ESTABILIDADE")
        self._validate_stability()
        
        # Fase 3 - Otimização
        print("\n⚡ FASE 3 - VALIDAÇÃO DE OTIMIZAÇÃO")
        self._validate_optimization()
        
        # Fase 4 - Melhorias Avançadas
        print("\n🚀 FASE 4 - VALIDAÇÃO DE MELHORIAS AVANÇADAS")
        self._validate_advanced()
        
        # Calcular score final
        self.results['overall_score'] = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        # Relatório final
        self._generate_report()
        
        return self.results
    
    def _validate_security(self):
        """Valida correções de segurança"""
        
        # 1. Verificar se senha hardcoded foi removida
        self._check("Senha hardcoded removida", self._check_no_hardcoded_password, 'security')
        
        # 2. Verificar .gitignore
        self._check("Arquivo .gitignore configurado", self._check_gitignore, 'security')
        
        # 3. Verificar CORS
        self._check("CORS configurado corretamente", self._check_cors_config, 'security')
        
        # 4. Verificar rate limiting
        self._check("Rate limiting configurado", self._check_rate_limiting, 'security')
        
        # 5. Verificar validação de entrada
        self._check("Validação de entrada implementada", self._check_input_validation, 'security')
    
    def _validate_stability(self):
        """Valida melhorias de estabilidade"""
        
        # 1. Verificar requirements.txt limpo
        self._check("Requirements.txt sem duplicações", self._check_clean_requirements, 'stability')
        
        # 2. Verificar logging estruturado
        self._check("Logging estruturado implementado", self._check_structured_logging, 'stability')
        
        # 3. Verificar tratamento de exceções
        self._check("Tratamento de exceções melhorado", self._check_exception_handling, 'stability')
    
    def _validate_optimization(self):
        """Valida otimizações implementadas"""
        
        # 1. Verificar circuit breaker
        self._check("Circuit breaker implementado", self._check_circuit_breaker, 'optimization')
        
        # 2. Verificar health check
        self._check("Health check aprimorado", self._check_health_endpoint, 'optimization')
    
    def _validate_advanced(self):
        """Valida melhorias avançadas"""
        
        # 1. Verificar dependências instaladas
        self._check("Dependências de monitoramento instaladas", self._check_monitoring_deps, 'advanced')
        
        # 2. Verificar estrutura de arquivos
        self._check("Novos arquivos de utilidade criados", self._check_utility_files, 'advanced')
    
    def _check(self, description: str, check_func, category: str):
        """Executa uma verificação individual"""
        self.total_checks += 1
        try:
            result = check_func()
            if result['passed']:
                print(f"✅ {description}")
                self.passed_checks += 1
            else:
                print(f"❌ {description}: {result['message']}")
            
            self.results[category].append({
                'description': description,
                'passed': result['passed'],
                'message': result['message'],
                'details': result.get('details')
            })
        except Exception as e:
            print(f"⚠️  {description}: Erro na validação - {str(e)}")
            self.results[category].append({
                'description': description,
                'passed': False,
                'message': f"Erro na validação: {str(e)}",
                'details': None
            })
    
    def _check_no_hardcoded_password(self) -> Dict[str, Any]:
        """Verifica se senha hardcoded foi removida"""
        settings_file = Path("src/settings.py")
        if not settings_file.exists():
            return {'passed': False, 'message': 'Arquivo settings.py não encontrado'}
        
        content = settings_file.read_text()
        if 'arrozefeijao13' in content:
            return {'passed': False, 'message': 'Senha hardcoded ainda presente'}
        
        if 'Field(env="NEO4J_PASSWORD")' in content:
            return {'passed': True, 'message': 'Senha hardcoded removida, usando variável de ambiente'}
        
        return {'passed': False, 'message': 'Configuração de senha não encontrada'}
    
    def _check_gitignore(self) -> Dict[str, Any]:
        """Verifica configuração do .gitignore"""
        gitignore_file = Path(".gitignore")
        if not gitignore_file.exists():
            return {'passed': False, 'message': 'Arquivo .gitignore não encontrado'}
        
        content = gitignore_file.read_text()
        required_entries = ['.env', '*.log', '*.pid', 'config/secrets.yaml']
        missing = [entry for entry in required_entries if entry not in content]
        
        if missing:
            return {'passed': False, 'message': f'Entradas ausentes: {missing}'}
        
        return {'passed': True, 'message': 'Arquivo .gitignore configurado corretamente'}
    
    def _check_cors_config(self) -> Dict[str, Any]:
        """Verifica configuração de CORS"""
        config_file = Path("config/config.yaml")
        if not config_file.exists():
            return {'passed': False, 'message': 'Arquivo config.yaml não encontrado'}
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        cors_origins = config.get('api', {}).get('cors_origins', [])
        if '*' in cors_origins:
            return {'passed': False, 'message': 'CORS ainda permite todas as origens (*)'}
        
        if cors_origins and all(origin.startswith('http://localhost') or origin.startswith('http://127.0.0.1') for origin in cors_origins):
            return {'passed': True, 'message': 'CORS configurado com origens específicas'}
        
        return {'passed': False, 'message': 'CORS não configurado adequadamente'}
    
    def _check_rate_limiting(self) -> Dict[str, Any]:
        """Verifica configuração de rate limiting"""
        config_file = Path("config/config.yaml")
        if not config_file.exists():
            return {'passed': False, 'message': 'Arquivo config.yaml não encontrado'}
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        rate_limit = config.get('api', {}).get('rate_limit', {})
        rpm = rate_limit.get('requests_per_minute', 60)
        
        if rpm <= 10:
            return {'passed': True, 'message': f'Rate limiting configurado para {rpm} req/min'}
        else:
            return {'passed': False, 'message': f'Rate limiting muito permissivo: {rpm} req/min'}
    
    def _check_input_validation(self) -> Dict[str, Any]:
        """Verifica validação de entrada na API"""
        api_file = Path("src/api/main.py")
        if not api_file.exists():
            return {'passed': False, 'message': 'Arquivo main.py da API não encontrado'}
        
        content = api_file.read_text()
        
        validators = ['@validator', 'Field(..., min_length', 'Field(..., max_length']
        if any(validator in content for validator in validators):
            return {'passed': True, 'message': 'Validação de entrada implementada'}
        
        return {'passed': False, 'message': 'Validação de entrada não encontrada'}
    
    def _check_clean_requirements(self) -> Dict[str, Any]:
        """Verifica se requirements.txt foi limpo"""
        req_file = Path("requirements.txt")
        if not req_file.exists():
            return {'passed': False, 'message': 'Arquivo requirements.txt não encontrado'}
        
        content = req_file.read_text()
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
        
        # Verificar duplicações
        packages = []
        for line in lines:
            if '==' in line or '>=' in line:
                package = line.split('==')[0].split('>=')[0].strip()
                packages.append(package)
        
        if len(packages) != len(set(packages)):
            return {'passed': False, 'message': 'Ainda existem duplicações'}
        
        return {'passed': True, 'message': f'Requirements limpo com {len(packages)} dependências únicas'}
    
    def _check_structured_logging(self) -> Dict[str, Any]:
        """Verifica se logging estruturado foi implementado"""
        logger_file = Path("src/utils/structured_logger.py")
        if logger_file.exists():
            return {'passed': True, 'message': 'Sistema de logging estruturado criado'}
        
        return {'passed': False, 'message': 'Sistema de logging estruturado não encontrado'}
    
    def _check_exception_handling(self) -> Dict[str, Any]:
        """Verifica melhorias no tratamento de exceções"""
        pipeline_file = Path("src/rag_pipeline.py")
        if not pipeline_file.exists():
            return {'passed': False, 'message': 'Arquivo rag_pipeline.py não encontrado'}
        
        content = pipeline_file.read_text()
        
        # Procurar por exceções específicas ao invés de genéricas
        specific_exceptions = ['OSError', 'PermissionError', 'ImportError', 'ConnectionError', 'TimeoutError']
        if any(exc in content for exc in specific_exceptions):
            return {'passed': True, 'message': 'Tratamento de exceções específicas implementado'}
        
        return {'passed': False, 'message': 'Tratamento de exceções ainda muito genérico'}
    
    def _check_circuit_breaker(self) -> Dict[str, Any]:
        """Verifica implementação do circuit breaker"""
        cb_file = Path("src/utils/circuit_breaker.py")
        if cb_file.exists():
            return {'passed': True, 'message': 'Circuit breaker implementado'}
        
        return {'passed': False, 'message': 'Circuit breaker não encontrado'}
    
    def _check_health_endpoint(self) -> Dict[str, Any]:
        """Verifica melhorias no endpoint de health"""
        api_file = Path("src/api/main.py")
        if not api_file.exists():
            return {'passed': False, 'message': 'Arquivo main.py da API não encontrado'}
        
        content = api_file.read_text()
        
        health_improvements = ['response_time_ms', 'components', 'timestamp']
        if any(improvement in content for improvement in health_improvements):
            return {'passed': True, 'message': 'Health check aprimorado'}
        
        return {'passed': False, 'message': 'Health check não foi aprimorado'}
    
    def _check_monitoring_deps(self) -> Dict[str, Any]:
        """Verifica se dependências de monitoramento foram instaladas"""
        req_file = Path("requirements.txt")
        if not req_file.exists():
            return {'passed': False, 'message': 'Arquivo requirements.txt não encontrado'}
        
        content = req_file.read_text()
        monitoring_deps = ['circuit-breaker', 'psutil', 'structlog']
        
        missing = [dep for dep in monitoring_deps if dep not in content]
        if missing:
            return {'passed': False, 'message': f'Dependências ausentes: {missing}'}
        
        return {'passed': True, 'message': 'Dependências de monitoramento instaladas'}
    
    def _check_utility_files(self) -> Dict[str, Any]:
        """Verifica se arquivos de utilidade foram criados"""
        utility_files = [
            'src/utils/structured_logger.py',
            'src/utils/circuit_breaker.py',
            'scripts/clean_requirements.py',
            'scripts/validate_system.py'
        ]
        
        existing = [f for f in utility_files if Path(f).exists()]
        
        if len(existing) >= 3:
            return {'passed': True, 'message': f'{len(existing)}/{len(utility_files)} arquivos criados'}
        
        return {'passed': False, 'message': f'Apenas {len(existing)}/{len(utility_files)} arquivos criados'}
    
    def _generate_report(self):
        """Gera relatório final da validação"""
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO FINAL DE VALIDAÇÃO")
        print("=" * 60)
        
        print(f"\n🎯 Score Geral: {self.results['overall_score']:.1f}%")
        print(f"✅ Verificações Aprovadas: {self.passed_checks}/{self.total_checks}")
        
        # Resumo por categoria
        for category, checks in self.results.items():
            if category == 'overall_score':
                continue
            
            passed = sum(1 for check in checks if check['passed'])
            total = len(checks)
            percentage = (passed / total) * 100 if total > 0 else 0
            
            print(f"\n{category.upper()}: {passed}/{total} ({percentage:.1f}%)")
            
            # Mostrar falhas
            failed = [check for check in checks if not check['passed']]
            if failed:
                print("  ❌ Falhas:")
                for fail in failed:
                    print(f"    - {fail['description']}: {fail['message']}")
        
        # Recomendações
        if self.results['overall_score'] < 100:
            print("\n🔧 RECOMENDAÇÕES:")
            print("- Corrigir as falhas identificadas acima")
            print("- Executar novamente a validação após correções")
            print("- Considerar implementação de testes automatizados")
        else:
            print("\n🎉 PARABÉNS! Todas as verificações foram aprovadas!")
        
        # Salvar relatório
        self._save_report()
    
    def _save_report(self):
        """Salva relatório em arquivo"""
        report_file = Path("validation_report.json")
        
        report_data = {
            **self.results,
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_checks': self.total_checks,
                'passed_checks': self.passed_checks,
                'overall_score': self.results['overall_score']
            }
        }
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Relatório salvo em: {report_file}")


def main():
    """Função principal"""
    validator = SystemValidator()
    results = validator.validate_all()
    
    # Exit code baseado no score
    exit_code = 0 if results['overall_score'] >= 80 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 