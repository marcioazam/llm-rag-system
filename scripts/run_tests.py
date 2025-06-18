#!/usr/bin/env python3
"""
Script de execução de testes para CI/CD
Versão Python compatível com Windows e Linux
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any


class Colors:
    """Cores para output no terminal"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class TestRunner:
    """Executor de testes para o sistema RAG"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent.parent
        self.reports_dir = self.project_dir / "reports"
        
    def log(self, level: str, message: str):
        """Log colorido para diferentes níveis"""
        colors = {
            'INFO': Colors.BLUE,
            'SUCCESS': Colors.GREEN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED
        }
        color = colors.get(level, Colors.NC)
        print(f"{color}[{level}]{Colors.NC} {message}")
    
    def run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Executa comando e retorna resultado"""
        self.log('INFO', f"Executando: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            self.log('ERROR', f"Comando falhou: {e}")
            self.log('ERROR', f"stdout: {e.stdout}")
            self.log('ERROR', f"stderr: {e.stderr}")
            raise
    
    def command_exists(self, cmd: str) -> bool:
        """Verifica se comando existe"""
        return shutil.which(cmd) is not None
    
    def check_dependencies(self) -> bool:
        """Verifica dependências básicas"""
        self.log('INFO', "Verificando dependências...")
        
        if not self.command_exists('python'):
            self.log('ERROR', "Python não encontrado")
            return False
        
        if not self.command_exists('pip'):
            self.log('ERROR', "pip não encontrado")
            return False
        
        # Verificar se pytest está disponível
        try:
            import pytest
            self.log('SUCCESS', "pytest disponível")
        except ImportError:
            self.log('WARNING', "pytest não encontrado, instalando...")
            self.run_command(['pip', 'install', 'pytest', 'pytest-cov', 'pytest-mock'])
        
        self.log('SUCCESS', "Dependências verificadas")
        return True
    
    def install_test_deps(self) -> bool:
        """Instala dependências de teste"""
        self.log('INFO', "Instalando dependências de teste...")
        
        try:
            self.run_command(['pip', 'install', '--upgrade', 'pip'])
            self.run_command(['pip', 'install', '-r', 'requirements.txt'])
            self.log('SUCCESS', "Dependências instaladas")
            return True
        except subprocess.CalledProcessError:
            self.log('ERROR', "Falha ao instalar dependências")
            return False
    
    def run_linting(self) -> bool:
        """Executa linting"""
        self.log('INFO', "Executando linting...")
        success = True
        
        # Black - formatação
        if self.command_exists('black'):
            self.log('INFO', "Executando Black...")
            try:
                self.run_command(['black', '--check', '--diff', 'src', 'tests', 'scripts'])
            except subprocess.CalledProcessError:
                self.log('ERROR', "Black encontrou problemas de formatação")
                success = False
        
        # isort - ordenação de imports
        if self.command_exists('isort'):
            self.log('INFO', "Executando isort...")
            try:
                self.run_command(['isort', '--check-only', '--diff', 'src', 'tests', 'scripts'])
            except subprocess.CalledProcessError:
                self.log('ERROR', "isort encontrou problemas")
                success = False
        
        # Flake8 - linting
        if self.command_exists('flake8'):
            self.log('INFO', "Executando Flake8...")
            try:
                self.run_command([
                    'flake8', 'src', 'tests', 'scripts',
                    '--max-line-length=100',
                    '--extend-ignore=E203,W503'
                ])
            except subprocess.CalledProcessError:
                self.log('ERROR', "Flake8 encontrou problemas")
                success = False
        
        # Ruff - linting rápido
        if self.command_exists('ruff'):
            self.log('INFO', "Executando Ruff...")
            try:
                self.run_command(['ruff', 'check', 'src', 'tests', 'scripts'])
            except subprocess.CalledProcessError:
                self.log('ERROR', "Ruff encontrou problemas")
                success = False
        
        if success:
            self.log('SUCCESS', "Linting concluído")
        return success
    
    def run_security_tests(self) -> bool:
        """Executa testes de segurança"""
        self.log('INFO', "Executando testes de segurança...")
        success = True
        
        # Bandit - análise de segurança
        if self.command_exists('bandit'):
            self.log('INFO', "Executando Bandit...")
            try:
                self.run_command([
                    'bandit', '-r', 'src',
                    '-f', 'json',
                    '-o', 'bandit-report.json'
                ], check=False)
            except subprocess.CalledProcessError:
                self.log('WARNING', "Bandit encontrou possíveis problemas de segurança")
        
        # Safety - verificação de vulnerabilidades
        if self.command_exists('safety'):
            self.log('INFO', "Executando Safety...")
            try:
                self.run_command([
                    'safety', 'check',
                    '--json',
                    '--output', 'safety-report.json'
                ], check=False)
            except subprocess.CalledProcessError:
                self.log('WARNING', "Safety encontrou vulnerabilidades conhecidas")
        
        # Testes de segurança com pytest
        self.log('INFO', "Executando testes de segurança...")
        try:
            self.run_command([
                'python', '-m', 'pytest',
                'tests/test_security.py',
                '-v', '--tb=short',
                '-m', 'security'
            ])
        except subprocess.CalledProcessError:
            self.log('ERROR', "Testes de segurança falharam")
            success = False
        
        if success:
            self.log('SUCCESS', "Testes de segurança concluídos")
        return success
    
    def run_unit_tests(self) -> bool:
        """Executa testes unitários"""
        self.log('INFO', "Executando testes unitários...")
        
        try:
            self.run_command([
                'python', '-m', 'pytest',
                'tests/', '-v', '--tb=short',
                '--cov=src',
                '--cov-report=xml',
                '--cov-report=html',
                '--cov-report=term-missing',
                '--cov-fail-under=70',
                '--junitxml=test-results.xml',
                '-m', 'unit or not slow',
                '--timeout=300'
            ])
            self.log('SUCCESS', "Testes unitários concluídos")
            return True
        except subprocess.CalledProcessError:
            self.log('ERROR', "Testes unitários falharam")
            return False
    
    def run_integration_tests(self) -> bool:
        """Executa testes de integração"""
        self.log('INFO', "Executando testes de integração...")
        
        try:
            self.run_command([
                'python', '-m', 'pytest',
                'tests/test_rag_integration.py',
                '-v', '--tb=short',
                '-m', 'integration',
                '--timeout=600'
            ])
            self.log('SUCCESS', "Testes de integração concluídos")
            return True
        except subprocess.CalledProcessError:
            self.log('ERROR', "Testes de integração falharam")
            return False
    
    def run_performance_tests(self) -> bool:
        """Executa testes de performance"""
        self.log('INFO', "Executando testes de performance...")
        
        try:
            self.run_command([
                'python', '-m', 'pytest',
                'tests/test_performance.py',
                '-v', '--tb=short',
                '-m', 'performance',
                '--benchmark-only',
                '--benchmark-json=benchmark-results.json'
            ])
            self.log('SUCCESS', "Testes de performance concluídos")
            return True
        except subprocess.CalledProcessError:
            self.log('ERROR', "Testes de performance falharam")
            return False
    
    def run_system_validation(self) -> bool:
        """Executa validação do sistema"""
        self.log('INFO', "Executando validação do sistema...")
        
        try:
            self.run_command(['python', 'scripts/validate_system.py'])
            self.log('SUCCESS', "Validação do sistema concluída")
            return True
        except subprocess.CalledProcessError:
            self.log('ERROR', "Validação do sistema falhou")
            return False
    
    def generate_report(self) -> None:
        """Gera relatório final"""
        self.log('INFO', "Gerando relatório final...")
        
        # Criar diretório de relatórios
        self.reports_dir.mkdir(exist_ok=True)
        
        # Mover arquivos de relatório
        report_files = [
            'test-results.xml',
            'coverage.xml',
            'benchmark-results.json',
            'bandit-report.json',
            'safety-report.json',
            'validation_report.json'
        ]
        
        for file in report_files:
            file_path = self.project_dir / file
            if file_path.exists():
                shutil.move(str(file_path), str(self.reports_dir / file))
        
        # Mover diretório htmlcov
        htmlcov_path = self.project_dir / 'htmlcov'
        if htmlcov_path.exists():
            shutil.move(str(htmlcov_path), str(self.reports_dir / 'htmlcov'))
        
        self.log('SUCCESS', f"Relatório gerado em {self.reports_dir}")
    
    def run_tests(self, test_type: str) -> bool:
        """Executa testes baseado no tipo"""
        
        self.log('INFO', f"Iniciando execução de testes: {test_type}")
        self.log('INFO', f"Diretório do projeto: {self.project_dir}")
        
        success = True
        
        if test_type in ['lint', 'all', 'ci']:
            if not self.check_dependencies():
                return False
            if not self.run_linting():
                success = False
        
        if test_type in ['security', 'all', 'ci']:
            if not self.check_dependencies():
                return False
            if not self.install_test_deps():
                return False
            if not self.run_security_tests():
                success = False
        
        if test_type in ['unit', 'all', 'ci']:
            if not self.check_dependencies():
                return False
            if not self.install_test_deps():
                return False
            if not self.run_unit_tests():
                success = False
        
        if test_type in ['integration', 'all']:
            if not self.check_dependencies():
                return False
            if not self.install_test_deps():
                return False
            if not self.run_integration_tests():
                success = False
        
        if test_type in ['performance']:
            if not self.check_dependencies():
                return False
            if not self.install_test_deps():
                return False
            if not self.run_performance_tests():
                success = False
        
        if test_type in ['validation', 'all', 'ci']:
            if not self.check_dependencies():
                return False
            if not self.install_test_deps():
                return False
            if not self.run_system_validation():
                success = False
        
        if test_type in ['all', 'ci']:
            self.generate_report()
        
        if success:
            self.log('SUCCESS', f"Execução de testes concluída: {test_type}")
        else:
            self.log('ERROR', f"Alguns testes falharam: {test_type}")
        
        return success


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Executor de testes para sistema RAG')
    parser.add_argument(
        'test_type',
        choices=['lint', 'security', 'unit', 'integration', 'performance', 'validation', 'all', 'ci'],
        default='all',
        nargs='?',
        help='Tipo de teste a executar'
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    success = runner.run_tests(args.test_type)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 