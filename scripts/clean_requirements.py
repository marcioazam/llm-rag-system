#!/usr/bin/env python3
"""
Script para limpar duplicações no requirements.txt
Remove dependências duplicadas e resolve conflitos de versão
"""

import re
from typing import Dict, List, Tuple
from pathlib import Path


def parse_requirement(line: str) -> Tuple[str, str, str]:
    """
    Parseia uma linha de requirement e retorna (package, operator, version)
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None, line
    
    # Padrões para diferentes formatos de requirements
    patterns = [
        r'^([a-zA-Z0-9_-]+)([><=!]+)([0-9.]+[a-zA-Z0-9.]*)',  # package>=1.0.0
        r'^([a-zA-Z0-9_-]+)',  # package (sem versão)
    ]
    
    for pattern in patterns:
        match = re.match(pattern, line)
        if match:
            if len(match.groups()) == 3:
                return match.group(1).lower(), match.group(2), match.group(3)
            else:
                return match.group(1).lower(), None, None
    
    return None, None, line


def resolve_version_conflict(existing: Tuple[str, str], new: Tuple[str, str]) -> Tuple[str, str]:
    """
    Resolve conflitos de versão entre duas especificações
    """
    existing_op, existing_ver = existing
    new_op, new_ver = new
    
    # Se uma é >= e outra é ==, prefere ==
    if existing_op == '>=' and new_op in ['==', '>=']:
        return new
    elif new_op == '>=' and existing_op in ['==', '>=']:
        return existing
    
    # Se ambas são >=, pega a versão maior
    if existing_op == '>=' and new_op == '>=':
        if existing_ver >= new_ver:
            return existing
        else:
            return new
    
    # Default: retorna a nova versão
    return new


def clean_requirements():
    """
    Limpa o arquivo requirements.txt removendo duplicações
    """
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("❌ Arquivo requirements.txt não encontrado!")
        return False
    
    print("🔧 Limpando duplicações no requirements.txt...")
    
    # Ler todas as linhas
    with open(requirements_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Processar requirements
    seen_packages: Dict[str, Tuple[str, str]] = {}  # package -> (operator, version)
    clean_lines: List[str] = []
    comments_and_sections: List[str] = []
    
    current_section = None
    
    for line_num, line in enumerate(lines, 1):
        original_line = line.rstrip()
        package, operator, version = parse_requirement(line)
        
        # Se é comentário ou linha vazia
        if package is None:
            # Se é uma seção de comentário
            if line.strip().startswith('#') and any(marker in line for marker in ['=======', '-------', 'DEPENDENCIES']):
                current_section = line.strip()
                comments_and_sections.append(original_line)
            elif line.strip().startswith('#') or not line.strip():
                comments_and_sections.append(original_line)
            continue
        
        # Se já vimos este pacote
        if package in seen_packages:
            existing_op, existing_ver = seen_packages[package]
            print(f"⚠️  Duplicação encontrada: {package}")
            print(f"   Existente: {package}{existing_op or ''}{existing_ver or ''}")
            print(f"   Nova: {package}{operator or ''}{version or ''}")
            
            # Resolver conflito de versão
            if operator and version:
                resolved = resolve_version_conflict((existing_op, existing_ver), (operator, version))
                seen_packages[package] = resolved
                print(f"   ✅ Resolvido: {package}{resolved[0] or ''}{resolved[1] or ''}")
            continue
        
        # Adicionar novo pacote
        seen_packages[package] = (operator, version)
    
    # Escrever arquivo limpo
    backup_path = requirements_path.with_suffix('.txt.backup')
    requirements_path.rename(backup_path)
    print(f"📋 Backup criado: {backup_path}")
    
    with open(requirements_path, 'w', encoding='utf-8') as f:
        # Escrever header
        f.write("# Sistema RAG - Dependências Limpas\n")
        f.write("# Arquivo limpo automaticamente - duplicações removidas\n")
        f.write("\n")
        
        # Agrupar por seções
        core_deps = []
        api_deps = []
        ml_deps = []
        dev_deps = []
        
        for package, (operator, version) in seen_packages.items():
            version_spec = f"{operator or ''}{version or ''}" if operator and version else ""
            line = f"{package}{version_spec}"
            
            # Categorizar dependências
            if package in ['fastapi', 'uvicorn', 'pydantic', 'requests', 'httpx', 'aiohttp']:
                api_deps.append(line)
            elif package in ['numpy', 'pandas', 'scikit-learn', 'torch', 'transformers', 'sentence-transformers']:
                ml_deps.append(line)
            elif package in ['pytest', 'pytest-cov', 'flake8', 'black', 'ruff']:
                dev_deps.append(line)
            else:
                core_deps.append(line)
        
        # Escrever seções organizadas
        if core_deps:
            f.write("# Core dependencies\n")
            for dep in sorted(core_deps):
                f.write(f"{dep}\n")
            f.write("\n")
        
        if api_deps:
            f.write("# API dependencies\n")
            for dep in sorted(api_deps):
                f.write(f"{dep}\n")
            f.write("\n")
        
        if ml_deps:
            f.write("# ML/AI dependencies\n")
            for dep in sorted(ml_deps):
                f.write(f"{dep}\n")
            f.write("\n")
        
        if dev_deps:
            f.write("# Development dependencies\n")
            for dep in sorted(dev_deps):
                f.write(f"{dep}\n")
    
    print(f"✅ Requirements.txt limpo! {len(seen_packages)} dependências únicas")
    print(f"📊 Estatísticas:")
    print(f"   - Core: {len(core_deps)}")
    print(f"   - API: {len(api_deps)}")
    print(f"   - ML/AI: {len(ml_deps)}")
    print(f"   - Dev: {len(dev_deps)}")
    
    return True


if __name__ == "__main__":
    clean_requirements() 