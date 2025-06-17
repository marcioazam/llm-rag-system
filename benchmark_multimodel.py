#!/usr/bin/env python3
"""
Benchmark para comparar diferentes configurações de modelos
Útil para encontrar a melhor combinação para seu hardware
"""

import sys
import os
import time
import psutil
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import RAGPipeline

console = Console()

# Queries de teste categorizadas
TEST_QUERIES = {
    'code_generation': [
        "Implemente um sistema de autenticação JWT completo em FastAPI",
        "Crie uma classe Python para gerenciar conexões com Redis",
        "Mostre como implementar rate limiting em uma API REST"
    ],
    'sql_queries': [
        "Crie uma query SQL para análise de vendas mensais com window functions",
        "Como otimizar uma query com múltiplos JOINs em PostgreSQL?",
        "Escreva uma stored procedure para atualização em lote"
    ],
    'architecture': [
        "Projete uma arquitetura de microserviços para um e-commerce",
        "Quais padrões usar para um sistema de mensageria distribuído?",
        "Como implementar CQRS e Event Sourcing?"
    ],
    'general': [
        "O que é Docker e quais suas vantagens?",
        "Explique o conceito de CI/CD",
        "Quais são as melhores práticas de segurança em APIs?"
    ]
}

def measure_resource_usage():
    """Mede uso de CPU e memória"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        'memory_percent': psutil.virtual_memory().percent
    }

def benchmark_query(rag: RAGPipeline, query: str, mode: str) -> Dict:
    """Executa benchmark para uma query específica"""
    start_time = time.time()
    start_resources = measure_resource_usage()
    
    # Executa query baseado no modo
    if mode == 'single':
        result = rag.query(query, k=3, use_hybrid=False)
    elif mode == 'hybrid':
        result = rag.query(query, k=3, use_hybrid=True)
    else:  # specific models
        result = rag.query_with_specific_models(query, models=['sql', 'code'], k=3)
    
    end_time = time.time()
    end_resources = measure_resource_usage()
    
    return {
        'time': end_time - start_time,
        'models_used': result.get('models_used', []),
        'answer_preview': result['answer'][:150] + '...',
        'cpu_delta': end_resources['cpu_percent'] - start_resources['cpu_percent'],
        'memory_delta': end_resources['memory_mb'] - start_resources['memory_mb'],
        'strategy': result.get('strategy', 'unknown'),
        'tasks': result.get('tasks_performed', [])
    }

def run_benchmark():
    """Executa benchmark completo"""
    console.print(Panel.fit("🏁 Benchmark do Sistema RAG Multi-Modelo", style="bold cyan"))
    
    # Inicializa RAG
    console.print("\n[bold]Inicializando sistema...[/bold]")
    rag = RAGPipeline(use_advanced_routing=True)
    
    # Adiciona documentos de teste
    test_docs = [
        {
            "content": "FastAPI é um framework web moderno para Python...",
            "source": "fastapi_docs.txt"
        },
        {
            "content": "SQL window functions permitem análises avançadas...",
            "source": "sql_guide.txt"
        }
    ]
    rag.add_documents(test_docs)
    
    # Verifica modelos disponíveis
    models_info = rag.get_available_models()
    if 'available' in models_info:
        console.print("\n[bold green]Modelos disponíveis:[/bold green]")
        for model, info in models_info['available'].items():
            console.print(f"  • {model}: {info.get('name', 'N/A')}")
    
    # Resultados por categoria
    results = {}
    
    # Testa cada categoria
    for category, queries in TEST_QUERIES.items():
        console.print(f"\n[bold yellow]Testando categoria: {category}[/bold yellow]")
        results[category] = {}
        
        for query in track(queries, description=f"Processando {category}..."):
            results[category][query] = {}
            
            # Testa diferentes modos
            if rag.routing_mode == 'simple':
                modes = ['single', 'hybrid']
            else:
                modes = ['single', 'hybrid']
            
            for mode in modes:
                try:
                    result = benchmark_query(rag, query, mode)
                    results[category][query][mode] = result
                except Exception as e:
                    console.print(f"[red]Erro em {mode}: {str(e)}[/red]")
                    results[category][query][mode] = {'error': str(e)}
    
    # Exibe resultados
    display_results(results)
    
    # Salva resultados
    save_results(results)

def display_results(results: Dict):
    """Exibe resultados do benchmark"""
    console.print("\n" + Panel.fit("📊 Resultados do Benchmark", style="bold green"))
    
    # Tabela resumo por categoria
    for category, queries in results.items():
        table = Table(title=f"\n{category.upper()}")
        table.add_column("Query", style="cyan", width=40)
        table.add_column("Modo", style="magenta")
        table.add_column("Tempo (s)", style="yellow")
        table.add_column("Modelos", style="green")
        table.add_column("CPU Δ%", style="blue")
        table.add_column("Mem ΔMB", style="red")
        
        for query, modes in queries.items():
            query_short = query[:37] + "..." if len(query) > 40 else query
            
            for mode, result in modes.items():
                if 'error' in result:
                    table.add_row(query_short, mode, "ERRO", "-", "-", "-")
                else:
                    table.add_row(
                        query_short,
                        mode,
                        f"{result['time']:.2f}",
                        ", ".join([m.split(':')[0] for m in result['models_used']]),
                        f"{result['cpu_delta']:.1f}",
                        f"{result['memory_delta']:.1f}"
                    )
        
        console.print(table)
    
    # Estatísticas gerais
    console.print("\n[bold]📈 Estatísticas Gerais:[/bold]")
    
    total_single = 0
    total_hybrid = 0
    count_single = 0
    count_hybrid = 0
    
    for category in results.values():
        for modes in category.values():
            if 'single' in modes and 'time' in modes['single']:
                total_single += modes['single']['time']
                count_single += 1
            if 'hybrid' in modes and 'time' in modes['hybrid']:
                total_hybrid += modes['hybrid']['time']
                count_hybrid += 1
    
    if count_single > 0:
        avg_single = total_single / count_single
        console.print(f"  • Tempo médio (modelo único): {avg_single:.2f}s")
    
    if count_hybrid > 0:
        avg_hybrid = total_hybrid / count_hybrid
        console.print(f"  • Tempo médio (híbrido): {avg_hybrid:.2f}s")
        
        if count_single > 0:
            overhead = ((avg_hybrid - avg_single) / avg_single) * 100
            console.print(f"  • Overhead do modo híbrido: {overhead:.1f}%")

def save_results(results: Dict):
    """Salva resultados em arquivo"""
    import json
    from datetime import datetime
    
    filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n💾 Resultados salvos em: {filename}")

def main():
    """Função principal"""
    try:
        # Verifica recursos antes de começar
        mem = psutil.virtual_memory()
        if mem.available < 8 * (1024**3):  # 8GB
            console.print("[yellow]⚠️  Aviso: Menos de 8GB de RAM disponível[/yellow]")
            if not console.input("Continuar mesmo assim? [s/N]: ").lower() == 's':
                return
        
        run_benchmark()
        
    except KeyboardInterrupt:
        console.print("\n[red]Benchmark interrompido pelo usuário[/red]")
    except Exception as e:
        console.print(f"\n[red]Erro: {str(e)}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
