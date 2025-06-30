#!/usr/bin/env python3

import click
import json
import sys
import os
from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

# Adiciona o diretório pai ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.client.rag_client import RAGClient
    from rag_pipeline import RAGPipeline
    from src.utils.document_loader import DocumentLoader
except ImportError as e:
    print(f"Erro ao importar dependências: {e}")
    sys.exit(1)

console = Console()

# ------------------------------------------------------------------
# Instância global do pipeline reutilizada pelos comandos
# ------------------------------------------------------------------

from src.config import get_cache_config

rag = RAGPipeline(config_path=None)
# Mesclar configurações de cache
try:
    cache_config = get_cache_config()
    rag.config.update(cache_config)
except Exception:
    pass  # Continue sem cache se houver problemas
rag._initialize_components()

@click.group()
def cli():
    """Sistema RAG com Roteamento Inteligente de Modelos"""
    pass

@cli.command()
@click.argument('question')
@click.option('--k', default=5, help='Número de documentos a recuperar')
@click.option('--json-output', is_flag=True, help='Output em JSON')
@click.option('--llm-only', is_flag=True, help='Usar apenas LLM sem buscar documentos')
@click.option('--hybrid/--no-hybrid', default=True, help='Usar modo híbrido com múltiplos modelos')
@click.option('--rich/--no-rich', default=True, help='Usar interface rica (Rich)')
def query(question, k, json_output, llm_only, hybrid, rich):
    """Fazer uma pergunta ao sistema RAG"""
    
    try:
        # Tenta usar RAGClient primeiro, depois RAGPipeline
        client = None
        result = None
        
        try:
            client = RAGClient()
            if llm_only:
                result = client.query_llm_only(question)
            else:
                result = client.query(question, k=k)
        except:
            # Fallback: usa instância global já criada
            global rag
            result = rag.query(query_text=question, k=k, use_hybrid=hybrid)

        if json_output:
            click.echo(json.dumps(result, indent=2))
            return

        if rich and not json_output:
            # Output rico
            mode = "Híbrido" if hybrid else "Simples"
            if llm_only:
                mode = "LLM Apenas"
            
            console.print(Panel(f"[bold]Modo:[/bold] {mode}\n[bold]Pergunta:[/bold] {question}", 
                              title="Consulta RAG"))
            
            # Exibe informações sobre os modelos usados
            if result.get('models_used'):
                models_info = Table(title="Modelos Utilizados")
                models_info.add_column("Modelo", style="cyan")
                models_info.add_column("Tipo", style="green")
                
                for model in result['models_used']:
                    if 'llama3.1' in model:
                        models_info.add_row(model, "Resposta Geral")
                    elif 'codellama' in model:
                        models_info.add_row(model, "Geração de Código")
                    else:
                        models_info.add_row(model, "Principal")
                
                console.print(models_info)
            
            # Exibe a resposta
            console.print("\n[bold cyan]Resposta:[/bold cyan]")
            md = Markdown(result['answer'])
            console.print(md)
            
            # Exibe informações adicionais
            if result.get('model'):
                console.print(f"\n[dim]Modelo: {result['model']}[/dim]")
            if result.get('response_mode'):
                console.print(f"[dim]Modo: {result['response_mode']}[/dim]")
            if result.get('strategy'):
                console.print(f"[dim]Estratégia: {result['strategy']}[/dim]")
            if result.get('note'):
                console.print(f"[dim]Nota: {result['note']}[/dim]")
            if result.get('needs_code'):
                console.print(f"[dim]Incluiu código: {'Sim' if result.get('needs_code') else 'Não'}[/dim]")
            
            # Exibe as fontes
            if result.get('sources'):
                console.print("\n[bold yellow]Fontes:[/bold yellow]")
                for i, source in enumerate(result['sources'], 1):
                    if isinstance(source, dict):
                        filename = source.get('metadata', {}).get('filename', 'Unknown')
                        score = source.get('score', 'N/A')
                        content = source.get('content', '')
                        
                        console.print(f"\n{i}. {filename}")
                        if score != 'N/A':
                            if isinstance(score, (int, float)):
                                console.print(f"   Score: {score:.2%}")
                            else:
                                console.print(f"   Score: {score}")
                        console.print(f"   Preview: {content[:200]}...")
                    else:
                        console.print(f"  • {source}")
        else:
            # Output simples
            click.echo(f"\nResposta: {result['answer']}\n")
            
            if result.get('model'):
                click.echo(f"Modelo: {result['model']}")
            if result.get('response_mode'):
                click.echo(f"Modo: {result['response_mode']}")
            click.echo()
            
            if result.get('note'):
                click.echo(f"Nota: {result['note']}\n")

            if result.get('sources'):
                click.echo("Fontes:")
                for i, source in enumerate(result['sources'], 1):
                    if isinstance(source, dict):
                        filename = source.get('metadata', {}).get('filename', 'Unknown')
                        score = source.get('score', 'N/A')
                        content = source.get('content', '')
                        
                        click.echo(f"\n{i}. {filename}")
                        if score != 'N/A':
                            if isinstance(score, (int, float)):
                                click.echo(f"   Score: {score:.2%}")
                            else:
                                click.echo(f"   Score: {score}")
                        click.echo(f"   Preview: {content[:500]}...")
                    else:
                        click.echo(f"  • {source}")

    except Exception as e:
        if rich:
            console.print(f"[bold red]Erro:[/bold red] {str(e)}")
        else:
            click.echo(f"Erro: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('files', nargs=-1, required=True)
@click.option('--chunking', '-c', default='recursive', 
              type=click.Choice(['recursive', 'semantic']),
              help='Estratégia de chunking')
@click.option('--chunk-size', '-s', default=500, help='Tamanho do chunk')
@click.option('--chunk-overlap', '-o', default=50, help='Sobreposição entre chunks')
@click.option('--rich/--no-rich', default=True, help='Usar interface rica (Rich)')
def index(files, chunking, chunk_size, chunk_overlap, rich):
    """Indexar documentos"""
    
    try:
        # Tenta RAGClient primeiro
        try:
            client = RAGClient()
            file_paths = [str(Path(f).absolute()) for f in files]
            result = client.index_documents(file_paths)
            
            if rich:
                console.print(f"[bold green]✓ Documentos indexados: {result['total_documents']}[/bold green]")
                console.print(f"[bold green]✓ Chunks criados: {result['total_chunks']}[/bold green]")
                
                if result.get('errors'):
                    console.print("\n[bold red]Erros:[/bold red]")
                    for error in result['errors']:
                        console.print(f"  - {error['document']}: {error['error']}")
            else:
                click.echo(f"Documentos indexados: {result['total_documents']}")
                click.echo(f"Chunks criados: {result['total_chunks']}")
                
                if result.get('errors'):
                    click.echo("\nErros:")
                    for error in result['errors']:
                        click.echo(f"  - {error['document']}: {error['error']}")
        
        except:
            # Fallback para DocumentLoader e RAGPipeline
            loader = DocumentLoader()
            documents = []
            
            for file_path in files:
                docs = loader.load_from_path(file_path)
                documents.extend(docs)
            
            if not documents:
                if rich:
                    console.print("[bold red]Nenhum documento encontrado![/bold red]")
                else:
                    click.echo("Nenhum documento encontrado!")
                return
            
            # Fallback: usa instância global já criada
            global rag
            
            if rich:
                with console.status(f"[bold green]Processando {len(documents)} documentos..."):
                    rag.add_documents(
                        documents=documents,
                        chunking_strategy=chunking,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                console.print(f"[bold green]✓ {len(documents)} documentos adicionados com sucesso![/bold green]")
            else:
                rag.add_documents(
                    documents=documents,
                    chunking_strategy=chunking,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                click.echo(f"Documentos indexados: {len(documents)}")

    except Exception as e:
        if rich:
            console.print(f"[bold red]Erro:[/bold red] {str(e)}")
        else:
            click.echo(f"Erro: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('file')
@click.option('--rich/--no-rich', default=True, help='Usar interface rica (Rich)')
def upload(file, rich):
    """Upload e indexação de arquivo"""
    
    try:
        client = RAGClient()
        result = client.upload_file(file)
        
        if rich:
            console.print(f"[bold green]✓ Arquivo enviado: {result['filename']}[/bold green]")
            console.print(f"[bold green]✓ Chunks criados: {result['indexing_result']['total_chunks']}[/bold green]")
        else:
            click.echo(f"Arquivo enviado: {result['filename']}")
            click.echo(f"Chunks criados: {result['indexing_result']['total_chunks']}")

    except Exception as e:
        if rich:
            console.print(f"[bold red]Erro:[/bold red] {str(e)}")
        else:
            click.echo(f"Erro: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--rich/--no-rich', default=True, help='Usar interface rica (Rich)')
def clear(rich):
    """Limpar índice"""
    
    if click.confirm('Tem certeza que deseja limpar todo o índice?'):
        try:
            # Tenta RAGClient primeiro
            try:
                client = RAGClient()
                result = client.clear_index()
                message = result['message']
            except:
                # Fallback para RAGPipeline
                global rag
                rag.clear_database()
                message = "Banco de dados limpo com sucesso!"
            
            if rich:
                console.print(f"[bold green]✓ {message}[/bold green]")
            else:
                click.echo(message)
                
        except Exception as e:
            if rich:
                console.print(f"[bold red]Erro:[/bold red] {str(e)}")
            else:
                click.echo(f"Erro: {e}", err=True)
            sys.exit(1)

@cli.command()
@click.option('--rich/--no-rich', default=True, help='Usar interface rica (Rich)')
def info(rich):
    """Mostrar informações do sistema"""
    
    try:
        # Tenta RAGClient primeiro
        try:
            client = RAGClient()
            info_data = client.get_info()
            
            if rich:
                table = Table(title="Informações do Sistema RAG")
                table.add_column("Propriedade", style="cyan", no_wrap=True)
                table.add_column("Valor", style="magenta")
                
                for key, value in info_data.items():
                    table.add_row(str(key), str(value))
                
                console.print(table)
            else:
                click.echo(json.dumps(info_data, indent=2))
                
        except:
            # Fallback para RAGPipeline
            global rag
            stats = rag.get_collection_stats()
            
            if rich:
                table = Table(title="Estatísticas do Sistema RAG")
                table.add_column("Métrica", style="cyan", no_wrap=True)
                table.add_column("Valor", style="magenta")
                
                table.add_row("Total de documentos", str(stats.get('total_documents', 'N/A')))
                table.add_row("Modelo geral", "llama3.1:8b-instruct-q4_K_M")
                table.add_row("Modelo de código", "codellama:7b-instruct")
                table.add_row("Modo híbrido", "Disponível")
                
                console.print(table)
            else:
                click.echo(json.dumps(stats, indent=2))
                
    except Exception as e:
        if rich:
            console.print(f"[bold red]Erro:[/bold red] {str(e)}")
        else:
            click.echo(f"Erro: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--query', '-q', required=True, help='Pergunta que requer código')
@click.option('--k', '-k', default=5, help='Número de documentos a recuperar')
@click.option('--rich/--no-rich', default=True, help='Usar interface rica (Rich)')
def code(query, k, rich):
    """Faz uma pergunta focada em geração de código"""
    
    try:
        global rag
        
        if rich:
            console.print(Panel(f"[bold]Modo:[/bold] Híbrido com foco em código\n[bold]Pergunta:[/bold] {query}", 
                              title="Consulta RAG - Código"))
            
            with console.status("[bold green]Gerando resposta com exemplos de código..."):
                result = rag.query_with_code_examples(query_text=query, k=k)
            
            console.print("\n[bold cyan]Resposta:[/bold cyan]")
            md = Markdown(result['answer'])
            console.print(md)
            
            console.print(f"\n[dim]Modelos usados: {', '.join(result.get('models_used', []))}[/dim]")
        else:
            result = rag.query_with_code_examples(query_text=query, k=k)
            click.echo(f"\nResposta: {result['answer']}")
            click.echo(f"Modelos usados: {', '.join(result.get('models_used', []))}")
        
    except Exception as e:
        if rich:
            console.print(f"[bold red]Erro:[/bold red] {str(e)}")
        else:
            click.echo(f"Erro: {e}", err=True)
        sys.exit(1)

@cli.command()
def demo():
    """Executa uma demonstração do sistema híbrido"""
    console.print("[bold cyan]Demonstração do Sistema RAG Híbrido[/bold cyan]\n")
    
    demo_query = "Como fazer um sistema web robusto com Python? Inclua exemplos de código."
    console.print(f"[bold]Pergunta de demonstração:[/bold] {demo_query}\n")
    
    try:
        global rag
        
        with console.status("[bold green]Processando..."):
            result = rag.query(query_text=demo_query, k=3, use_hybrid=True)
        
        console.print("[bold yellow]Processo:[/bold yellow]")
        console.print("1. Busca semântica no banco vetorial ✓")
        console.print("2. Detecção de necessidade de código ✓")
        console.print("3. Geração de resposta com Llama 3.1 ✓")
        console.print("4. Geração de exemplos com CodeLlama ✓\n")
        
        console.print("[bold cyan]Resposta Híbrida:[/bold cyan]")
        md = Markdown(result['answer'])
        console.print(md)
        
        console.print(f"\n[dim]Modelos utilizados: {', '.join(result.get('models_used', []))}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Erro:[/bold red] {str(e)}")

# Comando adicional para compatibilidade
@cli.command()
@click.option('--path', '-p', required=True, help='Caminho para arquivo ou diretório')
@click.option('--chunking', '-c', default='recursive', 
              type=click.Choice(['recursive', 'semantic']),
              help='Estratégia de chunking')
@click.option('--chunk-size', '-s', default=500, help='Tamanho do chunk')
@click.option('--chunk-overlap', '-o', default=50, help='Sobreposição entre chunks')
@click.option('--rich/--no-rich', default=True, help='Usar interface rica (Rich)')
def add(path, chunking, chunk_size, chunk_overlap, rich):
    """Adiciona documentos ao sistema RAG (alias para index)"""
    # Redireciona para o comando index
    ctx = click.get_current_context()
    ctx.invoke(index, files=[path], chunking=chunking, 
               chunk_size=chunk_size, chunk_overlap=chunk_overlap, rich=rich)

# Alias para stats
@cli.command()
@click.option('--rich/--no-rich', default=True, help='Usar interface rica (Rich)')
def stats(rich):
    """Exibe estatísticas do sistema (alias para info)"""
    ctx = click.get_current_context()
    ctx.invoke(info, rich=rich)

@cli.command()
@click.argument('project_path', type=click.Path(exists=True))
def analyze_project(project_path):
    """Analisar projeto (diretório) e popular grafo Neo4j"""
    if rag.graph_store is None:
        console.print("[bold red]Grafo não configurado (use_graph_store=False).[/bold red]")
        return
    try:
        from graphdb.code_analyzer import CodeAnalyzer

        analyzer = CodeAnalyzer(rag.graph_store)
        analyzer.analyze_project(project_path)
        console.print(f"[bold green]✓ Projeto analisado e inserido no grafo:[/bold green] {project_path}")
    except Exception as e:
        console.print(f"[bold red]Erro ao analisar projeto:[/bold red] {e}")

@cli.command()
@click.argument('query')
@click.option('--use-graph', is_flag=True, help='Usar contexto do grafo')
@click.option('--k', default=5, help='Número de documentos a recuperar')
def search(query, use_graph, k):
    """Realizar busca com ou sem grafo"""
    try:
        if use_graph:
            result = rag.query(query_text=query, k=k)
        else:
            # Desabilitar temporariamente grafo
            original_graph = rag.graph_store
            rag.graph_store = None
            result = rag.query(query_text=query, k=k)
            rag.graph_store = original_graph

        console.print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        console.print(f"[bold red]Erro:[/bold red] {e}")

if __name__ == '__main__':
    cli()
