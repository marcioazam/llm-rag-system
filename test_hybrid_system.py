#!/usr/bin/env python3
"""
Script de teste para o sistema RAG híbrido
Testa a integração entre Llama 3.1 e CodeLlama
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import RAGPipeline
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def test_hybrid_system():
    """Testa o sistema híbrido com diferentes tipos de perguntas"""
    
    # Inicializa o pipeline
    console.print("[bold cyan]Iniciando Sistema RAG Híbrido[/bold cyan]\n")
    rag = RAGPipeline()
    
    # Adiciona alguns documentos de teste
    test_documents = [
        {
            "content": """
            Para criar um sistema web robusto, é importante considerar:
            1. Arquitetura bem definida (MVC, microserviços, etc)
            2. Segurança (autenticação, autorização, HTTPS)
            3. Performance (cache, otimização de queries)
            4. Escalabilidade (horizontal e vertical)
            5. Monitoramento e logs
            6. Testes automatizados
            7. CI/CD pipeline
            """,
            "source": "web_systems_guide.txt",
            "metadata": {"type": "architecture"}
        },
        {
            "content": """
            FastAPI é excelente para construir APIs robustas em Python.
            Principais características:
            - Type hints e validação automática com Pydantic
            - Documentação automática (Swagger/OpenAPI)
            - Async/await nativo
            - Alta performance
            - Fácil de testar
            """,
            "source": "fastapi_overview.txt",
            "metadata": {"type": "framework"}
        },
        {
            "content": """
            Boas práticas de segurança para APIs:
            - Use HTTPS sempre
            - Implemente rate limiting
            - Valide todas as entradas
            - Use tokens JWT para autenticação
            - Implemente CORS corretamente
            - Mantenha dependências atualizadas
            """,
            "source": "api_security.txt",
            "metadata": {"type": "security"}
        }
    ]
    
    console.print("[bold]Adicionando documentos de teste...[/bold]")
    rag.add_documents(test_documents, chunking_strategy='recursive')
    console.print("[green]✓ Documentos adicionados[/green]\n")
    
    # Testes com diferentes tipos de queries
    test_queries = [
        {
            "query": "Quais são os principais aspectos de um sistema web robusto?",
            "expected_mode": "simple",
            "description": "Pergunta conceitual - deve usar apenas Llama 3.1"
        },
        {
            "query": "Como implementar autenticação JWT em FastAPI? Mostre um exemplo",
            "expected_mode": "hybrid",
            "description": "Pergunta com código - deve usar ambos os modelos"
        },
        {
            "query": "Crie uma API REST completa em FastAPI com autenticação e rate limiting",
            "expected_mode": "hybrid",
            "description": "Solicitação de código complexo - deve gerar muito código"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        console.print(Panel(
            f"[bold]Teste {i}:[/bold] {test['description']}\n"
            f"[bold]Pergunta:[/bold] {test['query']}",
            title=f"Teste {i}/3"
        ))
        
        # Executa a query
        result = rag.query(test['query'], k=3, use_hybrid=True)
        
        # Exibe os resultados
        console.print(f"\n[bold cyan]Estratégia usada:[/bold cyan] {result['strategy']}")
        console.print(f"[bold cyan]Modelos usados:[/bold cyan] {', '.join(result['models_used'])}")
        console.print(f"[bold cyan]Detectou necessidade de código:[/bold cyan] {'Sim' if result['needs_code'] else 'Não'}")
        
        console.print("\n[bold]Resposta:[/bold]")
        md = Markdown(result['answer'])
        console.print(md)
        
        console.print("\n" + "="*80 + "\n")
    
    # Teste específico do modo de código
    console.print(Panel(
        "[bold]Teste Especial: Forçando geração de código[/bold]",
        title="Teste Extra"
    ))
    
    code_result = rag.query_with_code_examples(
        "Explique o conceito de middleware em FastAPI",
        k=3
    )
    
    console.print("[bold]Resposta com código forçado:[/bold]")
    md = Markdown(code_result['answer'])
    console.print(md)
    
    # Estatísticas finais
    stats = rag.get_collection_stats()
    console.print(f"\n[bold green]Teste concluído![/bold green]")
    console.print(f"Total de documentos no sistema: {stats['total_documents']}")

if __name__ == "__main__":
    try:
        test_hybrid_system()
    except Exception as e:
        console.print(f"[bold red]Erro durante o teste:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()
