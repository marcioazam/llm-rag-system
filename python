#!/usr/bin/env python3
"""
Script para configurar e testar o sistema RAG multi-modelo
Otimizado para: i5 8gen, 20GB RAM, MX150
"""

import subprocess
import sys
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Modelos recomendados com suas características
RECOMMENDED_MODELS = {
    'llama3.1:8b-instruct-q4_K_M': {
        'size': '4.7GB',
        'purpose': 'Explicações gerais e documentação',
        'required': True
    },
    'codellama:7b-instruct': {
        'size': '3.8GB',
        'purpose': 'Geração de código',
        'required': True
    },
    'mistral:7b-instruct-q4_0': {
        'size': '4.1GB',
        'purpose': 'Arquitetura e design patterns',
        'required': False
    },
    'sqlcoder:7b-q4_0': {
        'size': '4.0GB',
        'purpose': 'Queries SQL complexas',
        'required': False
    },
    'phi:2.7b': {
        'size': '1.6GB',
        'purpose': 'Respostas rápidas',
        'required': False
    }
}

def check_system_resources():
    """Verifica recursos do sistema"""
    console.print(Panel.fit("🔍 Verificando Recursos do Sistema", style="bold blue"))
    
    # Verifica memória
    try:
        import psutil
        mem = psutil.virtual_memory()
        console.print(f"💾 RAM Total: {mem.total / (1024**3):.1f}GB")
        console.print(f"💾 RAM Disponível: {mem.available / (1024**3):.1f}GB")
        
        if mem.available < 8 * (1024**3):
            console.print("[yellow]⚠️  Recomenda-se ter pelo menos 8GB livres[/yellow]")
    except ImportError:
        console.print("[yellow]psutil não instalado - instale com: pip install psutil[/yellow]")
    
    # Verifica GPU (MX150)
    console.print("\n🎮 GPU: NVIDIA MX150 (2GB VRAM)")
    console.print("[dim]Nota: Modelos rodarão principalmente na CPU[/dim]")

def check_ollama_installation():
    """Verifica se Ollama está instalado"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            console.print("✅ Ollama está instalado")
            return True
    except FileNotFoundError:
        pass
    
    console.print("[red]❌ Ollama não está instalado![/red]")
    console.print("Instale com: curl -fsSL https://ollama.com/install.sh | sh")
    return False

def list_installed_models():
    """Lista modelos já instalados"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
    except:
        pass
    return ""

def check_and_install_models():
    """Verifica e sugere instalação de modelos"""
    console.print("\n" + Panel.fit("📦 Verificando Modelos LLM", style="bold blue"))
    
    installed = list_installed_models()
    
    table = Table(title="Status dos Modelos")
    table.add_column("Modelo", style="cyan")
    table.add_column("Tamanho", style="magenta")
    table.add_column("Propósito", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Ação", style="blue")
    
    total_size = 0
    to_install = []
    
    for model, info in RECOMMENDED_MODELS.items():
        model_base = model.split(':')[0]
        is_installed = model_base in installed
        
        status = "✅ Instalado" if is_installed else "❌ Não instalado"
        action = "-" if is_installed else f"ollama pull {model}"
        
        if not is_installed:
            if info['required']:
                to_install.append((model, info, True))
            else:
                to_install.append((model, info, False))
            total_size += float(info['size'].replace('GB', ''))
        
        table.add_row(
            model,
            info['size'],
            info['purpose'],
            status,
            action
        )
    
    console.print(table)
    
    if to_install:
        console.print(f"\n💽 Espaço necessário: ~{total_size:.1f}GB")
        console.print(f"⏱️  Tempo estimado: {int(total_size * 3)} minutos\n")
        
        # Modelos obrigatórios
        required = [m for m in to_install if m[2]]
        optional = [m for m in to_install if not m[2]]
        
        if required:
            console.print("[bold red]Modelos Obrigatórios:[/bold red]")
            for model, info, _ in required:
                console.print(f"  • {model} - {info['purpose']}")
        
        if optional:
            console.print("\n[bold yellow]Modelos Opcionais (Recomendados):[/bold yellow]")
            for model, info, _ in optional:
                console.print(f"  • {model} - {info['purpose']}")
        
        if console.input("\n🤔 Deseja instalar os modelos? [s/N]: ").lower() == 's':
            install_models(to_install)

def install_models(models_to_install):
    """Instala os modelos selecionados"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for model, info, required in models_to_install:
            if not required:
                if console.input(f"\nInstalar {model}? [s/N]: ").lower() != 's':
                    continue
            
            task = progress.add_task(f"Baixando {model}...", total=None)
            
            try:
                result = subprocess.run(
                    ['ollama', 'pull', model],
                    capture_output=True,
                    text=True
                )
                
                progress.remove_task(task)
                
                if result.returncode == 0:
                    console.print(f"✅ {model} instalado com sucesso!")
                else:
                    console.print(f"❌ Erro ao instalar {model}: {result.stderr}")
                    
            except Exception as e:
                progress.remove_task(task)
                console.print(f"❌ Erro: {str(e)}")

def create_model_config():
    """Cria arquivo de configuração para o sistema"""
    config = """# Configuração do Sistema RAG Multi-Modelo
# Otimizado para: i5 8gen, 20GB RAM, MX150

MODELS:
  general:
    name: llama3.1:8b-instruct-q4_K_M
    max_tokens: 2048
    temperature: 0.7
    
  code:
    name: codellama:7b-instruct
    max_tokens: 4096
    temperature: 0.3
    
  architecture:
    name: mistral:7b-instruct-q4_0
    max_tokens: 2048
    temperature: 0.8
    
  sql:
    name: sqlcoder:7b-q4_0
    max_tokens: 1024
    temperature: 0.1
    
  fast:
    name: phi:2.7b
    max_tokens: 512
    temperature: 0.5

PERFORMANCE:
  max_concurrent_models: 2  # Máximo de modelos rodando simultaneamente
  cpu_threads: 6  # Threads para CPU (i5 8gen tem 4 cores/8 threads)
  use_gpu: false  # MX150 tem pouca VRAM, melhor usar CPU
  
MEMORY:
  model_cache_size: 8192  # MB para cache de modelos
  chunk_size: 512  # Tamanho dos chunks de texto
  max_context: 4096  # Contexto máximo
"""
    
    config_path = "rag_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config)
    
    console.print(f"\n📄 Arquivo de configuração criado: {config_path}")

def test_multimodel_system():
    """Testa o sistema multi-modelo"""
    console.print("\n" + Panel.fit("🧪 Testando Sistema Multi-Modelo", style="bold green"))
    
    # Adiciona o path do src
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from models.model_router import AdvancedModelRouter
        
        router = AdvancedModelRouter()
        status = router.get_model_status()
        
        console.print("\n[bold]Modelos Disponíveis:[/bold]")
        for key, info in status['available'].items():
            console.print(f"  ✅ {key}: {info['name']}")
            console.print(f"     Tarefas: {', '.join(info['tasks'])}")
        
        if status['missing']:
            console.print("\n[bold]Modelos Não Encontrados:[/bold]")
            for key, info in status['missing'].items():
                console.print(f"  ❌ {key}: {info['name']}")
        
        # Teste rápido
        if len(status['available']) >= 2:
            console.print("\n[bold]Executando teste rápido...[/bold]")
            test_query = "Como implementar cache em uma API REST?"
            
            result = router.generate_advanced_response(
                query=test_query,
                context="",
                retrieved_docs=["APIs REST devem implementar cache para melhor performance"]
            )
            
            console.print(f"\nModelos usados: {', '.join(result['models_used'])}")
            console.print(f"Tarefas realizadas: {', '.join(result['tasks_performed'])}")
            console.print("\n✅ Sistema multi-modelo funcionando!")
            
    except ImportError:
        console.print("[red]❌ Erro ao importar módulos. Verifique a instalação.[/red]")
    except Exception as e:
        console.print(f"[red]❌ Erro no teste: {str(e)}[/red]")

def show_optimization_tips():
    """Mostra dicas de otimização para o hardware"""
    tips = """
🚀 Dicas de Otimização para seu Hardware:

1. **Gerenciamento de Memória**:
   - Rode apenas 1-2 modelos por vez
   - Use `ollama stop` para liberar memória
   - Configure swap de 8-10GB para segurança

2. **Performance da CPU**:
   - Configure CPU para modo performance:
     `sudo cpupower frequency-set -g performance`
   - Use 6 threads para melhor balanço

3. **Modelos Recomendados por Uso**:
   - Desenvolvimento geral: Llama3.1 + CodeLlama
   - SQL intensivo: Adicione SQLCoder
   - Respostas rápidas: Adicione Phi-2

4. **Otimizações do Sistema**:
   ```bash
   # Aumentar limites de memória
   sudo sysctl -w vm.max_map_count=262144
   
   # Desabilitar swap aggressiveness
   sudo sysctl -w vm.swappiness=10
   ```

5. **Monitoramento**:
   - Use `htop` para monitorar CPU/RAM
   - Use `ollama ps` para ver modelos ativos
"""
    
    console.print(Panel(tips, title="💡 Otimizações", style="blue"))

def main():
    """Função principal"""
    console.print(Panel.fit(
        "🤖 Setup do Sistema RAG Multi-Modelo\n" +
        "Otimizado para: Intel i5 8ª Gen, 20GB RAM, MX150",
        style="bold magenta"
    ))
    
    # 1. Verifica recursos
    check_system_resources()
    
    # 2. Verifica Ollama
    if not check_ollama_installation():
        return
    
    # 3. Verifica e instala modelos
    check_and_install_models()
    
    # 4. Cria configuração
    if console.input("\n📝 Criar arquivo de configuração? [s/N]: ").lower() == 's':
        create_model_config()
    
    # 5. Testa sistema
    if console.input("\n🧪 Testar sistema multi-modelo? [s/N]: ").lower() == 's':
        test_multimodel_system()
    
    # 6. Mostra dicas
    if console.input("\n💡 Ver dicas de otimização? [s/N]: ").lower() == 's':
        show_optimization_tips()
    
    console.print("\n✅ Setup concluído!")

if __name__ == "__main__":
    main()
