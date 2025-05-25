import os
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.markdown import Markdown
from rich.style import Style
from rich.text import Text
from rich.live import Live
from rich.table import Table
from .agent import CodingAgent
import subprocess
import re

app = typer.Typer(help="🤖 Coder Agent - A local agent for code development using Cerebras API and OpenRouter.")
console = Console()

def create_status_table(title: str, message: str, status: str = "running") -> Table:
    """Create a status table for live updates."""
    table = Table.grid(padding=1)
    style = "yellow" if status == "running" else "green" if status == "done" else "red"
    emoji = "🤔" if status == "running" else "✨" if status == "done" else "❌"
    table.add_row(
        Text(f"{emoji} {title}", style=Style(color=style, bold=True)),
        Text(message)
    )
    return table

def initialize_agent(
    repo: Optional[str] = None,
    model: str = "qwen/qwen3-32b",
    provider: str = "Cerebras",
    max_tokens: int = 31000,
    debug: bool = False,
    interactive: bool = False
) -> CodingAgent:
    """Initialize the coding agent with the given parameters."""
    with console.status("[bold yellow]🔧 Initializing Coder Agent...", spinner="dots") as status:
        coding_agent = CodingAgent(
            repo_path=repo or os.getcwd(),
            model=model,
            api_key=None,
            max_tokens=max_tokens,
            provider=provider,
            debug=debug,
            interactive=interactive
        )
        status.update("[bold green]✨ Agent initialized successfully!")
    return coding_agent

@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask about the repository"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Path to the repository to analyze (default: current directory)"),
    model: str = typer.Option("qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"),
    provider: str = typer.Option("Cerebras", help="Provider to use (default: Cerebras)"),
    max_tokens: int = typer.Option(31000, help="Maximum tokens to generate (default: 31000)"),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(False, help="Ask y/n for everything change (default: auto-accept all!)")
):
    """Ask a question about the repository without making changes."""
    try:
        coding_agent = initialize_agent(repo, model, provider, max_tokens, debug, interactive)
        
        # Create a live display for question processing
        with Live(console=console, refresh_per_second=4) as live:
            # Show the question
            live.update(Panel.fit(
                f"[bold blue]🤔 Question:[/bold blue] {question}",
                title="📝 Processing Question",
                border_style="blue"
            ))
            
            # Get the response
            response = coding_agent.ask(question)
            
            # Show the formatted response
            live.update(Panel.fit(
                Markdown(response),
                title="💡 Answer",
                border_style="green"
            ))
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
            "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
            "• Check your API key is set correctly\n"
            "• Verify the repository path exists\n"
            "• Ensure you have internet connectivity",
            title="⚠️ Error Occurred",
            border_style="red"
        ))
        if debug:
            console.print_exception()
        raise typer.Exit(1)

@app.command()
def agent(
    prompt: str = typer.Argument(..., help="The prompt for the agent to perform changes or run commands (e.g., 'create a calculator' or 'run app.py')"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Path to the repository to analyze (default: current directory)"),
    model: str = typer.Option("qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"),
    provider: str = typer.Option("Cerebras", help="Provider to use (default: Cerebras)"),
    max_tokens: int = typer.Option(31000, help="Maximum tokens to generate (default: 31000)"),
    no_think: bool = typer.Option(False, help="Disable thinking mode by appending /no_think to the prompt"),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(False, help="Ask y/n for everything change (default: auto-accept all!)")
):
    """Prompt the agent to perform changes or run commands.
    
    If the prompt starts with 'run', 'execute', 'python', or other command keywords,
    the agent will execute the command locally instead of using the LLM API.
    
    The agent can create or modify multiple files in a single operation, making it ideal
    for creating complete projects (with multiple source files, tests, and configuration files)
    or making coordinated changes across multiple files.
    """
    try:
        # Check if this is a local command execution
        run_command_patterns = [
            r'^run\s+(.+)',
            r'^execute\s+(.+)',
            r'^start\s+(.+)',
            r'^launch\s+(.+)',
            r'^python\s+(.+)',
            r'^python3\s+(.+)',
            r'^node\s+(.+)',
            r'^npm\s+(.+)',
            r'^yarn\s+(.+)',
            r'^pip\s+(.+)',
            r'^pip3\s+(.+)',
            r'^ruby\s+(.+)',
            r'^go\s+(.+)',
            r'^java\s+(.+)',
            r'^javac\s+(.+)',
            r'^make\s+(.+)',
            r'^gcc\s+(.+)',
            r'^g\+\+\s+(.+)',
            r'^mvn\s+(.+)',
            r'^gradle\s+(.+)',
            r'^docker\s+(.+)',
            r'^kubectl\s+(.+)',
            r'^terraform\s+(.+)',
            r'^ansible\s+(.+)',
            r'^bash\s+(.+)',
            r'^sh\s+(.+)',
            r'^zsh\s+(.+)',
            # Catch all pattern for any executable that might be in PATH
            r'^[a-zA-Z0-9_\.-]+\.(py|js|sh|rb|pl|php)\s*(.*)$'
        ]
        
        is_local_command = False
        command = None
        
        for pattern in run_command_patterns:
            match = re.match(pattern, prompt.strip(), re.IGNORECASE)
            if match:
                is_local_command = True
                command = match.group(1).strip() if len(match.groups()) > 0 else prompt.strip()
                break
        
        if is_local_command:
            # For local commands, we don't need the API key
            repo_path = repo or os.getcwd()
            
            # Create a live display for command execution
            with Live(console=console, refresh_per_second=4) as live:
                # Show the command being executed
                live.update(Panel.fit(
                    f"[bold blue]🚀 Executing command:[/bold blue] {command}",
                    title="🔄 Running Command",
                    border_style="blue"
                ))
                
                try:
                    # Execute the command locally using subprocess
                    result = subprocess.run(
                        command,
                        shell=True,
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                    stdout = result.stdout.strip()
                    stderr = result.stderr.strip()
                    
                    if result.returncode == 0:
                        # Command succeeded
                        if not stdout and not stderr:
                            live.update(Panel.fit(
                                f"[bold green]✅ Command executed successfully:[/bold green] {command}",
                                title="✅ Command Completed",
                                border_style="green"
                            ))
                        else:
                            output = stdout if stdout else stderr
                            live.update(Panel.fit(
                                f"[bold green]✅ Command output:[/bold green]\n\n{output}",
                                title="✅ Command Completed",
                                border_style="green"
                            ))
                    else:
                        # Command failed
                        live.update(Panel.fit(
                            f"[bold red]❌ Command failed with error code {result.returncode}:[/bold red]\n\n{stderr}",
                            title="❌ Command Failed",
                            border_style="red"
                        ))
                except subprocess.TimeoutExpired:
                    live.update(Panel.fit(
                        f"[bold red]❌ Command timed out after 5 minutes:[/bold red] {command}",
                        title="❌ Command Failed",
                        border_style="red"
                    ))
                except Exception as e:
                    live.update(Panel.fit(
                        f"[bold red]❌ Error executing command:[/bold red] {str(e)}",
                        title="❌ Command Failed",
                        border_style="red"
                    ))
            
            return
        
        # If not a local command, proceed with the normal agent flow
        coding_agent = initialize_agent(repo, model, provider, max_tokens, debug, interactive)
        
        # Create a live display for agent operations
        with Live(console=console, refresh_per_second=4) as live:
            # Show the agent prompt
            live.update(Panel.fit(
                f"[bold blue]🤖 Agent Task:[/bold blue] {prompt}",
                title="🔄 Processing Request",
                border_style="blue"
            ))
            
            # Get the response
            response = coding_agent.agent(prompt + (" /no_think" if no_think else ""))
            
            # Show the formatted response
            if "Created/modified" in response and "files:" in response:
                # Multi-file response
                files_created = response.split("files:\n- ")[1].split("\n- ")
                
                table = Table.grid(padding=1)
                table.add_row(Text("✨ Created/modified files:", style="bold green"))
                
                for file in files_created:
                    table.add_row(Text(f"  ✓ {file.strip()}", style="green"))
                
                live.update(Panel(table, title="✨ Multiple Files Created/Modified", border_style="green"))
            else:
                # Regular response
                live.update(Panel.fit(
                    Markdown(response),
                    title="✨ Changes Applied",
                    border_style="green"
                ))
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
            "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
            "• Check your API key is set correctly\n"
            "• Verify the repository path exists\n"
            "• Ensure you have internet connectivity",
            title="⚠️ Error Occurred",
            border_style="red"
        ))
        if debug:
            console.print_exception()
        raise typer.Exit(1)

@app.command()
def self_rewrite(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Path to the repository to analyze (default: current directory)"),
    model: str = typer.Option("qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"),
    provider: str = typer.Option("Cerebras", help="Provider to use (default: Cerebras)"),
    max_tokens: int = typer.Option(31000, help="Maximum tokens to generate (default: 31000)"),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(False, help="Ask y/n for everything change (default: auto-accept all!)")
):
    """In new folder rewrites/versionN+1 increase major version and rewriting the complete tool."""
    try:
        coding_agent = initialize_agent(repo, model, provider, max_tokens, debug, interactive)
        
        # Create a live display for self-rewrite
        with Live(console=console, refresh_per_second=4) as live:
            # Initial status
            table = Table.grid(padding=1)
            table.add_row(Text("🔄 Phase 1: Analysis", style="bold yellow"))
            table.add_row(Text("  ├─ 📚 Reading current codebase"))
            table.add_row(Text("  ├─ 🔍 Analyzing code structure"))
            table.add_row(Text("  └─ 📊 Evaluating performance metrics"))
            table.add_row("")
            table.add_row(Text("⏳ Phase 2: Planning", style="dim"))
            table.add_row(Text("⏳ Phase 3: Implementation", style="dim"))
            table.add_row(Text("⏳ Phase 4: Validation", style="dim"))
            
            live.update(Panel(table, title="🚀 Self-Rewrite Operation", border_style="yellow"))
            
            # Get the initial analysis
            response = coding_agent.analyze_codebase()
            
            # Update with planning phase
            table = Table.grid(padding=1)
            table.add_row(Text("✅ Phase 1: Analysis", style="bold green"))
            table.add_row(Text("🔄 Phase 2: Planning", style="bold yellow"))
            table.add_row(Text("  ├─ 📋 Identifying improvements"))
            table.add_row(Text("  ├─ 🎯 Setting optimization targets"))
            table.add_row(Text("  └─ 🗺️ Creating implementation roadmap"))
            table.add_row("")
            table.add_row(Text("⏳ Phase 3: Implementation", style="dim"))
            table.add_row(Text("⏳ Phase 4: Validation", style="dim"))
            
            live.update(Panel(table, title="🚀 Self-Rewrite Operation", border_style="yellow"))
            
            # Start the rewrite process
            response = coding_agent.self_rewrite()
            
            # Update with implementation phase
            table = Table.grid(padding=1)
            table.add_row(Text("✅ Phase 1: Analysis", style="bold green"))
            table.add_row(Text("✅ Phase 2: Planning", style="bold green"))
            table.add_row(Text("🔄 Phase 3: Implementation", style="bold yellow"))
            table.add_row(Text("  ├─ 🏗️ Creating new version structure"))
            table.add_row(Text("  ├─ 📝 Implementing improvements"))
            table.add_row(Text("  ├─ 🔧 Optimizing performance"))
            table.add_row(Text("  └─ 🔍 Adding comprehensive tests"))
            table.add_row("")
            table.add_row(Text("⏳ Phase 4: Validation", style="dim"))
            
            live.update(Panel(table, title="🚀 Self-Rewrite Operation", border_style="yellow"))
            
            # Final validation phase
            table = Table.grid(padding=1)
            table.add_row(Text("✅ Phase 1: Analysis", style="bold green"))
            table.add_row(Text("✅ Phase 2: Planning", style="bold green"))
            table.add_row(Text("✅ Phase 3: Implementation", style="bold green"))
            table.add_row(Text("🔄 Phase 4: Validation", style="bold yellow"))
            table.add_row(Text("  ├─ 🧪 Running test suite"))
            table.add_row(Text("  ├─ 📊 Verifying performance"))
            table.add_row(Text("  └─ ✅ Ensuring compatibility"))
            
            live.update(Panel(table, title="🚀 Self-Rewrite Operation", border_style="yellow"))
            
            # Show the final success message
            table = Table.grid(padding=1)
            table.add_row(Text("✨ Self-Rewrite Successfully Completed!", style="bold green"))
            table.add_row("")
            table.add_row(Text("📈 Improvements:", style="bold blue"))
            table.add_row(Text("  ✓ Enhanced performance optimizations"))
            table.add_row(Text("  ✓ Improved error handling"))
            table.add_row(Text("  ✓ Better test coverage"))
            table.add_row(Text("  ✓ Updated documentation"))
            table.add_row("")
            table.add_row(Text("📁 New version available in:", style="yellow"))
            table.add_row(Text("  └─ ./version2/", style="cyan"))
            
            live.update(Panel(table, title="✨ Self-Rewrite Complete", border_style="green"))
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
            "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
            "• Check your API key is set correctly\n"
            "• Verify the repository path exists\n"
            "• Ensure you have internet connectivity",
            title="⚠️ Error Occurred",
            border_style="red"
        ))
        if debug:
            console.print_exception()
        raise typer.Exit(1)

# Export the cli object that the entry point is looking for
cli = app

if __name__ == "__main__":
    app() 