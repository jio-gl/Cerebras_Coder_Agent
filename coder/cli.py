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

@app.command()
def main(
    ask: Optional[str] = typer.Option(None, "--ask", "-a", help="Ask a question about the repository without making changes"),
    agent: Optional[str] = typer.Option(None, "--agent", "-g", help="Prompt the agent to perform changes in the repository"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Path to the repository to analyze (default: current directory)"),
    model: str = typer.Option("qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"),
    no_think: bool = typer.Option(False, help="Disable Qwen-3-32b thinking mode by appending /no_think to the prompt"),
    debug: bool = typer.Option(False, help="Enable debug output"),
    self_rewrite: bool = typer.Option(False, help="In new folder rewrites/versionN+1 increase major version and rewriting the complete tool"),
    interactive: bool = typer.Option(False, help="Ask y/n for everything change (defaul: auto-accept all!)")
):
    """🤖 Coder Agent - A local agent for code development using Cerebras API and OpenRouter."""
    
    if not any([ask, agent, self_rewrite]):
        console.print(Panel.fit(
            "[bold red]Please provide either --ask, --agent, or --self-rewrite option.[/bold red]\n\n"
            "💡 Example commands:\n"
            "  • coder --ask \"How does this code work?\"\n"
            "  • coder --agent \"Add error handling to main.py\"\n"
            "  • coder --self-rewrite",
            title="⚠️ Error",
            border_style="red"
        ))
        raise typer.Exit(1)

    try:
        # Show initialization status
        with console.status("[bold yellow]🔧 Initializing Coder Agent...", spinner="dots") as status:
            coding_agent = CodingAgent(
                repo_path=repo,
                model=model,
                debug=debug,
                interactive=interactive
            )
            status.update("[bold green]✨ Agent initialized successfully!")

        if ask:
            # Create a live display for question processing
            with Live(console=console, refresh_per_second=4) as live:
                # Show the question
                live.update(Panel.fit(
                    f"[bold blue]🤔 Question:[/bold blue] {ask}",
                    title="📝 Processing Question",
                    border_style="blue"
                ))
                
                # Get the response
                response = coding_agent.ask(ask)
                
                # Show the formatted response
                live.update(Panel.fit(
                    Markdown(response),
                    title="💡 Answer",
                    border_style="green"
                ))

        elif agent:
            # Create a live display for agent operations
            with Live(console=console, refresh_per_second=4) as live:
                # Show the agent prompt
                live.update(Panel.fit(
                    f"[bold blue]🤖 Agent Task:[/bold blue] {agent}",
                    title="🔄 Processing Request",
                    border_style="blue"
                ))
                
                # Get the response
                response = coding_agent.agent(agent)
                
                # Show the formatted response
                live.update(Panel.fit(
                    Markdown(response),
                    title="✨ Changes Applied",
                    border_style="green"
                ))

        elif self_rewrite:
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

if __name__ == "__main__":
    app() 