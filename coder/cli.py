import os
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from .agent import CodingAgent

app = typer.Typer(help="Coder Agent - A local agent for code development using Cerebras API and OpenRouter.")
console = Console()

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
    """Coder Agent - A local agent for code development using Cerebras API and OpenRouter."""
    
    if not any([ask, agent, self_rewrite]):
        console.print(Panel.fit(
            "Please provide either --ask, --agent, or --self-rewrite option.",
            title="Error",
            border_style="red"
        ))
        raise typer.Exit(1)

    try:
        coding_agent = CodingAgent(
            repo_path=repo,
            model=model,
            debug=debug,
            interactive=interactive
        )

        if ask:
            console.print(Panel.fit(
                f"Question: {ask}",
                title="Asking",
                border_style="blue"
            ))
            response = coding_agent.ask(ask)
            console.print(Panel.fit(
                response,
                title="Answer",
                border_style="green"
            ))

        elif agent:
            console.print(Panel.fit(
                f"Agent Prompt: {agent}",
                title="Agent Operation",
                border_style="blue"
            ))
            response = coding_agent.agent(agent)
            console.print(Panel.fit(
                response,
                title="Result",
                border_style="green"
            ))

        elif self_rewrite:
            console.print(Panel.fit(
                "Starting self-rewrite operation...",
                title="Self-Rewrite",
                border_style="yellow"
            ))
            response = coding_agent.self_rewrite()
            console.print(Panel.fit(
                response,
                title="Self-Rewrite Complete",
                border_style="green"
            ))

    except Exception as e:
        console.print(Panel.fit(
            str(e),
            title="Error",
            border_style="red"
        ))
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 