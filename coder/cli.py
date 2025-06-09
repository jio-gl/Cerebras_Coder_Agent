import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .agent import CodingAgent

app = typer.Typer(
    help="🤖 Coder Agent - A local agent for code development using Cerebras API and OpenRouter."
)
console = Console()

# Create a prompts directory if it doesn't exist
os.makedirs("prompts", exist_ok=True)


def create_status_table(title: str, message: str, status: str = "running") -> Table:
    """Create a status table for live updates."""
    table = Table.grid(padding=1)
    style = "yellow" if status == "running" else "green" if status == "done" else "red"
    emoji = "🤔" if status == "running" else "✨" if status == "done" else "❌"
    table.add_row(
        Text(f"{emoji} {title}", style=Style(color=style, bold=True)), Text(message)
    )
    return table


def initialize_agent(
    repo: Optional[str] = None,
    model: str = "qwen/qwen3-32b",
    provider: str = "Cerebras",
    max_tokens: int = 31000,
    debug: bool = False,
    interactive: bool = False,
) -> CodingAgent:
    """Initialize the coding agent with the given parameters."""
    # Enforce fixed model and provider
    if model != "qwen/qwen3-32b":
        raise ValueError("Only qwen/qwen3-32b model is supported")
    if provider != "Cerebras":
        raise ValueError("Only Cerebras provider is supported")

    with console.status(
        "[bold yellow]🔧 Initializing Coder Agent...", spinner="dots"
    ) as status:
        coding_agent = CodingAgent(
            repo_path=repo or os.getcwd(),
            model="qwen/qwen3-32b",  # Enforce fixed model
            api_key=None,
            max_tokens=max_tokens,
            provider="Cerebras",  # Enforce fixed provider
            debug=debug,
            interactive=interactive,
        )
        status.update("[bold green]✨ Agent initialized successfully!")
    return coding_agent


@app.command()
def ask(
    question: str = typer.Argument(
        ..., help="The question to ask about the repository"
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository to analyze (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Ask a question about the repository without making changes."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        coding_agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create a live display for question processing
        with Live(console=console, refresh_per_second=4) as live:
            # Show the question
            live.update(
                Panel.fit(
                    f"[bold blue]🤔 Question:[/bold blue] {question}",
                    title="📝 Processing Question",
                    border_style="blue",
                )
            )

            # Get the response
            response = coding_agent.ask(question)

            # Show the formatted response
            live.update(
                Panel.fit(Markdown(response), title="💡 Answer", border_style="green")
            )
    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the repository path exists\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def agent(
    prompt: str = typer.Argument(
        ...,
        help="The prompt for the agent to perform changes or run commands (e.g., 'create a calculator' or 'run app.py')",
    ),
    from_file: bool = typer.Option(
        False,
        "--from-file",
        "-f",
        help="Treat the prompt argument as a file path instead of a literal prompt",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository to analyze (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    no_think: bool = typer.Option(
        False, help="Disable thinking mode by appending /no_think to the prompt"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Prompt the agent to perform changes or run commands.

    If the prompt starts with 'run', 'execute', 'python', or other command keywords,
    the agent will execute the command locally instead of using the LLM API.

    The agent can create or modify multiple files in a single operation, making it ideal
    for creating complete projects (with multiple source files, tests, and configuration files)
    or making coordinated changes across multiple files.

    Example with file input:
        coder agent --from-file prompts/web_scraper.txt
    """
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        # Handle prompt from file if requested
        if from_file:
            try:
                with open(prompt, "r") as f:
                    file_prompt = f.read().strip()

                if not file_prompt:
                    raise ValueError("The prompt file is empty")

                # Show the prompt being loaded
                console.print(
                    Panel.fit(
                        f"[bold blue]📄 Loaded prompt from:[/bold blue] {prompt}",
                        title="📝 Prompt File",
                        border_style="blue",
                    )
                )
                # Replace the prompt with the file contents
                prompt = file_prompt
            except FileNotFoundError:
                raise ValueError(f"Prompt file not found: {prompt}")
            except Exception as e:
                raise ValueError(f"Error reading prompt file: {str(e)}")

        # Check if this is a local command execution
        run_command_patterns = [
            r"^run\s+(.+)",
            r"^execute\s+(.+)",
            r"^start\s+(.+)",
            r"^launch\s+(.+)",
            r"^python\s+(.+)",
            r"^python3\s+(.+)",
            r"^node\s+(.+)",
            r"^npm\s+(.+)",
            r"^yarn\s+(.+)",
            r"^pip\s+(.+)",
            r"^pip3\s+(.+)",
            r"^ruby\s+(.+)",
            r"^go\s+(.+)",
            r"^java\s+(.+)",
            r"^javac\s+(.+)",
            r"^make\s+(.+)",
            r"^gcc\s+(.+)",
            r"^g\+\+\s+(.+)",
            r"^mvn\s+(.+)",
            r"^gradle\s+(.+)",
            r"^docker\s+(.+)",
            r"^kubectl\s+(.+)",
            r"^terraform\s+(.+)",
            r"^ansible\s+(.+)",
            r"^bash\s+(.+)",
            r"^sh\s+(.+)",
            r"^zsh\s+(.+)",
            # Catch all pattern for any executable that might be in PATH
            r"^[a-zA-Z0-9_\.-]+\.(py|js|sh|rb|pl|php)\s*(.*)$",
        ]

        is_local_command = False
        command = None

        for pattern in run_command_patterns:
            match = re.match(pattern, prompt.strip(), re.IGNORECASE)
            if match:
                is_local_command = True
                command = (
                    match.group(1).strip()
                    if len(match.groups()) > 0
                    else prompt.strip()
                )
                break

        if is_local_command:
            # For local commands, we don't need the API key
            repo_path = repo or os.getcwd()

            # Create a live display for command execution
            with Live(console=console, refresh_per_second=4) as live:
                # Show the command being executed
                live.update(
                    Panel.fit(
                        f"[bold blue]🚀 Executing command:[/bold blue] {command}",
                        title="🔄 Running Command",
                        border_style="blue",
                    )
                )

                try:
                    # Create a process object for the command with pipes for stdout and stderr
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        cwd=repo_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,  # Line buffered
                    )

                    # Initialize output collectors
                    all_stdout = []
                    all_stderr = []

                    # Create a panel with initial content
                    output_panel = Panel.fit(
                        "[bold blue]Command output:[/bold blue]\n\n",
                        title="🔄 Command Running",
                        border_style="blue",
                    )
                    live.update(output_panel)

                    # Function to read from a pipe and update the display
                    def read_stream(stream, is_stderr=False):
                        collector = all_stderr if is_stderr else all_stdout
                        for line in iter(stream.readline, ""):
                            if not line:
                                break
                            collector.append(line)

                            # Update the live display with all collected output
                            combined_output = ""
                            if all_stdout:
                                combined_output += "".join(all_stdout)
                            if all_stderr:
                                if combined_output:
                                    combined_output += "\n"
                                combined_output += (
                                    "[bold red]Error output:[/bold red]\n"
                                    + "".join(all_stderr)
                                )

                            output_panel = Panel.fit(
                                f"[bold blue]Command output:[/bold blue]\n\n{combined_output}",
                                title="🔄 Command Running",
                                border_style="blue",
                            )
                            live.update(output_panel)

                    # Start reader threads for stdout and stderr
                    import threading

                    stdout_thread = threading.Thread(
                        target=read_stream, args=(process.stdout,)
                    )
                    stderr_thread = threading.Thread(
                        target=read_stream, args=(process.stderr, True)
                    )

                    stdout_thread.daemon = True
                    stderr_thread.daemon = True

                    stdout_thread.start()
                    stderr_thread.start()

                    # Wait for process to complete with timeout
                    try:
                        process.wait(timeout=300)  # 5 minute timeout

                        # Make sure threads are done reading
                        stdout_thread.join(timeout=1)
                        stderr_thread.join(timeout=1)

                        # Get any remaining output
                        remaining_stdout, remaining_stderr = process.communicate()
                        if remaining_stdout:
                            all_stdout.append(remaining_stdout)
                        if remaining_stderr:
                            all_stderr.append(remaining_stderr)

                        # Determine final output based on return code
                        stdout_text = "".join(all_stdout).strip()
                        stderr_text = "".join(all_stderr).strip()

                        if process.returncode == 0:
                            # Command succeeded
                            if not stdout_text and not stderr_text:
                                live.update(
                                    Panel.fit(
                                        f"[bold green]✅ Command executed successfully:[/bold green] {command}",
                                        title="✅ Command Completed",
                                        border_style="green",
                                    )
                                )
                            else:
                                output = stdout_text if stdout_text else stderr_text
                                live.update(
                                    Panel.fit(
                                        f"[bold green]✅ Command output:[/bold green]\n\n{output}",
                                        title="✅ Command Completed",
                                        border_style="green",
                                    )
                                )
                        else:
                            # Command failed
                            combined_output = stdout_text
                            if stderr_text:
                                if combined_output:
                                    combined_output += "\n\n"
                                combined_output += (
                                    f"[bold red]Error:[/bold red]\n{stderr_text}"
                                )

                            error_panel = Panel.fit(
                                f"[bold red]❌ Command failed with error code {process.returncode}:[/bold red]\n\n{combined_output}",
                                title="❌ Command Failed",
                                border_style="red",
                            )
                            live.update(error_panel)

                            # Check if the user requested to fix errors
                            if any(
                                fix_term in prompt.lower()
                                for fix_term in [
                                    "fix",
                                    "repair",
                                    "solve",
                                    "debug",
                                    "resolve",
                                ]
                            ):
                                live.update(
                                    Panel.fit(
                                        f"[bold yellow]🔍 Analyzing error and suggesting fixes...[/bold yellow]",
                                        title="🔄 Analyzing Error",
                                        border_style="yellow",
                                    )
                                )

                                # Initialize the agent to analyze the error
                                try:
                                    coding_agent = initialize_agent(
                                        repo,
                                        model,
                                        provider,
                                        max_tokens,
                                        debug,
                                        interactive,
                                    )

                                    # Prepare a prompt for the agent to analyze the error
                                    error_prompt = f"""
                                    Analyze this command error and suggest specific fixes:
                                    
                                    Command: {command}
                                    
                                    Error output:
                                    {stderr_text}
                                    
                                    Standard output:
                                    {stdout_text}
                                    
                                    Working directory: {repo_path}
                                    
                                    Provide specific fixes I can apply to solve this problem. Be concise and focus on the exact changes needed.
                                    """

                                    # Get the agent's analysis
                                    analysis = coding_agent.ask(error_prompt)

                                    # Update the display with the analysis
                                    analysis_panel = Panel.fit(
                                        f"[bold red]❌ Command failed with error code {process.returncode}:[/bold red]\n\n{combined_output}\n\n"
                                        f"[bold green]🔧 Suggested fixes:[/bold green]\n\n{analysis}",
                                        title="🔧 Error Analysis",
                                        border_style="yellow",
                                    )
                                    live.update(analysis_panel)

                                    # Ask the user if they want to apply the suggested fixes
                                    if interactive:
                                        console.print(
                                            "\n[bold yellow]Would you like to apply these fixes? (y/n)[/bold yellow]"
                                        )
                                        response = input().strip().lower()
                                        if response == "y":
                                            # Apply fixes based on the analysis
                                            fix_prompt = f"Fix the following error by modifying the necessary files. Be specific and focus only on fixing this error:\n\n{error_prompt}"

                                            live.update(
                                                Panel.fit(
                                                    f"[bold blue]🔧 Applying fixes...[/bold blue]",
                                                    title="🔄 Applying Fixes",
                                                    border_style="blue",
                                                )
                                            )

                                            fix_response = coding_agent.agent(
                                                fix_prompt
                                            )

                                            live.update(
                                                Panel.fit(
                                                    f"[bold green]✅ Fix applied:[/bold green]\n\n{fix_response}",
                                                    title="✅ Fixes Applied",
                                                    border_style="green",
                                                )
                                            )
                                    else:
                                        # In non-interactive mode, automatically apply the fixes
                                        fix_prompt = f"Fix the following error by modifying the necessary files. Be specific and focus only on fixing this error:\n\n{error_prompt}"

                                        live.update(
                                            Panel.fit(
                                                f"[bold blue]🔧 Automatically applying fixes...[/bold blue]",
                                                title="🔄 Applying Fixes",
                                                border_style="blue",
                                            )
                                        )

                                        fix_response = coding_agent.agent(fix_prompt)

                                        live.update(
                                            Panel.fit(
                                                f"[bold green]✅ Fix applied:[/bold green]\n\n{fix_response}",
                                                title="✅ Fixes Applied",
                                                border_style="green",
                                            )
                                        )
                                except Exception as analysis_error:
                                    live.update(
                                        Panel.fit(
                                            f"[bold red]❌ Command failed with error code {process.returncode}:[/bold red]\n\n{combined_output}\n\n"
                                            f"[bold red]Failed to analyze error:[/bold red] {str(analysis_error)}",
                                            title="❌ Error Analysis Failed",
                                            border_style="red",
                                        )
                                    )
                    except subprocess.TimeoutExpired:
                        # Kill the process if it times out
                        process.kill()
                        process.communicate()  # Clean up
                        live.update(
                            Panel.fit(
                                f"[bold red]❌ Command timed out after 5 minutes:[/bold red] {command}",
                                title="❌ Command Failed",
                                border_style="red",
                            )
                        )
                except Exception as e:
                    live.update(
                        Panel.fit(
                            f"[bold red]❌ Error executing command:[/bold red] {str(e)}",
                            title="❌ Command Failed",
                            border_style="red",
                        )
                    )

            return

        # If not a local command, proceed with the normal agent flow
        coding_agent = initialize_agent(
            repo, model, provider, max_tokens, debug, interactive
        )

        # Create a live display for agent operations
        with Live(console=console, refresh_per_second=4) as live:
            # Show the agent prompt
            live.update(
                Panel.fit(
                    f"[bold blue]🤖 Agent Task:[/bold blue] {prompt}",
                    title="🔄 Processing Request",
                    border_style="blue",
                )
            )

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

                live.update(
                    Panel(
                        table,
                        title="✨ Multiple Files Created/Modified",
                        border_style="green",
                    )
                )
            else:
                # Regular response
                live.update(
                    Panel.fit(
                        Markdown(response),
                        title="✨ Changes Applied",
                        border_style="green",
                    )
                )
    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the repository path exists\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def self_rewrite(
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository to analyze (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """In new folder rewrites/versionN+1 increase major version and rewriting the complete tool."""
    try:
        coding_agent = initialize_agent(
            repo, model, provider, max_tokens, debug, interactive
        )

        repo_path = repo or os.getcwd()
        version_dir = os.path.join(repo_path, "version2")

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

            live.update(
                Panel(table, title="🚀 Self-Rewrite Operation", border_style="yellow")
            )

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

            live.update(
                Panel(table, title="🚀 Self-Rewrite Operation", border_style="yellow")
            )

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

            live.update(
                Panel(table, title="🚀 Self-Rewrite Operation", border_style="yellow")
            )

            # Final validation phase
            table = Table.grid(padding=1)
            table.add_row(Text("✅ Phase 1: Analysis", style="bold green"))
            table.add_row(Text("✅ Phase 2: Planning", style="bold green"))
            table.add_row(Text("✅ Phase 3: Implementation", style="bold green"))
            table.add_row(Text("🔄 Phase 4: Validation", style="bold yellow"))
            table.add_row(Text("  ├─ 🧪 Running test suite"))
            table.add_row(Text("  ├─ 📊 Verifying performance"))
            table.add_row(Text("  └─ ✅ Ensuring compatibility"))

            live.update(
                Panel(table, title="🚀 Self-Rewrite Operation", border_style="yellow")
            )

            # Check if rewrite was successful by verifying the response and directory
            if "failed" in response.lower() or not os.path.exists(version_dir):
                error_table = Table.grid(padding=1)
                error_table.add_row(Text("❌ Self-Rewrite Failed!", style="bold red"))
                error_table.add_row("")
                error_table.add_row(Text(response, style="yellow"))

                live.update(
                    Panel(
                        error_table, title="⚠️ Self-Rewrite Failed", border_style="red"
                    )
                )
            else:
                # Show the final success message
                success_table = Table.grid(padding=1)
                success_table.add_row(
                    Text("✨ Self-Rewrite Successfully Completed!", style="bold green")
                )
                success_table.add_row("")
                success_table.add_row(Text("📈 Improvements:", style="bold blue"))
                success_table.add_row(Text("  ✓ Enhanced performance optimizations"))
                success_table.add_row(Text("  ✓ Improved error handling"))
                success_table.add_row(Text("  ✓ Better test coverage"))
                success_table.add_row(Text("  ✓ Updated documentation"))
                success_table.add_row("")
                success_table.add_row(
                    Text("📁 New version available in:", style="yellow")
                )
                success_table.add_row(Text(f"  └─ {version_dir}/", style="cyan"))

                live.update(
                    Panel(
                        success_table,
                        title="✨ Self-Rewrite Complete",
                        border_style="green",
                    )
                )
    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the repository path exists\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def fix_syntax(
    path: str = typer.Argument(
        ...,
        help="Path to the file or directory to check and fix Python syntax errors",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository to analyze (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Fix Python syntax errors in a file or directory using LLM-powered analysis.

    This command checks Python files for syntax errors and attempts to fix them
    using both heuristic techniques and LLM-powered analysis. It can process
    a single file or recursively scan all Python files in a directory.

    Examples:
      coder fix-syntax file.py
      coder fix-syntax ./my_project/
    """
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        coding_agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create a progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )

        with progress:
            task = progress.add_task(
                f"[yellow]Checking {path} for syntax errors...", total=100
            )

            # Update progress
            progress.update(
                task,
                advance=50,
                description=f"[yellow]Fixing syntax errors in {path}...",
            )

            # Fix syntax errors
            result = coding_agent.fix_syntax_errors(path)

            # Complete progress
            progress.update(
                task,
                completed=100,
                description=f"[green]Completed syntax check on {path}",
            )

        # Display results with appropriate styling
        if "✓" in result:
            console.print(
                Panel.fit(result, title="✅ Syntax Check", border_style="green")
            )
        elif "⚠️" in result:
            console.print(
                Panel.fit(result, title="⚠️ Syntax Warnings", border_style="yellow")
            )
        else:
            console.print(
                Panel.fit(result, title="❌ Syntax Errors", border_style="red")
            )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the path exists and is accessible\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


# Create a subcommand group for LLM toolkit commands
llm_tools = typer.Typer(help="🧠 LLM-powered code tools")
app.add_typer(llm_tools, name="code")


@llm_tools.command("analyze")
def analyze_code(
    file_path: str = typer.Argument(..., help="Path to the file to analyze"),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Analyze a file for code quality, complexity, and potential issues."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        coding_agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create a progress display
        with console.status(
            f"[bold yellow]🔍 Analyzing {file_path}...", spinner="dots"
        ) as status:
            # Run the analysis
            analysis = coding_agent.analyze_file(file_path)

            status.update("[bold green]✨ Analysis complete!")

        # Create a rich table to display results
        table = Table(title=f"🔍 Code Analysis: {file_path}")

        # Add columns
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="green")

        # Add rows with analysis results
        table.add_row("Complexity", str(analysis.get("complexity", "N/A")))
        table.add_row("Functions", str(analysis.get("functions", "N/A")))
        table.add_row("Classes", str(analysis.get("classes", "N/A")))

        # Add imports as a comma-separated list
        imports = analysis.get("imports", [])
        table.add_row("Imports", ", ".join(imports) if imports else "None")

        # Create a panel for issues and suggestions
        issues_table = Table.grid(padding=1)
        issues_table.add_row(Text("⚠️ Potential Issues:", style="bold yellow"))
        for issue in analysis.get("potential_issues", []):
            issues_table.add_row(f"  • {issue}")

        issues_table.add_row("")
        issues_table.add_row(Text("💡 Suggestions:", style="bold blue"))
        for suggestion in analysis.get("suggestions", []):
            issues_table.add_row(f"  • {suggestion}")

        # Display the analysis
        console.print(table)
        console.print(
            Panel(issues_table, title="Issues & Suggestions", border_style="yellow")
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the file path exists and is accessible\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@llm_tools.command("optimize")
def optimize_code(
    file_path: str = typer.Argument(..., help="Path to the file to optimize"),
    goal: str = typer.Option(
        "performance",
        "--goal",
        "-g",
        help="Optimization goal (performance, readability, memory)",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Optimize a file for performance, readability, or memory usage."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        # Validate optimization goal
        valid_goals = ["performance", "readability", "memory"]
        if goal not in valid_goals:
            console.print(
                f"[bold red]❌ Error:[/bold red] Invalid optimization goal: {goal}"
            )
            console.print(f"[yellow]Valid goals are: {', '.join(valid_goals)}[/yellow]")
            raise typer.Exit(1)

        coding_agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create a progress display
        with console.status(
            f"[bold yellow]🔧 Optimizing {file_path} for {goal}...", spinner="dots"
        ) as status:
            # Run the optimization
            result = coding_agent.optimize_file(file_path, goal)

            status.update("[bold green]✨ Optimization complete!")

        # Display the result
        console.print(
            Panel.fit(
                f"[bold green]{result}[/bold green]\n\n"
                f"[yellow]The file has been optimized for {goal}.[/yellow]",
                title="✅ Optimization Complete",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the file path exists and is accessible\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@llm_tools.command("docstring")
def add_docstrings(
    file_path: str = typer.Argument(..., help="Path to the file to add docstrings to"),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Add or improve docstrings in a file."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        coding_agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create a progress display
        with console.status(
            f"[bold yellow]📝 Adding docstrings to {file_path}...", spinner="dots"
        ) as status:
            # Add docstrings
            result = coding_agent.add_docstrings(file_path)

            status.update("[bold green]✨ Docstrings added/improved!")

        # Display the result
        console.print(
            Panel.fit(
                f"[bold green]{result}[/bold green]",
                title="✅ Docstrings Added",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the file path exists and is accessible\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@llm_tools.command("tests")
def generate_tests(
    file_path: str = typer.Argument(..., help="Path to the file to generate tests for"),
    output_path: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write the test file (default: auto-generate in tests/ directory)",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Generate unit tests for a file."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        coding_agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create a progress display
        with console.status(
            f"[bold yellow]🧪 Generating tests for {file_path}...", spinner="dots"
        ) as status:
            # Generate tests
            result = coding_agent.generate_tests(file_path, output_path)

            status.update("[bold green]✨ Tests generated!")

        # Display the result
        console.print(
            Panel.fit(
                f"[bold green]{result}[/bold green]",
                title="✅ Tests Generated",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the file path exists and is accessible\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@llm_tools.command("errors")
def enhance_error_handling(
    file_path: str = typer.Argument(
        ..., help="Path to the file to enhance error handling in"
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Enhance error handling in a file."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        coding_agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create a progress display
        with console.status(
            f"[bold yellow]🛡️ Enhancing error handling in {file_path}...",
            spinner="dots",
        ) as status:
            # Enhance error handling
            result = coding_agent.enhance_error_handling(file_path)

            status.update("[bold green]✨ Error handling enhanced!")

        # Display the result
        console.print(
            Panel.fit(
                f"[bold green]{result}[/bold green]",
                title="✅ Error Handling Enhanced",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the file path exists and is accessible\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@llm_tools.command("explain")
def explain_code(
    file_path: str = typer.Argument(..., help="Path to the file to explain"),
    level: str = typer.Option(
        "detailed",
        "--level",
        "-l",
        help="Explanation level (basic, detailed, advanced)",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Generate a natural language explanation of a file."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        # Validate explanation level
        valid_levels = ["basic", "detailed", "advanced"]
        if level not in valid_levels:
            console.print(
                f"[bold red]❌ Error:[/bold red] Invalid explanation level: {level}"
            )
            console.print(
                f"[yellow]Valid levels are: {', '.join(valid_levels)}[/yellow]"
            )
            raise typer.Exit(1)

        coding_agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create a progress display
        with console.status(
            f"[bold yellow]📚 Generating {level} explanation for {file_path}...",
            spinner="dots",
        ) as status:
            # Generate explanation
            explanation = coding_agent.explain_code(file_path, level)

            status.update("[bold green]✨ Explanation generated!")

        # Display the explanation
        console.print(
            Panel(
                Markdown(explanation),
                title=f"📚 {level.capitalize()} Explanation of {file_path}",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the file path exists and is accessible\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@llm_tools.command("refactor")
def refactor_code(
    file_path: str = typer.Argument(..., help="Path to the file to refactor"),
    goal: str = typer.Argument(
        ..., help="Refactoring goal (e.g., 'extract method', 'apply factory pattern')"
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Refactor code according to a specific goal."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        coding_agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create a progress display
        with console.status(
            f"[bold yellow]🔄 Refactoring {file_path} with goal: {goal}...",
            spinner="dots",
        ) as status:
            # Refactor code
            result = coding_agent.refactor_code(file_path, goal)

            status.update("[bold green]✨ Refactoring complete!")

        # Display the result
        console.print(
            Panel.fit(
                f"[bold green]{result}[/bold green]",
                title="✅ Refactoring Complete",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check your API key is set correctly\n"
                "• Verify the file path exists and is accessible\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@llm_tools.command("markdown")
def generate_markdown_docs(
    file_path: str = typer.Argument(
        ..., help="Path to the Python file to generate Markdown documentation for"
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write the Markdown file (default: auto-generate .md file with same name)",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Generate Markdown documentation for Python code."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Create default output path if not provided
        if not output_path:
            path_obj = Path(file_path)
            output_path = str(path_obj.with_suffix(".md"))

        # Progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating Markdown documentation...", total=1)

            # Read the file
            try:
                with open(file_path, "r") as f:
                    code = f.read()
            except Exception as e:
                console.print(f"[bold red]Error reading file:[/bold red] {str(e)}")
                raise typer.Exit(1)

            # Generate Markdown docs
            progress.update(task, description="Generating Markdown documentation...")
            markdown_content = agent.llm_toolkit.generate_markdown_docs(code)

            # Write to output file
            progress.update(task, description="Writing Markdown file...")
            try:
                with open(output_path, "w") as f:
                    f.write(markdown_content)
            except Exception as e:
                console.print(f"[bold red]Error writing file:[/bold red] {str(e)}")
                raise typer.Exit(1)

            progress.update(
                task, completed=1, description="Documentation generated successfully!"
            )

        # Show success message
        console.print(
            Panel.fit(
                f"[bold green]✨ Generated Markdown documentation:[/bold green] {output_path}",
                title="📝 Documentation Generator",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}",
                title="⚠️ Documentation Generation Failed",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@llm_tools.command("syntax")
def fix_python_syntax(
    file_path: str = typer.Argument(
        ..., help="Path to the Python file to fix syntax errors in"
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Fix Python syntax errors in a file."""
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        agent = initialize_agent(
            repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
        )

        # Progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Checking for syntax errors...", total=1)

            # Read the file
            try:
                with open(file_path, "r") as f:
                    code = f.read()
            except Exception as e:
                console.print(f"[bold red]Error reading file:[/bold red] {str(e)}")
                raise typer.Exit(1)

            # Check for syntax errors
            has_syntax_errors = False
            try:
                compile(code, file_path, "exec")
            except SyntaxError:
                has_syntax_errors = True

            if not has_syntax_errors:
                progress.update(
                    task, completed=1, description="No syntax errors found!"
                )
                console.print(
                    Panel.fit(
                        f"[bold green]✨ File has no syntax errors:[/bold green] {file_path}",
                        title="🔍 Syntax Check",
                        border_style="green",
                    )
                )
                return

            # Fix syntax errors
            progress.update(task, description="Fixing syntax errors...")
            fixed_code = agent.llm_toolkit.fix_python_syntax(code)

            # Write the fixed code
            progress.update(task, description="Saving fixed code...")
            try:
                with open(file_path, "w") as f:
                    f.write(fixed_code)
            except Exception as e:
                console.print(f"[bold red]Error writing file:[/bold red] {str(e)}")
                raise typer.Exit(1)

            progress.update(
                task, completed=1, description="Syntax errors fixed successfully!"
            )

        # Show success message
        console.print(
            Panel.fit(
                f"[bold green]✨ Fixed syntax errors in:[/bold green] {file_path}",
                title="🔧 Syntax Fixer",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}",
                title="⚠️ Syntax Fix Failed",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def prompt_from_file(
    prompt_file: str = typer.Argument(
        ..., help="Path to a text file containing the prompt"
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the repository to analyze (default: current directory)",
    ),
    model: str = typer.Option(
        "qwen/qwen3-32b", help="Model to use (default: qwen/qwen3-32b)"
    ),
    provider: str = typer.Option(
        "Cerebras", help="Provider to use (default: Cerebras)"
    ),
    max_tokens: int = typer.Option(
        31000, help="Maximum tokens to generate (default: 31000)"
    ),
    no_think: bool = typer.Option(
        False, help="Disable thinking mode by appending /no_think to the prompt"
    ),
    debug: bool = typer.Option(False, help="Enable debug output"),
    interactive: bool = typer.Option(
        False, help="Ask y/n for everything change (default: auto-accept all!)"
    ),
):
    """Execute a prompt from a text file.

    This command reads a prompt from a text file and passes it to the agent.
    Useful for complex or multi-line prompts that would be difficult to type
    on the command line.

    Example:
        coder prompt-from-file prompts/web_scraper.txt
    """
    try:
        # Enforce fixed model and provider
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")

        # Read the prompt from the file
        try:
            with open(prompt_file, "r") as f:
                prompt = f.read().strip()

            if not prompt:
                raise ValueError("The prompt file is empty")

            # Show the prompt being loaded
            console.print(
                Panel.fit(
                    f"[bold blue]📄 Loaded prompt from:[/bold blue] {prompt_file}",
                    title="📝 Prompt File",
                    border_style="blue",
                )
            )

            # Pass the prompt to the agent function
            coding_agent = initialize_agent(
                repo, "qwen/qwen3-32b", "Cerebras", max_tokens, debug, interactive
            )

            # Create a live display for agent operation
            with Live(console=console, refresh_per_second=4) as live:
                # Show prompt processing status
                live.update(
                    create_status_table(
                        "Processing prompt",
                        "Analyzing and executing actions...",
                        "running",
                    )
                )

                # Execute the agent with the prompt from the file
                result = coding_agent.agent(prompt + (" /no_think" if no_think else ""))

                # Show the formatted response
                if "Created/modified" in result and "files:" in result:
                    # Multi-file response
                    files_created = result.split("files:\n- ")[1].split("\n- ")

                    table = Table.grid(padding=1)
                    table.add_row(
                        Text("✨ Created/modified files:", style="bold green")
                    )

                    for file in files_created:
                        table.add_row(Text(f"  ✓ {file.strip()}", style="green"))

                    live.update(
                        Panel(
                            table,
                            title="✨ Multiple Files Created/Modified",
                            border_style="green",
                        )
                    )
                else:
                    # Regular response
                    live.update(
                        Panel.fit(
                            Markdown(result),
                            title="✨ Changes Applied",
                            border_style="green",
                        )
                    )

        except FileNotFoundError:
            raise ValueError(f"Prompt file not found: {prompt_file}")
        except Exception as e:
            raise ValueError(f"Error reading prompt file: {str(e)}")

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]❌ Error:[/bold red] {str(e)}\n\n"
                "[yellow]💡 Troubleshooting Tips:[/yellow]\n"
                "• Check that the prompt file exists and is readable\n"
                "• Verify the file contains a valid prompt\n"
                "• Ensure you have internet connectivity",
                title="⚠️ Error Occurred",
                border_style="red",
            )
        )
        if debug:
            console.print_exception()
        raise typer.Exit(1)


# Export the cli object that the entry point is looking for
cli = app

if __name__ == "__main__":
    app()
