"""Prompt templates for the CodingAgent."""


def get_improved_specs_prompt(
    current_specs: str, current_version: int, next_version: int
) -> str:
    """Generate a prompt for improved specifications.

    Args:
        current_specs: Current specifications text
        current_version: Current version number
        next_version: Next version number

    Returns:
        Prompt for generating improved specifications
    """
    return f"""
You are an expert software architect tasked with creating improved specifications for a coding agent.

# Current Specifications (v{current_version})
{current_specs}

# Task
Generate improved specifications for version {next_version} of the coding agent.
The specifications should include:
1. A clear title with the version number
2. An introduction describing the purpose of the coding agent
3. A list of features (including improvements over the previous version)
4. Architecture details
5. Implementation requirements

Make significant improvements to the architecture and features compared to v{current_version}.
Ensure the specifications are detailed and implementable.

# Output Format
The output should be in Markdown format with clear sections.
Start with the title "# Coding Agent v{next_version} Specification"
"""


def get_file_generation_prompt(file_path: str, specs: str, version: int) -> str:
    """Generate a prompt for file content generation.

    Args:
        file_path: Path to the file to generate
        specs: Specifications for the file
        version: Version number

    Returns:
        Prompt for generating file content
    """
    return f"""
You are an expert Python developer tasked with implementing a file for the Coding Agent v{version}.

# File Path
{file_path}

# Specifications
{specs}

# Task
Generate the complete content for this file based on the specifications.
The file should be high-quality, well-documented, and include all necessary imports.
Make sure the code adheres to best practices and is consistent with the version {version} architecture.

# Output Format
Output only the file content, no explanations.
"""


def get_completion_summary_prompt(version: int, files: list) -> str:
    """Generate a prompt for completion summary.

    Args:
        version: Version number
        files: List of generated files

    Returns:
        Prompt for generating completion summary
    """
    file_list = "\n".join([f"- {file}" for file in files])

    return f"""
Summarize the completion of the self-rewrite process for Coding Agent v{version}.

# Files Generated
{file_list}

# Task
Create a concise summary of the self-rewrite, highlighting:
1. The successful completion
2. The version number ({version})
3. Key improvements in this version
4. Suggested next steps

# Output Format
Output in Markdown format, starting with "# Self-Rewrite Completed Successfully!"
"""


def get_file_specific_instructions(file_path: str, version: int) -> str:
    """Get file-specific instructions based on the file path.

    Args:
        file_path: Path to the file
        version: Version number

    Returns:
        File-specific instructions
    """
    # Instructions for different file types
    if file_path.endswith("agent.py"):
        return f"""
For coder/agent.py:
- Implement the CodingAgent class as the core of the application
- Include methods for interacting with the OpenRouter API
- Add file manipulation methods for reading, editing, and generating code
- Ensure all methods are well-documented with docstrings
- Version: {version}
"""
    elif file_path.endswith("api.py"):
        return f"""
For coder/api.py:
- Implement the OpenRouterClient class for API communication
- Include methods for chat completions and tool calls
- Add error handling and retry logic
- Version: {version}
"""
    elif file_path.endswith("cli.py"):
        return f"""
For coder/cli.py:
- Implement a Typer-based CLI interface
- Include commands for asking questions, running the agent, and self-rewrite
- Add rich output formatting for a better user experience
- Version: {version}
"""
    elif file_path.endswith("utils/version.py"):
        return f"""
For coder/utils/version.py:
- Implement the VersionManager class for managing version directories
- Include methods for creating version directories and tracking versions
- Version: {version}
"""
    elif file_path.endswith("setup.py"):
        return f"""
For setup.py:
- Create a proper setuptools configuration
- Set version to {version}.0.0
- Include all dependencies
- Configure entry points for CLI commands
"""
    elif file_path.endswith("README.md"):
        return f"""
For README.md:
- Create a comprehensive README for version {version}
- Include installation instructions
- Add usage examples for all commands
- List key features and improvements
"""
    elif file_path.startswith("tests/"):
        return f"""
For {file_path}:
- Create thorough tests for the corresponding module
- Include unit tests and integration tests
- Use pytest fixtures and mocks appropriately
- Version: {version}
"""
    else:
        return f"Generate appropriate content for {file_path} based on the specifications for version {version}."
