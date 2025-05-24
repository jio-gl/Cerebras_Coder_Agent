# Coding Agent v1 Specification

## Overview

The Coding Agent is a minimalistic, language-agnostic, self-improving coding assistant that leverages the OpenRouter API (with Cerebras provider) for LLM-powered code generation, editing, and repository tooling. It is designed for robust, automated, and test-driven codebase management.

## Goals
- Minimal, clean, and robust design
- Language-agnostic file and directory tooling
- Self-rewrite and self-healing capabilities
- Comprehensive test coverage
- Backward compatibility between versions

## Core Features
1. **OpenRouter API Integration**: Uses OpenRouter API with Cerebras provider (qwen/qwen-3-32b) for all LLM calls.
2. **File Operations**: Core file reading, writing, and directory listing capabilities.
3. **Self-Rewrite**: Ability to rewrite its own codebase with improvements.
4. **Language Agnostic**: All file and directory operations work with any text-based files.
5. **Error Handling**: Comprehensive error handling for all file operations.
6. **Testing**: Unit tests for all core functionality.
7. **Environment Management**: API key handling through environment variables.

## API & Tooling
- **OpenRouterClient**: Handles API communication and response parsing
- **CodingAgent**: Main agent class with the following methods:
  - `ask(question: str) -> str`: Ask questions about the codebase
  - `agent(prompt: str) -> str`: Execute agent operations
  - `self_rewrite() -> str`: Perform self-rewrite with improvements
  - `analyze_codebase() -> Dict[str, Any]`: Get codebase metrics
- **Core Tools**:
  - `read_file(target_file: str) -> str`: Read file contents
  - `list_directory(relative_workspace_path: str) -> List[str]`: List directory contents
  - `edit_file(target_file: str, content: str) -> bool`: Create or edit files

## Self-Rewrite Process
1. **Version Detection**: Reads current version from SPECS.md
2. **Specification Generation**: Creates improved specifications for next version
3. **Code Generation**: Generates new code files based on improved specs
4. **Validation**: 
   - Installs dependencies
   - Runs linting checks (flake8)
   - Performs type checking (mypy)
   - Executes test suite with coverage
   - Iteratively fixes issues (max 5 iterations)

## Environment Variables
- `OPENROUTER_API_KEY`: Required for API access
- `DEFAULT_MODEL`: Optional model override (default: qwen/qwen-3-32b)

## Project Structure
```
.
├── coder/
│   ├── __init__.py
│   ├── agent.py      # Main agent implementation
│   ├── api.py        # OpenRouter API client
│   └── cli.py        # CLI interface
├── tests/
│   ├── test_agent.py
│   ├── test_agent_tools.py
│   └── test_integration_agent.py
├── SPECS.md          # This specification file
├── README.md         # Usage documentation
├── requirements.txt  # Dependencies
└── setup.py         # Package configuration
```

## Implementation Details
- Uses qwen/qwen-3-32b as default model
- Type-annotated codebase
- Modular architecture with clear separation of concerns
- Tool-based approach for file operations
- Comprehensive error handling and validation
- Automated test suite
- Self-improvement capabilities through rewrite process

---

**This SPECS.md represents the current state of version 1. Future versions will build upon these features while maintaining backward compatibility.** 