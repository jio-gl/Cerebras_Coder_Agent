# Coding Agent v1 Specification

## Overview

The Coding Agent is a minimalistic, language-agnostic, self-improving coding assistant that leverages the OpenRouter API (with Cerebras provider) for LLM-powered code generation, editing, and repository tooling. It is inspired by the Cursor Coding Agent and is designed for robust, automated, and test-driven codebase management.

## Goals
- Minimal, clean, and robust design
- Language-agnostic file and directory tooling
- Self-rewrite and self-healing capabilities
- Comprehensive, automated test coverage
- Modern CLI interface
- Backward compatibility with previous agent versions

## Core Features
1. **OpenRouter API Integration**: Uses OpenRouter API with Cerebras provider for all LLM calls.
2. **Auto-Accept Changes**: All code changes are auto-applied unless explicitly disabled.
3. **Self-Rewrite**: The agent can rewrite its own codebase based on a specification (SPECS.md).
4. **Language Agnostic**: All file and directory operations work for any programming language or text file.
5. **Tooling**: Core tools include reading files, editing files, listing directories, and executing tool calls.
6. **Robust Error Handling**: All operations include detailed error messages and validation.
7. **Comprehensive Testing**: All features are covered by unit and integration tests, including edge cases.
8. **CLI Interface**: Typer-based CLI for all agent operations, including self-rewrite.
9. **Environment Management**: Uses `.env` for API keys and configuration.
10. **Documentation**: All code and tests are well-documented and type-annotated.

## API & Tooling
- **OpenRouterClient**: Handles all API requests, error handling, and response parsing.
- **CodingAgent**: Main agent class, manages repo path, tools, and LLM interactions.
- **Tools**:
  - `read_file(target_file: str) -> str`
  - `edit_file(target_file: str, content: str) -> bool`
  - `list_directory(relative_workspace_path: str) -> List[str]`
  - `_execute_tool_call(tool_call: dict) -> Any`
- **Self-Rewrite**: Reads SPECS.md, generates new code/tests in a new folder, and ensures all requirements are met.

## CLI
- `--ask`: Ask questions about the repo
- `--agent`: Prompt the agent to make changes
- `--self-rewrite`: Rewrite the agent using the current SPECS.md
- `--repo`: Path to the target repository
- `--model`: Model selection (default: qwen/qwen-3-32b)
- `--debug`: Enable debug output
- `--interactive`: Enable/disable interactive mode

## Environment
- `.env` file with `OPENROUTER_API_KEY`, `DEFAULT_MODEL`, and `DEBUG`

## Test Suite (Current)
### Unit Tests (test_agent_tools.py)
- File operations:
  - Read file (basic, JSON, non-existent)
  - Edit file (create, update, nested, invalid path)
- Directory operations:
  - List directory (root, subdir, non-existent)
- Tool call execution:
  - Execute read_file, list_directory, edit_file, invalid tool
- Tool interaction:
  - Read then edit
  - List then create
  - Complex workflow (multi-step file/directory operations)

### Agent Tests (test_agent.py)
- Agent initialization and properties
- File reading (root, subdir, non-existent)
- Directory listing (root, subdir, non-existent)
- File editing (update, create new, create in subdir, invalid path)
- Tool call execution (read_file, list_directory, edit_file, invalid tool)
- LLM tool call integration (ask/agent with tool calls, multiple tool calls)

### Integration Tests (test_integration_agent.py)
- Node.js repo bug fix (edit file, check output)
- Run npm install and npm start, check output
- Generate and run shell command, check output

## Requirements for Next Rewrite
- All current features and tests must be preserved or improved.
- Add new tests for any new features or edge cases discovered during rewrite.
- All code must be type-annotated and documented.
- CLI must remain minimal and user-friendly.
- Self-rewrite must be fully automated and verifiable by running the new test suite.
- The new version must be placed in a new folder (e.g., `agent_v3/`) and must not overwrite the current version until all tests pass.
- The new SPECS.md must be used as the single source of truth for the rewrite.
- All dependencies must be listed in requirements.txt and setup.py.
- README.md must be updated to reflect all new features and usage.

## Deliverables
- New folder with complete, tested, and documented agent codebase
- Updated SPECS.md
- Updated README.md
- requirements.txt and setup.py
- All tests passing

## Test List (for rewrite)
- [ ] All tests from test_agent_tools.py
- [ ] All tests from test_agent.py
- [ ] All tests from test_integration_agent.py
- [ ] New tests for any new features or edge cases
- [ ] 100% code coverage for all core modules

---

**This SPECS.md will be used as the basis for the next self-rewrite. All requirements and tests listed here must be met or exceeded in the new version.** 