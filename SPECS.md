# Coding Agent v1 Specification

## Overview

The Coding Agent is a powerful CLI tool that leverages OpenRouter's API to assist in code development tasks. It features semantic code understanding, file manipulation capabilities, and a robust testing infrastructure with equivalence checking.

## Core Components

### 1. CLI Interface (`coder/cli.py`)
- Rich-based beautiful command-line interface
- Command options:
  - `--ask, -a`: Ask questions about the codebase
  - `--agent, -g`: Request code changes
  - `--repo, -r`: Specify repository path
  - `--model`: Choose LLM model (default: qwen/qwen-3-32b)
  - `--no-think`: Disable thinking mode
  - `--debug`: Enable debug output
  - `--interactive`: Enable confirmation prompts
- Progress tracking with live updates
- Error handling with helpful messages

### 2. Agent Core (`coder/agent.py`)
- Manages LLM interactions via OpenRouter API
- Handles file operations:
  - Reading file contents
  - Listing directory contents
  - Creating/editing files
- Tool-based architecture for extensibility
- Interactive mode support
- Debug logging capabilities

### 3. API Client (`coder/api.py`)
- OpenRouter API integration
- Model selection support
- Robust error handling
- Rate limiting and retry logic
- Response parsing and validation

### 4. Equivalence System (`coder/utils/equivalence.py`)
- Semantic equivalence checking for:
  - JSON content (order-independent)
  - Text content (whitespace-normalized)
  - Code snippets (comment-aware)
  - File structures (path-normalized)
- Configurable similarity thresholds
- Detailed difference reporting
- Edge case handling

## Testing Infrastructure

### 1. Unit Tests
- `tests/test_equivalence.py`: Tests for equivalence checker
  - JSON equivalence scenarios
  - Text comparison cases
  - Code comparison logic
  - File structure validation
  - Edge case handling

### 2. Integration Tests
- `tests/test_cli_integration.py`: End-to-end CLI testing
  - Command execution validation
  - File creation/modification tests
  - Response validation
  - Error handling verification
  - Environment setup handling

### 3. Test Configuration
- `tests/conftest.py`: Test environment setup
  - API key validation
  - Environment variable management
  - Skip conditions for missing credentials

## Environment Management

### Required Environment Variables
- `OPENROUTER_API_KEY`: API key for OpenRouter

### Optional Environment Variables
- `MODEL`: Default LLM model
- `DEBUG`: Debug mode flag
- `INTERACTIVE`: Interactive mode flag

## Installation Requirements

### Core Dependencies
- `typer`: CLI framework
- `rich`: Terminal formatting
- `requests`: HTTP client
- `python-dotenv`: Environment management
- `pydantic`: Data validation

### Development Dependencies
- `pytest`: Testing framework
- `pytest-asyncio`: Async test support
- `pytest-benchmark`: Performance testing
- `hypothesis`: Property-based testing

## Current Test Coverage

### Equivalence Tests
- [x] JSON equivalence checking
- [x] Text similarity comparison
- [x] Code structure analysis
- [x] File system structure validation
- [x] Edge case handling

### CLI Integration Tests
- [x] Help command functionality
- [x] Question asking capability
- [x] File creation operations
- [x] File modification operations
- [x] Error handling scenarios
- [x] Environment setup validation

## Security Considerations

1. API Key Management
   - Keys stored in environment variables
   - No hardcoding in source code
   - Example configurations provided safely

2. File Operations
   - Path traversal prevention
   - Permission checking
   - Safe file handling

3. Error Handling
   - Informative error messages
   - No sensitive data exposure
   - Graceful failure modes

## Performance Considerations

1. API Usage
   - Efficient prompt construction
   - Response caching when appropriate
   - Rate limit awareness

2. File Operations
   - Minimal file system operations
   - Efficient directory traversal
   - Binary file handling

3. Testing
   - Fast test execution
   - Isolated test environments
   - Cleanup after tests

## Future Improvements

1. Enhanced Equivalence Checking
   - AST-based code comparison
   - Semantic code analysis
   - Multi-language support

2. CLI Enhancements
   - Progress bar improvements
   - More interactive features
   - Configuration file support

3. Testing Infrastructure
   - Expanded test coverage
   - Performance benchmarks
   - Property-based testing

4. Documentation
   - API documentation
   - Usage examples
   - Contributing guidelines

## Version Control

- Git-based version control
- Semantic versioning (1.0.0)
- Feature branch workflow
- Pull request process

## Documentation

1. README.md
   - Installation instructions
   - Usage examples
   - Environment setup
   - Testing guide

2. Code Documentation
   - Docstrings
   - Type hints
   - Comments for complex logic

3. SPECS.md (this file)
   - Detailed specifications
   - Component documentation
   - Test coverage tracking 