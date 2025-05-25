# Coding Agent v1 Specification

## Overview
The Coding Agent is an AI-powered tool designed to assist developers in writing, testing, and maintaining code. It provides intelligent code generation, real-time error detection, and automated testing capabilities.

## Core Features

### 1. Code Generation
- Generate code snippets, functions, and complete modules based on natural language descriptions
- Support for multiple programming languages
- Context-aware code suggestions
- Template-based code generation

### 2. Code Analysis
- Static code analysis
- Syntax error detection and correction using both heuristic techniques and LLM-powered analysis
- Code quality assessment
- Performance optimization suggestions

### 3. Testing Support
- Automated test generation
- Test coverage analysis
- Test case optimization
- Integration test support

### 4. Self-Improvement Capabilities
- Self-rewrite functionality to improve its own codebase
- Version management for tracking improvements
- Automated error fixing and syntax correction
- Test coverage preservation during rewrites
- Iterative improvement through versioning

### 5. Version Management
- Version directory structure (v1, v2, etc.)
- Version-specific specifications
- Backward compatibility maintenance
- Test coverage tracking across versions

## Technical Requirements

### 1. Code Structure
- Modular design for easy extension
- Clear separation of concerns
- Well-documented code
- Type hints and docstrings

### 2. Testing Requirements
- Comprehensive test coverage
- Unit tests for all components
- Integration tests for workflows
- Test-driven development approach

### 3. Error Handling
- Robust error detection
- Automated error correction with LLM-powered syntax fixing
- Syntax validation
- Type checking

### 4. Self-Rewrite Process
- Version directory creation
- Specification preservation
- Test coverage validation
- Error fixing capabilities
- Iterative improvement tracking

## Implementation Guidelines

### 1. Version Management
- Each version in its own directory (v1, v2, etc.)
- Version-specific SPECS.md in version directory
- Root SPECS.md remains at v1
- Version tracking in code

### 2. Testing Strategy
- Maintain or increase test coverage
- Preserve existing test functionality
- Add new tests for new features
- Validate test results
- Automatic syntax error fixing during test validation

### 3. Error Handling
- Syntax error detection and automated fixing using both heuristic and LLM-powered techniques
- Validation before commits
- Error logging and tracking
- Dedicated syntax fixing tool for developers

### 4. Self-Improvement
- Iterative code enhancement
- Automated error fixing
- Test coverage preservation
- Version-specific improvements

## Development Workflow

### 1. Version Creation
- Create new version directory
- Copy and update SPECS.md
- Implement new features
- Add/update tests

### 2. Testing Process
- Run existing tests
- Add new tests
- Validate coverage
- Fix any issues

### 3. Error Handling
- Detect syntax errors
- Apply fixes
- Validate changes
- Update documentation

### 4. Self-Rewrite Process
- Create new version
- Copy existing code
- Apply improvements
- Fix errors
- Validate changes
- Update tests

## Quality Assurance

### 1. Code Quality
- Follow PEP 8 guidelines
- Maintain documentation
- Use type hints
- Write clear docstrings

### 2. Testing Quality
- Comprehensive coverage
- Clear test cases
- Proper assertions
- Error case handling

### 3. Error Handling
- Robust error detection
- Clear error messages
- Proper exception handling
- Logging and tracking

### 4. Version Management
- Clear version tracking
- Proper documentation
- Backward compatibility
- Test coverage preservation

## Future Improvements

### 1. Planned Features
- Enhanced code generation
- Improved error detection
- Better test coverage
- More language support

### 2. Technical Improvements
- Performance optimization
- Better error handling
- Enhanced testing
- Improved documentation

### 3. Self-Improvement
- Better error fixing
- Automated testing
- Version management
- Documentation updates

### 4. User Experience
- Better error messages
- Clearer documentation
- Improved workflows
- Enhanced usability

## Model and Provider Requirements

### Fixed Configuration
- **Provider**: Cerebras (exclusively)
- **Model**: qwen/qwen3-32b (exclusively)
- **Rationale**: To maintain consistency and focus on core functionality improvements rather than provider/model management

## Core Components

### 1. CLI Interface (`coder/cli.py`)
- Rich-based beautiful command-line interface
- Command options:
  - `--ask, -a`: Ask questions about the codebase
  - `--agent, -g`: Request code changes
  - `--repo, -r`: Specify repository path
  - `--debug`: Enable debug output
  - `--interactive`: Enable confirmation prompts
  - `fix-syntax`: Fix Python syntax errors in files or directories
- Progress tracking with live updates
- Error handling with helpful messages

### 2. Agent Core (`coder/agent.py`)
- Manages LLM interactions via OpenRouter API
- Handles file operations:
  - Reading file contents
  - Listing directory contents
  - Creating/editing files
  - Fixing syntax errors in Python files
- Tool-based architecture for extensibility
- Interactive mode support
- Debug logging capabilities
- Smart caching for repeated operations
- Optimized prompt management

### 3. API Client (`coder/api.py`)
- OpenRouter API integration with Cerebras provider
- Robust error handling
- Rate limiting and retry logic
- Response parsing and validation
- Connection pooling for better performance
- Request batching when possible

### 4. Equivalence System (`coder/utils/equivalence.py`)
- Semantic equivalence checking for:
  - JSON content (order-independent)
  - Text content (whitespace-normalized)
  - Code snippets (comment-aware)
  - File structures (path-normalized)
- Configurable similarity thresholds
- Detailed difference reporting
- Edge case handling
- Performance optimizations for large files

## Testing Infrastructure

### 1. Unit Tests
- `tests/test_equivalence.py`: Tests for equivalence checker
  - JSON equivalence scenarios
  - Text comparison cases
  - Code comparison logic
  - File structure validation
  - Edge case handling
  - Performance benchmarks

### 2. Integration Tests
- `tests/test_cli_integration.py`: End-to-end CLI testing
  - Command execution validation
  - File creation/modification tests
  - Response validation
  - Error handling verification
  - Environment setup handling
  - Performance profiling

### 3. Test Configuration
- `tests/conftest.py`: Test environment setup
  - API key validation
  - Environment variable management
  - Skip conditions for missing credentials
  - Performance test configurations

## Environment Management

### Required Environment Variables
- `OPENROUTER_API_KEY`: API key for OpenRouter

### Optional Environment Variables
- `DEBUG`: Debug mode flag
- `INTERACTIVE`: Interactive mode flag

## Installation Requirements

### Core Dependencies
- `typer`: CLI framework
- `rich`: Terminal formatting
- `requests`: HTTP client
- `python-dotenv`: Environment management
- `pydantic`: Data validation
- `cachetools`: Smart caching
- `aiohttp`: Async HTTP operations

### Development Dependencies
- `pytest`: Testing framework
- `pytest-asyncio`: Async test support
- `pytest-benchmark`: Performance testing
- `hypothesis`: Property-based testing
- `memory-profiler`: Memory usage analysis

## Current Test Coverage

### Equivalence Tests
- [x] JSON equivalence checking
- [x] Text similarity comparison
- [x] Code structure analysis
- [x] File system structure validation
- [x] Edge case handling
- [x] Performance benchmarks

### CLI Integration Tests
- [x] Help command functionality
- [x] Question asking capability
- [x] File creation operations
- [x] File modification operations
- [x] Error handling scenarios
- [x] Environment setup validation
- [x] Performance metrics

## Security Considerations

1. API Key Management
   - Keys stored in environment variables
   - No hardcoding in source code
   - Example configurations provided safely

2. File Operations
   - Path traversal prevention
   - Permission checking
   - Safe file handling
   - Atomic operations where possible

3. Error Handling
   - Informative error messages
   - No sensitive data exposure
   - Graceful failure modes
   - Automatic recovery strategies

## Performance Considerations

1. API Usage
   - Efficient prompt construction
   - Response caching with TTL
   - Rate limit awareness
   - Connection pooling
   - Request batching
   - Async operations where beneficial

2. File Operations
   - Minimal file system operations
   - Efficient directory traversal
   - Binary file handling
   - Memory-mapped files for large operations
   - Buffered I/O optimization

3. Testing
   - Fast test execution
   - Isolated test environments
   - Cleanup after tests
   - Performance regression checks

## Version Control

- Git-based version control
- Semantic versioning (1.0.0)
- Feature branch workflow
- Pull request process
- Performance regression tracking

## Documentation

1. README.md
   - Installation instructions
   - Usage examples
   - Environment setup
   - Testing guide
   - Performance tips

2. Code Documentation
   - Docstrings
   - Type hints
   - Comments for complex logic
   - Performance considerations

3. SPECS.md (this file)
   - Detailed specifications
   - Component documentation
   - Test coverage tracking
   - Performance metrics

## Performance Goals

1. Response Time
   - CLI commands: <100ms
   - Simple queries: <1s
   - Complex operations: <5s
   - File operations: <100ms for files under 1MB

2. Resource Usage
   - Memory: <100MB base, <500MB peak
   - CPU: <10% idle, <50% active
   - Disk: <100MB excluding caches
   - Network: Efficient batching

3. Scalability
   - Handle repositories up to 1GB
   - Support concurrent operations
   - Graceful degradation under load
   - Smart resource management

## Self-Rewrite Requirements

### Test Coverage Validation
- **Test Count Preservation**: New versions must maintain or increase the number of test files and test functions
- **Current Baseline**: 12 test files, 143 test functions (minimum)
- **Validation Process**: Automated counting during self-rewrite validation
- **Failure Handling**: Rollback if test count decreases

### Environment Configuration
- **API Key Management**: Copy `.env` file to new version directory for real API testing
- **Provider Consistency**: All versions must use only Cerebras provider with qwen/qwen3-32b model
- **Configuration Validation**: Ensure API access works in new version

### Quality Assurance
- **Incremental Improvements**: Focus on small, safe refactoring rather than major structural changes
- **Backward Compatibility**: Maintain existing functionality while adding improvements
- **Performance Focus**: Prioritize performance optimizations and smart features over adding new providers/models

## LLM Toolkit Features

The Coding Agent now includes a powerful LLM-based toolkit for advanced code operations:

### 1. Code Analysis
- Analyze code complexity, structure, and quality
- Identify potential issues and suggest improvements
- Extract metrics like function count, class count, and imports
- Provide detailed reports for code review

### 2. Code Optimization
- Optimize code for performance, readability, or memory usage
- Improve algorithmic efficiency
- Enhance code structure for better maintainability
- Apply best practices for the target optimization goal

### 3. Documentation Enhancement
- Add or improve docstrings throughout codebase
- Generate comprehensive documentation for functions, classes, and modules
- Follow PEP 257 and Google-style docstring conventions
- Ensure documentation completeness and accuracy

### 4. Test Generation
- Generate comprehensive unit tests for code
- Create test cases for normal scenarios, edge cases, and error conditions
- Support pytest and unittest frameworks
- Automatically place tests in appropriate directories

### 5. Error Handling Enhancement
- Add robust error handling to existing code
- Implement appropriate try/except blocks
- Add useful error messages and logging
- Handle edge cases and unexpected inputs

### 6. Code Explanation
- Generate natural language explanations of code
- Support different detail levels (basic, detailed, advanced)
- Create documentation for complex algorithms
- Assist with code review and onboarding

### 7. Code Refactoring
- Apply common refactoring patterns (extract method, rename, etc.)
- Implement design patterns (factory, singleton, etc.)
- Improve code organization
- Enhance code quality while preserving behavior

### Integration Points
- Self-rewrite process uses LLM toolkit to enhance generated files
- CLI commands expose toolkit functionality directly
- Agent interface provides programmatic access to all toolkit features
- All tools are available via the agent API and CLI 