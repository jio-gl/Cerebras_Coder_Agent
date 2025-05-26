# Coding Agent v1 Specification

## Overview
The Coding Agent is an AI-powered tool designed to assist developers in writing, testing, and maintaining Python code. It provides intelligent code generation, real-time error detection, and automated testing capabilities specifically optimized for Python and Markdown documentation.

## Core Features

### 1. Python Code Generation
- Generate Python code snippets, functions, and complete modules based on natural language descriptions
- Context-aware code suggestions
- Minimalistic, clean Python code generation
- PEP 8 compliant output

### 2. Python Code Analysis
- Static code analysis for Python
- Python syntax error detection and correction
- Code quality assessment
- Performance optimization suggestions for Python

### 3. Python Testing Support
- Automated pytest test generation
- Test coverage analysis
- Test case optimization
- Integration test support

### 4. Self-Improvement Capabilities
- Self-rewrite functionality to improve its own Python codebase
- Version management for tracking improvements
- Automated Python error fixing and syntax correction
- Test coverage preservation during rewrites
- Iterative improvement through versioning

### 5. Version Management
- Version directory structure (v1, v2, etc.)
- Version-specific specifications
- Backward compatibility maintenance
- Test coverage tracking across versions

### 6. Markdown Documentation
- Generate comprehensive Markdown documentation for Python code
- Function and class documentation
- Parameter and return value descriptions
- Usage examples in Markdown format

## LLM Toolkit for Python

The agent includes a specialized LLM toolkit focused on Python code operations:

### 1. Python Code Analysis
```python
def analyze_code(code: str) -> Dict[str, Any]:
    """Analyze Python code structure and quality."""
    # Returns dict with complexity, functions, classes, issues, etc.
```

### 2. Python Code Optimization
```python
def optimize_code(code: str, optimization_goal: str = "performance") -> str:
    """Optimize Python code for performance, readability, or memory usage."""
    # Returns optimized Python code
```

### 3. Python Docstrings
```python
def generate_docstring(code: str) -> str:
    """Generate minimal but effective Python docstrings."""
    # Returns code with added docstrings
```

### 4. Python Test Generation
```python
def generate_unit_tests(code: str) -> str:
    """Generate pytest unit tests for Python code."""
    # Returns pytest test code
```

### 5. Python Error Handling
```python
def enhance_error_handling(code: str) -> str:
    """Add Python-specific error handling to code."""
    # Returns code with enhanced error handling
```

### 6. Markdown Documentation
```python
def generate_markdown_docs(code: str) -> str:
    """Generate Markdown documentation for Python code."""
    # Returns Markdown documentation
```

### 7. Python Code Explanation
```python
def explain_code(code: str, explanation_level: str = "detailed") -> str:
    """Generate a natural language explanation of Python code."""
    # Returns text explanation
```

### 8. Python Code Refactoring
```python
def refactor_code(code: str, refactoring_goal: str) -> str:
    """Refactor Python code according to specific goal."""
    # Returns refactored code
```

### 9. Python Syntax Fixing
```python
def fix_python_syntax(code: str) -> str:
    """Fix Python syntax errors in code."""
    # Returns corrected Python code
```

## Technical Requirements

### 1. Python Code Structure
- Modular design for easy extension
- Clear separation of concerns
- Minimalistic Python code with limited comments
- Type hints where helpful

### 2. Testing Requirements
- Comprehensive pytest test coverage
- Unit tests for all components
- Integration tests for workflows
- Test-driven development approach

### 3. Error Handling
- Robust error detection in Python code
- Automated error correction with LLM-powered syntax fixing
- Python syntax validation
- Type checking

### 4. Self-Rewrite Process
- Version directory creation
- Specification preservation
- Test coverage validation
- Error fixing capabilities
- Iterative improvement tracking

## Development Workflow

### 1. Version Creation
- Create new version directory
- Copy and update SPECS.md
- Implement new features in Python
- Add/update pytest tests

### 2. Testing Process
- Run existing pytest tests
- Add new tests
- Validate coverage
- Fix any issues

### 3. Error Handling
- Detect Python syntax errors
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

### 1. Python Code Quality
- Follow PEP 8 guidelines
- Maintain minimalistic documentation
- Use type hints when beneficial
- Write clear but concise docstrings

### 2. Testing Quality
- Comprehensive pytest coverage
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

## Model and Provider Requirements

### Fixed Configuration
- **Provider**: Cerebras (exclusively)
- **Model**: qwen/qwen3-32b (exclusively)
- **Rationale**: To maintain consistency and focus on core functionality improvements rather than provider/model management

## Python-Focused Tools

### 1. Python Code Analysis
```python
def analyze_code(code: str) -> Dict[str, Any]:
    """Analyze Python code structure and quality."""
    # Returns dict with complexity, functions, classes, issues, etc.
```

### 2. Python Code Optimization
```python
def optimize_code(code: str, optimization_goal: str = "performance") -> str:
    """Optimize Python code for performance, readability, or memory usage."""
    # Returns optimized Python code
```

### 3. Python Docstrings
```python
def generate_docstring(code: str) -> str:
    """Generate minimal but effective Python docstrings."""
    # Returns code with added docstrings
```

### 4. Python Test Generation
```python
def generate_unit_tests(code: str) -> str:
    """Generate pytest unit tests for Python code."""
    # Returns pytest test code
```

### 5. Python Error Handling
```python
def enhance_error_handling(code: str) -> str:
    """Add Python-specific error handling to code."""
    # Returns code with enhanced error handling
```

### 6. Markdown Documentation
```python
def generate_markdown_docs(code: str) -> str:
    """Generate Markdown documentation for Python code."""
    # Returns Markdown documentation
```

### 7. Python Code Explanation
```python
def explain_code(code: str, explanation_level: str = "detailed") -> str:
    """Generate a natural language explanation of Python code."""
    # Returns text explanation
```

### 8. Python Code Refactoring
```python
def refactor_code(code: str, refactoring_goal: str) -> str:
    """Refactor Python code according to specific goal."""
    # Returns refactored code
```

### 9. Python Syntax Fixing
```python
def fix_python_syntax(code: str) -> str:
    """Fix Python syntax errors in code."""
    # Returns corrected Python code
```

## Environment Setup

### Required Environment Variables
- `OPENROUTER_API_KEY`: API key for OpenRouter

### Core Dependencies
- `typer`: CLI framework
- `rich`: Terminal formatting
- `requests`: HTTP client
- `python-dotenv`: Environment management
- `pytest`: Testing framework

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