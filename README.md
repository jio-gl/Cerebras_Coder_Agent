# Coder Agent

A powerful coding agent that uses OpenRouter API to help with code development.

## Installation

```bash
pip install -e .
```

## Environment Setup

Before using the agent or running tests, you need to set up your environment:

1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Create a `.env` file in your project root with:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
   
Optional environment variables:
- `MODEL`: Default model to use (default: qwen/qwen-3-32b)
- `DEBUG`: Enable debug mode (true/false)
- `INTERACTIVE`: Enable interactive mode (true/false)

## Usage

```bash
# Ask a question about your code
coder --ask "How does this function work?"

# Make changes to your code
coder --agent "Add error handling to main.py"

# Get help
coder --help
```

## Running Tests

To run the tests, make sure you have set up your environment variables and then run:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_equivalence.py
pytest tests/test_cli_integration.py

# Run with verbose output
pytest -v
```

Note: Integration tests require a valid OpenRouter API key to be set in your environment.

## Features

- Semantic code understanding and modification
- File creation and editing
- Interactive mode for controlled changes
- Comprehensive test suite with equivalence checking
- Beautiful CLI interface with progress tracking

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Run the tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details 