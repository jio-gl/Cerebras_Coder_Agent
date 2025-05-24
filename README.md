# Coder Agent

A minimalistic coding agent using OpenRouter API and Cerebras provider for LLM API calls. The agent's primary feature is its ability to self-rewrite and improve its own codebase, making it a truly evolving system. While inspired by Cursor's coding agent, it's designed with self-improvement as its core capability.

## Key Feature: Self-Rewrite

The agent's most powerful feature is its ability to rewrite and improve itself. Through a sophisticated process, it:
1. Analyzes its current implementation
2. Generates improved specifications
3. Creates a new version with enhancements
4. Validates the new version through comprehensive testing
5. Ensures backward compatibility

This self-improvement cycle allows the agent to evolve and adapt to new requirements while maintaining stability.

## Additional Features

- Ask questions about your codebase
- Make changes through natural language prompts
- Language-agnostic code analysis and modification
- Automatic command execution and iteration
- Interactive mode for manual confirmation of changes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/coder-agent.git
cd coder-agent
```

2. Install the package:
```bash
pip install -e .
```

3. Create a `.env` file with your OpenRouter API key:
```bash
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

### Self-Rewrite (Primary Feature)

```bash
coder --self-rewrite
```

This command initiates the self-improvement process, creating a new version with enhanced capabilities.

### Other Operations

Ask Questions:
```bash
coder --ask "What does this function do?"
```

Make Changes:
```bash
coder --agent "Add error handling to this function"
```

### Additional Options

- `--repo` or `-r`: Specify the repository path (default: current directory)
- `--model`: Choose the model to use (default: qwen/qwen-3-32b)
- `--no-think`: Disable thinking mode for Qwen-3-32b
- `--debug`: Enable debug output
- `--interactive`: Ask for confirmation before making changes

## Development

### Running Tests

```bash
pytest
```

### Code Style

The project uses:
- Black for code formatting
- Flake8 for linting
- MyPy for type checking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 