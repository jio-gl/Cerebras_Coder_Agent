# Coder Agent

A minimalistic coding agent using OpenRouter API and Cerebras provider for LLM API calls. This agent is inspired by Cursor's coding agent but designed to be simpler and more focused.

## Features

- Ask questions about your codebase
- Make changes to your codebase through natural language prompts
- Self-rewrite capability to improve itself
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

The agent can be used in several ways:

### Ask Questions

```bash
coder --ask "What does this function do?"
```

### Make Changes

```bash
coder --agent "Add error handling to this function"
```

### Self-Rewrite

```bash
coder --self-rewrite
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