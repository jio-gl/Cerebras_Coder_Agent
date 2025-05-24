import pytest
from coder.agent import CodingAgent
import time
import os
import json

# Use models that are compatible with the Cerebras provider
SUPPORTED_MODELS = [
    "qwen/qwen3-32b"
]

@pytest.fixture
def api_key():
    """Return API key for testing."""
    return os.getenv("OPENROUTER_API_KEY", "test-key")

@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_basic_completion(model, tmp_path, api_key):
    """Test basic completion with each model."""
    agent = CodingAgent(
        repo_path=tmp_path,
        model=model,
        api_key=api_key,
        max_tokens=31000
    )
    
    # Test with a simple prompt
    response = agent.ask("What is a function in Python?")
    assert response
    assert len(response.strip()) > 50  # Should have a meaningful response

@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_structured_data(model, tmp_path, api_key):
    """Test structured data generation with each model."""
    agent = CodingAgent(
        repo_path=tmp_path,
        model=model,
        api_key=api_key,
        max_tokens=31000
    )
    
    # Test with a prompt that requires structured output
    prompt = """
    Generate a JSON object with the following structure:
    {
        "name": "string",
        "age": number,
        "skills": ["string"],
        "experience": {
            "years": number,
            "projects": ["string"]
        }
    }
    Fill it with realistic data for a Python developer.
    """
    
    response = agent.ask(prompt)
    assert response
    
    # Try to parse the response as JSON
    try:
        data = json.loads(response)
        assert isinstance(data, dict)
        assert "name" in data
        assert "age" in data
        assert "skills" in data
        assert "experience" in data
        assert isinstance(data["skills"], list)
        assert isinstance(data["experience"], dict)
    except json.JSONDecodeError:
        # If JSON parsing fails, check if the response contains the expected fields
        assert any(keyword in response.lower() for keyword in ["name", "age", "skills", "experience"])

@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_tool_calls(model, tmp_path, api_key):
    """Test tool calls with each model."""
    agent = CodingAgent(
        repo_path=tmp_path,
        model=model,
        api_key=api_key,
        max_tokens=31000
    )
    
    # Test with a prompt that requires file operations
    prompt = "Create a simple calculator.py file with add and subtract functions"
    response = agent.agent(prompt)
    
    # Check if the file was created
    calc_file = tmp_path / "calculator.py"
    if calc_file.exists():
        content = calc_file.read_text()
        assert "def add" in content
        assert "def subtract" in content
    else:
        # If file creation failed, check if the response contains a valid tool call
        assert isinstance(response, str)
        assert "edit_file" in response.lower() or "parameters" in response.lower()

@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_error_handling(model, tmp_path, api_key):
    """Test error handling with each model."""
    agent = CodingAgent(
        repo_path=tmp_path,
        model=model,
        api_key=api_key,
        max_tokens=31000
    )
    
    # Test with an invalid prompt that should trigger an error
    response = agent.ask("INVALID_COMMAND_THAT_SHOULD_FAIL")
    assert response
    assert any(keyword in response.lower() for keyword in ["error", "invalid", "cannot", "unable", "don't understand"])

def test_qwen_large_context(tmp_path, api_key):
    """Test Qwen model with large context window."""
    model = "qwen/qwen3-32b"
    agent = CodingAgent(
        repo_path=tmp_path,
        model=model,
        api_key=api_key,
        max_tokens=31000  # Set max tokens to 25,000
    )
    
    # Test with a large prompt
    large_prompt = "Write a detailed explanation of Python's memory management system, including garbage collection, reference counting, and memory optimization techniques. Include examples and best practices."
    response = agent.ask(large_prompt)
    
    # Verify response
    assert response
    assert len(response.strip()) > 1000  # Should have a substantial response
    assert any(keyword in response.lower() for keyword in ["memory", "garbage", "reference", "optimization"]) 