import os
import tempfile
from pathlib import Path

import pytest

from coder.agent import CodingAgent


@pytest.fixture
def sample_file_with_syntax_error():
    """Create a temporary file with a Python syntax error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file with syntax error
        error_file = Path(temp_dir) / "syntax_error.py"
        with open(error_file, "w") as f:
            # Missing closing parenthesis
            f.write(
                """
def test_function(a, b):
    result = (a + b * (3 - 1
    return result

print(test_function(1, 2))
"""
            )
        yield str(error_file)


@pytest.fixture
def sample_file_with_unbalanced_quotes():
    """Create a temporary file with unbalanced quotes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file with unbalanced quotes
        error_file = Path(temp_dir) / "quote_error.py"
        with open(error_file, "w") as f:
            # Unterminated string
            f.write(
                """
def greet(name):
    return f"Hello {name}!

print(greet("World"))
"""
            )
        yield str(error_file)


@pytest.fixture
def sample_directory_with_errors():
    """Create a temporary directory with multiple Python files, some with errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file with no errors
        good_file = Path(temp_dir) / "good.py"
        with open(good_file, "w") as f:
            f.write(
                """
def add(a, b):
    return a + b

print(add(1, 2))
"""
            )

        # Create a file with syntax error
        error_file1 = Path(temp_dir) / "error1.py"
        with open(error_file1, "w") as f:
            # Missing closing parenthesis
            f.write(
                """
def subtract(a, b):
    return a - b

print(subtract(5, 3)
"""
            )

        # Create another file with different syntax error
        error_file2 = Path(temp_dir) / "error2.py"
        with open(error_file2, "w") as f:
            # Unclosed brackets
            f.write(
                """
def process_list(items):
    result = [
        item.upper() for item in items
    
    return result

print(process_list(["a", "b", "c"]))
"""
            )

        yield str(temp_dir)


class TestSyntaxFixing:
    """Test the syntax error fixing functionality."""

    def test_fix_missing_parenthesis(self, sample_file_with_syntax_error, monkeypatch):
        """Test fixing a file with a missing parenthesis."""

        # Mock API call to avoid actual API usage in tests
        def mock_chat_completion(*args, **kwargs):
            # Return a fixed version of the code
            return {
                "choices": [
                    {
                        "message": {
                            "content": """
def test_function(a, b):
    result = (a + b * (3 - 1))
    return result

print(test_function(1, 2))
"""
                        }
                    }
                ]
            }

        # Get the directory of the file for repo path
        file_dir = os.path.dirname(sample_file_with_syntax_error)

        # Create agent with mocked API and set repo_path to include the test file
        monkeypatch.setenv("OPENROUTER_API_KEY", "mock_key")
        agent = CodingAgent(repo_path=file_dir, debug=True)
        monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
        monkeypatch.setattr(
            agent.client,
            "get_completion",
            lambda x: x["choices"][0]["message"]["content"],
        )

        # Fix the file
        file_name = os.path.basename(sample_file_with_syntax_error)
        result = agent.fix_syntax_errors(file_name)

        # Check that the fix was successful
        assert "Fixed syntax errors" in result or "✓" in result

        # Verify the file now has correct syntax
        import subprocess

        check_result = subprocess.run(
            ["python", "-m", "py_compile", sample_file_with_syntax_error],
            capture_output=True,
            text=True,
        )
        assert check_result.returncode == 0

    def test_fix_unbalanced_quotes(
        self, sample_file_with_unbalanced_quotes, monkeypatch
    ):
        """Test fixing a file with unbalanced quotes."""

        # Mock API call to avoid actual API usage in tests
        def mock_chat_completion(*args, **kwargs):
            # Return a fixed version of the code
            return {
                "choices": [
                    {
                        "message": {
                            "content": """
def greet(name):
    return f"Hello {name}!"

print(greet("World"))
"""
                        }
                    }
                ]
            }

        # Get the directory of the file for repo path
        file_dir = os.path.dirname(sample_file_with_unbalanced_quotes)

        # Create agent with mocked API and set repo_path to include the test file
        monkeypatch.setenv("OPENROUTER_API_KEY", "mock_key")
        agent = CodingAgent(repo_path=file_dir, debug=True)
        monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
        monkeypatch.setattr(
            agent.client,
            "get_completion",
            lambda x: x["choices"][0]["message"]["content"],
        )

        # Fix the file
        file_name = os.path.basename(sample_file_with_unbalanced_quotes)
        result = agent.fix_syntax_errors(file_name)

        # Check that the fix was successful
        assert "Fixed syntax errors" in result or "✓" in result

        # Verify the file now has correct syntax
        import subprocess

        check_result = subprocess.run(
            ["python", "-m", "py_compile", sample_file_with_unbalanced_quotes],
            capture_output=True,
            text=True,
        )
        assert check_result.returncode == 0

    def test_fix_directory(self, sample_directory_with_errors, monkeypatch):
        """Test fixing all Python files in a directory."""

        # Mock API call to avoid actual API usage in tests
        def mock_chat_completion(*args, **kwargs):
            # Return a fixed version of the code based on the prompt content
            prompt = kwargs.get("messages", [{}])[-1].get("content", "")

            if "error1.py" in prompt:
                fixed_code = """
def subtract(a, b):
    return a - b

print(subtract(5, 3))
"""
            elif "error2.py" in prompt:
                fixed_code = """
def process_list(items):
    result = [
        item.upper() for item in items
    ]
    
    return result

print(process_list(["a", "b", "c"]))
"""
            else:
                fixed_code = "# No fixes needed"

            return {"choices": [{"message": {"content": fixed_code}}]}

        # Create agent with mocked API and set repo_path to include the test directory
        monkeypatch.setenv("OPENROUTER_API_KEY", "mock_key")
        agent = CodingAgent(repo_path=sample_directory_with_errors, debug=True)
        monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
        monkeypatch.setattr(
            agent.client,
            "get_completion",
            lambda x: x["choices"][0]["message"]["content"],
        )

        # Fix all files in the directory
        result = agent.fix_syntax_errors(".")

        # Check that the fix was successful
        assert "Fixed syntax errors" in result or "✓" in result

        # Verify all files now have correct syntax
        import subprocess

        for py_file in Path(sample_directory_with_errors).glob("**/*.py"):
            check_result = subprocess.run(
                ["python", "-m", "py_compile", str(py_file)],
                capture_output=True,
                text=True,
            )
            assert (
                check_result.returncode == 0
            ), f"File {py_file} still has syntax errors"
