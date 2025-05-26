"""Unit tests for the CodingAgent's integration with LLM toolkit."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from coder.agent import CodingAgent


@pytest.fixture
def mock_llm_toolkit():
    """Create a mock LLMToolkit."""
    mock_toolkit = Mock()
    mock_toolkit.analyze_code.return_value = {
        "complexity": 5,
        "functions": 2,
        "classes": 1,
        "imports": ["os", "sys"],
        "potential_issues": ["Unused import 'sys'"],
        "suggestions": ["Remove unused imports"],
    }
    mock_toolkit.optimize_code.return_value = "optimized code"
    mock_toolkit.generate_docstring.return_value = "code with docstrings"
    mock_toolkit.generate_unit_tests.return_value = "test code"
    mock_toolkit.enhance_error_handling.return_value = "code with error handling"
    mock_toolkit.explain_code.return_value = "code explanation"
    mock_toolkit.refactor_code.return_value = "refactored code"
    return mock_toolkit


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    mock_client = Mock()
    mock_client.chat_completion.return_value = {"choices": [{"message": {"content": "test response"}}]}
    mock_client.get_completion.return_value = "test response"
    return mock_client


@pytest.fixture
def sample_python_file(tmp_path, monkeypatch):
    """Create a temporary Python file for testing within the repository directory."""
    # Create a temp directory in the repository
    test_dir = Path("tests/temp")
    test_dir.mkdir(exist_ok=True)

    # Create a temporary file in the test directory
    temp_file = test_dir / "test_sample.py"
    with open(temp_file, "w") as f:
        f.write(
            """
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
"""
        )

    # Yield the file path
    yield str(temp_file)

    # Clean up after test
    try:
        if temp_file.exists():
            temp_file.unlink()
    except:
        pass


class TestAgentLLMTools:
    """Tests for the CodingAgent's integration with LLM toolkit."""

    @pytest.mark.requires_api_key
    def test_analyze_file(self, sample_python_file, mock_llm_toolkit, mock_api_client):
        """Test the analyze_file method."""
        # Create agent with mocked toolkit and API client
        with patch('coder.agent.OpenRouterClient', return_value=mock_api_client):
            agent = CodingAgent(debug=True, api_key="test-key")
            agent.llm_toolkit = mock_llm_toolkit

            # Call the analyze_file method
            result = agent.analyze_file(sample_python_file)

            # Verify the toolkit was called correctly
            assert mock_llm_toolkit.analyze_code.called

            # Verify the result has the expected fields
            assert "complexity" in result
            assert "functions" in result
            assert "classes" in result
            assert "imports" in result
            assert "potential_issues" in result
            assert "suggestions" in result
            assert "file_path" in result
            assert result["file_path"] == sample_python_file

    @pytest.mark.requires_api_key
    def test_optimize_file(self, sample_python_file, mock_llm_toolkit, mock_api_client):
        """Test the optimize_file method."""
        # Create agent with mocked toolkit and API client
        with patch('coder.agent.OpenRouterClient', return_value=mock_api_client):
            agent = CodingAgent(debug=True, api_key="test-key")
            agent.llm_toolkit = mock_llm_toolkit

            # Create a mock for _edit_file to avoid actually modifying the file
            with patch.object(agent, "_edit_file", return_value=True):
                # Call the optimize_file method
                result = agent.optimize_file(sample_python_file, "performance")

                # Verify the toolkit was called correctly
                mock_llm_toolkit.optimize_code.assert_called_once()

                # Verify _edit_file was called with the optimized code
                agent._edit_file.assert_called_once_with(
                    sample_python_file, "optimized code"
                )

                # Verify the result
                assert "✨ Optimized" in result
                assert "performance" in result

    @pytest.mark.requires_api_key
    def test_add_docstrings(self, sample_python_file, mock_llm_toolkit, mock_api_client):
        """Test the add_docstrings method."""
        # Create agent with mocked toolkit and API client
        with patch('coder.agent.OpenRouterClient', return_value=mock_api_client):
            agent = CodingAgent(debug=True, api_key="test-key")
            agent.llm_toolkit = mock_llm_toolkit

            # Create a mock for _edit_file to avoid actually modifying the file
            with patch.object(agent, "_edit_file", return_value=True):
                # Call the add_docstrings method
                result = agent.add_docstrings(sample_python_file)

                # Verify the toolkit was called correctly
                mock_llm_toolkit.generate_docstring.assert_called_once()

                # Verify _edit_file was called with the documented code
                agent._edit_file.assert_called_once_with(
                    sample_python_file, "code with docstrings"
                )

                # Verify the result
                assert "✨ Added/improved docstrings" in result

    @pytest.mark.requires_api_key
    def test_generate_tests(self, sample_python_file, mock_llm_toolkit, mock_api_client):
        """Test the generate_tests method."""
        # Create agent with mocked toolkit and API client
        with patch('coder.agent.OpenRouterClient', return_value=mock_api_client):
            agent = CodingAgent(debug=True, api_key="test-key")
            agent.llm_toolkit = mock_llm_toolkit

            # Create a mock for _edit_file to avoid actually creating a test file
            with patch.object(agent, "_edit_file", return_value=True):
                # Call the generate_tests method with explicit output path
                output_path = sample_python_file.replace(".py", "_test.py")
                result = agent.generate_tests(sample_python_file, output_path)

                # Verify the toolkit was called correctly
                mock_llm_toolkit.generate_unit_tests.assert_called_once()

                # Verify _edit_file was called with the test code
                agent._edit_file.assert_called_once_with(output_path, "test code")

                # Verify the result
                assert "✨ Generated tests" in result

    @pytest.mark.requires_api_key
    def test_enhance_error_handling(self, sample_python_file, mock_llm_toolkit, mock_api_client):
        """Test the enhance_error_handling method."""
        # Create agent with mocked toolkit and API client
        with patch('coder.agent.OpenRouterClient', return_value=mock_api_client):
            agent = CodingAgent(debug=True, api_key="test-key")
            agent.llm_toolkit = mock_llm_toolkit

            # Create a mock for _edit_file to avoid actually modifying the file
            with patch.object(agent, "_edit_file", return_value=True):
                # Call the enhance_error_handling method
                result = agent.enhance_error_handling(sample_python_file)

                # Verify the toolkit was called correctly
                mock_llm_toolkit.enhance_error_handling.assert_called_once()

                # Verify _edit_file was called with the enhanced code
                agent._edit_file.assert_called_once_with(
                    sample_python_file, "code with error handling"
                )

                # Verify the result
                assert "✨ Enhanced error handling" in result

    @pytest.mark.requires_api_key
    def test_explain_code(self, sample_python_file, mock_llm_toolkit, mock_api_client):
        """Test the explain_code method."""
        # Create agent with mocked toolkit and API client
        with patch('coder.agent.OpenRouterClient', return_value=mock_api_client):
            agent = CodingAgent(debug=True, api_key="test-key")
            agent.llm_toolkit = mock_llm_toolkit

            # Call the explain_code method
            result = agent.explain_code(sample_python_file, "detailed")

            # Verify the toolkit was called correctly
            mock_llm_toolkit.explain_code.assert_called_once()

            # Verify the result
            assert result == "code explanation"

    @pytest.mark.requires_api_key
    def test_refactor_code(self, sample_python_file, mock_llm_toolkit, mock_api_client):
        """Test the refactor_code method."""
        # Create agent with mocked toolkit and API client
        with patch('coder.agent.OpenRouterClient', return_value=mock_api_client):
            agent = CodingAgent(debug=True, api_key="test-key")
            agent.llm_toolkit = mock_llm_toolkit

            # Create a mock for _edit_file to avoid actually modifying the file
            with patch.object(agent, "_edit_file", return_value=True):
                # Call the refactor_code method
                result = agent.refactor_code(sample_python_file, "extract method")

                # Verify the toolkit was called correctly
                mock_llm_toolkit.refactor_code.assert_called_once()

                # Verify _edit_file was called with the refactored code
                agent._edit_file.assert_called_once_with(
                    sample_python_file, "refactored code"
                )

                # Verify the result
                assert "✨ Refactored" in result
                assert "extract method" in result
