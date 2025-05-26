"""Unit tests for the LLMToolkit class."""

import pytest
from unittest.mock import Mock, patch

from coder.utils.llm_tools import LLMToolkit


@pytest.fixture
def mock_api_client():
    """Create a mock API client for testing."""
    mock_client = Mock()
    mock_client.chat_completion.return_value = {
        "choices": [{"message": {"content": "Sample response"}}]
    }
    mock_client.get_completion.return_value = "Sample response"
    return mock_client


@pytest.fixture
def llm_toolkit(mock_api_client):
    """Create a LLMToolkit instance with a mock API client."""
    return LLMToolkit(
        api_client=mock_api_client,
        model="qwen/qwen3-32b",
        provider="Cerebras",
        max_tokens=1000,
        debug=True,
    )


class TestLLMToolkit:
    """Tests for the LLMToolkit class."""

    def test_initialization(self, mock_api_client):
        """Test initialization of the LLMToolkit class."""
        toolkit = LLMToolkit(
            api_client=mock_api_client,
            model="qwen/qwen3-32b",
            provider="Cerebras",
            max_tokens=1000,
            debug=True,
        )

        assert toolkit.api_client == mock_api_client
        assert toolkit.model == "qwen/qwen3-32b"
        assert toolkit.provider == "Cerebras"
        assert toolkit.max_tokens == 1000
        assert toolkit.debug is True

    def test_analyze_code(self, llm_toolkit, mock_api_client):
        """Test the analyze_code method."""
        mock_api_client.get_completion.return_value = """
        {
            "complexity": 5,
            "functions": 2,
            "classes": 1,
            "imports": ["os", "sys"],
            "potential_issues": ["Unused import 'sys'"],
            "suggestions": ["Remove unused imports"]
        }
        """

        code = """
        import os
        import sys
        
        class Example:
            def __init__(self):
                self.value = 0
                
            def increment(self):
                self.value += 1
                return self.value
        """

        result = llm_toolkit.analyze_code(code)

        assert mock_api_client.chat_completion.called
        assert isinstance(result, dict)
        assert "complexity" in result
        assert "functions" in result
        assert "classes" in result
        assert "imports" in result
        assert "potential_issues" in result
        assert "suggestions" in result

    def test_basic_code_analysis(self, llm_toolkit):
        """Test the _basic_code_analysis method for fallback."""
        # Enable debug mode
        llm_toolkit.debug = True

        code = """
        import os
        import sys
        
        class Example:
            def __init__(self):
                self.value = 0
                
            def increment(self):
                self.value += 1
                return self.value
        """

        # Force fallback to basic analysis
        with patch.object(llm_toolkit, "_call_llm", side_effect=Exception("API error")):
            result = llm_toolkit.analyze_code(code)

            assert isinstance(result, dict)
            assert result["functions"] == 2  # __init__ and increment
            assert result["classes"] == 1  # Example
            assert "os" in result["imports"]
            assert "sys" in result["imports"]

    def test_fix_code_semantics(self, llm_toolkit, mock_api_client):
        """Test the fix_code_semantics method."""
        mock_api_client.get_completion.return_value = """
        ```python
        def calculate_average(numbers):
            if not numbers:
                return 0
            return sum(numbers) / len(numbers)
        ```
        """

        code = """
        def calculate_average(numbers):
            return sum(numbers) / len(numbers)
        """

        issue = "Function crashes on empty list"

        result = llm_toolkit.fix_code_semantics(code, issue)

        assert mock_api_client.chat_completion.called
        assert "if not numbers:" in result
        assert "return 0" in result

    def test_optimize_code(self, llm_toolkit, mock_api_client):
        """Test the optimize_code method."""
        mock_api_client.get_completion.return_value = """
        ```python
        def fibonacci(n):
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a
        ```
        """

        code = """
        def fibonacci(n):
            if n <= 0:
                return 0
            elif n == 1:
                return 1
            else:
                return fibonacci(n-1) + fibonacci(n-2)
        """

        result = llm_toolkit.optimize_code(code, "performance")

        assert mock_api_client.chat_completion.called
        assert "a, b = 0, 1" in result
        assert "for" in result
        assert "return a" in result

    def test_generate_docstring(self, llm_toolkit, mock_api_client):
        """Test the generate_docstring method."""
        mock_api_client.get_completion.return_value = """
        ```python
        def calculate_average(numbers):
            \"\"\"Calculate the average of a list of numbers.
            
            Args:
                numbers: A list of numeric values.
                
            Returns:
                The average of the input values, or 0 for empty lists.
            \"\"\"
            if not numbers:
                return 0
            return sum(numbers) / len(numbers)
        ```
        """

        code = """
        def calculate_average(numbers):
            if not numbers:
                return 0
            return sum(numbers) / len(numbers)
        """

        result = llm_toolkit.generate_docstring(code)

        assert mock_api_client.chat_completion.called
        assert "Calculate the average" in result
        assert "Args:" in result
        assert "Returns:" in result

    def test_generate_unit_tests(self, llm_toolkit, mock_api_client):
        """Test the generate_unit_tests method."""
        mock_api_client.get_completion.return_value = """
        ```python
        import pytest
        
        def test_calculate_average_normal():
            assert calculate_average([1, 2, 3]) == 2
            
        def test_calculate_average_empty():
            assert calculate_average([]) == 0
        ```
        """

        code = """
        def calculate_average(numbers):
            if not numbers:
                return 0
            return sum(numbers) / len(numbers)
        """

        result = llm_toolkit.generate_unit_tests(code)

        assert mock_api_client.chat_completion.called
        assert "import pytest" in result
        assert "test_calculate_average_normal" in result
        assert "test_calculate_average_empty" in result

    def test_enhance_error_handling(self, llm_toolkit, mock_api_client):
        """Test the enhance_error_handling method."""
        mock_api_client.get_completion.return_value = """
        ```python
        def calculate_average(numbers):
            try:
                if not numbers:
                    return 0
                return sum(numbers) / len(numbers)
            except TypeError:
                raise TypeError("Input must be a list of numbers")
            except ZeroDivisionError:
                return 0
        ```
        """

        code = """
        def calculate_average(numbers):
            if not numbers:
                return 0
            return sum(numbers) / len(numbers)
        """

        result = llm_toolkit.enhance_error_handling(code)

        assert mock_api_client.chat_completion.called
        assert "try:" in result
        assert "except TypeError:" in result

    def test_extract_code_requirements(self, llm_toolkit, mock_api_client):
        """Test the extract_code_requirements method."""
        mock_api_client.get_completion.return_value = """
        {
            "core_functionality": ["Calculate average of numbers", "Handle empty lists"],
            "interfaces": ["Function that takes a list and returns a float"],
            "dependencies": ["None required"],
            "constraints": ["Must handle empty lists", "Must handle non-numeric values"],
            "test_scenarios": ["Test with normal list", "Test with empty list", "Test with invalid inputs"]
        }
        """

        description = "Create a function to calculate the average of a list of numbers. It should handle empty lists by returning 0."

        result = llm_toolkit.extract_code_requirements(description)

        assert mock_api_client.chat_completion.called
        assert isinstance(result, dict)
        assert "core_functionality" in result
        assert "interfaces" in result
        assert "dependencies" in result
        assert "constraints" in result
        assert "test_scenarios" in result

    def test_explain_code(self, llm_toolkit, mock_api_client):
        """Test the explain_code method."""
        mock_api_client.get_completion.return_value = "This function calculates the average of a list of numbers. It handles empty lists by returning 0."

        code = """
        def calculate_average(numbers):
            if not numbers:
                return 0
            return sum(numbers) / len(numbers)
        """

        result = llm_toolkit.explain_code(code)

        assert mock_api_client.chat_completion.called
        assert "calculates the average" in result
        assert "handles empty lists" in result

    def test_refactor_code(self, llm_toolkit, mock_api_client):
        """Test the refactor_code method."""
        mock_api_client.get_completion.return_value = """
        ```python
        def is_empty(numbers):
            return not numbers
            
        def safe_average(numbers):
            if is_empty(numbers):
                return 0
            return sum(numbers) / len(numbers)
        ```
        """

        code = """
        def calculate_average(numbers):
            if not numbers:
                return 0
            return sum(numbers) / len(numbers)
        """

        result = llm_toolkit.refactor_code(code, "extract method")

        assert mock_api_client.chat_completion.called
        assert "def is_empty" in result
        assert "def safe_average" in result
