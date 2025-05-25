"""Tests for the equivalence checker module."""

import json

import pytest

from coder.utils.equivalence import EquivalenceChecker


class TestEquivalenceChecker:
    @pytest.fixture
    def checker(self):
        return EquivalenceChecker()

    def test_json_equivalence(self, checker):
        """Test JSON equivalence checking."""
        json1 = {"a": 1, "b": 2}
        json2 = {"b": 2, "a": 1}  # Same content, different order

        result = checker.check_json_equivalence(json1, json2)
        assert result.is_equivalent
        assert result.similarity_score == 1.0

        # Test with string input
        json1_str = '{"a": 1, "b": 2}'
        result = checker.check_json_equivalence(json1_str, json2)
        assert result.is_equivalent

        # Test with different values
        json3 = {"a": 1, "b": 3}
        result = checker.check_json_equivalence(json1, json3)
        assert not result.is_equivalent
        assert result.similarity_score < 1.0

    def test_text_equivalence(self, checker):
        """Test text equivalence checking."""
        text1 = "Hello  World"
        text2 = "Hello World"  # Extra space normalized

        result = checker.check_text_equivalence(text1, text2)
        assert result.is_equivalent
        assert result.similarity_score == 1.0

        # Test with different text
        text3 = "Hello Universe"
        result = checker.check_text_equivalence(text1, text3)
        assert not result.is_equivalent
        assert result.similarity_score < 1.0

        # Test with custom threshold
        text4 = "Hello World!"
        result = checker.check_text_equivalence(text1, text4, similarity_threshold=0.8)
        assert result.is_equivalent  # Should pass with lower threshold

    def test_code_equivalence(self, checker):
        """Test code equivalence checking."""
        code1 = """
        def hello():
            # This is a comment
            print('Hello')  # Another comment
        """

        code2 = """
        def hello():
            print('Hello')
        """

        result = checker.check_code_equivalence(code1, code2)
        assert result.is_equivalent  # Comments should be ignored

        # Test with different code
        code3 = """
        def hello():
            print('Hi')
        """

        result = checker.check_code_equivalence(code1, code3)
        assert not result.is_equivalent
        assert result.similarity_score < 1.0

    def test_file_structure_equivalence(self, checker):
        """Test file structure equivalence checking."""
        structure1 = {
            "src": {"main.py": "content", "utils/": {"helper.py": "helper content"}}
        }

        structure2 = {
            "src": {
                "main.py": "content",
                "utils": {"helper.py": "helper content"},  # No trailing slash
            }
        }

        result = checker.check_file_structure_equivalence(structure1, structure2)
        assert result.is_equivalent  # Should normalize paths

        # Test with different structure
        structure3 = {
            "src": {
                "main.py": "different content",
                "utils": {"helper.py": "helper content"},
            }
        }

        result = checker.check_file_structure_equivalence(structure1, structure3)
        assert not result.is_equivalent
        assert result.similarity_score < 1.0

    def test_edge_cases(self, checker):
        """Test edge cases and error handling."""
        # Invalid JSON
        result = checker.check_json_equivalence("invalid json", '{"a": 1}')
        assert not result.is_equivalent
        assert result.similarity_score < 0.5  # Just ensure low similarity

        # Empty inputs
        result = checker.check_text_equivalence("", "")
        assert result.is_equivalent

        # None values in structure
        structure1 = {"file": None}
        structure2 = {"file": None}
        result = checker.check_file_structure_equivalence(structure1, structure2)
        assert result.is_equivalent
