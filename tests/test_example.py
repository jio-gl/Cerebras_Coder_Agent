"""
Tests for the example module.
"""

from coder.example import hello


def test_hello():
    """Test the hello function."""
    assert hello("World") == "Hello, World!"
    assert hello("Python") == "Hello, Python!"
