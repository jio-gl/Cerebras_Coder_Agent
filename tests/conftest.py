"""Test configuration and fixtures."""

import os

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    """Configure test environment."""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Check for required API key
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip(
            "OPENROUTER_API_KEY environment variable not set. "
            "Please set it to run integration tests."
        )

    # Register custom marks
    config.addinivalue_line(
        "markers", "integration: mark a test as an integration test"
    )
    config.addinivalue_line("markers", "slow: mark a test as a slow-running test")
