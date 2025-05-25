"""Test configuration and fixtures."""

import os

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def pytest_configure(config):
    """Configure test environment."""
    # Register custom marks
    config.addinivalue_line(
        "markers", "integration: mark a test as an integration test"
    )
    config.addinivalue_line("markers", "slow: mark a test as a slow-running test")
    config.addinivalue_line(
        "markers", "requires_api_key: mark a test as requiring an API key"
    )

def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on environment conditions."""
    # Check for API key presence
    api_key_missing = not os.getenv("OPENROUTER_API_KEY")
    
    # If API key is missing, skip tests that require it
    if api_key_missing:
        skip_marker = pytest.mark.skip(
            reason="OPENROUTER_API_KEY environment variable not set. "
            "Please set it to run integration tests."
        )
        for item in items:
            # Skip tests marked with 'integration' or 'requires_api_key'
            if any(mark in item.keywords for mark in ['integration', 'requires_api_key']):
                item.add_marker(skip_marker)
