"""
Coder package.
"""

from .agent import CodingAgent
from .api import OpenRouterClient
from .cli import app

__version__ = "0.1.0"
__all__ = ["OpenRouterClient", "CodingAgent", "app"]
