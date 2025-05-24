"""
Coder package.
"""

from .api import OpenRouterClient
from .agent import CodingAgent
from .cli import app

__version__ = "0.1.0"
__all__ = ["OpenRouterClient", "CodingAgent", "app"] 