import os
from typing import Dict, List, Optional, Union
import requests
from dotenv import load_dotenv

load_dotenv()

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "qwen/qwen3-32b",
        temperature: float = 0.7,
        max_tokens: int = 16000,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None
    ) -> Dict:
        """
        Make a chat completion request to OpenRouter API.
        
        Args:
            messages: List of message dictionaries with role and content
            model: Model to use (default: qwen/qwen3-32b)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools for function calling
            tool_choice: Optional tool choice strategy
            response_format: Optional response format specification
            
        Returns:
            Dict containing the API response
        """
        data = {
            "model": model,
            "provider": {
                "only": ["Cerebras"]
            },
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice
        if response_format:
            data["response_format"] = response_format

        # Debug logging
        print(f"Request payload: {data}")
        response = requests.post(self.base_url, headers=self.headers, json=data)
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        return response.json()

    def get_completion(self, response: Dict) -> str:
        """Extract the completion text from the API response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return ""

    def get_tool_calls(self, response: Dict) -> List[Dict]:
        """Extract tool calls from the API response."""
        try:
            return response["choices"][0]["message"].get("tool_calls", [])
        except (KeyError, IndexError):
            return [] 