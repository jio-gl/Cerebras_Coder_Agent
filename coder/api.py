import os
from typing import Dict, List, Optional, Union, Iterator
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class OpenRouterClient:
    def __init__(self, api_key: str, provider: str = "Cerebras", max_tokens: int = 31000):
        self.api_key = api_key
        self.provider = provider
        self.max_tokens = max_tokens
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
        model: str = "meta-llama/llama-3.1-8b-instruct",
        temperature: float = 0.7,
        max_tokens: int = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        stream: bool = True,
        provider: str = None
    ) -> Dict:
        """
        Make a chat completion request to OpenRouter API.
        
        Args:
            messages: List of message dictionaries with role and content
            model: Model to use (default: meta-llama/llama-3.1-8b-instruct)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools for function calling
            tool_choice: Optional tool choice strategy
            response_format: Optional response format specification
            stream: Whether to use streaming (required for tools with Cerebras)
            provider: Optional provider to use
            
        Returns:
            Dict containing the API response
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if provider is None:
            provider = self.provider

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "provider": {"only": [provider]}
        }

        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice
        if response_format:
            data["response_format"] = response_format

        # Debug logging
        print(f"Request payload: {data}")
        
        if stream:
            return self._handle_streaming_response(data)
        else:
            return self._handle_non_streaming_response(data)
    
    def _handle_non_streaming_response(self, data: dict) -> Dict:
        """Handle non-streaming API responses."""
        response = requests.post(self.base_url, headers=self.headers, json=data)
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        return response.json()
    
    def _handle_streaming_response(self, data: dict) -> Dict:
        """Handle streaming API responses and reconstruct the full response."""
        response = requests.post(
            self.base_url, 
            headers=self.headers, 
            json=data, 
            stream=True
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Response body: {response.text}")
            raise Exception(f"API request failed: {response.text}")
        
        # Reconstruct the response from streaming chunks
        full_content = ""
        tool_calls = []
        finish_reason = None
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        if 'choices' in chunk_data and chunk_data['choices']:
                            delta = chunk_data['choices'][0].get('delta', {})
                            
                            # Handle content
                            if 'content' in delta and delta['content']:
                                full_content += delta['content']
                            
                            # Handle tool calls
                            if 'tool_calls' in delta:
                                for tool_call in delta['tool_calls']:
                                    # Find existing tool call or create new one
                                    call_index = tool_call.get('index', 0)
                                    while len(tool_calls) <= call_index:
                                        tool_calls.append({
                                            "id": "",
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        })
                                    
                                    if 'id' in tool_call:
                                        tool_calls[call_index]['id'] = tool_call['id']
                                    
                                    if 'function' in tool_call:
                                        if 'name' in tool_call['function']:
                                            tool_calls[call_index]['function']['name'] = tool_call['function']['name']
                                        if 'arguments' in tool_call['function']:
                                            tool_calls[call_index]['function']['arguments'] += tool_call['function']['arguments']
                            
                            # Handle finish reason
                            if 'finish_reason' in chunk_data['choices'][0]:
                                finish_reason = chunk_data['choices'][0]['finish_reason']
                                
                    except json.JSONDecodeError:
                        continue
        
        # Reconstruct the full response format
        reconstructed_response = {
            "choices": [{
                "message": {
                    "content": full_content,
                    "role": "assistant"
                },
                "finish_reason": finish_reason
            }]
        }
        
        if tool_calls:
            reconstructed_response["choices"][0]["message"]["tool_calls"] = tool_calls
        
        print(f"Reconstructed response: {reconstructed_response}")
        return reconstructed_response

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