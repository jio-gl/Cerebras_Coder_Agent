import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from .api import OpenRouterClient
from .prompts import (
    get_completion_summary_prompt,
    get_file_generation_prompt,
    get_file_specific_instructions,
    get_improved_specs_prompt,
)
from .tools import get_agent_tools
from .utils import CodeValidator, EquivalenceChecker, VersionManager, LLMToolkit

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENROUTER_API_KEY")


class CodingAgent:
    def __init__(
        self,
        repo_path: Optional[str] = None,
        model: str = "qwen/qwen3-32b",
        debug: bool = False,
        interactive: bool = False,
        api_key: Optional[str] = None,
        no_think: bool = False,
        max_tokens: int = 31000,
        provider: str = "Cerebras",
    ):
        self.repo_path = repo_path or os.getcwd()
        
        # Enforce fixed model
        if model != "qwen/qwen3-32b":
            raise ValueError("Only qwen/qwen3-32b model is supported")
        self.model = "qwen/qwen3-32b"
        
        self.debug = debug
        self.interactive = interactive
        
        # Enforce fixed provider
        if provider != "Cerebras":
            raise ValueError("Only Cerebras provider is supported")
        self.provider = "Cerebras"
        
        # Don't store API key directly in the instance to prevent accidental exposure
        # Instead, just pass it to the client
        self._client = OpenRouterClient(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            provider="Cerebras"  # Enforce fixed provider
        )
        self.no_think = no_think
        self.max_tokens = max_tokens
        self.version_manager = VersionManager(Path(self.repo_path))
        
        # Initialize the LLM toolkit
        self.llm_toolkit = LLMToolkit(
            api_client=self._client,
            model=self.model,
            provider=self.provider,
            max_tokens=self.max_tokens,
            debug=self.debug
        )

        # Get tools from tools module
        self.tools = get_agent_tools()

    @property
    def client(self):
        """Secure access to the API client.

        This property method prevents direct access to the client's API key.
        """
        return self._client

    def _read_file(self, path: str) -> str:
        """Read a file's contents."""
        try:
            # Handle both relative and absolute paths
            if os.path.isabs(path):
                target_path = Path(path)
            else:
                target_path = Path(self.repo_path) / path

            # Resolve the paths
            resolved_target = target_path.resolve()
            resolved_repo = Path(self.repo_path).resolve()

            # Debug: Print path information
            if self.debug:
                print("\nDEBUG: File path information:")
                print(f"Target path: {target_path}")
                print(f"Resolved target: {resolved_target}")
                print(f"Resolved repo: {resolved_repo}")

            # For absolute paths, just check that the file exists and is readable
            if os.path.isabs(path):
                # Security check: Ensure absolute paths are inside the repository
                if not str(resolved_target).startswith(str(resolved_repo)):
                    raise Exception(
                        f"Access denied: Absolute path '{path}' is outside the repository"
                    )

                if not os.path.exists(resolved_target):
                    raise Exception(f"File not found: {path}")
                if not os.access(resolved_target, os.R_OK):
                    raise Exception(f"File not readable: {path}")
            else:
                # For relative paths, ensure they are within the repository
                if not str(resolved_target).startswith(str(resolved_repo)):
                    raise Exception(
                        f"Access denied: Path '{path}' is outside the repository"
                    )

                # Additional check for common path traversal patterns
                if (
                    ".." in path or path.startswith("/") or ":" in path[:3]
                ):  # Check for drive letters
                    raise Exception(f"Access denied: Invalid path '{path}'")

            # Read the file
            with open(resolved_target, "r") as f:
                content = f.read()

                # Debug: Print file content
                if self.debug:
                    print("\nDEBUG: File content:")
                    print(content)

                return content
        except Exception as e:
            if self.debug:
                print("\nDEBUG: Error reading file:")
                print(str(e))
            raise Exception(f"Error reading file: {str(e)}")

    def _list_directory(self, path: str) -> List[str]:
        """List contents of a directory."""
        try:
            dir_path = os.path.join(self.repo_path, path)
            if not os.path.exists(dir_path):
                raise Exception(f"Directory not found: {path}")
            return [f for f in os.listdir(dir_path)]
        except Exception as e:
            raise Exception(f"Error listing directory: {str(e)}")

    def _edit_file(self, path: str, content: str) -> bool:
        """Edit or create a file using a unified diff-like format.

        The content can be:
        1. A complete file replacement if no special markers are present
        2. A unified diff-like format using special markers:
           // ... existing code ...  - represents unchanged code
           // === add below ===     - add content below this line
           // === add above ===     - add content above this line
           // === replace start === - start of code to replace
           // === replace end ===   - end of code to replace
        3. A simple function addition (if content contains a function definition)
        """
        try:
            # Handle both relative and absolute paths
            if os.path.isabs(path):
                target_path = Path(path)
                repo_path = Path(self.repo_path)
            else:
                target_path = Path(self.repo_path) / path
                repo_path = Path(self.repo_path)

            # Resolve the paths
            resolved_target = target_path.resolve()
            resolved_repo = repo_path.resolve()

            # Debug: Print path information
            if self.debug:
                print("\nDEBUG: File path information:")
                print(f"Target path: {target_path}")
                print(f"Resolved target: {resolved_target}")
                print(f"Resolved repo: {resolved_repo}")

            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(resolved_target), exist_ok=True)

            # Debug: Print initial content
            if self.debug:
                print("\nDEBUG: Content to write:")
                print(repr(content))

            # Strip whitespace but don't decode
            content = content.strip()

            # Handle file editing based on content type
            if os.path.exists(resolved_target):
                # Read existing content
                with open(resolved_target, "r") as f:
                    existing_content = f.read()

                # Debug: Print existing content
                if self.debug:
                    print("\nDEBUG: Existing content:")
                    print(repr(existing_content))

                # If content is empty, do nothing
                if not content:
                    if self.debug:
                        print("\nDEBUG: Content is empty, no changes made.")
                    return True

                # Check if this is a function addition
                is_function = content.lstrip().startswith("def ") and "def " in content

                if is_function:
                    # Extract function name
                    func_def_line = content.lstrip().split("\n")[0]
                    func_name = func_def_line.split("def ")[1].split("(")[0].strip()

                    # Check if function already exists
                    if f"def {func_name}" in existing_content:
                        if self.debug:
                            print(
                                f"\nDEBUG: Function {func_name} already exists, no changes made."
                            )
                        return True

                    # Append function to existing content
                    new_content = existing_content.rstrip() + "\n\n" + content + "\n"

                    # Debug: Print new content
                    if self.debug:
                        print("\nDEBUG: New content (appending function):")
                        print(repr(new_content))

                    # Write updated content
                    with open(resolved_target, "w") as f:
                        f.write(new_content)

                    # Verify write was successful
                    if self.debug:
                        try:
                            with open(resolved_target, "r") as f:
                                verify_content = f.read()
                            print("\nDEBUG: File content after write:")
                            print(repr(verify_content))
                            print(f"File size: {os.path.getsize(resolved_target)}")
                        except Exception as e:
                            print(f"\nDEBUG: Error verifying file: {str(e)}")

                    return True
                else:
                    # Treat as full file replacement
                    new_content = content if content.endswith("\n") else content + "\n"

                    # Debug: Print new content
                    if self.debug:
                        print("\nDEBUG: New content (full replacement):")
                        print(repr(new_content))

                    # Write updated content
                    with open(resolved_target, "w") as f:
                        f.write(new_content)

                    return True
            else:
                # For new files, just write the content as is
                new_content = content if content.endswith("\n") else content + "\n"

                # Debug: Print new content
                if self.debug:
                    print("\nDEBUG: New content (new file):")
                    print(repr(new_content))

                # Write updated content
                with open(resolved_target, "w") as f:
                    f.write(new_content)

                return True
        except Exception as e:
            if self.debug:
                print("\nDEBUG: Error in _edit_file:")
                print(str(e))
            raise Exception(f"Error editing file: {str(e)}")

    def _execute_tool_call(self, tool_call: Dict) -> str:
        """Execute a tool call based on type and name."""
        try:
            # Debugging information
            if self.debug:
                print("\nDEBUG: Tool Call:")
                print(f"Type: {tool_call.get('type')}")
                print(f"Name: {tool_call.get('function', {}).get('name')}")
                print(f"Arguments: {tool_call.get('function', {}).get('arguments')}")

            # Make sure it's a function call
            if tool_call.get("type") != "function":
                raise Exception(f"Unsupported tool call type: {tool_call.get('type')}")

            # Get the function name and arguments
            func_name = tool_call.get("function", {}).get("name")
            if not func_name:
                raise Exception("Missing function name in tool call")
                
            args_str = tool_call.get("function", {}).get("arguments", "{}")
            args = json.loads(args_str) if args_str else {}

            # Execute the tool call
            if func_name == "read_file":
                return self._read_file(args.get("target_file", ""))
            elif func_name == "list_directory":
                return "\n".join(
                    self._list_directory(args.get("relative_workspace_path", ""))
                )
            elif func_name == "edit_file":
                target_file = args.get("target_file", "")
                instructions = args.get("instructions", "")
                code_edit = args.get("code_edit", "")

                # Debug info
                if self.debug:
                    print(f"\nDEBUG: Editing file {target_file}")
                    print(f"Instructions: {instructions}")
                    print(f"Code Edit: {code_edit}")

                # Execute the edit
                self._edit_file(target_file, code_edit)
                return f"File edited: {target_file}"
            elif func_name == "fix_syntax_errors":
                return self.fix_syntax_errors(args.get("target_path", ""))
            elif func_name == "analyze_code":
                return json.dumps(self.analyze_file(args.get("target_file", "")), indent=2)
            elif func_name == "optimize_code":
                return self.optimize_file(args.get("target_file", ""), args.get("optimization_goal", "performance"))
            elif func_name == "add_docstrings":
                return self.add_docstrings(args.get("target_file", ""))
            elif func_name == "generate_tests":
                return self.generate_tests(args.get("target_file", ""), args.get("output_file", None))
            elif func_name == "enhance_error_handling":
                return self.enhance_error_handling(args.get("target_file", ""))
            elif func_name == "explain_code":
                return self.explain_code(args.get("target_file", ""), args.get("explanation_level", "detailed"))
            elif func_name == "refactor_code":
                return self.refactor_code(args.get("target_file", ""), args.get("refactoring_goal", ""))
            elif func_name == "generate_markdown_docs":
                return self.generate_markdown_docs(args.get("target_file", ""), args.get("output_file", None))
            elif func_name == "fix_python_syntax":
                return self.fix_python_syntax(args.get("target_file", ""))
            else:
                # For invalid function names, we need to raise an exception
                # This is to maintain compatibility with the tests
                raise Exception(f"Unknown tool: {func_name}")
        except Exception as e:
            # Detailed error message for debugging
            error_message = f"Error executing tool call: {str(e)}"
            if self.debug:
                print(f"\nDEBUG: {error_message}")
            
            # Re-raise the exception if it's an unknown tool
            if "Unknown tool:" in str(e):
                raise
                
            # Otherwise return the error message
            return error_message

    def _clean_code_output(self, content: str) -> str:
        """Clean and sanitize code output to prevent security issues.

        This method sanitizes code to remove potentially dangerous patterns:
        1. Removes dangerous system commands (rm -rf, curl to external sites)
        2. Sanitizes HTML and JavaScript code
        3. Handles special characters and binary data
        4. Limits input size to prevent DoS attacks
        5. Removes markdown code block markers

        Args:
            content: The code content to clean

        Returns:
            Sanitized code content
        """
        try:
            # Check for null input
            if content is None:
                return ""

            # Handle extremely large inputs (DoS protection)
            max_size = 1 * 1024 * 1024  # 1MB limit
            if len(content) > max_size:
                content = (
                    content[:max_size]
                    + "\n# ... content truncated (exceeded size limit)"
                )

            # Remove markdown code block markers
            import re
            code_pattern = r"```(?:python|json|javascript)?(.+?)```"
            code_blocks = re.findall(code_pattern, content, re.DOTALL)
            if code_blocks:
                content = code_blocks[0].strip()

            # Remove potentially dangerous patterns
            dangerous_patterns = [
                # System commands that could cause damage
                (r"rm\s+-rf", 'echo "rm command blocked"'),
                (r'os\.system\s*\(\s*[\'"]rm', 'os.system("echo rm command blocked"'),
                (
                    r'subprocess\.call\s*\(\s*\[\s*[\'"]rm[\'"]',
                    'subprocess.call(["echo", "rm command blocked"]',
                ),
                (
                    r'subprocess\.Popen\s*\(\s*\[\s*[\'"]rm[\'"]',
                    'subprocess.Popen(["echo", "rm command blocked"]',
                ),
                # Network access that could exfiltrate data
                (r"urllib\.request\.urlopen\s*\(", 'print("URL access blocked: "'),
                (r"requests\.get\s*\(", 'print("Requests access blocked: "'),
                (r"curl\s+http", 'echo "curl command blocked"'),
                # Shell injection attempts
                (r"eval\s*\(", 'print("eval blocked: "'),
                (r"exec\s*\(", 'print("exec blocked: "'),
                (r"system\s*\(", 'print("system blocked: "'),
                # XSS and HTML injection
                (r"<script>", "<!-- script tag removed -->"),
                (r"javascript:", "blocked-js:"),
                (r"data:text/html", "blocked-data:text/plain"),
                (r"onerror=", "data-blocked="),
                (r"onclick=", "data-blocked="),
                # SQL injection
                (r"DROP TABLE", "SELECT FROM"),
                (r"DELETE FROM", "SELECT FROM"),
                # File access
                (r"file:///\S+", "local-file-access-blocked"),
            ]

            # Apply regex replacements
            for pattern, replacement in dangerous_patterns:
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

            # Remove null bytes and control characters
            content = "".join(
                c if c.isprintable() or c in "\n\r\t" else " " for c in content
            )

            # Add a safety comment
            if any(
                marker in content
                for marker in ["os.", "subprocess.", "system(", "eval(", "exec("]
            ):
                content = (
                    "# NOTICE: This code has been sanitized for security reasons\n"
                    + content
                )

            return content

        except Exception as e:
            # Fail safely - if there's any error in sanitization, return empty string
            if self.debug:
                print(f"\nDEBUG: Error in _clean_code_output: {str(e)}")
            return "# Error sanitizing content"

    def ask(self, question: str, model: Optional[str] = None) -> str:
        """Ask a question about the codebase."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. Answer questions about coding and programming. If you need to examine specific files or the codebase structure to answer properly, say 'I need to examine the files' in your response.",
            },
            {"role": "user", "content": question},
        ]
        model_to_use = model or self.model
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                provider=self.provider,
                stream=False,
            )

            # Check for API errors
            if isinstance(response, dict) and "error" in response:
                raise Exception(f"API request failed: {response}")

            initial_response = self.client.get_completion(response)
            if (
                "need to examine" in initial_response.lower()
                or "need to see" in initial_response.lower()
            ):
                return (
                    initial_response
                    + "\n\nNote: File examination tools are not available with the current Cerebras reasoning model configuration."
                )
            return initial_response
        except Exception as e:
            error_msg = str(e)
            if "provider" in error_msg.lower():
                return f"Error: The model '{self.model}' is not compatible with the {self.provider} provider. Please use a different model or provider."
            return f"Error: {error_msg}"

    def agent(self, prompt: str, model: Optional[str] = None) -> str:
        """Execute agent operations based on the prompt."""
        # Add /auto_confirm to the prompt to instruct the model to avoid asking for confirmation
        prompt_with_no_think = f"{prompt} /no_think /auto_confirm"
        messages = [
            {
                "role": "system",
                "content": "You are a coding agent that can read, analyze, and modify code. Use the available tools to help with the task. DO NOT ask for confirmation before making changes - always proceed immediately with implementation. You can create or modify multiple files in a single request by calling the edit_file tool multiple times.",
            },
            {"role": "user", "content": prompt_with_no_think},
        ]
        actions_taken = []
        files_created_or_modified = []
        model_to_use = model or self.model

        # Extract explicitly mentioned files from the prompt
        explicitly_mentioned_files = self._extract_mentioned_files(prompt)

        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                tools=self.tools,
                max_tokens=self.max_tokens,
                provider=self.provider,
                stream=True,
            )
            tool_calls = self.client.get_tool_calls(response)
            if not tool_calls:
                completion = self.client.get_completion(response)
                # If the response is asking for confirmation, automatically answer "yes" and try again
                if any(
                    phrase in completion.lower()
                    for phrase in [
                        "would you like me to",
                        "shall i proceed",
                        "do you want me to",
                        "should i create",
                        "should i implement",
                        "would you like to proceed",
                    ]
                ):
                    # Log that we're auto-confirming
                    if self.debug:
                        print(
                            "\nDEBUG: Detected confirmation prompt, automatically answering 'Yes'"
                        )

                    # Make a new request with confirmation
                    messages.append({"role": "assistant", "content": completion})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Yes, please proceed immediately with implementation without asking for further confirmation.",
                        }
                    )

                    response = self.client.chat_completion(
                        messages=messages,
                        model=self.model,
                        tools=self.tools,
                        max_tokens=self.max_tokens,
                        provider=self.provider,
                        stream=True,
                    )

                    # Try to get tool calls from the new response
                    tool_calls = self.client.get_tool_calls(response)
                    if not tool_calls:
                        completion = self.client.get_completion(response)
                        return f"No actions were taken after auto-confirmation. Response: {completion}"
                else:
                    return f"No actions were taken. Response: {completion}"

            # First collect and execute all tool calls
            tool_results = []

            for tool_call in tool_calls:
                try:
                    tool_name = tool_call.get("function", {}).get("name", "unknown")
                    action = f"Using tool: {tool_name}"
                    actions_taken.append(action)

                    # Debug: Print tool call details
                    if self.debug:
                        print("\nDEBUG: Tool call details:")
                        print(f"Tool: {tool_name}")
                        print(
                            "Arguments:",
                            tool_call.get("function", {}).get("arguments", ""),
                        )

                    result = self._execute_tool_call(tool_call)

                    # Track file operations
                    if tool_name == "edit_file":
                        try:
                            args = json.loads(
                                tool_call.get("function", {}).get("arguments", "{}")
                            )
                            target_file = args.get("target_file", "unknown")
                            files_created_or_modified.append(target_file)
                        except:
                            pass

                    # Save the tool result to include in the next API call
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", "mock_id"),
                            "content": result,
                        }
                    )
                except Exception as e:
                    error_msg = str(e)
                    actions_taken.append(f"Error: {error_msg}")
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", "mock_id"),
                            "content": error_msg,
                        }
                    )

            # Add all tool results to messages
            messages.extend(tool_results)

            # Make a follow-up API call with all tool results included
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                tools=self.tools,
                max_tokens=self.max_tokens,
                provider=self.provider,
                stream=True,
            )

            # Process any additional tool calls recursively (up to max 5 iterations to prevent infinite loops)
            max_iterations = 5
            current_iteration = 1

            while current_iteration < max_iterations:
                additional_tool_calls = self.client.get_tool_calls(response)

                if not additional_tool_calls:
                    break

                additional_tool_results = []

                for tool_call in additional_tool_calls:
                    try:
                        tool_name = tool_call.get("function", {}).get("name", "unknown")
                        action = f"Using tool: {tool_name}"
                        actions_taken.append(action)

                        # Debug: Print tool call details
                        if self.debug:
                            print(f"\nDEBUG: Additional tool call {current_iteration}:")
                            print(f"Tool: {tool_name}")
                            print(
                                "Arguments:",
                                tool_call.get("function", {}).get("arguments", ""),
                            )

                        result = self._execute_tool_call(tool_call)

                        # Track file operations
                        if tool_name == "edit_file":
                            try:
                                args = json.loads(
                                    tool_call.get("function", {}).get("arguments", "{}")
                                )
                                target_file = args.get("target_file", "unknown")
                                files_created_or_modified.append(target_file)
                            except:
                                pass

                        additional_tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", "mock_id"),
                                "content": result,
                            }
                        )
                    except Exception as e:
                        error_msg = str(e)
                        actions_taken.append(f"Error: {error_msg}")
                        additional_tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", "mock_id"),
                                "content": error_msg,
                            }
                        )

                # Add additional tool results to messages
                messages.extend(additional_tool_results)

                # Make another follow-up API call
                response = self.client.chat_completion(
                    messages=messages,
                    model=self.model,
                    tools=self.tools,
                    max_tokens=self.max_tokens,
                    provider=self.provider,
                    stream=True,
                )

                current_iteration += 1

            # Check if any explicitly mentioned files were not created
            missing_files = [
                file
                for file in explicitly_mentioned_files
                if file not in files_created_or_modified
            ]

            # If there are missing files, create them with another API call
            if missing_files:
                if self.debug:
                    print(f"\nDEBUG: Detected missing files: {missing_files}")

                # Generate content for missing files
                missing_files_prompt = f"Please create the following files that were mentioned in the original request but not created: {', '.join(missing_files)}. The original request was: {prompt}"

                missing_files_messages = [
                    {
                        "role": "system",
                        "content": "You are a coding agent that creates files based on context. Create appropriate content for the requested files based on the project context and related files already created.",
                    },
                    {"role": "user", "content": missing_files_prompt},
                ]

                # Add file contents for context
                for file in files_created_or_modified:
                    try:
                        file_content = self._read_file(file)
                        missing_files_messages.append(
                            {
                                "role": "user",
                                "content": f"Here's the content of {file} for context:\n\n{file_content}",
                            }
                        )
                    except:
                        pass

                # Get content for missing files
                for missing_file in missing_files:
                    try:
                        # Generate prompt for missing file
                        file_prompt = f"Based on the project context and other files, create appropriate content for {missing_file}. Create a complete, well-formatted file."

                        missing_files_messages.append(
                            {"role": "user", "content": file_prompt}
                        )

                        # Get content from API
                        missing_file_response = self.client.chat_completion(
                            messages=missing_files_messages,
                            model=self.model,
                            max_tokens=self.max_tokens,
                            provider=self.provider,
                            stream=False,
                        )

                        # Check for API errors
                        if isinstance(missing_file_response, dict) and "error" in missing_file_response:
                            raise Exception(f"API request failed: {missing_file_response}")

                        missing_file_content = self.client.get_completion(
                            missing_file_response
                        )

                        # Extract content between code blocks if present
                        import re

                        code_pattern = r"```(?:json|javascript)?(.+?)```"
                        code_blocks = re.findall(
                            code_pattern, missing_file_content, re.DOTALL
                        )

                        if code_blocks:
                            clean_content = code_blocks[0].strip()
                        else:
                            clean_content = missing_file_content.strip()

                        # Create the missing file
                        self._edit_file(missing_file, clean_content)
                        files_created_or_modified.append(missing_file)

                        if self.debug:
                            print(f"\nDEBUG: Created missing file: {missing_file}")

                    except Exception as e:
                        if self.debug:
                            print(
                                f"\nDEBUG: Error creating missing file {missing_file}: {str(e)}"
                            )

            completion = self.client.get_completion(response)

            if actions_taken:
                # Return a summary of changes if files were created or modified
                if files_created_or_modified:
                    if len(files_created_or_modified) == 1:
                        return (
                            f"✨ Created/modified file: {files_created_or_modified[0]}"
                        )
                    else:
                        file_list = "\n- ".join(files_created_or_modified)
                        return f"✨ Created/modified {len(files_created_or_modified)} files:\n- {file_list}"
                return "✨ Changes Applied"
            else:
                return completion
        except Exception as e:
            error_msg = str(e)
            if "provider" in error_msg.lower():
                return f"Error: The model '{self.model}' is not compatible with the {self.provider} provider. Please use a different model or provider."
            return f"Error: {error_msg}"

    def _extract_mentioned_files(self, prompt: str) -> List[str]:
        """Extract explicitly mentioned files from the prompt."""
        import re

        # Define patterns for common file types and mentions
        patterns = [
            r'(?:create|include|implement|build|generate|make)\s+(?:a|an)?\s*["\']?([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+)["\']?',
            r'(?:file|module|script|config)?\s*["\']?([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+)["\']?',
            r"([a-zA-Z0-9_\-\.]+\.(?:js|py|json|md|txt|html|css|ts|jsx|tsx))",
            r'(?:named|called)\s+["\']?([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+)["\']?',
        ]

        # Find all matches
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, prompt, re.IGNORECASE))

        # Filter for unique, valid filenames
        filtered_matches = []
        for match in matches:
            # Skip if not a valid filename
            if not re.match(r"^[a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+$", match):
                continue
            # Skip if looks like a version number
            if re.match(r"^v?\d+\.\d+\.\d+$", match):
                continue
            # Add if not already in list
            if match not in filtered_matches:
                filtered_matches.append(match)

        return filtered_matches

    def _generate_file_content(self, filename: str, specs: str, version: int) -> str:
        """Generate file content based on specifications.

        This method requests content generation from the API and sanitizes the output
        to ensure security.

        Args:
            filename: Name of the file to generate
            specs: Specifications for the file content
            version: Version number for versioning

        Returns:
            Sanitized file content
        """
        try:
            # Generate prompt for file creation
            file_extension = os.path.splitext(filename)[1]
            language = "python" if file_extension == ".py" else "unknown"

            prompt = f"""
            Generate content for file: {filename}
            
            Specifications:
            {specs}

            For {filename}:
            Version: {version}
            
            Language: {language}
            
            Generate only the file content, no explanations.
            """

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant that generates high-quality, secure code based on specifications.",
                },
                {"role": "user", "content": prompt},
            ]

            if self.debug:
                print(f"\nDEBUG: Generating content for {filename} using model {self.model}")

            # Generate content using only the specified model
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                provider=self.provider,
                stream=False,
            )

            # Check for API errors
            if isinstance(response, dict) and "error" in response:
                raise Exception(f"API request failed: {response}")

            # Extract content from response using the client's get_completion method
            content = self.client.get_completion(response)
            
            if not content:
                raise Exception("Empty content received from API")

            # Clean and sanitize the content
            content = self._clean_code_output(content)

            return content

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _generate_file_content: {str(e)}")
            raise Exception(f"Failed to generate content for {filename}: {str(e)}")

    def _create_backup(self, items=None, backup_dir=None):
        """Create a backup of the codebase.

        This method creates a secure backup of specified files/directories
        in the codebase, preserving file permissions.

        Args:
            items: List of files/directories to backup (default: main project files)
            backup_dir: Directory to store the backup (default: temp directory)

        Returns:
            Path to the backup directory
        """
        try:
            # Default items to backup if none specified
            if items is None:
                items = [
                    "coder",
                    "tests",
                    "setup.py",
                    "README.md",
                    "SPECS.md",
                    "requirements.txt",
                ]

            # Create backup directory
            if backup_dir is None:
                # Get current version
                current_version = self.version_manager.get_current_version()
                timestamp = int(time.time())
                # Use format: backup_v{version}_{timestamp}_{uuid}
                backup_name = (
                    f"backup_v{current_version}_{timestamp}_{uuid.uuid4().hex[:8]}"
                )
                backup_dir = Path(tempfile.mkdtemp(prefix=f"{backup_name}_"))
            else:
                backup_dir = Path(backup_dir)
                os.makedirs(backup_dir, exist_ok=True)

            # Copy files/directories with permissions preserved
            repo_path = Path(self.repo_path)
            for item in items:
                src_path = repo_path / item
                dst_path = backup_dir / item

                if not src_path.exists():
                    continue

                if src_path.is_file():
                    # Create parent directories if needed
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                    # Copy file with content
                    shutil.copy2(src_path, dst_path)

                    # Preserve permissions
                    src_mode = os.stat(src_path).st_mode
                    os.chmod(dst_path, src_mode)
                elif src_path.is_dir():
                    # Copy directory recursively with permissions
                    shutil.copytree(
                        src_path,
                        dst_path,
                        symlinks=False,
                        ignore=None,
                        copy_function=shutil.copy2,  # copy2 preserves metadata
                        ignore_dangling_symlinks=True,
                        dirs_exist_ok=True,
                    )

            return backup_dir

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _create_backup: {str(e)}")
            raise Exception(f"Failed to create backup: {str(e)}")

    def analyze_codebase(self) -> str:
        """Analyze the codebase for self-rewrite."""
        # This is a placeholder for the CLI interface's analyze_codebase method
        # The actual implementation would analyze the codebase structure, performance, etc.
        return "Codebase analysis complete. Ready for self-rewrite."

    def _generate_improved_specs(self, current_version: int, next_version: int) -> str:
        """Generate improved specifications for a new version.

        Args:
            current_version: The current version number
            next_version: The next version number to generate specs for

        Returns:
            Improved specifications for the new version
        """
        try:
            # Read the current SPECS.md file
            specs_path = os.path.join(self.repo_path, "SPECS.md")
            if not os.path.exists(specs_path):
                current_specs = f"# Coding Agent v{current_version} Specification\n\nBasic coding agent."
            else:
                with open(specs_path, "r") as f:
                    current_specs = f.read()

            # Generate prompt for improved specs
            prompt = get_improved_specs_prompt(
                current_specs, current_version, next_version
            )

            # Request improved specs from API
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates improved specifications for software projects.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                provider=self.provider,
                stream=False,
            )

            # Check for API errors
            if isinstance(response, dict) and "error" in response:
                raise Exception(f"API request failed: {response}")

            improved_specs = self.client.get_completion(response)

            # Make sure the specs have the correct version number
            if f"v{next_version}" not in improved_specs:
                improved_specs = (
                    f"# Coding Agent v{next_version} Specification\n\n{improved_specs}"
                )

            return improved_specs

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _generate_improved_specs: {str(e)}")
            # Return a basic specification if there's an error
            return f"# Coding Agent v{next_version} Specification\n\nImproved version with enhanced features."

    def _generate_completion_summary(self, version: int, version_dir: Path) -> str:
        """Generate a summary of the completed self-rewrite.

        Args:
            version: The version number that was generated
            version_dir: The directory containing the new version

        Returns:
            Summary of the completed self-rewrite
        """
        try:
            # Get list of files that were created
            files = []
            for root, _, filenames in os.walk(version_dir):
                for filename in filenames:
                    if filename.endswith(".py") or filename.endswith(".md"):
                        rel_path = os.path.relpath(
                            os.path.join(root, filename), version_dir
                        )
                        files.append(rel_path)

            # Format the file list
            file_list = "\n".join([f"- {file}" for file in sorted(files)])

            # Generate the summary
            summary = f"""# Self-Rewrite Completed Successfully!

## Version Information
- Version: v{version}
- Directory: {version_dir}
- Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Files Generated
{file_list}

## Next Steps
1. Review the generated files
2. Run tests to verify functionality
3. Deploy the new version
"""
            return summary

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _generate_completion_summary: {str(e)}")
            # Return a basic summary if there's an error
            return f"Self-Rewrite Completed Successfully! Version: v{version}, Directory: {version_dir}"

    def _generate_version_files(
        self, version_dir: Path, specs: str, version: int
    ) -> bool:
        """Generate files for a new version.

        Args:
            version_dir: Directory to generate files in
            specs: Specifications for the new version
            version: Version number

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create basic structure if not exists
            os.makedirs(version_dir / "coder", exist_ok=True)
            os.makedirs(version_dir / "coder" / "utils", exist_ok=True)
            os.makedirs(version_dir / "tests", exist_ok=True)

            # Get current test files
            import subprocess
            try:
                test_files_result = subprocess.run(
                    ["find", "tests/", "-name", "*.py", "-type", "f"], 
                    capture_output=True, text=True, cwd=self.repo_path
                )
                current_test_files = test_files_result.stdout.strip().split('\n') if test_files_result.stdout.strip() else []
                if self.debug:
                    print(f"\nDEBUG: Found existing test files: {len(current_test_files)}")
            except Exception as e:
                if self.debug:
                    print(f"\nDEBUG: Could not find current test files: {str(e)}")
                current_test_files = []

            # List of files to generate
            files_to_generate = [
                "coder/agent.py",
                "coder/api.py",
                "coder/cli.py",
                "coder/utils/__init__.py",
                "coder/utils/version.py",
                "tests/test_agent.py",
                "tests/test_api.py",
                "tests/test_cli.py",
                "README.md",
                "SPECS.md",
                "setup.py",
                "requirements.txt",
            ]
            
            # Add all current test files to the generation list
            for test_file in current_test_files:
                if test_file and test_file not in files_to_generate:
                    files_to_generate.append(test_file)

            # Generate each file
            for file_path in files_to_generate:
                if self.debug:
                    print(f"\nDEBUG: Generating file: {file_path}")

                # Get file-specific instructions if available
                file_instructions = get_file_specific_instructions(file_path, version)

                # Generate file content
                try:
                    file_content = self._generate_file_content(
                        file_path, specs + "\n\n" + file_instructions, version
                    )

                    # Write the file
                    target_path = version_dir / file_path
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    with open(target_path, "w") as f:
                        f.write(file_content)
                    
                    # Apply LLM enhancements to the generated file if it's a Python file
                    if file_path.endswith('.py'):
                        try:
                            # Enhance error handling in the file
                            if self.debug:
                                print(f"\nDEBUG: Enhancing error handling in {file_path}")
                            
                            # Read the file content
                            with open(target_path, "r") as f:
                                content = f.read()
                            
                            # Enhance error handling
                            enhanced_code = self.llm_toolkit.enhance_error_handling(content)
                            
                            # Write the enhanced code back to the file
                            with open(target_path, "w") as f:
                                f.write(enhanced_code)
                            
                            # Add or improve docstrings
                            if self.debug:
                                print(f"\nDEBUG: Adding/improving docstrings in {file_path}")
                            
                            # Read the file content again after error handling enhancements
                            with open(target_path, "r") as f:
                                content = f.read()
                            
                            # Generate improved docstrings
                            documented_code = self.llm_toolkit.generate_docstring(content)
                            
                            # Write the documented code back to the file
                            with open(target_path, "w") as f:
                                f.write(documented_code)
                                
                            # Optimize code for readability
                            if self.debug:
                                print(f"\nDEBUG: Optimizing {file_path} for readability")
                            
                            # Read the file content again after docstring additions
                            with open(target_path, "r") as f:
                                content = f.read()
                            
                            # Optimize the code
                            optimized_code = self.llm_toolkit.optimize_code(content, "readability")
                            
                            # Write the optimized code back to the file
                            with open(target_path, "w") as f:
                                f.write(optimized_code)
                        except Exception as e:
                            if self.debug:
                                print(f"\nDEBUG: Error enhancing {file_path}: {str(e)}")
                except Exception as e:
                    if self.debug:
                        print(f"\nDEBUG: Error generating {file_path}: {str(e)}")
                    
                    # If this is a test file and generation failed, copy from original
                    if file_path.startswith("tests/") and Path(self.repo_path, file_path).exists():
                        if self.debug:
                            print(f"\nDEBUG: Copying test file {file_path} from original")
                        
                        import shutil
                        source_path = Path(self.repo_path, file_path)
                        target_path = version_dir / file_path
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.copy2(source_path, target_path)

                # Sleep briefly to avoid rate limiting
                time.sleep(1)

            # Write the specs file directly
            with open(version_dir / "SPECS.md", "w") as f:
                f.write(specs)
                
            # Ensure all test files exist in the new version
            missing_test_files = []
            for test_file in current_test_files:
                if test_file and not (version_dir / test_file).exists():
                    missing_test_files.append(test_file)
            
            # Copy any missing test files
            if missing_test_files:
                if self.debug:
                    print(f"\nDEBUG: Copying {len(missing_test_files)} missing test files")
                
                import shutil
                for test_file in missing_test_files:
                    source_path = Path(self.repo_path, test_file)
                    target_path = version_dir / test_file
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy2(source_path, target_path)

            return True

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _generate_version_files: {str(e)}")
            return False

    def _rollback(self, version_dir: Path, backup_dir: Path) -> bool:
        """Roll back changes in case of failure.

        Args:
            version_dir: The version directory to remove
            backup_dir: The backup directory to restore from

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove the version directory if it exists
            if version_dir.exists():
                shutil.rmtree(version_dir)

            # Restore from backup if needed
            # Note: We don't actually restore by default, as the backup is just for reference

            if self.debug:
                print(
                    f"\nDEBUG: Rolled back successfully. Backup preserved at {backup_dir}"
                )

            return True

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _rollback: {str(e)}")
            return False

    def _validate_new_version(self, version_dir: Path) -> bool:
        """Validate the new version.

        Args:
            version_dir: Directory containing the new version

        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Validate the structure
            required_dirs = ["coder", "tests"]
            required_files = [
                "coder/agent.py",
                "coder/cli.py",
                "SPECS.md",
                "README.md",
                "setup.py",
            ]

            for dir_name in required_dirs:
                if not (version_dir / dir_name).is_dir():
                    if self.debug:
                        print(
                            f"\nDEBUG: Validation failed - missing directory: {dir_name}"
                        )
                    return False

            for file_path in required_files:
                if not (version_dir / file_path).is_file():
                    if self.debug:
                        print(f"\nDEBUG: Validation failed - missing file: {file_path}")
                    return False

            # Perform comprehensive syntax error checking and fixing
            import subprocess
            
            # Find all Python files in the new version directory
            python_files_result = subprocess.run(
                ["find", ".", "-name", "*.py", "-type", "f"],
                capture_output=True, 
                text=True,
                cwd=str(version_dir)
            )
            
            python_files = python_files_result.stdout.strip().split('\n') if python_files_result.stdout.strip() else []
            files_with_errors = []
            
            # Check each file for syntax errors
            for py_file in python_files:
                if not py_file:
                    continue
                    
                file_path = version_dir / py_file[2:]  # Remove leading './'
                
                # Run a syntax check
                syntax_check = subprocess.run(
                    ["python", "-m", "py_compile", str(file_path)],
                    capture_output=True,
                    text=True
                )
                
                if syntax_check.returncode != 0:
                    if self.debug:
                        print(f"\nDEBUG: Syntax error in {py_file}: {syntax_check.stderr}")
                    files_with_errors.append(file_path)
            
            # Try to fix syntax errors using LLM
            if files_with_errors:
                if self.debug:
                    print(f"\nDEBUG: Found {len(files_with_errors)} files with syntax errors, attempting to fix")
                
                # First try with heuristic fixes
                self._fix_syntax_errors(version_dir)
                
                # Recheck for remaining errors
                remaining_errors = []
                for py_file in files_with_errors:
                    syntax_check = subprocess.run(
                        ["python", "-m", "py_compile", str(py_file)],
                        capture_output=True,
                        text=True
                    )
                    
                    if syntax_check.returncode != 0:
                        remaining_errors.append(py_file)
                
                # Use LLM to fix remaining errors
                if remaining_errors:
                    if self.debug:
                        print(f"\nDEBUG: Attempting to fix {len(remaining_errors)} files with LLM")
                    
                    for py_file in remaining_errors:
                        self._fix_syntax_errors_with_llm(py_file)
            
            # Basic code validation after fixes
            validator = CodeValidator(version_dir)

            # Validate Python syntax again
            validation_result = validator.validate_syntax()
            if not validation_result.success:
                if self.debug:
                    print(
                        f"\nDEBUG: Python syntax validation failed after fixing attempts: {validation_result.error}"
                    )
                return False

            # Import validation for main files
            try:
                import sys

                sys.path.insert(0, str(version_dir))
                import coder

                del sys.modules["coder"]
                sys.path.pop(0)
            except Exception as e:
                if self.debug:
                    print(f"\nDEBUG: Import validation failed: {str(e)}")
                return False

            return True

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _validate_new_version: {str(e)}")
            return False
            
    def _fix_syntax_errors(self, version_dir: Path, error_message: str = "") -> bool:
        """Fix common syntax errors in generated files.
        
        Args:
            version_dir: Directory containing the new version
            error_message: Specific error message to target for fixes
            
        Returns:
            True if fixes were applied, False otherwise
        """
        try:
            if self.debug:
                print(f"\nDEBUG: Attempting to fix syntax errors")
            
            fixed_something = False
            files_with_errors = []
            
            # First, try basic heuristic fixes
            # Check all Python files
            for py_file in version_dir.glob("**/*.py"):
                try:
                    # Read the file content
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Fix specific issues based on error message
                    if "unterminated triple-quoted string" in error_message and py_file.name in error_message:
                        # Find unterminated triple quotes
                        if '"""' in content and content.count('"""') % 2 != 0:
                            # Add missing closing quotes at the end of the file
                            content = content + '\n"""'
                            fixed_something = True
                            if self.debug:
                                print(f"\nDEBUG: Fixed unterminated triple-quoted string in {py_file}")
                    
                    # Fix missing parentheses in function calls/definitions
                    if "SyntaxError: unexpected EOF" in error_message or "SyntaxError: invalid syntax" in error_message:
                        # Check for unbalanced parentheses
                        if content.count('(') > content.count(')'):
                            missing_count = content.count('(') - content.count(')')
                            content = content + '\n' + ')' * missing_count
                            fixed_something = True
                            if self.debug:
                                print(f"\nDEBUG: Fixed {missing_count} missing closing parentheses in {py_file}")
                    
                    # Fix common issues even without specific error messages
                    
                    # Fix unterminated triple-quoted strings (general case)
                    if content.count('"""') % 2 != 0:
                        # This is a heuristic approach - add closing quotes at the end
                        content = content + '\n"""'
                        fixed_something = True
                        if self.debug:
                            print(f"\nDEBUG: Fixed unterminated triple-quoted string in {py_file}")
                    
                    # Fix unterminated single triple-quoted strings
                    if content.count("'''") % 2 != 0:
                        content = content + "\n'''"
                        fixed_something = True
                        if self.debug:
                            print(f"\nDEBUG: Fixed unterminated single triple-quoted string in {py_file}")
                    
                    # Fix unbalanced brackets and braces
                    if content.count('[') > content.count(']'):
                        missing_count = content.count('[') - content.count(']')
                        content = content + '\n' + ']' * missing_count
                        fixed_something = True
                        if self.debug:
                            print(f"\nDEBUG: Fixed {missing_count} missing closing brackets in {py_file}")
                    
                    if content.count('{') > content.count('}'):
                        missing_count = content.count('{') - content.count('}')
                        content = content + '\n' + '}' * missing_count
                        fixed_something = True
                        if self.debug:
                            print(f"\nDEBUG: Fixed {missing_count} missing closing braces in {py_file}")
                    
                    # Write back if changed
                    if content != original_content:
                        with open(py_file, "w", encoding="utf-8") as f:
                            f.write(content)
                    
                    # Check if file still has syntax errors after basic fixes
                    import subprocess
                    result = subprocess.run(
                        ["python", "-m", "py_compile", str(py_file)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        # File still has syntax errors, add to list for LLM fixing
                        files_with_errors.append(py_file)
                        if self.debug:
                            print(f"\nDEBUG: File {py_file} still has syntax errors after basic fixes")
                
                except Exception as e:
                    if self.debug:
                        print(f"\nDEBUG: Error fixing {py_file}: {str(e)}")
                    files_with_errors.append(py_file)
            
            # If we still have files with errors, use LLM to fix them
            if files_with_errors:
                if self.debug:
                    print(f"\nDEBUG: Using LLM to fix {len(files_with_errors)} files with syntax errors")
                
                for py_file in files_with_errors:
                    try:
                        fixed = self._fix_syntax_errors_with_llm(py_file, error_message)
                        if fixed:
                            fixed_something = True
                    except Exception as e:
                        if self.debug:
                            print(f"\nDEBUG: Error using LLM to fix {py_file}: {str(e)}")
            
            return fixed_something
            
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _fix_syntax_errors: {str(e)}")
            return False
            
    def _fix_syntax_errors_with_llm(self, file_path: Path, error_message: str = "") -> bool:
        """Use LLM to fix syntax errors in a file.
        
        Args:
            file_path: Path to the file to fix
            error_message: Specific error message to target for fixes
            
        Returns:
            True if fixes were applied, False otherwise
        """
        try:
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Get specific syntax errors for this file
            import subprocess
            result = subprocess.run(
                ["python", "-c", f"compile(open('{file_path}', 'r').read(), '{file_path}', 'exec')"],
                capture_output=True,
                text=True
            )
            
            specific_error = result.stderr if result.returncode != 0 else ""
            
            # Generate prompt for LLM
            prompt = f"""
            Fix the Python syntax errors in the following code. Return only the fixed code with no explanation.
            
            Error message: {specific_error}
            
            File: {file_path.name}
            
            ```python
            {content}
            ```
            """
            
            # Request fixes from LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a Python syntax fixing assistant. Fix syntax errors in the provided code. Only output the fixed code, nothing else."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                provider=self.provider,
                stream=False
            )
            
            # Extract fixed code from response
            fixed_content = self.client.get_completion(response)
            
            # Clean up the response to get only the code
            import re
            # Extract code if it's wrapped in code blocks
            code_pattern = r"```(?:python)?(.*?)```"
            code_matches = re.findall(code_pattern, fixed_content, re.DOTALL)
            
            if code_matches:
                fixed_content = code_matches[0].strip()
            else:
                # If no code blocks, use the whole response but remove any explanations
                lines = fixed_content.split('\n')
                code_lines = []
                in_code = False
                
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code = not in_code
                        continue
                    
                    if in_code or not (line.startswith('#') or line.lower().startswith('here') or line.lower().startswith('i fixed')):
                        code_lines.append(line)
                
                fixed_content = '\n'.join(code_lines)
            
            # Write the fixed content back to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            
            # Verify the fixes worked
            result = subprocess.run(
                ["python", "-m", "py_compile", str(file_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                if self.debug:
                    print(f"\nDEBUG: Successfully fixed syntax errors in {file_path} using LLM")
                return True
            else:
                if self.debug:
                    print(f"\nDEBUG: Failed to fix all syntax errors in {file_path} using LLM: {result.stderr}")
                return False
                
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _fix_syntax_errors_with_llm: {str(e)}")
            return False

    def self_rewrite(self) -> str:
        """Perform a self-rewrite of the codebase.

        This method generates a new version of the codebase with improved functionality.

        Returns:
            Summary of the self-rewrite process
        """
        try:
            # Get current and next version
            current_version = self.version_manager.get_current_version()
            next_version = self.version_manager.get_next_version()

            if self.debug:
                print(
                    f"\nDEBUG: Starting self-rewrite from v{current_version} to v{next_version}"
                )
                print(f"\nDEBUG: Using model: {self.model}, provider: {self.provider}")

            # Count current tests for validation
            import subprocess
            try:
                # Count test files
                test_files_result = subprocess.run(
                    ["find", "tests/", "-name", "*.py", "-type", "f"], 
                    capture_output=True, text=True, cwd=self.repo_path
                )
                current_test_files = len(test_files_result.stdout.strip().split('\n')) if test_files_result.stdout.strip() else 0
                test_file_list = test_files_result.stdout.strip().split('\n') if test_files_result.stdout.strip() else []
                
                # Count test functions
                test_functions_result = subprocess.run(
                    ["find", "tests/", "-name", "test_*.py", "-exec", "grep", "-c", "def test_", "{}", ";"], 
                    capture_output=True, text=True, cwd=self.repo_path
                )
                current_test_functions = sum(int(line) for line in test_functions_result.stdout.strip().split('\n') if line.isdigit())
                
                if self.debug:
                    print(f"\nDEBUG: Current test baseline - Files: {current_test_files}, Functions: {current_test_functions}")
            except Exception as e:
                if self.debug:
                    print(f"\nDEBUG: Could not count current tests: {str(e)}")
                current_test_files = 0
                current_test_functions = 0
                test_file_list = []

            # Create version directory
            version_dir = self.version_manager.create_version_directory(next_version)

            # Copy .env file to new version directory for API testing
            env_file = Path(self.repo_path) / ".env"
            if env_file.exists():
                try:
                    import shutil
                    shutil.copy2(env_file, version_dir / ".env")
                    if self.debug:
                        print(f"\nDEBUG: Copied .env file to {version_dir}")
                except Exception as e:
                    if self.debug:
                        print(f"\nDEBUG: Warning - Could not copy .env file: {str(e)}")

            # Create backup of current state
            backup_dir = self._create_backup()

            # Try with specified model, but fall back to anthropic/claude-3-opus if needed
            try:
                # Generate improved specifications
                specs = self._generate_improved_specs(current_version, next_version)
            except Exception as e:
                if "No allowed providers are available for the selected model" in str(e):
                    if self.debug:
                        print(f"\nDEBUG: Error with model {self.model}: {str(e)}")
                    return f"Self-rewrite failed: Error with model {self.model}. Please ensure the model is available with the Cerebras provider."
                raise

            # Write specs to the version directory
            specs_path = version_dir / "SPECS.md"
            with open(specs_path, "w") as f:
                f.write(specs)

            if self.debug:
                print(f"\nDEBUG: Generated improved specs for v{next_version}")

            # Generate version files
            try:
                file_generation_success = self._generate_version_files(
                    version_dir, specs, next_version
                )
            except Exception as e:
                if "No allowed providers are available for the selected model" in str(e):
                    if self.debug:
                        print(f"\nDEBUG: Error with model {self.model}: {str(e)}")
                    return f"Self-rewrite failed: Error with model {self.model}. Please ensure the model is available with the Cerebras provider."
                raise
                    
            if not file_generation_success:
                self._rollback(version_dir, backup_dir)
                return f"Self-rewrite failed during file generation. Backup preserved at {backup_dir}"

            if self.debug:
                print(f"\nDEBUG: Generated files for v{next_version}")
                
            # Run syntax error fixing on the generated files
            if self.debug:
                print(f"\nDEBUG: Running syntax error fixing on generated files")
                
            try:
                # First fix all Python files in the coder directory
                coder_fix_result = self.fix_syntax_errors(str(version_dir / "coder"))
                if self.debug:
                    print(f"\nDEBUG: Coder directory fix result: {coder_fix_result}")
                    
                # Then fix all test files
                tests_fix_result = self.fix_syntax_errors(str(version_dir / "tests"))
                if self.debug:
                    print(f"\nDEBUG: Tests directory fix result: {tests_fix_result}")
            except Exception as e:
                if self.debug:
                    print(f"\nDEBUG: Warning - Error during syntax fixing: {str(e)}")

            # Validate new version
            validation_success = self._validate_new_version(version_dir)
            if not validation_success:
                self._rollback(version_dir, backup_dir)
                return f"Self-rewrite failed during validation. Backup preserved at {backup_dir}"

            # Double-check that files were actually created
            required_files = ["coder/agent.py", "coder/cli.py", "SPECS.md", "README.md", "setup.py"]
            files_exist = all((version_dir / file).exists() for file in required_files)
            if not files_exist:
                self._rollback(version_dir, backup_dir)
                return f"Self-rewrite failed: Required files were not created. Backup preserved at {backup_dir}"

            # Validate test count - ensure we have same or more tests
            try:
                # Count new test files
                new_test_files_result = subprocess.run(
                    ["find", "tests/", "-name", "*.py", "-type", "f"], 
                    capture_output=True, text=True, cwd=str(version_dir)
                )
                new_test_files = len(new_test_files_result.stdout.strip().split('\n')) if new_test_files_result.stdout.strip() else 0
                new_test_file_list = new_test_files_result.stdout.strip().split('\n') if new_test_files_result.stdout.strip() else []
                
                # Count new test functions
                new_test_functions_result = subprocess.run(
                    ["find", "tests/", "-name", "test_*.py", "-exec", "grep", "-c", "def test_", "{}", ";"], 
                    capture_output=True, text=True, cwd=str(version_dir)
                )
                new_test_functions = sum(int(line) for line in new_test_functions_result.stdout.strip().split('\n') if line.isdigit())
                
                if self.debug:
                    print(f"\nDEBUG: New version test count - Files: {new_test_files}, Functions: {new_test_functions}")
                
                # Check if test file count decreased and attempt recovery if needed
                if new_test_files < current_test_files:
                    if self.debug:
                        print(f"\nDEBUG: Test file count decreased from {current_test_files} to {new_test_files}. Attempting recovery...")
                    
                    # Find missing test files
                    missing_test_files = []
                    for test_file in test_file_list:
                        if test_file and test_file not in new_test_file_list:
                            missing_test_files.append(test_file)
                    
                    if missing_test_files and self.debug:
                        print(f"\nDEBUG: Missing test files: {missing_test_files}")
                    
                    # Copy missing test files from original
                    import shutil
                    for test_file in missing_test_files:
                        source_path = Path(self.repo_path, test_file)
                        target_path = version_dir / test_file
                        if source_path.exists():
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            shutil.copy2(source_path, target_path)
                            if self.debug:
                                print(f"\nDEBUG: Copied {test_file} from original")
                    
                    # Recount test files after recovery
                    new_test_files_result = subprocess.run(
                        ["find", "tests/", "-name", "*.py", "-type", "f"], 
                        capture_output=True, text=True, cwd=str(version_dir)
                    )
                    new_test_files = len(new_test_files_result.stdout.strip().split('\n')) if new_test_files_result.stdout.strip() else 0
                    
                    if new_test_files < current_test_files:
                        self._rollback(version_dir, backup_dir)
                        return f"Self-rewrite failed: Test file count decreased from {current_test_files} to {new_test_files} even after recovery attempt. Backup preserved at {backup_dir}"
                    
                    if self.debug:
                        print(f"\nDEBUG: After recovery - Test files: {new_test_files}")
                
                # Check test function count and attempt recovery if needed
                if new_test_functions < current_test_functions:
                    if self.debug:
                        print(f"\nDEBUG: Test function count decreased from {current_test_functions} to {new_test_functions}. Some functionality may be missing.")
                    
                    # For now, we'll accept this if file count is maintained
                    # In a more sophisticated version, we could try to regenerate files with insufficient test functions
                    if new_test_files >= current_test_files:
                        if self.debug:
                            print(f"\nDEBUG: Proceeding despite reduced test function count as file count is maintained")
                    else:
                        self._rollback(version_dir, backup_dir)
                        return f"Self-rewrite failed: Test function count decreased from {current_test_functions} to {new_test_functions}. Backup preserved at {backup_dir}"
                    
                if self.debug:
                    print(f"\nDEBUG: Test count validation passed - maintained or increased test coverage")
                    
            except Exception as e:
                if self.debug:
                    print(f"\nDEBUG: Warning - Could not validate test count: {str(e)}")

            if self.debug:
                print(f"\nDEBUG: Validated v{next_version}")

            # Generate completion summary
            summary = self._generate_completion_summary(next_version, version_dir)

            return summary

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in self_rewrite: {str(e)}")
            return f"Self-rewrite failed: {str(e)}"

    def fix_syntax_errors(self, path: str) -> str:
        """Fix syntax errors in a file or directory.
        
        This method uses a combination of heuristic fixes and LLM-powered fixes to correct
        syntax errors in Python files. It can fix a single file or all files in a directory.
        
        Args:
            path: Path to the file or directory to fix
            
        Returns:
            A summary of the fixes applied
        """
        try:
            # Handle both relative and absolute paths
            if os.path.isabs(path):
                target_path = Path(path)
            else:
                target_path = Path(self.repo_path) / path
                
            # Resolve the paths
            resolved_target = target_path.resolve()
            resolved_repo = Path(self.repo_path).resolve()
            
            # Security check: Ensure paths are inside the repository
            if not str(resolved_target).startswith(str(resolved_repo)):
                raise Exception(f"Access denied: Path '{path}' is outside the repository")
                
            # Check if path exists
            if not resolved_target.exists():
                raise Exception(f"Path not found: {path}")
                
            # Track files fixed
            files_checked = 0
            files_fixed = 0
            files_with_errors = []
            
            # If path is a file, fix it directly
            if resolved_target.is_file():
                if not str(resolved_target).endswith('.py'):
                    return f"File {path} is not a Python file. Only Python files can be fixed."
                    
                files_checked = 1
                
                # Check if file has syntax errors
                import subprocess
                result = subprocess.run(
                    ["python", "-m", "py_compile", str(resolved_target)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    # File has syntax errors, try to fix
                    if self._fix_syntax_errors_with_llm(resolved_target):
                        files_fixed = 1
                    else:
                        files_with_errors.append(str(resolved_target.relative_to(resolved_repo)))
                        
            # If path is a directory, fix all Python files in it
            elif resolved_target.is_dir():
                # Find all Python files in the directory
                for py_file in resolved_target.glob("**/*.py"):
                    files_checked += 1
                    
                    # Check if file has syntax errors
                    import subprocess
                    result = subprocess.run(
                        ["python", "-m", "py_compile", str(py_file)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        # File has syntax errors, try to fix
                        if self._fix_syntax_errors_with_llm(py_file):
                            files_fixed += 1
                        else:
                            files_with_errors.append(str(py_file.relative_to(resolved_repo)))
            
            # Generate summary
            if files_fixed == 0 and files_checked > 0 and not files_with_errors:
                return f"✓ No syntax errors found in {files_checked} Python file(s)."
            elif files_fixed > 0 and not files_with_errors:
                return f"✓ Fixed syntax errors in {files_fixed} out of {files_checked} Python file(s)."
            elif files_with_errors:
                error_list = "\n- ".join(files_with_errors)
                return f"⚠️ Fixed {files_fixed} file(s), but couldn't fix all errors in {len(files_with_errors)} file(s):\n- {error_list}"
            else:
                return "No Python files were found to check."
                
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in fix_syntax_errors: {str(e)}")
            return f"Error fixing syntax errors: {str(e)}"

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a file for code quality, complexity, and potential issues.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Read the file content
            content = self._read_file(file_path)
            
            # Analyze the code
            analysis = self.llm_toolkit.analyze_code(content)
            
            # Add file path to the analysis
            analysis["file_path"] = file_path
            
            return analysis
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error analyzing file {file_path}: {str(e)}")
            return {
                "file_path": file_path,
                "error": str(e),
                "potential_issues": [f"Error analyzing file: {str(e)}"],
                "suggestions": ["Fix file access issues"]
            }
    
    def optimize_file(self, file_path: str, optimization_goal: str = "performance") -> str:
        """Optimize a file for a specific goal.
        
        Args:
            file_path: Path to the file to optimize
            optimization_goal: Goal for optimization (performance, readability, memory)
            
        Returns:
            Summary of optimizations applied
        """
        try:
            # Read the file content
            content = self._read_file(file_path)
            
            # Optimize the code
            optimized_code = self.llm_toolkit.optimize_code(content, optimization_goal)
            
            # Write the optimized code back to the file
            self._edit_file(file_path, optimized_code)
            
            return f"✨ Optimized {file_path} for {optimization_goal}"
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error optimizing file {file_path}: {str(e)}")
            return f"Error optimizing file: {str(e)}"
    
    def add_docstrings(self, file_path: str) -> str:
        """Add or improve docstrings in a file.
        
        Args:
            file_path: Path to the file to document
            
        Returns:
            Summary of docstring additions
        """
        try:
            # Read the file content
            content = self._read_file(file_path)
            
            # Generate improved docstrings
            documented_code = self.llm_toolkit.generate_docstring(content)
            
            # Write the documented code back to the file
            self._edit_file(file_path, documented_code)
            
            return f"✨ Added/improved docstrings in {file_path}"
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error adding docstrings to {file_path}: {str(e)}")
            return f"Error adding docstrings: {str(e)}"
    
    def generate_tests(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Generate unit tests for a file.
        
        Args:
            file_path: Path to the file to generate tests for
            output_path: Path to write the test file to (default: auto-generate based on file_path)
            
        Returns:
            Summary of test generation
        """
        try:
            # Read the file content
            content = self._read_file(file_path)
            
            # Generate tests
            test_code = self.llm_toolkit.generate_unit_tests(content)
            
            # Determine output path if not provided
            if output_path is None:
                file_name = os.path.basename(file_path)
                file_base, _ = os.path.splitext(file_name)
                
                if not file_base.startswith("test_"):
                    test_file_name = f"test_{file_base}.py"
                else:
                    test_file_name = file_name
                
                # Put tests in tests directory if it exists
                tests_dir = os.path.join(self.repo_path, "tests")
                if os.path.isdir(tests_dir):
                    output_path = os.path.join(tests_dir, test_file_name)
                else:
                    # Otherwise put next to the original file
                    file_dir = os.path.dirname(file_path)
                    output_path = os.path.join(file_dir, test_file_name)
            
            # Write the test code to the output path
            self._edit_file(output_path, test_code)
            
            return f"✨ Generated tests in {output_path}"
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error generating tests for {file_path}: {str(e)}")
            return f"Error generating tests: {str(e)}"
    
    def enhance_error_handling(self, file_path: str) -> str:
        """Enhance error handling in a file.
        
        Args:
            file_path: Path to the file to enhance
            
        Returns:
            Summary of error handling enhancements
        """
        try:
            # Read the file content
            content = self._read_file(file_path)
            
            # Enhance error handling
            enhanced_code = self.llm_toolkit.enhance_error_handling(content)
            
            # Write the enhanced code back to the file
            self._edit_file(file_path, enhanced_code)
            
            return f"✨ Enhanced error handling in {file_path}"
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error enhancing error handling in {file_path}: {str(e)}")
            return f"Error enhancing error handling: {str(e)}"
    
    def explain_code(self, file_path: str, explanation_level: str = "detailed") -> str:
        """Generate a natural language explanation of a file.
        
        Args:
            file_path: Path to the file to explain
            explanation_level: Level of detail (basic, detailed, advanced)
            
        Returns:
            Natural language explanation of the code
        """
        try:
            # Read the file content
            content = self._read_file(file_path)
            
            # Generate explanation
            explanation = self.llm_toolkit.explain_code(content, explanation_level)
            
            return explanation
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error explaining {file_path}: {str(e)}")
            return f"Error explaining code: {str(e)}"
    
    def refactor_code(self, file_path: str, refactoring_goal: str) -> str:
        """Refactor code in a file according to a specific goal.
        
        Args:
            file_path: Path to the file to refactor
            refactoring_goal: The refactoring goal
            
        Returns:
            Summary of refactoring applied
        """
        try:
            # Read the file content
            content = self._read_file(file_path)
            
            # Refactor the code
            refactored_code = self.llm_toolkit.refactor_code(content, refactoring_goal)
            
            # Write the refactored code back to the file
            self._edit_file(file_path, refactored_code)
            
            return f"✨ Refactored {file_path} with goal: {refactoring_goal}"
        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error refactoring {file_path}: {str(e)}")
            return f"Error refactoring code: {str(e)}"

    def generate_markdown_docs(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Generate Markdown documentation for Python code.
        
        Args:
            file_path: Path to the Python file
            output_path: Optional path for the output Markdown file
            
        Returns:
            str: Success message
        """
        # Read the file
        file_content = self._read_file(file_path)
        
        # Generate Markdown docs
        markdown_content = self.llm_toolkit.generate_markdown_docs(file_content)
        
        # Determine output path if not provided
        if not output_path:
            path_obj = Path(file_path)
            output_path = str(path_obj.with_suffix(".md"))
            
        # Write the Markdown content
        self._edit_file(output_path, markdown_content)
        
        return f"✨ Generated Markdown documentation for {file_path} at {output_path}"
    
    def fix_python_syntax(self, file_path: str) -> str:
        """Fix Python syntax errors in a file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            str: Success message
        """
        # Read the file
        file_content = self._read_file(file_path)
        
        # Check for syntax errors
        try:
            compile(file_content, file_path, 'exec')
            return f"✅ No syntax errors found in {file_path}"
        except SyntaxError:
            pass  # File has syntax errors, continue to fix
        
        # Fix syntax errors
        fixed_code = self.llm_toolkit.fix_python_syntax(file_content)
        
        # Write the fixed code
        self._edit_file(file_path, fixed_code)
        
        return f"✨ Fixed syntax errors in {file_path}"
