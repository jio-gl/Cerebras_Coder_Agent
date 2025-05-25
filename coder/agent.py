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
from .utils import CodeValidator, EquivalenceChecker, VersionManager

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
        self.model = model
        self.debug = debug
        self.interactive = interactive
        # Don't store API key directly in the instance to prevent accidental exposure
        # Instead, just pass it to the client
        self._client = OpenRouterClient(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY")
        )
        self.no_think = no_think
        self.max_tokens = max_tokens
        self.provider = provider
        self.version_manager = VersionManager(Path(self.repo_path))

        # Validate model configuration
        if not self.model:
            self.model = "llama3.1-8b"

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
        """Execute a tool call and return the result."""
        try:
            # Extract function name and arguments
            if isinstance(tool_call["function"], str):
                function_name = tool_call["function"]
                arguments = tool_call["arguments"]
            else:
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

            # Parse arguments if they're a string
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            # Debug: Print tool call details
            if self.debug:
                print("\nDEBUG: Tool call details:")
                print(f"Function: {function_name}")
                print("Arguments:", arguments)

            # Execute the appropriate tool
            if function_name == "read_file":
                return self._read_file(arguments["target_file"])
            elif function_name == "list_directory":
                return json.dumps(
                    self._list_directory(arguments["relative_workspace_path"])
                )
            elif function_name == "edit_file":
                # Debug: Print edit file details
                if self.debug:
                    print("\nDEBUG: Edit file details:")
                    print(f"Target file: {arguments['target_file']}")
                    print("Instructions:", arguments["instructions"])
                    print("Code edit:", repr(arguments["code_edit"]))

                # Clean the code edit content for security
                safe_content = self._clean_code_output(arguments["code_edit"])

                # Debug: Print safe content after cleaning
                if self.debug:
                    print("\nDEBUG: Safe content after cleaning:")
                    print(repr(safe_content))

                success = self._edit_file(arguments["target_file"], safe_content)
                if not success:
                    raise Exception("Failed to edit file")

                # Debug: Print final file content
                if self.debug:
                    print("\nDEBUG: Final file content after edit:")
                    try:
                        if os.path.isabs(arguments["target_file"]):
                            file_path = arguments["target_file"]
                        else:
                            file_path = os.path.join(
                                self.repo_path, arguments["target_file"]
                            )
                        with open(file_path, "r") as f:
                            file_content = f.read()
                        print(repr(file_content))
                    except Exception as e:
                        print(f"Error reading final content: {str(e)}")

                return "File edited successfully"
            else:
                raise ValueError(f"Unknown tool: {function_name}")
        except ValueError as e:
            raise  # Re-raise ValueError to preserve the original exception
        except Exception as e:
            if self.debug:
                print("\nDEBUG: Error in _execute_tool_call:")
                print(str(e))
            raise Exception(f"Error executing {function_name}: {str(e)}")

    def _clean_code_output(self, content: str) -> str:
        """Clean and sanitize code output to prevent security issues.

        This method sanitizes code to remove potentially dangerous patterns:
        1. Removes dangerous system commands (rm -rf, curl to external sites)
        2. Sanitizes HTML and JavaScript code
        3. Handles special characters and binary data
        4. Limits input size to prevent DoS attacks

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
            import re

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
            
            Version: {version}
            
            Language: {language}
            
            Generate only the file content, no explanations.
            """

            # Request content from API
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant that generates high-quality, secure code based on specifications.",
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
            content = self.client.get_completion(response)

            # Clean and sanitize the content
            sanitized_content = self._clean_code_output(content)

            return sanitized_content

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in _generate_file_content: {str(e)}")
            return f"# Error generating content: {str(e)}"

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

            # Generate each file
            for file_path in files_to_generate:
                if self.debug:
                    print(f"\nDEBUG: Generating file: {file_path}")

                # Get file-specific instructions if available
                file_instructions = get_file_specific_instructions(file_path, version)

                # Generate file content
                file_content = self._generate_file_content(
                    file_path, specs + "\n\n" + file_instructions, version
                )

                # Write the file
                target_path = version_dir / file_path
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                with open(target_path, "w") as f:
                    f.write(file_content)

                # Sleep briefly to avoid rate limiting
                time.sleep(1)

            # Write the specs file directly
            with open(version_dir / "SPECS.md", "w") as f:
                f.write(specs)

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

            # Basic code validation
            validator = CodeValidator(version_dir)

            # Validate Python syntax
            validation_result = validator.validate_python_syntax()
            if not validation_result.success:
                if self.debug:
                    print(
                        f"\nDEBUG: Python syntax validation failed: {validation_result.error}"
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

            # Create version directory
            version_dir = self.version_manager.create_version_directory(next_version)

            # Create backup of current state
            backup_dir = self._create_backup()

            # Generate improved specifications
            specs = self._generate_improved_specs(current_version, next_version)

            # Write specs to the version directory
            specs_path = version_dir / "SPECS.md"
            with open(specs_path, "w") as f:
                f.write(specs)

            if self.debug:
                print(f"\nDEBUG: Generated improved specs for v{next_version}")

            # Generate version files
            file_generation_success = self._generate_version_files(
                version_dir, specs, next_version
            )
            if not file_generation_success:
                self._rollback(version_dir, backup_dir)
                return f"Self-rewrite failed during file generation. Backup preserved at {backup_dir}"

            if self.debug:
                print(f"\nDEBUG: Generated files for v{next_version}")

            # Validate new version
            validation_success = self._validate_new_version(version_dir)
            if not validation_success:
                self._rollback(version_dir, backup_dir)
                return f"Self-rewrite failed during validation. Backup preserved at {backup_dir}"

            if self.debug:
                print(f"\nDEBUG: Validated v{next_version}")

            # Generate completion summary
            summary = self._generate_completion_summary(next_version, version_dir)

            return summary

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in self_rewrite: {str(e)}")
            return f"Self-rewrite failed: {str(e)}"
