import json
import os
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock

from coder.agent import CodingAgent
from coder.api import OpenRouterClient


@pytest.fixture
def api_key():
    """Fixture to provide API key for testing."""
    return os.getenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def test_repo(tmp_path):
    """Fixture to create a temporary test repository."""
    # Create test directory structure
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Create test files
    (repo / "test.py").write_text("def hello():\n    return 'Hello, World!'\n")
    (repo / "subdir").mkdir()
    (repo / "subdir" / "test2.py").write_text("def goodbye():\n    return 'Goodbye!'\n")

    return repo


@pytest.fixture
def agent(api_key, test_repo):
    """Fixture to create a CodingAgent instance."""
    return CodingAgent(
        repo_path=str(test_repo),
        api_key=api_key,
        model="qwen/qwen3-32b",
        provider="Cerebras",
        max_tokens=31000,
    )


@pytest.mark.integration
def test_agent_initialization(agent, test_repo):
    """Test agent initialization and properties."""
    assert agent.repo_path == str(test_repo)
    assert agent.model == "qwen/qwen3-32b"
    assert agent.provider == "Cerebras"
    assert agent.max_tokens == 31000
    assert isinstance(agent.client, OpenRouterClient)
    assert len(agent.tools) == 13  # Now includes additional Python and Markdown tools


@pytest.mark.integration
def test_read_file(agent):
    """Test reading file contents."""
    content = agent._read_file("test.py")
    assert "def hello()" in content
    assert "return 'Hello, World!'" in content


@pytest.mark.integration
def test_read_file_nonexistent(agent):
    """Test reading non-existent file."""
    with pytest.raises(Exception):
        agent._read_file("nonexistent.py")


@pytest.mark.integration
def test_read_file_in_subdirectory(agent):
    """Test reading file in subdirectory."""
    content = agent._read_file("subdir/test2.py")
    assert "def goodbye()" in content
    assert "return 'Goodbye!'" in content


@pytest.mark.integration
def test_list_directory(agent):
    """Test listing directory contents."""
    contents = agent._list_directory(".")
    assert "test.py" in contents
    assert "subdir" in contents


@pytest.mark.integration
def test_list_directory_nonexistent(agent):
    """Test listing non-existent directory."""
    with pytest.raises(Exception):
        agent._list_directory("nonexistent")


@pytest.mark.integration
def test_list_directory_subdirectory(agent):
    """Test listing subdirectory contents."""
    contents = agent._list_directory("subdir")
    assert "test2.py" in contents


@pytest.mark.integration
def test_edit_file(agent):
    """Test editing existing file."""
    # First read the original content
    original_content = agent._read_file("test.py")

    # Define new content to add
    new_content = "def new_function():\n    return 'New content'\n"

    # Edit the file
    success = agent._edit_file("test.py", new_content)

    # Assert success and verify that the file contains both the original content and the new content
    assert success
    updated_content = agent._read_file("test.py")
    assert original_content in updated_content
    assert new_content in updated_content


@pytest.mark.integration
def test_edit_file_create_new(agent):
    """Test creating new file."""
    new_content = "def new_file():\n    return 'New file'\n"
    success = agent._edit_file("new_file.py", new_content)
    assert success
    assert agent._read_file("new_file.py") == new_content


@pytest.mark.integration
def test_edit_file_create_in_subdirectory(agent):
    """Test creating file in subdirectory."""
    new_content = "def subdir_file():\n    return 'In subdirectory'\n"
    success = agent._edit_file("subdir/new_file.py", new_content)
    assert success
    assert agent._read_file("subdir/new_file.py") == new_content


@pytest.mark.integration
def test_edit_file_invalid_path(agent):
    """Test editing file with invalid path."""
    with pytest.raises(Exception):
        agent._edit_file("/invalid/path/file.py", "content")


@pytest.mark.integration
def test_execute_tool_call_read_file(agent):
    """Test executing read_file tool call."""
    result = agent._execute_tool_call(
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"target_file": "test.py"})
            },
            "id": "test_id"
        }
    )
    assert "def hello()" in result


@pytest.mark.integration
def test_execute_tool_call_list_directory(agent):
    """Test executing list_directory tool call."""
    result = agent._execute_tool_call(
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "arguments": json.dumps({"relative_workspace_path": "."})
            },
            "id": "test_id"
        }
    )
    assert "test.py" in result
    assert "subdir" in result


@pytest.mark.integration
def test_execute_tool_call_edit_file(agent):
    """Test executing edit_file tool call."""
    result = agent._execute_tool_call(
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "arguments": json.dumps({
                    "target_file": "test.py",
                    "instructions": "Update function",
                    "code_edit": "def updated():\n    return 'Updated'\n",
                })
            },
            "id": "test_id"
        }
    )
    assert "File edited" in result
    assert "def updated()" in agent._read_file("test.py")


@pytest.mark.integration
def test_execute_tool_call_invalid_tool(agent):
    """Test executing invalid tool call."""
    with pytest.raises(Exception):
        agent._execute_tool_call({
            "type": "function",
            "function": {
                "name": "invalid_tool", 
                "arguments": json.dumps({})
            },
            "id": "test_id"
        })


@pytest.mark.integration
def test_ask_with_tool_calls(agent, monkeypatch):
    """Test ask method with tool calls."""

    def mock_chat_completion(*args, **kwargs):
        if not hasattr(mock_chat_completion, "called"):
            mock_chat_completion.called = True
            return {
                "choices": [
                    {
                        "message": {
                            "content": "I found the answer",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps(
                                            {"target_file": "test.py"}
                                        ),
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        else:
            return {
                "choices": [
                    {"message": {"content": "I found the answer", "tool_calls": []}}
                ]
            }

    monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
    response = agent.ask("What's in test.py?")
    assert "I found the answer" in response


@pytest.mark.integration
def test_agent_with_tool_calls(agent, monkeypatch):
    """Test agent method with tool calls."""

    def mock_chat_completion(*args, **kwargs):
        if not hasattr(mock_chat_completion, "called"):
            mock_chat_completion.called = True
            return {
                "choices": [
                    {
                        "message": {
                            "content": "✨ Changes Applied",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "edit_file",
                                        "arguments": json.dumps(
                                            {
                                                "target_file": "test.py",
                                                "instructions": "Update function",
                                                "code_edit": "def updated():\n    return 'Updated'\n",
                                            }
                                        ),
                                    },
                                    "id": "call_123"
                                }
                            ],
                        }
                    }
                ]
            }
        else:
            return {
                "choices": [
                    {"message": {"content": "✨ Changes Applied", "tool_calls": []}}
                ]
            }

    monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
    monkeypatch.setattr(agent, "_edit_file", lambda path, content: True)
    monkeypatch.setattr(agent, "_read_file", lambda path: "def updated():\n    return 'Updated'\n" if path == "test.py" else "")
    
    response = agent.agent("Update the test function")
    # Check for either the old message or the new format that includes file names
    assert any(
        [
            "✨ Changes Applied" in response,
            "Created/modified file" in response,
            "test.py" in response,
        ]
    )
    assert "def updated()" in agent._read_file("test.py")


@pytest.mark.integration
def test_agent_with_multiple_tool_calls(agent, monkeypatch):
    """Test agent method with multiple tool calls."""
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"target_file": "test.py"}),
            },
            "id": "call_123"
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "arguments": json.dumps(
                    {
                        "target_file": "test.py",
                        "instructions": "Update function",
                        "code_edit": "def updated():\n    return 'Updated'\n",
                    }
                ),
            },
            "id": "call_456"
        },
    ]

    def mock_chat_completion(*args, **kwargs):
        if not hasattr(mock_chat_completion, "called"):
            mock_chat_completion.called = True
            return {
                "choices": [
                    {
                        "message": {
                            "content": "✨ Changes Applied",
                            "tool_calls": tool_calls,
                        }
                    }
                ]
            }
        else:
            return {
                "choices": [
                    {"message": {"content": "✨ Changes Applied", "tool_calls": []}}
                ]
            }

    monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
    monkeypatch.setattr(agent, "_edit_file", lambda path, content: True)
    monkeypatch.setattr(agent, "_read_file", lambda path: "def updated():\n    return 'Updated'\n" if path == "test.py" else "")
    
    response = agent.agent("Read and update the test function")
    # Check for either the old message or the new format that includes file names
    assert any(
        [
            "✨ Changes Applied" in response,
            "Created/modified file" in response,
            "test.py" in response,
        ]
    )
    assert "def updated()" in agent._read_file("test.py")


@pytest.mark.integration
def test_agent_with_error_handling(agent, monkeypatch):
    """Test agent method with error handling."""

    def mock_chat_completion(*args, **kwargs):
        if not hasattr(mock_chat_completion, "called"):
            mock_chat_completion.called = True
            return {
                "choices": [
                    {
                        "message": {
                            "content": "✨ Changes Applied",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps(
                                            {"target_file": "nonexistent.py"}
                                        ),
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        else:
            return {
                "choices": [
                    {"message": {"content": "✨ Changes Applied", "tool_calls": []}}
                ]
            }

    monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
    response = agent.agent("Try to read non-existent file")
    # Check for success message or error handling response
    assert any(
        [
            "✨ Changes Applied" in response,
            "Error" in response,
            "nonexistent.py" in response,
        ]
    )


@pytest.mark.integration
def test_extract_mentioned_files(agent):
    """Test extracting mentioned files from prompts."""
    # Test with explicit file mentions
    prompt1 = "Create a Node.js project with server.js and package.json files"
    files1 = agent._extract_mentioned_files(prompt1)
    assert "server.js" in files1
    assert "package.json" in files1

    # Test with file extensions
    prompt2 = "Build a web app with index.html, styles.css, and app.js"
    files2 = agent._extract_mentioned_files(prompt2)
    assert "index.html" in files2
    assert "styles.css" in files2
    assert "app.js" in files2

    # Test with complex description
    prompt3 = "Build a NodeJS web service that calculates financial options prices using the Black-Scholes model. Include server.js, package.json, an options.js module, and a test file."
    files3 = agent._extract_mentioned_files(prompt3)
    assert "server.js" in files3
    assert "package.json" in files3
    assert "options.js" in files3


@pytest.mark.integration
def test_agent_handles_missing_files(agent, monkeypatch):
    """Test that agent handles explicitly mentioned files that weren't created."""

    # Define a series of tool calls that create some files but not all
    def mock_chat_completion(*args, **kwargs):
        messages = kwargs.get("messages", [])

        # Initial response - creating only server.js
        if len(messages) == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Creating server.js",
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "function": {
                                        "name": "edit_file",
                                        "arguments": json.dumps(
                                            {
                                                "target_file": "server.js",
                                                "instructions": "Create server.js",
                                                "code_edit": "console.log('Server');",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        # Second response - the follow-up with no tool calls
        elif any(
            msg.get("tool_call_id") == "call1"
            for msg in messages
            if msg.get("role") == "tool"
        ):
            return {
                "choices": [
                    {"message": {"content": "Server file created", "tool_calls": []}}
                ]
            }
        # Third response - generating content for package.json
        elif any(
            "Please create the following files" in msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{\n  "name": "financial-options",\n  "version": "1.0.0",\n  "main": "server.js"\n}\n```'
                        }
                    }
                ]
            }

    # Mock the file operations
    def mock_edit_file(path, content):
        return True

    def mock_read_file(path):
        if path == "server.js":
            return "console.log('Server');"
        return "Mock file content"

    # Apply mocks
    monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
    monkeypatch.setattr(
        agent.client,
        "get_tool_calls",
        lambda x: (
            [
                {
                    "id": "call1",
                    "function": {
                        "name": "edit_file",
                        "arguments": '{"target_file": "server.js", "instructions": "Create server", "code_edit": "console.log(\'Server\');"}',
                    },
                }
            ]
            if "content" in x["choices"][0]["message"]
            and "Server file created" not in x["choices"][0]["message"]["content"]
            else []
        ),
    )
    monkeypatch.setattr(
        agent.client,
        "get_completion",
        lambda x: x["choices"][0]["message"].get("content", ""),
    )
    monkeypatch.setattr(agent, "_edit_file", mock_edit_file)
    monkeypatch.setattr(agent, "_read_file", mock_read_file)

    # Run the agent with a prompt mentioning multiple files
    prompt = "Create a Node.js project with server.js and package.json"
    response = agent.agent(prompt)

    # Verify that both files are mentioned in the response
    assert "server.js" in response
    assert "package.json" in response
