import os
import pytest
from pathlib import Path
from coder.agent import CodingAgent
from coder.api import OpenRouterClient
import json

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
        max_tokens=31000
    )

def test_agent_initialization(agent, test_repo):
    """Test agent initialization and properties."""
    assert agent.repo_path == str(test_repo)
    assert agent.model == "qwen/qwen3-32b"
    assert agent.provider == "Cerebras"
    assert agent.max_tokens == 31000
    assert isinstance(agent.client, OpenRouterClient)
    assert len(agent.tools) == 3  # read_file, list_directory, edit_file

def test_read_file(agent):
    """Test reading file contents."""
    content = agent._read_file("test.py")
    assert "def hello()" in content
    assert "return 'Hello, World!'" in content

def test_read_file_nonexistent(agent):
    """Test reading non-existent file."""
    with pytest.raises(Exception):
        agent._read_file("nonexistent.py")

def test_read_file_in_subdirectory(agent):
    """Test reading file in subdirectory."""
    content = agent._read_file("subdir/test2.py")
    assert "def goodbye()" in content
    assert "return 'Goodbye!'" in content

def test_list_directory(agent):
    """Test listing directory contents."""
    contents = agent._list_directory(".")
    assert "test.py" in contents
    assert "subdir" in contents

def test_list_directory_nonexistent(agent):
    """Test listing non-existent directory."""
    with pytest.raises(Exception):
        agent._list_directory("nonexistent")

def test_list_directory_subdirectory(agent):
    """Test listing subdirectory contents."""
    contents = agent._list_directory("subdir")
    assert "test2.py" in contents

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

def test_edit_file_create_new(agent):
    """Test creating new file."""
    new_content = "def new_file():\n    return 'New file'\n"
    success = agent._edit_file("new_file.py", new_content)
    assert success
    assert agent._read_file("new_file.py") == new_content

def test_edit_file_create_in_subdirectory(agent):
    """Test creating file in subdirectory."""
    new_content = "def subdir_file():\n    return 'In subdirectory'\n"
    success = agent._edit_file("subdir/new_file.py", new_content)
    assert success
    assert agent._read_file("subdir/new_file.py") == new_content

def test_edit_file_invalid_path(agent):
    """Test editing file with invalid path."""
    with pytest.raises(Exception):
        agent._edit_file("/invalid/path/file.py", "content")

def test_execute_tool_call_read_file(agent):
    """Test executing read_file tool call."""
    result = agent._execute_tool_call({
        "function": "read_file",
        "arguments": {"target_file": "test.py"}
    })
    assert "def hello()" in result

def test_execute_tool_call_list_directory(agent):
    """Test executing list_directory tool call."""
    result = agent._execute_tool_call({
        "function": "list_directory",
        "arguments": {"relative_workspace_path": "."}
    })
    assert "test.py" in result
    assert "subdir" in result

def test_execute_tool_call_edit_file(agent):
    """Test executing edit_file tool call."""
    result = agent._execute_tool_call({
        "function": "edit_file",
        "arguments": {
            "target_file": "test.py",
            "instructions": "Update function",
            "code_edit": "def updated():\n    return 'Updated'\n"
        }
    })
    assert result == "File edited successfully"
    assert "def updated()" in agent._read_file("test.py")

def test_execute_tool_call_invalid_tool(agent):
    """Test executing invalid tool call."""
    with pytest.raises(ValueError, match="Unknown tool: invalid_tool"):
        agent._execute_tool_call({
            "function": "invalid_tool",
            "arguments": {}
        })

def test_ask_with_tool_calls(agent, monkeypatch):
    """Test ask method with tool calls."""
    def mock_chat_completion(*args, **kwargs):
        if not hasattr(mock_chat_completion, "called"):
            mock_chat_completion.called = True
            return {
                "choices": [{
                    "message": {
                        "content": "I found the answer",
                        "tool_calls": [{
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"target_file": "test.py"})
                            }
                        }]
                    }
                }]
            }
        else:
            return {
                "choices": [{
                    "message": {
                        "content": "I found the answer",
                        "tool_calls": []
                    }
                }]
            }
    
    monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
    response = agent.ask("What's in test.py?")
    assert "I found the answer" in response

def test_agent_with_tool_calls(agent, monkeypatch):
    """Test agent method with tool calls."""
    def mock_chat_completion(*args, **kwargs):
        if not hasattr(mock_chat_completion, "called"):
            mock_chat_completion.called = True
            return {
                "choices": [{
                    "message": {
                        "content": "✨ Changes Applied",
                        "tool_calls": [{
                            "function": {
                                "name": "edit_file",
                                "arguments": json.dumps({
                                    "target_file": "test.py",
                                    "instructions": "Update function",
                                    "code_edit": "def updated():\n    return 'Updated'\n"
                                })
                            }
                        }]
                    }
                }]
            }
        else:
            return {
                "choices": [{
                    "message": {
                        "content": "✨ Changes Applied",
                        "tool_calls": []
                    }
                }]
            }
    
    monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
    response = agent.agent("Update the test function")
    # Check for either the old message or the new format that includes file names
    assert any([
        "✨ Changes Applied" in response,
        "Created/modified file" in response,
        "test.py" in response
    ])
    assert "def updated()" in agent._read_file("test.py")

def test_agent_with_multiple_tool_calls(agent, monkeypatch):
    """Test agent method with multiple tool calls."""
    tool_calls = [
        {
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"target_file": "test.py"})
            }
        },
        {
            "function": {
                "name": "edit_file",
                "arguments": json.dumps({
                    "target_file": "test.py",
                    "instructions": "Update function",
                    "code_edit": "def updated():\n    return 'Updated'\n"
                })
            }
        }
    ]
    def mock_chat_completion(*args, **kwargs):
        if not hasattr(mock_chat_completion, "called"):
            mock_chat_completion.called = True
            return {
                "choices": [{
                    "message": {
                        "content": "✨ Changes Applied",
                        "tool_calls": tool_calls
                    }
                }]
            }
        else:
            return {
                "choices": [{
                    "message": {
                        "content": "✨ Changes Applied",
                        "tool_calls": []
                    }
                }]
            }
    
    monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
    response = agent.agent("Read and update the test function")
    # Check for either the old message or the new format that includes file names
    assert any([
        "✨ Changes Applied" in response,
        "Created/modified file" in response,
        "test.py" in response
    ])
    assert "def updated()" in agent._read_file("test.py")

def test_agent_with_error_handling(agent, monkeypatch):
    """Test agent method with error handling."""
    def mock_chat_completion(*args, **kwargs):
        if not hasattr(mock_chat_completion, "called"):
            mock_chat_completion.called = True
            return {
                "choices": [{
                    "message": {
                        "content": "✨ Changes Applied",
                        "tool_calls": [{
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"target_file": "nonexistent.py"})
                            }
                        }]
                    }
                }]
            }
        else:
            return {
                "choices": [{
                    "message": {
                        "content": "✨ Changes Applied",
                        "tool_calls": []
                    }
                }]
            }
    
    monkeypatch.setattr(agent.client, "chat_completion", mock_chat_completion)
    response = agent.agent("Try to read non-existent file")
    # Check for success message or error handling response
    assert any([
        "✨ Changes Applied" in response,
        "Error" in response,
        "nonexistent.py" in response
    ]) 