import json
import os
from pathlib import Path

import pytest

from coder.agent import CodingAgent


@pytest.fixture
def test_workspace(tmp_path):
    """Create a minimal test workspace with various file types and structures."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()

    # Create a simple project structure
    (workspace / "src").mkdir()
    (workspace / "tests").mkdir()
    (workspace / "docs").mkdir()

    # Create some test files
    (workspace / "src" / "main.txt").write_text("Main content\n")
    (workspace / "src" / "config.json").write_text('{"key": "value"}\n')
    (workspace / "tests" / "test.txt").write_text("Test content\n")
    (workspace / "docs" / "readme.txt").write_text("Documentation\n")

    return workspace


@pytest.fixture
def api_key():
    """Return API key for testing."""
    return os.getenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def agent(test_workspace, api_key):
    """Create a CodingAgent instance for the test workspace."""
    return CodingAgent(repo_path=str(test_workspace), api_key=api_key)


@pytest.mark.integration
class TestFileOperations:
    """Test file operations."""

    def test_read_file_basic(self, agent):
        """Test basic file reading functionality."""
        content = agent._read_file("src/main.txt")
        assert "Main content" in content

    def test_read_file_json(self, agent):
        """Test reading and parsing JSON files."""
        content = agent._read_file("src/config.json")
        data = json.loads(content)
        assert data["key"] == "value"

    def test_read_file_nonexistent(self, agent):
        """Test reading non-existent file."""
        with pytest.raises(Exception):
            agent._read_file("nonexistent.txt")

    def test_edit_file_create(self, agent):
        """Test creating a new file."""
        content = "New file content\n"
        success = agent._edit_file("src/new.txt", content)
        assert success
        assert agent._read_file("src/new.txt") == content

    def test_edit_file_update(self, agent):
        """Test updating an existing file."""
        new_content = "Updated content\n"
        success = agent._edit_file("src/main.txt", new_content)
        assert success
        assert agent._read_file("src/main.txt") == new_content

    def test_edit_file_create_nested(self, agent):
        """Test creating a file in a nested directory."""
        content = "Nested content\n"
        success = agent._edit_file("src/nested/deep/file.txt", content)
        assert success
        assert agent._read_file("src/nested/deep/file.txt") == content


@pytest.mark.integration
class TestDirectoryOperations:
    """Test directory operations."""

    def test_list_directory_root(self, agent):
        """Test listing root directory contents."""
        contents = agent._list_directory(".")
        assert "src" in contents
        assert "tests" in contents
        assert "docs" in contents

    def test_list_directory_subdir(self, agent):
        """Test listing subdirectory contents."""
        contents = agent._list_directory("src")
        assert "main.txt" in contents
        assert "config.json" in contents

    def test_list_directory_nonexistent(self, agent):
        """Test listing non-existent directory."""
        with pytest.raises(Exception):
            agent._list_directory("nonexistent")


@pytest.mark.integration
class TestToolCallExecution:
    """Test tool call execution."""

    def test_execute_read_file_tool(self, agent):
        """Test executing read_file tool call."""
        tool_call = {
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"target_file": "src/main.txt"}),
            }
        }
        result = agent._execute_tool_call(tool_call)
        assert "Main content" in result

    def test_execute_list_directory_tool(self, agent):
        """Test executing list_directory tool call."""
        tool_call = {
            "function": {
                "name": "list_directory",
                "arguments": json.dumps({"relative_workspace_path": "src"}),
            }
        }
        result = agent._execute_tool_call(tool_call)
        contents = json.loads(result)
        assert "main.txt" in contents
        assert "config.json" in contents

    def test_execute_edit_file_tool(self, agent):
        """Test executing edit_file tool call."""
        tool_call = {
            "function": {
                "name": "edit_file",
                "arguments": json.dumps(
                    {
                        "target_file": "src/main.txt",
                        "instructions": "Update content",
                        "code_edit": "Updated via tool\n",
                    }
                ),
            }
        }
        result = agent._execute_tool_call(tool_call)
        assert result == "File edited successfully"
        assert agent._read_file("src/main.txt") == "Updated via tool\n"

    def test_execute_invalid_tool(self, agent):
        """Test executing invalid tool call."""
        tool_call = {"function": {"name": "invalid_tool", "arguments": json.dumps({})}}
        with pytest.raises(ValueError, match="Unknown tool: invalid_tool"):
            agent._execute_tool_call(tool_call)


@pytest.mark.integration
class TestToolInteraction:
    """Test tool interactions."""

    def test_read_then_edit(self, agent):
        """Test reading a file and then editing it."""
        # First read
        content = agent._read_file("src/main.txt")
        assert "Main content" in content

        # Then edit
        new_content = f"{content.strip()}\nAdded content\n"
        success = agent._edit_file("src/main.txt", new_content)
        assert success

        # Verify
        updated = agent._read_file("src/main.txt")
        assert "Main content" in updated
        assert "Added content" in updated

    def test_list_then_create(self, agent):
        """Test listing directory and then creating a file."""
        # First list
        contents = agent._list_directory("src")
        assert "main.txt" in contents

        # Then create new file
        success = agent._edit_file("src/new.txt", "New file\n")
        assert success

        # Verify new file appears in listing
        updated_contents = agent._list_directory("src")
        assert "new.txt" in updated_contents

    def test_complex_workflow(self, agent):
        """Test a complex workflow using multiple tools."""
        # Create a new directory structure
        success = agent._edit_file("src/newdir/file1.txt", "File 1\n")
        assert success
        success = agent._edit_file("src/newdir/file2.txt", "File 2\n")
        assert success

        # List the new directory
        contents = agent._list_directory("src/newdir")
        assert "file1.txt" in contents
        assert "file2.txt" in contents

        # Read and update files
        content1 = agent._read_file("src/newdir/file1.txt")
        content2 = agent._read_file("src/newdir/file2.txt")

        success = agent._edit_file(
            "src/newdir/file1.txt", f"{content1.strip()}\nUpdated 1\n"
        )
        assert success
        success = agent._edit_file(
            "src/newdir/file2.txt", f"{content2.strip()}\nUpdated 2\n"
        )
        assert success

        # Verify final state
        final1 = agent._read_file("src/newdir/file1.txt")
        final2 = agent._read_file("src/newdir/file2.txt")
        assert "File 1" in final1
        assert "Updated 1" in final1
        assert "File 2" in final2
        assert "Updated 2" in final2
