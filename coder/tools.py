"""Tool definitions for the CodingAgent."""
from typing import List, Dict

def get_agent_tools() -> List[Dict]:
    """Return the list of tools available to the agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to read"
                        }
                    },
                    "required": ["target_file"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List contents of a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relative_workspace_path": {
                            "type": "string",
                            "description": "Path to the directory to list"
                        }
                    },
                    "required": ["relative_workspace_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit or create a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to edit"
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Instructions for the edit"
                        },
                        "code_edit": {
                            "type": "string",
                            "description": "New content for the file"
                        }
                    },
                    "required": ["target_file", "instructions", "code_edit"]
                }
            }
        }
    ] 