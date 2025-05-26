"""Tool definitions for the CodingAgent."""

from typing import Dict, List


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
                            "description": "Path to the file to read",
                        }
                    },
                    "required": ["target_file"],
                },
            },
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
                            "description": "Path to the directory to list",
                        }
                    },
                    "required": ["relative_workspace_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit or create a file. You can call this function multiple times to create or modify multiple files for the same project (e.g., create server.js, package.json, README.md, and test files in a single response). You should aim to create all necessary files for a complete project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to edit or create",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Instructions for the edit or file creation",
                        },
                        "code_edit": {
                            "type": "string",
                            "description": "New content for the file",
                        },
                    },
                    "required": ["target_file", "instructions", "code_edit"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fix_syntax_errors",
                "description": "Fix syntax errors in Python files using both heuristic and LLM-powered methods. Can fix a single file or all Python files in a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Path to the file or directory to check and fix",
                        }
                    },
                    "required": ["target_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_code",
                "description": "Analyze a file for code quality, complexity, and potential issues",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to analyze",
                        }
                    },
                    "required": ["target_file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "optimize_code",
                "description": "Optimize a file for a specific goal like performance, readability, or memory usage",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to optimize",
                        },
                        "optimization_goal": {
                            "type": "string",
                            "description": "Goal for optimization (performance, readability, memory)",
                            "enum": ["performance", "readability", "memory"],
                        }
                    },
                    "required": ["target_file", "optimization_goal"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_docstrings",
                "description": "Add or improve docstrings in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to document",
                        }
                    },
                    "required": ["target_file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_tests",
                "description": "Generate unit tests for a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to generate tests for",
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path to write the test file to (optional, will auto-generate if not provided)",
                        }
                    },
                    "required": ["target_file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "enhance_error_handling",
                "description": "Enhance error handling in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to enhance",
                        }
                    },
                    "required": ["target_file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "explain_code",
                "description": "Generate a natural language explanation of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to explain",
                        },
                        "explanation_level": {
                            "type": "string",
                            "description": "Level of detail (basic, detailed, advanced)",
                            "enum": ["basic", "detailed", "advanced"],
                        }
                    },
                    "required": ["target_file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "refactor_code",
                "description": "Refactor a file according to a specific goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the file to refactor",
                        },
                        "refactoring_goal": {
                            "type": "string",
                            "description": "Goal for refactoring (e.g., 'extract method', 'apply factory pattern')",
                        }
                    },
                    "required": ["target_file", "refactoring_goal"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_markdown_docs",
                "description": "Generate comprehensive Markdown documentation for a Python file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the Python file to document",
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path to write the Markdown file to (optional, will auto-generate if not provided)",
                        }
                    },
                    "required": ["target_file"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fix_python_syntax",
                "description": "Fix Python syntax errors in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_file": {
                            "type": "string",
                            "description": "Path to the Python file to fix",
                        }
                    },
                    "required": ["target_file"],
                },
            },
        },
    ]
