"""Version management utilities for the coding agent."""
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VersionInfo:
    """Information about a version."""
    version: int
    path: Path
    specs_content: str
    is_valid: bool
    test_status: Optional[str] = None

class VersionManager:
    """Manages version creation and validation."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
    
    def get_current_version(self) -> int:
        """Get the current version number from SPECS.md."""
        specs_path = self.base_path / "SPECS.md"
        if not specs_path.exists():
            return 1
        
        try:
            content = specs_path.read_text()
            match = re.search(r'# Coding Agent v(\d+) Specification', content)
            return int(match.group(1)) if match else 1
        except Exception:
            return 1
    
    def get_next_version(self) -> int:
        """Get the next version number."""
        return self.get_current_version() + 1
    
    def create_version_directory(self, version: int) -> Path:
        """Create a new version directory."""
        version_dir = self.base_path / f"version{version}"
        if version_dir.exists():
            raise ValueError(f"Version {version} directory already exists")
        
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Create necessary subdirectories
        (version_dir / "coder").mkdir(exist_ok=True)
        (version_dir / "coder" / "utils").mkdir(exist_ok=True)
        (version_dir / "tests").mkdir(exist_ok=True)
        
        return version_dir
    
    def get_files_to_generate(self) -> List[str]:
        """Get the list of files that need to be generated for a new version."""
        return [
            "README.md",
            "requirements.txt", 
            "setup.py",
            "coder/__init__.py",
            "coder/api.py",
            "coder/agent.py",
            "coder/cli.py",
            "coder/utils/__init__.py",
            "coder/utils/equivalence.py",
            "coder/utils/version_manager.py",
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/test_agent.py",
            "tests/test_equivalence.py",
            "tests/test_cli_integration.py",
        ] 