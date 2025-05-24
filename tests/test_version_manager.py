"""Tests for the VersionManager utility class."""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from coder.utils.version_manager import VersionManager, VersionInfo


class TestVersionManager:
    """Test class for VersionManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.vm = VersionManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init_with_default_path(self):
        """Test VersionManager initialization with default path."""
        vm = VersionManager()
        assert vm.base_path == Path.cwd()
    
    def test_init_with_custom_path(self):
        """Test VersionManager initialization with custom path."""
        custom_path = Path("/tmp/test")
        vm = VersionManager(custom_path)
        assert vm.base_path == custom_path
    
    def test_get_current_version_no_specs(self):
        """Test getting current version when SPECS.md doesn't exist."""
        version = self.vm.get_current_version()
        assert version == 1
    
    def test_get_current_version_with_specs(self):
        """Test getting current version when SPECS.md exists."""
        specs_content = "# Coding Agent v3 Specification\n\nSome content..."
        specs_path = self.temp_dir / "SPECS.md"
        specs_path.write_text(specs_content)
        
        version = self.vm.get_current_version()
        assert version == 3
    
    def test_get_current_version_invalid_specs(self):
        """Test getting current version with invalid SPECS.md."""
        specs_content = "# Some Random File\n\nNo version info..."
        specs_path = self.temp_dir / "SPECS.md"
        specs_path.write_text(specs_content)
        
        version = self.vm.get_current_version()
        assert version == 1
    
    def test_get_current_version_corrupted_specs(self):
        """Test getting current version with corrupted SPECS.md."""
        specs_path = self.temp_dir / "SPECS.md"
        # Write binary data that will cause read error
        specs_path.write_bytes(b'\x00\x01\x02\x03')
        
        version = self.vm.get_current_version()
        assert version == 1
    
    def test_get_next_version(self):
        """Test getting next version number."""
        # Mock current version to 5
        with patch.object(self.vm, 'get_current_version', return_value=5):
            next_version = self.vm.get_next_version()
            assert next_version == 6
    
    def test_create_version_directory_new(self):
        """Test creating a new version directory."""
        version_dir = self.vm.create_version_directory(2)
        
        assert version_dir.exists()
        assert version_dir.name == "version2"
        assert (version_dir / "coder").exists()
        assert (version_dir / "coder" / "utils").exists()
        assert (version_dir / "tests").exists()
    
    def test_create_version_directory_exists(self):
        """Test creating version directory when it already exists."""
        # Create directory first
        existing_dir = self.temp_dir / "version2"
        existing_dir.mkdir()
        
        with pytest.raises(ValueError, match="Version 2 directory already exists"):
            self.vm.create_version_directory(2)
    
    def test_get_files_to_generate(self):
        """Test getting list of files to generate."""
        files = self.vm.get_files_to_generate()
        
        expected_files = [
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
        
        assert all(f in files for f in expected_files)
        assert len(files) >= len(expected_files)


class TestVersionInfo:
    """Test class for VersionInfo dataclass."""
    
    def test_version_info_creation(self):
        """Test creating VersionInfo instance."""
        path = Path("/tmp/version1")
        specs = "# Version 1 specs"
        
        info = VersionInfo(
            version=1,
            path=path,
            specs_content=specs,
            is_valid=True,
            test_status="passed"
        )
        
        assert info.version == 1
        assert info.path == path
        assert info.specs_content == specs
        assert info.is_valid is True
        assert info.test_status == "passed"
    
    def test_version_info_default_test_status(self):
        """Test VersionInfo with default test_status."""
        info = VersionInfo(
            version=1,
            path=Path("/tmp"),
            specs_content="specs",
            is_valid=True
        )
        
        assert info.test_status is None
    
    def test_version_info_equality(self):
        """Test VersionInfo equality comparison."""
        path = Path("/tmp/version1")
        specs = "specs"
        
        info1 = VersionInfo(version=1, path=path, specs_content=specs, is_valid=True)
        info2 = VersionInfo(version=1, path=path, specs_content=specs, is_valid=True)
        info3 = VersionInfo(version=2, path=path, specs_content=specs, is_valid=True)
        
        assert info1 == info2
        assert info1 != info3


@pytest.mark.integration
class TestVersionManagerIntegration:
    """Integration tests for VersionManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.vm = VersionManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_version_workflow(self):
        """Test complete version management workflow."""
        # Start with no specs
        assert self.vm.get_current_version() == 1
        assert self.vm.get_next_version() == 2
        
        # Create first version specs
        specs_v1 = "# Coding Agent v1 Specification\n\nInitial version"
        (self.temp_dir / "SPECS.md").write_text(specs_v1)
        
        assert self.vm.get_current_version() == 1
        assert self.vm.get_next_version() == 2
        
        # Create version 2 directory
        v2_dir = self.vm.create_version_directory(2)
        assert v2_dir.exists()
        assert v2_dir.name == "version2"
        
        # Update specs to version 2
        specs_v2 = "# Coding Agent v2 Specification\n\nImproved version"
        (self.temp_dir / "SPECS.md").write_text(specs_v2)
        
        assert self.vm.get_current_version() == 2
        assert self.vm.get_next_version() == 3
    
    def test_multiple_version_directories(self):
        """Test creating multiple version directories."""
        # Create multiple versions
        for version in range(1, 5):
            version_dir = self.vm.create_version_directory(version)
            assert version_dir.exists()
            assert version_dir.name == f"version{version}"
            
            # Check subdirectories exist
            assert (version_dir / "coder").exists()
            assert (version_dir / "coder" / "utils").exists()
            assert (version_dir / "tests").exists()
        
        # Check all directories exist
        for version in range(1, 5):
            assert (self.temp_dir / f"version{version}").exists()
    
    def test_version_parsing_edge_cases(self):
        """Test version parsing with various edge cases."""
        test_cases = [
            ("# Coding Agent v10 Specification", 10),
            ("# Coding Agent v999 Specification\nMore content", 999),
            ("Some text\n# Coding Agent v5 Specification\nMore", 5),
            ("# Coding Agent v0 Specification", 0),
            ("# Other title\n# Coding Agent v7 Specification", 7),
        ]
        
        for specs_content, expected_version in test_cases:
            specs_path = self.temp_dir / "SPECS.md"
            specs_path.write_text(specs_content)
            
            version = self.vm.get_current_version()
            assert version == expected_version, f"Failed for: {specs_content[:50]}..."
    
    def test_concurrent_version_creation(self):
        """Test handling concurrent version creation attempts."""
        version = 3
        
        # Create directory in a separate thread simulation
        version_dir1 = self.temp_dir / f"version{version}"
        version_dir1.mkdir(parents=True)
        
        # Now try to create the same version
        with pytest.raises(ValueError, match=f"Version {version} directory already exists"):
            self.vm.create_version_directory(version)
    
    def test_file_permissions_handling(self):
        """Test handling of file permission issues."""
        # Create a read-only SPECS.md file
        specs_path = self.temp_dir / "SPECS.md"
        specs_path.write_text("# Coding Agent v1 Specification")
        
        # Make it read-only
        specs_path.chmod(0o444)
        
        try:
            # Should still be able to read version
            version = self.vm.get_current_version()
            assert version == 1
        finally:
            # Restore permissions for cleanup
            specs_path.chmod(0o644) 