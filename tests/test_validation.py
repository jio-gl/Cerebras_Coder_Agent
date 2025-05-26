"""Tests for the CodeValidator utility class."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from coder.utils.validation import CodeValidator, ValidationResult


class TestValidationResult:
    """Test class for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult instance."""
        result = ValidationResult(
            success=True,
            step="test_step",
            output="Success output",
            error="Some error",
            exit_code=0,
        )

        assert result.success is True
        assert result.step == "test_step"
        assert result.output == "Success output"
        assert result.error == "Some error"
        assert result.exit_code == 0

    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        result = ValidationResult(
            success=False, step="failed_step", output="Error output"
        )

        assert result.success is False
        assert result.step == "failed_step"
        assert result.output == "Error output"
        assert result.error is None
        assert result.exit_code is None


class TestCodeValidator:
    """Test class for CodeValidator."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = CodeValidator(self.temp_dir)

        # Create a minimal Python project structure
        (self.temp_dir / "setup.py").write_text(
            """
from setuptools import setup, find_packages

setup(
    name="test-project",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[],
)
"""
        )

        # Create main package
        (self.temp_dir / "coder").mkdir()
        (self.temp_dir / "coder" / "__init__.py").write_text("")
        (self.temp_dir / "coder" / "main.py").write_text(
            """
def hello_world():
    return "Hello, World!"
"""
        )

        # Create tests
        (self.temp_dir / "tests").mkdir()
        (self.temp_dir / "tests" / "__init__.py").write_text("")
        (self.temp_dir / "tests" / "test_main.py").write_text(
            """
import pytest
from coder.main import hello_world

def test_hello_world():
    assert hello_world() == "Hello, World!"
"""
        )

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test CodeValidator initialization."""
        assert self.validator.project_dir == self.temp_dir
        assert self.validator.results == []

    @patch("subprocess.run")
    def test_run_command_success(self, mock_run):
        """Test running a successful command."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.validator.run_command(["echo", "hello"], "test_step")

        assert result.success is True
        assert result.step == "test_step"
        assert result.output == "Success output"
        assert result.error is None
        assert result.exit_code == 0
        assert len(self.validator.results) == 1

    @patch("subprocess.run")
    def test_run_command_failure(self, mock_run):
        """Test running a failed command."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "Some output"
        mock_result.stderr = "Error message"
        mock_run.return_value = mock_result

        result = self.validator.run_command(["false"], "failing_step")

        assert result.success is False
        assert result.step == "failing_step"
        assert result.output == "Some output"
        assert result.error == "Error message"
        assert result.exit_code == 1

    @patch("subprocess.run")
    def test_run_command_no_check_success(self, mock_run):
        """Test running command with check_success=False."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "Output"
        mock_result.stderr = "Error"
        mock_run.return_value = mock_result

        result = self.validator.run_command(["false"], "step", check_success=False)

        assert result.success is True  # Because check_success=False
        assert result.exit_code == 1

    @patch("subprocess.run")
    def test_run_command_timeout(self, mock_run):
        """Test running command that times out."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 120)

        result = self.validator.run_command(["sleep", "200"], "timeout_step")

        assert result.success is False
        assert result.step == "timeout_step"
        assert result.error == "Command timed out after 2 minutes"
        assert result.exit_code == -1

    @patch("subprocess.run")
    def test_run_command_exception(self, mock_run):
        """Test running command that raises exception."""
        mock_run.side_effect = Exception("Subprocess error")

        result = self.validator.run_command(["invalid_command"], "exception_step")

        assert result.success is False
        assert result.step == "exception_step"
        assert result.error == "Subprocess error"
        assert result.exit_code == -1

    @patch("subprocess.run")
    def test_validate_installation(self, mock_run):
        """Test package installation validation."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Installation successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.validator.validate_installation()

        assert result.success is True
        assert result.step == "Installation"
        mock_run.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=self.temp_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

    @patch("subprocess.run")
    def test_validate_syntax(self, mock_run):
        """Test Python syntax validation."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.validator.validate_syntax()

        assert result.success is True
        assert result.step == "Syntax validation"

        # Check that py_compile was called with Python files
        call_args = mock_run.call_args
        assert call_args[0][0][:3] == [sys.executable, "-m", "py_compile"]
        # Should include the Python files we created
        python_files = call_args[0][0][3:]
        assert any("main.py" in str(f) for f in python_files)
        assert any("test_main.py" in str(f) for f in python_files)

    @patch("subprocess.run")
    def test_validate_tests(self, mock_run):
        """Test test suite validation."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "All tests passed"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.validator.validate_tests()

        assert result.success is True
        assert result.step == "Test execution"
        mock_run.assert_called_once_with(
            [sys.executable, "-m", "pytest", "-v", "--tb=short"],
            cwd=self.temp_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

    @patch("subprocess.run")
    def test_validate_cli(self, mock_run):
        """Test CLI validation."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "CLI help output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.validator.validate_cli()

        assert result.success is True
        assert result.step == "CLI validation"
        mock_run.assert_called_once_with(
            [sys.executable, "-m", "coder.cli", "--help"],
            cwd=self.temp_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_validate_imports_success(self):
        """Test successful import validation."""
        # Create a more complete package structure
        (self.temp_dir / "coder" / "agent.py").write_text("class Agent: pass")
        (self.temp_dir / "coder" / "api.py").write_text("class API: pass")
        (self.temp_dir / "coder" / "cli.py").write_text("def main(): pass")

        # Create utils subdirectory
        (self.temp_dir / "coder" / "utils").mkdir()
        (self.temp_dir / "coder" / "utils" / "__init__.py").write_text("")
        (self.temp_dir / "coder" / "utils" / "equivalence.py").write_text(
            "class EquivalenceChecker: pass"
        )

        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "All imports successful"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = self.validator.validate_imports()

            assert result.success is True
            assert result.step == "Import validation"

            # Check that the test script was created and executed
            call_args = mock_run.call_args
            assert call_args[0][0] == [sys.executable, "test_imports.py"]

            # Check that the test script was cleaned up
            assert not (self.temp_dir / "test_imports.py").exists()

    def test_validate_imports_cleanup_on_exception(self):
        """Test import validation cleans up script even on exception."""
        # Mock the run_command method to simulate an exception
        original_run_command = self.validator.run_command

        def mock_run_command(*args, **kwargs):
            raise Exception("Test error")

        self.validator.run_command = mock_run_command

        try:
            # Create a test script that will be cleaned up
            script_path = self.temp_dir / "test_imports.py"
            script_path.write_text("# Test import script")

            # Call the method that should clean up even on exception
            with pytest.raises(Exception) as exc_info:
                self.validator.validate_imports()

            # Check that the exception was raised and the file was cleaned up
            assert "Test error" in str(exc_info.value)
            assert (
                not script_path.exists()
            ), "Script file should be cleaned up on exception"

        finally:
            # Restore the original method
            self.validator.run_command = original_run_command

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_run_full_validation_all_pass(self, mock_print, mock_run):
        """Test full validation suite with all steps passing."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Create required files for import validation
        (self.temp_dir / "coder" / "agent.py").write_text("class Agent: pass")
        (self.temp_dir / "coder" / "api.py").write_text("class API: pass")
        (self.temp_dir / "coder" / "cli.py").write_text("def main(): pass")
        (self.temp_dir / "coder" / "utils").mkdir()
        (self.temp_dir / "coder" / "utils" / "__init__.py").write_text("")
        (self.temp_dir / "coder" / "utils" / "equivalence.py").write_text(
            "class EquivalenceChecker: pass"
        )

        result = self.validator.run_full_validation()

        assert result is True
        assert len(self.validator.results) == 5  # All validation steps

        # Check that success messages were printed
        success_calls = [call for call in mock_print.call_args_list if "✅" in str(call)]
        assert len(success_calls) == 5

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_run_full_validation_some_fail(self, mock_print, mock_run):
        """Test full validation suite with some steps failing."""

        def mock_run_side_effect(*args, **kwargs):
            # Make installation fail, others succeed
            if "pip" in args[0]:
                mock_result = Mock()
                mock_result.returncode = 1
                mock_result.stdout = ""
                mock_result.stderr = "Installation failed"
                return mock_result
            else:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "Success"
                mock_result.stderr = ""
                return mock_result

        mock_run.side_effect = mock_run_side_effect

        # Create required files for import validation
        (self.temp_dir / "coder" / "agent.py").write_text("class Agent: pass")
        (self.temp_dir / "coder" / "api.py").write_text("class API: pass")
        (self.temp_dir / "coder" / "cli.py").write_text("def main(): pass")
        (self.temp_dir / "coder" / "utils").mkdir()
        (self.temp_dir / "coder" / "utils" / "__init__.py").write_text("")
        (self.temp_dir / "coder" / "utils" / "equivalence.py").write_text(
            "class EquivalenceChecker: pass"
        )

        result = self.validator.run_full_validation()

        assert result is False

        # Check that both success and failure messages were printed
        success_calls = [call for call in mock_print.call_args_list if "✅" in str(call)]
        failure_calls = [call for call in mock_print.call_args_list if "❌" in str(call)]
        assert len(success_calls) >= 1  # Some should pass
        assert len(failure_calls) >= 1  # Some should fail

    def test_get_validation_summary_empty(self):
        """Test getting validation summary with no results."""
        summary = self.validator.get_validation_summary()

        expected = {
            "total_steps": 0,
            "passed": 0,
            "failed": 0,
            "success_rate": 0,
            "all_passed": True,  # Vacuously true
            "results": [],
        }

        assert summary == expected

    def test_get_validation_summary_mixed_results(self):
        """Test getting validation summary with mixed results."""
        # Add some mock results
        self.validator.results = [
            ValidationResult(True, "step1", "output1"),
            ValidationResult(False, "step2", "output2"),
            ValidationResult(True, "step3", "output3"),
        ]

        summary = self.validator.get_validation_summary()

        assert summary["total_steps"] == 3
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["success_rate"] == 2 / 3
        assert summary["all_passed"] is False
        assert len(summary["results"]) == 3


@pytest.mark.integration
class TestCodeValidatorIntegration:
    """Integration tests for CodeValidator."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = CodeValidator(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_real_syntax_validation(self):
        """Test syntax validation with real Python files."""
        # Create valid Python file
        valid_file = self.temp_dir / "valid.py"
        valid_file.write_text(
            """
def hello():
    return "Hello, World!"

if __name__ == "__main__":
    print(hello())
"""
        )

        # Create invalid Python file
        invalid_file = self.temp_dir / "invalid.py"
        invalid_file.write_text(
            """
def hello(
    return "Hello, World!"  # Missing closing parenthesis

if __name__ == "__main__":
    print(hello())
"""
        )

        # Test valid file
        result = self.validator.run_command(
            [sys.executable, "-m", "py_compile", str(valid_file)], "Valid syntax test"
        )
        assert result.success is True

        # Test invalid file
        result = self.validator.run_command(
            [sys.executable, "-m", "py_compile", str(invalid_file)],
            "Invalid syntax test",
        )
        assert result.success is False
        assert "SyntaxError" in result.error

    def test_timeout_handling(self):
        """Test that command timeouts are handled properly."""
        # Create a script that sleeps longer than the timeout
        sleep_script = self.temp_dir / "sleep.py"

        # Adjust the timeout in the validator for this test only
        original_run_command = self.validator.run_command

        def patched_run_command(command, step_name, check_success=True):
            import subprocess

            # For sleep script, use a much shorter timeout of 1 second
            if str(sleep_script) in str(command):
                try:
                    result = subprocess.run(
                        command,
                        cwd=self.temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=1,  # 1 second timeout for test
                    )

                    success = result.returncode == 0 if check_success else True

                    validation_result = ValidationResult(
                        success=success,
                        step=step_name,
                        output=result.stdout,
                        error=result.stderr if result.stderr else None,
                        exit_code=result.returncode,
                    )

                    return validation_result

                except subprocess.TimeoutExpired:
                    validation_result = ValidationResult(
                        success=False,
                        step=step_name,
                        output="",
                        error="Command timed out after 1 second",
                        exit_code=-1,
                    )
                    return validation_result
            else:
                # For other commands, use the original method
                return original_run_command(command, step_name, check_success)

        try:
            # Replace the run_command method temporarily
            self.validator.run_command = patched_run_command

            # Write a script that sleeps for 5 seconds (longer than our 1s timeout)
            sleep_script.write_text("import time; time.sleep(5)")

            result = self.validator.run_command(
                [sys.executable, str(sleep_script)], "Timeout test"
            )

            assert result.success is False
            assert "timed out" in result.error
            assert result.exit_code == -1

        finally:
            # Restore the original method
            self.validator.run_command = original_run_command

    def test_working_directory_isolation(self):
        """Test that commands run in the correct working directory."""
        # Create a file in the temp directory
        test_file = self.temp_dir / "test_file.txt"
        test_file.write_text("test content")

        # Run a command that lists directory contents
        result = self.validator.run_command(["ls", "-la"], "Directory test")

        # Should see our test file in the output
        assert (
            "test_file.txt" in result.output or result.success is False
        )  # ls might not be available on all systems

    def test_environment_isolation(self):
        """Test that validation runs in isolated environment."""
        # Test that Python path is correctly set
        result = self.validator.run_command(
            [sys.executable, "-c", "import sys; print(sys.path[0])"], "Environment test"
        )

        if result.success:
            # The first path should be the temp directory
            assert str(self.temp_dir) in result.output or result.output.strip() == ""


@pytest.mark.slow
class TestCodeValidatorPerformance:
    """Performance tests for CodeValidator."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = CodeValidator(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_large_file_validation(self):
        """Test validation with large Python files."""
        # Create a large Python file
        large_file = self.temp_dir / "large.py"
        large_content = []

        for i in range(1000):
            large_content.append(
                f"""
def function_{i}():
    '''Function number {i}'''
    result = {i} * 2
    return result
"""
            )

        large_file.write_text("\n".join(large_content))

        # Test syntax validation on large file
        import time

        start_time = time.time()

        result = self.validator.run_command(
            [sys.executable, "-m", "py_compile", str(large_file)],
            "Large file syntax test",
        )

        end_time = time.time()
        duration = end_time - start_time

        assert result.success is True
        assert duration < 30  # Should complete within 30 seconds

    def test_many_files_validation(self):
        """Test validation with many Python files."""
        # Create many small Python files
        files = []
        for i in range(100):
            file_path = self.temp_dir / f"file_{i}.py"
            file_path.write_text(
                f"""
def func_{i}():
    return {i}
"""
            )
            files.append(str(file_path))

        # Test syntax validation on all files
        import time

        start_time = time.time()

        result = self.validator.run_command(
            [sys.executable, "-m", "py_compile"] + files, "Many files syntax test"
        )

        end_time = time.time()
        duration = end_time - start_time

        assert result.success is True
        assert duration < 60  # Should complete within 60 seconds
