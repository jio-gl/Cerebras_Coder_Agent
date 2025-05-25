"""Validation utilities for self-rewrite process."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ValidationResult:
    """Result of code validation."""

    success: bool
    step: str
    output: str
    error: Optional[str] = None
    exit_code: Optional[int] = None


class CodeValidator:
    """Validates generated code through multiple checks."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.results: List[ValidationResult] = []

    def run_command(
        self, command: List[str], step_name: str, check_success: bool = True
    ) -> ValidationResult:
        """Run a command and return the result."""
        try:
            result = subprocess.run(
                command,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            success = result.returncode == 0 if check_success else True

            validation_result = ValidationResult(
                success=success,
                step=step_name,
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                exit_code=result.returncode,
            )

            self.results.append(validation_result)
            return validation_result

        except subprocess.TimeoutExpired:
            validation_result = ValidationResult(
                success=False,
                step=step_name,
                output="",
                error="Command timed out after 2 minutes",
                exit_code=-1,
            )
            self.results.append(validation_result)
            return validation_result

        except Exception as e:
            validation_result = ValidationResult(
                success=False, step=step_name, output="", error=str(e), exit_code=-1
            )
            self.results.append(validation_result)
            return validation_result

    def validate_installation(self) -> ValidationResult:
        """Validate that the package can be installed."""
        return self.run_command(
            [sys.executable, "-m", "pip", "install", "-e", "."], "Installation"
        )

    def validate_imports(self) -> ValidationResult:
        """Validate that main modules can be imported."""
        import_script = """
import sys
sys.path.insert(0, '.')
try:
    import coder
    import coder.agent
    import coder.api
    import coder.cli
    import coder.utils.equivalence
    print("All imports successful")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)
"""

        # Write script to temporary file
        script_path = self.project_dir / "test_imports.py"
        script_path.write_text(import_script)

        try:
            result = self.run_command(
                [sys.executable, "test_imports.py"], "Import validation"
            )
            # Clean up
            script_path.unlink()
            return result
        except:
            # Clean up even if there's an error
            if script_path.exists():
                script_path.unlink()
            raise

    def validate_syntax(self) -> ValidationResult:
        """Validate Python syntax of all files."""
        return self.run_command(
            [sys.executable, "-m", "py_compile"]
            + [str(f) for f in self.project_dir.rglob("*.py")],
            "Syntax validation",
        )

    def validate_tests(self) -> ValidationResult:
        """Run the test suite."""
        return self.run_command(
            [sys.executable, "-m", "pytest", "-v", "--tb=short"], "Test execution"
        )

    def validate_cli(self) -> ValidationResult:
        """Validate CLI functionality."""
        return self.run_command(
            [sys.executable, "-m", "coder.cli", "--help"], "CLI validation"
        )

    def run_full_validation(self) -> bool:
        """Run complete validation suite."""
        validation_steps = [
            self.validate_syntax,
            self.validate_installation,
            self.validate_imports,
            self.validate_cli,
            self.validate_tests,
        ]

        all_passed = True
        for step in validation_steps:
            result = step()
            if not result.success:
                all_passed = False
                print(f"❌ {result.step} failed:")
                print(f"   {result.error or result.output}")
            else:
                print(f"✅ {result.step} passed")

        return all_passed

    def get_validation_summary(self) -> Dict[str, any]:
        """Get a summary of validation results."""
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)

        return {
            "total_steps": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "all_passed": passed == total,
            "results": self.results,
        }
