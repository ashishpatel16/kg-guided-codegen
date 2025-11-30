import os
import shutil
import subprocess
import json
import tempfile
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TriageEnvironment:
    """
    Enhanced class to handle SWE-Bench issue triaging, repository setup,
    environment management, and test execution without Docker.
    """

    def __init__(self, issue_dict: dict):
        self.repo = issue_dict.get("repo")

        self.instance_id = issue_dict.get("instance_id")
        self.patch = issue_dict.get("patch")

        self.base_commit = issue_dict.get("base_commit")
        self.version = issue_dict.get("version")
        self.environment_setup_commit = issue_dict.get(
            "environment_setup_commit", self.base_commit
        )

        self.fail_to_pass = json.loads(issue_dict.get("FAIL_TO_PASS", "[]"))
        self.pass_to_pass = json.loads(issue_dict.get("PASS_TO_PASS", "[]"))

        self.work_dir = None
        self.venv_dir = None
        self.repo_dir = None
        self.pip_path = None
        self.python_path = None

    def setup_workspace(self, work_root: str = None) -> str:
        """
        Create a workspace and clone the repository.
        """
        if work_root is None:
            self.work_dir = tempfile.mkdtemp(prefix=f"swe_env_{self.instance_id}_")
        else:
            work_root = Path(work_root)
            self.work_dir = str(work_root / f"swe_env_{self.instance_id}")
            os.makedirs(self.work_dir, exist_ok=True)
            assert os.path.exists(
                self.work_dir
            ), f"Working directory {self.work_dir} does not exist"

        logger.info(f"Working directory created at {self.work_dir}")

        repo_url = f"https://github.com/{self.repo}.git"
        self.repo_dir = os.path.join(self.work_dir, self.repo.split("/")[-1])

        if os.path.exists(self.repo_dir):
            logger.info(f"Repository already exists at {self.repo_dir}, skipping clone")
        else:
            logger.info(f"Cloning {repo_url} to {self.repo_dir}")
            subprocess.run(["git", "clone", repo_url, self.repo_dir], check=True)

        commit = self.environment_setup_commit or self.base_commit
        logger.info(f"Checking out commit {commit}")
        subprocess.run(["git", "checkout", commit], cwd=self.repo_dir, check=True)

        self.venv_dir = os.path.join(self.repo_dir, "venv")
        return self.repo_dir

    def setup_environment(self) -> str:
        """
        Sets up the virtual environment and installs dependencies.
        Returns the path to the virtual environment.
        """
        if not self.work_dir:
            raise RuntimeError("Workspace not set up. Call setup_workspace first.")

        logger.info(f"Creating virtual environment at {self.venv_dir}")

        subprocess.run(["python3", "-m", "venv", self.venv_dir], check=True)

        self.pip_path = os.path.join(self.venv_dir, "bin", "pip3")
        self.python_path = os.path.join(self.venv_dir, "bin", "python3")

        # Upgrade core build tools
        subprocess.run(
            [self.pip_path, "install", "--upgrade", "pip", "setuptools", "wheel"],
            check=True,
        )

        # Install dependencies
        self._install_dependencies()

        # Verify installation
        self._verify_installation()

        return self.venv_dir

    def _install_dependencies(self):
        """
        Attempt to install dependencies using various standard files.
        """
        logger.info(f"Installing dependencies from {self.repo_dir}")

        # 1. Try requirements.txt
        req_path = os.path.join(self.repo_dir, "requirements.txt")
        if os.path.exists(req_path):
            logger.info("Found requirements.txt, installing...")
            try:
                subprocess.run([self.pip_path, "install", "-r", req_path], check=True)
            except subprocess.CalledProcessError:
                logger.warning(
                    "Failed to install some requirements from requirements.txt"
                )

        # 2. Try setup.py or pyproject.toml for editable install
        if os.path.exists(os.path.join(self.repo_dir, "setup.py")) or os.path.exists(
            os.path.join(self.repo_dir, "pyproject.toml")
        ):
            logger.info("Found setup.py/pyproject.toml, installing in editable mode...")
            try:
                subprocess.run(
                    [self.pip_path, "install", "-e", self.repo_dir], check=True
                )
            except subprocess.CalledProcessError:
                logger.error("Failed to install package in editable mode")

        # 3. Try to install specific version if provided and not already installed
        if self.version:
            try:
                package_name = self.repo.split("/")[-1]
                subprocess.run(
                    [self.pip_path, "install", f"{package_name}=={self.version}"],
                    check=True,
                )
            except subprocess.SubprocessError:
                logger.warning(f"Failed to install explicit version {self.version}")

        # 4. Install test dependencies (pytest is common)
        subprocess.run([self.pip_path, "install", "pytest"], check=False)

    def _verify_installation(self):
        """
        Verify that the main package can be imported.
        """
        package_name = self.repo.split("/")[-1].replace("-", "_")
        logger.info(f"Verifying installation by importing {package_name}...")
        try:
            subprocess.run(
                [
                    self.python_path,
                    "-c",
                    f"import {package_name}; print('Import successful')",
                ],
                check=True,
                capture_output=True,
            )
            logger.info("Package import successful.")
        except subprocess.CalledProcessError:
            logger.warning(
                f"Could not import {package_name}. The package name might differ from the repo name or installation failed."
            )

    def apply_patch(self) -> bool:
        """
        Apply the solution patch to the repository.
        """
        if not self.repo_dir or not self.patch:
            return False

        patch_file = os.path.join(self.work_dir, "solution.patch")
        with open(patch_file, "w") as f:
            f.write(self.patch)

        try:
            subprocess.run(
                ["git", "apply", patch_file],
                cwd=self.repo_dir,
                check=True,
                stderr=subprocess.PIPE,
                text=True,
            )
            logger.info("Patch applied successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply patch: {e.stderr}")
            return False

    def run_tests(self, tests: List[str] = None) -> Dict[str, str]:
        """
        Run specified tests or the fail_to_pass/pass_to_pass tests.
        Returns a dictionary mapping test names to their output/status.
        """
        if not self.venv_dir:
            raise RuntimeError("Environment not set up. Call setup_environment first.")

        tests_to_run = tests if tests else (self.fail_to_pass + self.pass_to_pass)
        results = {}

        if not tests_to_run:
            logger.warning("No tests specified to run.")
            return results

        logger.info(f"Running {len(tests_to_run)} tests...")

        # Simple heuristic: if it looks like a file path, run pytest on it.
        # If it looks like a python module path, try to run it as a module or with pytest.

        for test in tests_to_run:
            logger.info(f"Running test: {test}")
            try:
                # Try running with pytest first as it's most common for SWE-bench
                # We assume the test string is something pytest can handle (file path or nodeid)
                cmd = [self.python_path, "-m", "pytest", test]

                result = subprocess.run(
                    cmd,
                    cwd=self.repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout per test
                )

                status = "PASSED" if result.returncode == 0 else "FAILED"
                results[test] = {
                    "status": status,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                logger.info(f"Test {test} finished: {status}")

            except subprocess.TimeoutExpired:
                results[test] = {"status": "TIMEOUT", "output": "Test timed out"}
                logger.error(f"Test {test} timed out")
            except Exception as e:
                results[test] = {"status": "ERROR", "output": str(e)}
                logger.error(f"Error running test {test}: {e}")

        return results

    def cleanup(self):
        """
        Clean up the workspace.
        """
        if self.work_dir and os.path.exists(self.work_dir):
            logger.info(f"Cleaning up work directory {self.work_dir}")
            shutil.rmtree(self.work_dir)
            self.work_dir = None
            self.venv_dir = None
            self.repo_dir = None


if __name__ == "__main__":
    from dataset_loader import load_swebench

    dataset = load_swebench(split="test")
    triage_env = TriageEnvironment(dataset[0])
    triage_env.setup_workspace(work_root="tests/swe-bench-exp")
    triage_env.setup_environment()
