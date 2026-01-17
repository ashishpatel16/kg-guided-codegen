import argparse
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def setup_bugsinpy_case(
    project_name: str,
    bug_id: str,
    experiments_dir: str = "experiments",
    bugsinpy_root: str = "datasets/BugsInPy",
) -> Path:
    """
    Sets up a BugsInPy bug case by checking out the buggy version (v0).
    """
    # Resolve paths
    experiments_path = Path(experiments_dir).resolve()
    bugsinpy_root_path = Path(bugsinpy_root).resolve()

    # Bugsinpy binaries
    bugsinpy_bin_dir = bugsinpy_root_path / "framework" / "bin"
    checkout_cmd = bugsinpy_bin_dir / "bugsinpy-checkout"
    compile_cmd = bugsinpy_bin_dir / "bugsinpy-compile"

    if not checkout_cmd.exists() or not compile_cmd.exists():
        print(f"Error: binaries not found in {bugsinpy_bin_dir}")
        print("Please check --bugsinpy-root path.")
        sys.exit(1)

    # Ensure experiments directory exists
    experiments_path.mkdir(parents=True, exist_ok=True)

    case_dir_name = f"{project_name}_{bug_id}"
    work_dir = experiments_path / case_dir_name

    # The actual project content is checked out into a subdirectory named after the project
    project_root = work_dir / project_name

    print(f"Setting up {project_name} bug {bug_id} in {work_dir}...")

    if project_root.exists() and any(project_root.iterdir()):
        print(f"Warning: Directory {project_root} exists and is not empty.")

    cmd_checkout = [
        str(checkout_cmd),
        "-p",
        project_name,
        "-i",
        str(bug_id),
        "-v",
        "0",
        "-w",
        str(work_dir),
    ]

    try:
        env = os.environ.copy()
        env["PATH"] = f"{str(bugsinpy_bin_dir)}:{env.get('PATH', '')}"

        # 1. Checkout
        print(f"Running checkout...")
        subprocess.run(cmd_checkout, env=env, check=True)

        # Verify checkout
        if not (project_root / "bugsinpy_run_test.sh").exists():
            print(
                f"Error: Checkout finished but {project_root} does not contain bugsinpy_run_test.sh"
            )
            print(
                "Possible causes: Invalid project name, invalid bug ID, or bugsInPy error."
            )
            sys.exit(1)

        print(f"Successfully checked out {project_name} bug {bug_id} to {project_root}")

        # 2. Compile
        # bugsinpy-compile -w work_dir
        cmd_compile = [
            str(compile_cmd),
            "-w",
            str(project_root),  # Compile the inner directory
        ]
        print(f"Compiling project...")
        subprocess.run(cmd_compile, env=env, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error setting up bug case: {e}")
        sys.exit(1)

    return project_root


@dataclass
class TestExecutionResult:
    success: bool
    stdout: str
    stderr: str
    fail_details: Optional[str] = None


def run_bugsinpy_case(
    work_dir: Path, bugsinpy_root: str = "datasets/BugsInPy", run_relevant: bool = True
) -> TestExecutionResult:
    """
    Runs the relevant test cases for a BugsInPy bug case.
    Returns TestExecutionResult with detailed info.
    """
    work_dir = Path(work_dir).resolve()
    bugsinpy_root_path = Path(bugsinpy_root).resolve()
    bugsinpy_bin_dir = bugsinpy_root_path / "framework" / "bin"
    test_cmd = bugsinpy_bin_dir / "bugsinpy-test"

    if not test_cmd.exists():
        msg = f"Error: bugsinpy-test binary not found at {test_cmd}"
        print(msg)
        return TestExecutionResult(success=False, stdout="", stderr=msg)

    cmd = [str(test_cmd), "-w", str(work_dir), "-r" if run_relevant else "-a"]

    print(f"Running tests in {work_dir}...")
    try:
        # bugsinpy-test prints output to stdout/stderr and creates bugsinpy_fail.txt on failure.
        env = os.environ.copy()
        env["PATH"] = f"{str(bugsinpy_bin_dir)}:{env.get('PATH', '')}"

        # Capture output
        result = subprocess.run(
            cmd, env=env, check=False, capture_output=True, text=True
        )
        stdout = result.stdout
        stderr = result.stderr

    except Exception as e:
        msg = f"Error running bugsinpy-test: {e}"
        print(msg)
        return TestExecutionResult(success=False, stdout="", stderr=msg)

    # Check for failure file
    fail_file = work_dir / "bugsinpy_fail.txt"
    fail_details = None
    success = True

    if fail_file.exists():
        success = False
        try:
            fail_details = fail_file.read_text(encoding="utf-8")
            print(f"Tests failed. See {fail_file} for details.")
        except Exception as e:
            fail_details = f"Could not read fail file: {e}"
    else:
        print("Tests passed (no failure file found).")

    return TestExecutionResult(
        success=success, stdout=stdout, stderr=stderr, fail_details=fail_details
    )


if __name__ == "__main__":
    # Example usage: setup and run tests for a bug
    # 'youtube-dl' is a valid project name (hyphen, not underscore)
    project = "youtube-dl"
    bug_id = 1

    try:
        repo_dir = setup_bugsinpy_case(project, bug_id)
        print(f"Repo setup at: {repo_dir}")

        result = run_bugsinpy_case(repo_dir)
        print(f"Test execution success: {result.success}")
        if not result.success:
            print(f"Failure details: {result.fail_details}")

    except SystemExit:
        print("Setup failed.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Cleanup (optional, commented out)
    # import shutil
    # shutil.rmtree(repo_dir)
