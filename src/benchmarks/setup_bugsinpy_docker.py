import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from src.docker_utils.basic_container import SimpleDockerSandbox

logger = logging.getLogger(__name__)

class BugsInPyDockerSandbox:
    """
    A specialized sandbox for BugsInPy that handles dataset mounting and framework setup.
    """
    def __init__(
        self,
        project_name: str,
        bug_id: str,
        image_name: str = "python:3.11-slim",
        bugsinpy_root: str = "datasets/BugsInPy",
        experiments_dir: str = "experiments",
    ):
        self.project_name = project_name
        self.bug_id = bug_id
        
        # Absolute paths on host
        self.host_bugsinpy_root = Path(bugsinpy_root).resolve()
        self.host_experiments_dir = Path(experiments_dir).resolve()
        self.host_repo_root = Path(__file__).resolve().parent.parent.parent
        
        # Paths in container
        self.container_bugsinpy_home = "/home/bugsinpy"
        self.container_workspace = "/home/workspace"
        self.container_project_root = f"{self.container_workspace}/{project_name}"
        self.container_debugger_root = "/home/debugger"
        
        # Volumes to mount
        volumes = {
            str(self.host_bugsinpy_root / "framework"): {
                "bind": f"{self.container_bugsinpy_home}/framework",
                "mode": "ro",
            },
            str(self.host_bugsinpy_root / "projects"): {
                "bind": f"{self.container_bugsinpy_home}/projects",
                "mode": "rw",
            },
            str(self.host_experiments_dir): {
                "bind": self.container_workspace,
                "mode": "rw",
            },
            str(self.host_repo_root): {
                "bind": self.container_debugger_root,
                "mode": "ro",
            },
        }
        
        self.sandbox = SimpleDockerSandbox(
            image_name=image_name,
            sandbox_dir=self.container_workspace,
            volumes=volumes,
            keep_alive=True
        )

    def __enter__(self):
        self.sandbox.start()
        self._setup_environment()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sandbox.stop()

    def _setup_environment(self):
        """Sets up the BugsInPy environment inside the container."""
        # Install dependencies
        logger.info("Installing BugsInPy dependencies in container...")
        self.sandbox.run_command("apt-get update && apt-get install -y git dos2unix", verbose=True)
        
        # Install tracer dependencies in system python (for general use)
        self.sandbox.run_command("pip install networkx pydantic pytest tree-sitter tree-sitter-python docker", verbose=True)
        
        # Set environment variables
        self.bugsinpy_bin = f"{self.container_bugsinpy_home}/framework/bin"
        self.env_vars = {
            "BUGSINPY_HOME": self.container_bugsinpy_home,
            "PATH": f"{self.bugsinpy_bin}:/usr/local/bin:/usr/bin:/bin"
        }

    def run_dynamic_tracer(self, output_file: str = "call_graph.json") -> Tuple[int, str, str]:
        """
        Runs the dynamic tracer on the checked-out bug case.
        """
        logger.info(f"Running dynamic tracer for {self.project_name}...")
        
        # 1. Get test info from bugsinpy_bug.info
        exit_code, stdout, _ = self.sandbox.run_command(
            f"grep 'test_file=' {self.container_project_root}/bugsinpy_bug.info"
        )
        if exit_code != 0:
            return 1, "", "Could not find test_file info"
        
        # Parse test_file="path/to/test.py"
        test_file_line = stdout.strip()
        test_file = test_file_line.split('"')[1].split(';')[0] # Take first test file if multiple
        
        # 2. Prepare paths
        venv_python = f"{self.container_project_root}/env/bin/python"
        debugger_module = "src.program_analysis.dynamic_call_graph"
        output_path = f"{self.container_project_root}/{output_file}"
        
        # 3. Install tracer dependencies in the project venv
        logger.info("Installing tracer dependencies in project venv...")
        self.sandbox.run_command(f"{venv_python} -m pip install networkx pydantic pytest tree-sitter tree-sitter-python docker")
        
        # 4. Run the tracer
        # We need to set PYTHONPATH to include the debugger root and the repo root
        run_cmd = (
            f"export PYTHONPATH=$PYTHONPATH:{self.container_debugger_root}:{self.container_project_root} && "
            f"{venv_python} -m {debugger_module} "
            f"--repo {self.container_project_root} "
            f"--scripts {test_file} "
            f"--output {output_path} "
            f"--test-mode"
        )
        
        return self.sandbox.run_command(run_cmd, verbose=True)
        
    def _run_bugsinpy_cmd(self, cmd_name: str, args: List[str], workdir: Optional[str] = None, verbose: bool = False) -> Tuple[int, str, str]:
        """Runs a bugsinpy command with the correct environment."""
        env_str = " ".join([f"{k}={v}" for k, v in self.env_vars.items()])
        full_cmd = f"{env_str} {cmd_name} {' '.join(args)}"
        return self.sandbox.run_command(full_cmd, workdir=workdir, verbose=verbose)

    def checkout(self, version: int = 0) -> Tuple[int, str, str]:
        """
        Checks out the project version.
        version: 0 for buggy, 1 for fixed.
        """
        logger.info(f"Checking out {self.project_name} bug {self.bug_id} (version {version})...")
        args = [
            "-p", self.project_name,
            "-i", str(self.bug_id),
            "-v", str(version),
            "-w", self.container_workspace
        ]
        return self._run_bugsinpy_cmd("bugsinpy-checkout", args)

    def compile(self, verbose: bool = False) -> Tuple[int, str, str]:
        """Compiles the project inside the container."""
        logger.info(f"Compiling {self.project_name}...")
        # bugsinpy-compile uses the current directory or -w
        args = ["-w", self.container_project_root]
        return self._run_bugsinpy_cmd("bugsinpy-compile", args, verbose=verbose)

    def test(self, relevant: bool = True, verbose: bool = False) -> Tuple[int, str, str]:
        """
        Runs tests for the project.
        Returns (0, stdout, stderr) if tests pass, (1, stdout, stderr) if they fail.
        """
        logger.info(f"Running tests for {self.project_name}...")
        args = ["-w", self.container_project_root]
        if relevant:
            args.append("-r")
        else:
            args.append("-a")
        
        exit_code, stdout, stderr = self._run_bugsinpy_cmd("bugsinpy-test", args, verbose=verbose)
        
        # BugsInPy bugsinpy-test often returns 0 even if tests fail, 
        # but it creates bugsinpy_fail.txt on failure.
        fail_check_code, _, _ = self.sandbox.run_command(
            f"[ -f {self.container_project_root}/bugsinpy_fail.txt ]"
        )
        if fail_check_code == 0:
            return 1, stdout, stderr
            
        return exit_code, stdout, stderr

    def get_test_command(self) -> Optional[str]:
        """
        Retrieves the exact test command from bugsinpy_run_test.sh.
        This can be used as an entry point for tracing.
        """
        # Read bugsinpy_run_test.sh from the project root in the container
        exit_code, stdout, stderr = self.sandbox.run_command(
            f"cat {self.container_project_root}/bugsinpy_run_test.sh"
        )
        if exit_code == 0:
            return stdout.strip()
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    project = "youtube-dl"
    bug = "1"
    
    with BugsInPyDockerSandbox(project, bug) as bspy:
        exit_code, out, err = bspy.checkout(version=0)
        if exit_code != 0:
            print(f"Checkout failed: {err}")
        else:
            print(f"Checkout successful.")
            
            exit_code, out, err = bspy.compile(verbose=True)
            print(f"Compile exit code: {exit_code}")
            
            test_cmd = bspy.get_test_command()
            print(f"Test command: {test_cmd}")
            
            exit_code, out, err = bspy.test(verbose=True)
            print(f"Test exit code: {exit_code}")
            if exit_code != 0:
                # Check for failure file
                _, fail_details, _ = bspy.sandbox.run_command(
                    f"cat {bspy.container_project_root}/bugsinpy_fail.txt"
                )
                print(f"Failure details:\n{fail_details}")

            # NEW: Run the dynamic tracer
            print("\n" + "="*50)
            print("RUNNING DYNAMIC TRACER")
            print("="*50)
            exit_code, out, err = bspy.run_dynamic_tracer()
            print(f"Tracer exit code: {exit_code}")
            if exit_code == 0:
                print("Tracer finished successfully.")
                # The output is at experiments/youtube-dl_1/youtube-dl/call_graph.json on host
            else:
                print(f"Tracer failed: {err}")
