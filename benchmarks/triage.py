import os
import shutil
import subprocess
import venv
import json
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Triage:
    """
    Class to handle SWE-Bench issue triaging, repository setup, and environment management.
    """
    
    def __init__(self, issue_dict: dict):
        self.instance_id = issue_dict.get('instance_id')
        self.patch = issue_dict.get('patch')
        self.repo = issue_dict.get('repo')
        self.base_commit = issue_dict.get('base_commit')
        self.hints_text = issue_dict.get('hints_text')
        self.created_at = issue_dict.get('created_at')
        self.test_patch = issue_dict.get('test_patch')
        self.problem_statement = issue_dict.get('problem_statement')
        self.version = issue_dict.get('version')
        self.environment_setup_commit = issue_dict.get('environment_setup_commit', self.base_commit)
        
        self.fail_to_pass = json.loads(issue_dict.get('FAIL_TO_PASS', '[]'))
        self.pass_to_pass = json.loads(issue_dict.get('PASS_TO_PASS', '[]'))
        
        self.work_dir = None
        self.venv_dir = None
    
    def setup_repo(self, work_root: str = None) -> str:
        """
        Create a new folder and clone the repository.
        
        Args:
            work_root (str, optional): Root directory for the workspace. 
                                      If None, uses a temporary directory.
        
        Returns:
            str: Path to the cloned repository
        """
        if work_root is None:
            self.work_dir = tempfile.mkdtemp(prefix=f"swe_{self.instance_id}_")
        else:
            work_root = Path(work_root)
            self.work_dir = str(work_root / f"swe_{self.instance_id}")
            os.makedirs(self.work_dir, exist_ok=True)
            
        logger.info(f"Working directory created at {self.work_dir}")
        
        repo_url = f"https://github.com/{self.repo}.git"
        repo_dir = os.path.join(self.work_dir, self.repo.split('/')[-1])
        
        logger.info(f"Cloning {repo_url} to {repo_dir}")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        
        # TODO: Checkout the difference between the environment setup commit and the base commit
        commit = self.environment_setup_commit or self.base_commit
        logger.info(f"Checking out commit {commit}")
        subprocess.run(["git", "checkout", commit], cwd=repo_dir, check=True)
        
        return repo_dir
    
    def setup_environment(self) -> str:
        if not self.work_dir:
            raise RuntimeError("Repository not set up. Call setup_repo first.")
        
        repo_dir = os.path.join(self.work_dir, self.repo.split('/')[-1])
        self.venv_dir = os.path.join(self.work_dir, "venv")
        
        logger.info(f"Creating virtual environment at {self.venv_dir}")
        venv.create(self.venv_dir, with_pip=True)
        
        if os.name == 'nt':
            pip_path = os.path.join(self.venv_dir, "Scripts", "pip")
        else:
            pip_path = os.path.join(self.venv_dir, "bin", "pip")
        
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        logger.info(f"Installing package from {repo_dir}")
        if os.path.exists(os.path.join(repo_dir, "requirements.txt")):
            subprocess.run([pip_path, "install", "-r", os.path.join(repo_dir, "requirements.txt")], check=True)
        
        if os.path.exists(os.path.join(repo_dir, "setup.py")):
            subprocess.run([pip_path, "install", "-e", repo_dir], check=True)
        elif os.path.exists(os.path.join(repo_dir, "pyproject.toml")):
            subprocess.run([pip_path, "install", "-e", repo_dir], check=True)
        
        if self.version:
            try:
                package_name = self.repo.split('/')[-1]
                subprocess.run([pip_path, "install", f"{package_name}=={self.version}"], check=True)
            except subprocess.SubprocessError:
                logger.warning(f"Failed to install version {self.version}, using development version")
        
        return self.venv_dir
    
    def cleanup(self) -> bool:
        if self.work_dir and os.path.exists(self.work_dir):
            logger.info(f"Cleaning up work directory {self.work_dir}")
            shutil.rmtree(self.work_dir)
            self.work_dir = None
            self.venv_dir = None
            return True
        return False

    def apply_patch(self) -> bool:
        if not self.work_dir or not self.patch:
            return False
        
        repo_dir = os.path.join(self.work_dir, self.repo.split('/')[-1])
        patch_file = os.path.join(self.work_dir, "solution.patch")
        
        with open(patch_file, 'w') as f:
            f.write(self.patch)
        
        try:
            result = subprocess.run(
                ["git", "apply", patch_file],
                cwd=repo_dir,
                check=True,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info("Patch applied successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply patch: {e.stderr}")
            return False

    def __str__(self) -> str:
        """String representation of the issue."""
        return f"SWE-Bench Issue: {self.instance_id} ({self.repo})"

    def __repr__(self):
        """Representation of the issue."""
        return f"Triage(instance_id='{self.instance_id}', repo='{self.repo}')"


# Example usage
if __name__ == "__main__":
    import json
    
    # Load example data
    with open('../swebench_ex1.json', 'r') as f:
        issue_data = json.load(f)
    
    # Create triage instance
    triage = Triage(issue_data)
    print(f"Loaded issue: {triage}")
    
    # Setup repository and environment
    try:
        repo_dir = triage.setup_repo(work_root="./workspaces")
        venv_dir = triage.setup_environment()
        print(f"Repository set up at: {repo_dir}")
        print(f"Virtual environment created at: {venv_dir}")
        
        # Apply the gold patch
        if triage.apply_patch():
            print("Patch applied successfully!")
        
    finally:
        # Clean up
        if input("Clean up workspace? (y/n): ").lower() == 'y':
            triage.cleanup()
            print("Workspace cleaned up")
