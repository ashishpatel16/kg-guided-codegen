import os
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch
import pytest
from src.benchmarks.setup_bugsinpy_docker import BugsInPyDockerSandbox

@pytest.fixture
def mock_docker_client():
    """Patch docker.from_env so no real Docker daemon is needed."""
    with patch("src.docker_utils.basic_container.docker.from_env") as mock_env:
        mock_client = MagicMock()
        mock_env.return_value = mock_client
        yield mock_client


@pytest.fixture
def sandbox(mock_docker_client) -> BugsInPyDockerSandbox:
    """Create a BugsInPyDockerSandbox for youtube-dl bug 1."""
    return BugsInPyDockerSandbox(
        project_name="youtube-dl",
        bug_id="1",
        bugsinpy_root="datasets/BugsInPy",
        experiments_dir="experiments",
    )


def test_container_project_root(sandbox: BugsInPyDockerSandbox):
    """container_project_root should be /home/workspace/<project_name>."""
    expected: str = "/home/workspace/youtube-dl"
    assert sandbox.container_project_root == expected


def test_container_bugsinpy_home(sandbox: BugsInPyDockerSandbox):
    """container_bugsinpy_home should be /home/bugsinpy."""
    assert sandbox.container_bugsinpy_home == "/home/bugsinpy"


def test_container_workspace(sandbox: BugsInPyDockerSandbox):
    """container_workspace should be /home/workspace."""
    assert sandbox.container_workspace == "/home/workspace"


def test_host_bugsinpy_root_is_absolute(sandbox: BugsInPyDockerSandbox):
    """host_bugsinpy_root should be resolved to an absolute Path."""
    assert sandbox.host_bugsinpy_root.is_absolute()


def test_host_experiments_dir_is_absolute(sandbox: BugsInPyDockerSandbox):
    """host_experiments_dir should be resolved to an absolute Path."""
    assert sandbox.host_experiments_dir.is_absolute()


def test_volume_mounts_contain_framework(sandbox: BugsInPyDockerSandbox):
    """The framework dir should be mounted read-only into the container."""
    volumes: Dict = sandbox.sandbox.volumes
    framework_host: str = str(sandbox.host_bugsinpy_root / "framework")
    assert framework_host in volumes
    assert volumes[framework_host]["mode"] == "ro"


def test_volume_mounts_contain_projects(sandbox: BugsInPyDockerSandbox):
    """The projects dir should be mounted read-write into the container."""
    volumes: Dict = sandbox.sandbox.volumes
    projects_host: str = str(sandbox.host_bugsinpy_root / "projects")
    assert projects_host in volumes
    assert volumes[projects_host]["mode"] == "rw"


def test_volume_mounts_contain_experiments(sandbox: BugsInPyDockerSandbox):
    """The experiments dir should be mounted rw at container_workspace."""
    volumes: Dict = sandbox.sandbox.volumes
    exp_host: str = str(sandbox.host_experiments_dir)
    assert exp_host in volumes
    bind_path: str = volumes[exp_host]["bind"]
    assert bind_path == sandbox.container_workspace


def test_volume_mounts_contain_debugger(sandbox: BugsInPyDockerSandbox):
    """The repo root should be mounted read-only for the debugger."""
    volumes: Dict = sandbox.sandbox.volumes
    repo_host: str = str(sandbox.host_repo_root)
    assert repo_host in volumes
    assert volumes[repo_host]["mode"] == "ro"


def test_checkout_buggy_version(sandbox: BugsInPyDockerSandbox):
    """checkout(version=0) should call bugsinpy-checkout with -v 0."""
    sandbox.sandbox.run_command = MagicMock(return_value=(0, "", ""))
    sandbox._setup_environment = MagicMock()
    sandbox.env_vars = {"BUGSINPY_HOME": "/home/bugsinpy", "PATH": "/usr/bin"}

    sandbox.checkout(version=0)

    call_args: str = sandbox.sandbox.run_command.call_args[0][0]
    assert "bugsinpy-checkout" in call_args
    assert "-p youtube-dl" in call_args
    assert "-i 1" in call_args
    assert "-v 0" in call_args


def test_checkout_fixed_version(sandbox: BugsInPyDockerSandbox):
    """checkout(version=1) should pass -v 1."""
    sandbox.sandbox.run_command = MagicMock(return_value=(0, "", ""))
    sandbox.env_vars = {"BUGSINPY_HOME": "/home/bugsinpy", "PATH": "/usr/bin"}

    sandbox.checkout(version=1)

    call_args: str = sandbox.sandbox.run_command.call_args[0][0]
    assert "-v 1" in call_args


def test_compile_passes_project_root(sandbox: BugsInPyDockerSandbox):
    """compile() should target container_project_root with -w."""
    sandbox.sandbox.run_command = MagicMock(return_value=(0, "", ""))
    sandbox.env_vars = {"BUGSINPY_HOME": "/home/bugsinpy", "PATH": "/usr/bin"}

    sandbox.compile()

    call_args: str = sandbox.sandbox.run_command.call_args[0][0]
    assert "bugsinpy-compile" in call_args
    assert sandbox.container_project_root in call_args


def test_test_relevant_flag(sandbox: BugsInPyDockerSandbox):
    """test(relevant=True) should pass -r flag."""
    sandbox.sandbox.run_command = MagicMock(return_value=(0, "", ""))
    sandbox.env_vars = {"BUGSINPY_HOME": "/home/bugsinpy", "PATH": "/usr/bin"}

    sandbox.test(relevant=True)

    call_args: str = sandbox.sandbox.run_command.call_args_list[0][0][0]
    assert "-r" in call_args


def test_test_all_flag(sandbox: BugsInPyDockerSandbox):
    """test(relevant=False) should pass -a flag."""
    sandbox.sandbox.run_command = MagicMock(return_value=(0, "", ""))
    sandbox.env_vars = {"BUGSINPY_HOME": "/home/bugsinpy", "PATH": "/usr/bin"}

    sandbox.test(relevant=False)

    call_args: str = sandbox.sandbox.run_command.call_args_list[0][0][0]
    assert "-a" in call_args


def test_run_dynamic_tracer_command_shape(sandbox: BugsInPyDockerSandbox):
    """
    run_dynamic_tracer should build a command that includes:
      - the venv python path
      - the dynamic_call_graph module
      - --repo pointing at container_project_root
      - --test-mode flag
    """
    sandbox.sandbox.run_command = MagicMock(
        side_effect=[
            (0, 'test_file="tests/test_download.py"', ""),  # grep for test file
            (0, "", ""),  # pip install
            (0, "", ""),  # tracer run
        ]
    )

    sandbox.run_dynamic_tracer(output_file="cg.json")

    tracer_call_args: str = sandbox.sandbox.run_command.call_args_list[2][0][0]
    assert "src.program_analysis.dynamic_call_graph" in tracer_call_args
    assert f"--repo {sandbox.container_project_root}" in tracer_call_args
    assert "--test-mode" in tracer_call_args
    assert "cg.json" in tracer_call_args
