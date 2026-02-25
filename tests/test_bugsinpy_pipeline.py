"""
Integration tests for the BugsInPy debugging pipeline (main_bugsinpy.py).

Verifies that `run_bugsinpy_debugging` wires up the Docker sandbox and the
debugging agent correctly, using mocks to avoid real Docker/LLM calls.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.main_bugsinpy import run_bugsinpy_debugging


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_CALL_GRAPH: Dict[str, Any] = {
    "nodes": [
        {
            "fqn": "youtube_dl.extractor.YoutubeIE._real_extract",
            "file": "/home/workspace/youtube-dl/youtube_dl/extractor/youtube.py",
            "start_line": 100,
            "end_line": 200,
            "suspiciousness": 0.8,
        },
        {
            "fqn": "youtube_dl.utils.sanitize_filename",
            "file": "/home/workspace/youtube-dl/youtube_dl/utils.py",
            "start_line": 50,
            "end_line": 70,
            "suspiciousness": 0.2,
        },
    ],
    "edges": [
        {
            "source": "youtube_dl.extractor.YoutubeIE._real_extract",
            "target": "youtube_dl.utils.sanitize_filename",
        }
    ],
}


@pytest.fixture
def mock_sandbox():
    """
    Returns a mock BugsInPyDockerSandbox that simulates a successful
    checkout → compile → trace pipeline.
    """
    mock_bspy = MagicMock()
    mock_bspy.checkout.return_value = (0, "", "")
    mock_bspy.compile.return_value = (0, "", "")
    mock_bspy.run_dynamic_tracer.return_value = (0, "", "")
    mock_bspy.container_project_root = "/home/workspace/youtube-dl"
    mock_bspy.host_experiments_dir = Path("/tmp/experiments")
    mock_bspy.sandbox = MagicMock()
    mock_bspy.sandbox.container = MagicMock()
    mock_bspy.sandbox.container.id = "abc123"
    # Make it work as a context manager
    mock_bspy.__enter__ = MagicMock(return_value=mock_bspy)
    mock_bspy.__exit__ = MagicMock(return_value=False)
    return mock_bspy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("src.main_bugsinpy.debugging_agent")
@patch("src.main_bugsinpy.BugsInPyDockerSandbox")
def test_pipeline_passes_correct_state_keys(
    mock_sandbox_cls: MagicMock,
    mock_agent: MagicMock,
    mock_sandbox: MagicMock,
    tmp_path: Path,
):
    """
    The initial_state dict given to the agent must contain all required keys:
    call_graph, container_id, use_docker, host_workspace, test_command, etc.
    """
    mock_sandbox_cls.return_value = mock_sandbox

    # Write a fake call graph JSON
    project_dir: Path = tmp_path / "youtube-dl"
    project_dir.mkdir()
    cg_path: Path = project_dir / "call_graph_youtube-dl_1.json"
    cg_path.write_text(json.dumps(FAKE_CALL_GRAPH))

    # Override host_experiments_dir to point at tmp_path
    mock_sandbox.host_experiments_dir = tmp_path

    # Make the agent return a state with the nodes
    mock_agent.invoke.return_value = {
        "call_graph": FAKE_CALL_GRAPH,
        "reflection": "Found a bug.",
    }

    run_bugsinpy_debugging("youtube-dl", "1")

    # Assert agent was called exactly once
    mock_agent.invoke.assert_called_once()

    # Extract the state that was passed to the agent
    state: Dict[str, Any] = mock_agent.invoke.call_args[0][0]

    # Verify required keys
    expected_keys = [
        "call_graph",
        "score_delta",
        "test_command",
        "container_id",
        "container_workspace",
        "host_workspace",
        "use_docker",
        "llm_calls",
    ]
    for key in expected_keys:
        assert key in state, f"Missing key '{key}' in agent initial_state"

    assert state["use_docker"] is True
    assert state["container_id"] == "abc123"
    assert state["container_workspace"] == "/home/workspace/youtube-dl"
    assert isinstance(state["call_graph"], dict)
    assert "nodes" in state["call_graph"]


@patch("src.main_bugsinpy.debugging_agent")
@patch("src.main_bugsinpy.BugsInPyDockerSandbox")
def test_pipeline_remaps_container_paths_to_host(
    mock_sandbox_cls: MagicMock,
    mock_agent: MagicMock,
    mock_sandbox: MagicMock,
    tmp_path: Path,
):
    """
    Node file paths should be remapped from container paths to host paths.
    """
    mock_sandbox_cls.return_value = mock_sandbox

    project_dir: Path = tmp_path / "youtube-dl"
    project_dir.mkdir()
    cg_path: Path = project_dir / "call_graph_youtube-dl_1.json"
    cg_path.write_text(json.dumps(FAKE_CALL_GRAPH))

    mock_sandbox.host_experiments_dir = tmp_path

    mock_agent.invoke.return_value = {
        "call_graph": FAKE_CALL_GRAPH,
        "reflection": "Done.",
    }

    run_bugsinpy_debugging("youtube-dl", "1")

    state: Dict[str, Any] = mock_agent.invoke.call_args[0][0]
    for node in state["call_graph"]["nodes"]:
        file_path: str = node.get("file", "")
        # Should NOT start with the container path anymore
        assert not file_path.startswith("/home/workspace/youtube-dl"), (
            f"Node file path was not remapped: {file_path}"
        )
        # Should be an absolute host path
        assert os.path.isabs(file_path), f"Node file path is not absolute: {file_path}"


@patch("src.main_bugsinpy.debugging_agent")
@patch("src.main_bugsinpy.BugsInPyDockerSandbox")
def test_pipeline_aborts_on_checkout_failure(
    mock_sandbox_cls: MagicMock,
    mock_agent: MagicMock,
    mock_sandbox: MagicMock,
):
    """
    If checkout fails, the agent should NOT be invoked.
    """
    mock_sandbox_cls.return_value = mock_sandbox
    mock_sandbox.checkout.return_value = (1, "", "checkout error")

    run_bugsinpy_debugging("youtube-dl", "1")

    mock_agent.invoke.assert_not_called()


@patch("src.main_bugsinpy.debugging_agent")
@patch("src.main_bugsinpy.BugsInPyDockerSandbox")
def test_pipeline_aborts_on_compile_failure(
    mock_sandbox_cls: MagicMock,
    mock_agent: MagicMock,
    mock_sandbox: MagicMock,
):
    """
    If compile fails, the agent should NOT be invoked.
    """
    mock_sandbox_cls.return_value = mock_sandbox
    mock_sandbox.compile.return_value = (1, "", "compile error")

    run_bugsinpy_debugging("youtube-dl", "1")

    mock_agent.invoke.assert_not_called()


@patch("src.main_bugsinpy.debugging_agent")
@patch("src.main_bugsinpy.BugsInPyDockerSandbox")
def test_pipeline_aborts_on_tracer_failure(
    mock_sandbox_cls: MagicMock,
    mock_agent: MagicMock,
    mock_sandbox: MagicMock,
):
    """
    If the dynamic tracer fails, the agent should NOT be invoked.
    """
    mock_sandbox_cls.return_value = mock_sandbox
    mock_sandbox.run_dynamic_tracer.return_value = (1, "", "tracer error")

    run_bugsinpy_debugging("youtube-dl", "1")

    mock_agent.invoke.assert_not_called()
