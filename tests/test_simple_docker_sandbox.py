import pytest
from src.docker_utils.basic_container import SimpleDockerSandbox
import docker

# Use a lightweight image for faster tests
TEST_IMAGE = "alpine:latest"


def test_initialization():
    sandbox = SimpleDockerSandbox(image_name=TEST_IMAGE, sandbox_dir="/test_dir")
    assert sandbox.image_name == TEST_IMAGE
    assert sandbox.sandbox_dir == "/test_dir"


def test_start_pulls_image_if_missing():
    try:
        docker_client = docker.from_env()
        docker_client.images.remove(TEST_IMAGE, force=False)
    except Exception:
        pass
    with SimpleDockerSandbox(image_name=TEST_IMAGE) as sandbox:
        assert sandbox.container is not None
        assert sandbox.container.image.tags == [TEST_IMAGE]


def test_lifecycle_real_container():
    with SimpleDockerSandbox(image_name=TEST_IMAGE) as sandbox:
        assert sandbox.container is not None
        sandbox.container.reload()  # Refresh state
        assert sandbox.container.status == "running"

        container_id = sandbox.container.id

    # After exiting the context manager, container should be gone (or at least stopped/removed)
    client = docker.from_env()
    try:
        # Try to get the container - it should raise NotFound if it was removed
        client.containers.get(container_id)
        assert False, "Container should have been removed"
    except docker.errors.NotFound:
        pass


def test_run_command_files_real():
    with SimpleDockerSandbox(image_name="python:3.11-slim") as sandbox:
        # 1. Simple echo
        exit_code, stdout, stderr = sandbox.run_command("echo 'Hello World'")
        assert exit_code == 0
        assert "Hello World" in stdout.strip()

        # 2. Create a file in the default sandbox_dir (/codebase)
        exit_code, _, _ = sandbox.run_command("echo 'content' > test_file.txt")
        assert exit_code == 0

        # 3. Verify file exists and content matches
        exit_code, stdout, _ = sandbox.run_command("cat test_file.txt")
        assert exit_code == 0
        assert "content" in stdout.strip()

        # 4. Verify working directory is indeed /codebase
        exit_code, stdout, _ = sandbox.run_command("pwd")
        assert stdout.strip() == "/codebase"


def test_run_command_failure_real():
    with SimpleDockerSandbox(image_name="python:3.11-slim") as sandbox:
        exit_code, stdout, stderr = sandbox.run_command("non_existent_command")
        assert exit_code != 0
        assert "command not found" in stderr.lower() or "not found" in stderr.lower()


def test_run_command_without_start_raises_error():
    sandbox = SimpleDockerSandbox()
    with pytest.raises(RuntimeError, match="Container is not running"):
        sandbox.run_command("ls")


def test_stop_handles_errors_gracefully():
    sandbox = SimpleDockerSandbox(image_name=TEST_IMAGE)
    sandbox.start()
    sandbox.container.remove(
        force=True
    )  # Force a removal so sandbox.stop() fails on .stop() or .remove()
    try:
        sandbox.stop()  # Now calling stop() should trigger the exception block but not crash
    except Exception:
        pytest.fail("sandbox.stop() raised an exception instead of handling it")

    assert sandbox.container is None
