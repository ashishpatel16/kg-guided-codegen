import docker
from typing import Tuple


class SimpleDockerSandbox:
    """
    A simple Docker sandbox that can be used to run commands in a container. This initializes a container with the given image and keeps it running.
    This class implements the context manager protocol, so it can be used with the with statement. This means that the container is started when the context is entered and stopped when the context is exited.

    Args:
        image_name: The name of the image to use. Defaults to "python:3.11-slim".

    Attributes:
        image_name: The name of the image to use.
        client: The Docker client.
        container: The container object.

    Methods:
        start: Starts the container in detached mode.
        stop: Stops and removes the container.
        run_command: Executes a bash command in the container.

    Example:
        with SimpleDockerSandbox() as sandbox:
            sandbox.run_command("echo 'Hello, World!'")
    """

    def __init__(
        self, image_name: str = "python:3.11-slim", sandbox_dir: str = "/codebase"
    ):
        self.image_name = image_name
        self.sandbox_dir = sandbox_dir  # Ensures that the code is always written to the same directory in the container, default is /codebase
        self.client = (
            docker.from_env()
        )  # uses the default Docker client configuration on the machine
        self.container = None

    def start(self):
        """Starts the container in detached mode."""
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            print(f"Pulling image {self.image_name}...")
            self.client.images.pull(self.image_name)

        self.container = self.client.containers.run(
            self.image_name,
            command="tail -f /dev/null",  # Keep the container alive
            detach=True,
            tty=True,
            working_dir=self.sandbox_dir,  # Sets default dir and creates it if missing
        )

    def stop(self):
        """Stops and removes the container."""
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
            except Exception as e:
                print(f"Error stopping container: {e}")
            finally:
                self.container = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run_command(self, command: str, workdir: str = None) -> Tuple[int, str, str]:
        """
        Executes a bash command in the container.

        Args:
            command: The command to run.
            workdir: Optional working directory. Defaults to sandbox_dir.

        Returns:
            Tuple containing (exit_code, stdout, stderr).
        """
        if not self.container:
            raise RuntimeError("Container is not running. Call start() first.")

        # Default to sandbox_dir if workdir is not specified
        if workdir is None:
            workdir = self.sandbox_dir

        # Use bash to execute the command to support shell features
        result = self.container.exec_run(
            cmd=["bash", "-c", command], workdir=workdir, demux=True
        )

        stdout = result.output[0] if result.output[0] else b""
        stderr = result.output[1] if result.output[1] else b""

        return (
            result.exit_code,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
