import docker
from typing import Tuple
import logging
import tarfile
import io
import os


class SimpleDockerSandbox:
    """
    A simple Docker sandbox that can be used to run commands in a container. This initializes a container with the given image and keeps it running.
    This class implements the context manager protocol, so it can be used with the with statement. This means that the container is started when the context is entered and stopped when the context is exited.

    Args:
        image_name: The name of the image to use. Defaults to "python:3.11-slim".
        sandbox_dir: The directory inside the container to use as the default working directory.
        keep_alive: If True, the container will not be stopped or removed when the context exits.

    Attributes:
        image_name: The name of the image to use.
        client: The Docker client.
        container: The container object.

    Methods:
        start: Starts the container in detached mode.
        stop: Stops and removes the container (unless keep_alive is True).
        run_command: Executes a bash command in the container.
    """

    def __init__(
        self,
        image_name: str = "python:3.11-slim",
        sandbox_dir: str = "/codebase",
        keep_alive: bool = False,
    ):
        self.image_name = image_name
        self.sandbox_dir = sandbox_dir
        self.keep_alive = keep_alive
        self.client = docker.from_env()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='[sandbox_%(levelname)s] %(message)s'
        )
        self.container = None

    def start(self):
        """Starts the container in detached mode."""
        if self.container:
            self.logger.warning("Container already running.")
            return self.container

        self.logger.info(f"Starting container {self.image_name} in detached mode...")
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            self.logger.info(f"Pulling image {self.image_name}...")
            self.client.images.pull(self.image_name)

        self.container = self.client.containers.run(
            self.image_name,
            command="tail -f /dev/null",  # Keep the container alive
            detach=True,
            tty=True,
            working_dir=self.sandbox_dir,  # Sets default dir and creates it if missing
            labels={"created_by": "SimpleDockerSandbox"},
        )
        self.logger.info(f"Container {self.container.id} started")
        return self.container

    def stop(self):
        """Stops and removes the container."""
        if self.container:
            if self.keep_alive:
                self.logger.info(f"Keeping container {self.container.id} alive...")
                return

            self.logger.info(f"Stopping container {self.container.id}...")
            try:
                self.container.stop()
                self.container.remove()
            except Exception as e:
                self.logger.error(f"Error stopping container: {e}")
            finally:
                self.container = None
        else:
            self.logger.debug("Stop called but no container is running.")

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

        if workdir is None:
            workdir = self.sandbox_dir

        result = self.container.exec_run(
            cmd=["bash", "-c", command], workdir=workdir, demux=True
        )
        self.logger.info(f"Command '{command}' executed with exit code {result.exit_code}")

        stdout_bytes = result.output[0] if result.output[0] else b""
        stderr_bytes = result.output[1] if result.output[1] else b""

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        return (
            result.exit_code,
            stdout,
            stderr,
        )

    def copy_to(self, src: str, dest: str, workdir: str = None):
        """
        Copies a file or directory from the host to the container.

        Args:
            src: Source path on the host.
            dest: Destination path in the container.
            workdir: Optional working directory in the container. Defaults to sandbox_dir.
        """
        if not self.container:
            raise RuntimeError("Container is not running.")

        if workdir is None:
            workdir = self.sandbox_dir

        # Ensure dest is scoped to workdir
        if not dest.startswith(workdir):
            dest = os.path.join(workdir, dest.lstrip("/"))

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            if os.path.isfile(src):
                target_path = os.path.dirname(dest)
                if not target_path:
                    target_path = "."
                tar.add(src, arcname=os.path.basename(dest))
            else:
                target_path = dest
                for item in os.listdir(src):
                    item_path = os.path.join(src, item)
                    tar.add(item_path, arcname=item)

        self.run_command(f"mkdir -p {target_path}")
        tar_stream.seek(0)
        self.container.put_archive(target_path, tar_stream)

    def copy_from(self, src: str, dest: str, workdir: str = None):
        """
        Copies a file or directory from the container to the host.

        Args:
            src: Source path in the container.
            dest: Destination path on the host.
            workdir: Optional working directory in the container. Defaults to sandbox_dir.
        """
        if not self.container:
            raise RuntimeError("Container is not running.")

        if workdir is None:
            workdir = self.sandbox_dir

        # Ensure src is scoped to workdir
        if not src.startswith(workdir):
            src = os.path.join(workdir, src.lstrip("/"))

        bits, _ = self.container.get_archive(src)
        tar_stream = io.BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)
        tar_stream.seek(0)

        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            tar.extractall(path=dest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[sandbox_%(levelname)s] %(message)s")
    with SimpleDockerSandbox() as sandbox:
        sandbox.copy_to("README.md", "/README_root.md")
        sandbox.copy_to("README.md", "README_relative.md")
        current_dir = os.path.dirname(__file__)
        sandbox.copy_to(current_dir, "/utils")
        sandbox.run_command("ls -l /")
        sandbox.run_command(f"ls -l {sandbox.sandbox_dir}")
        sandbox.run_command("ls -l /utils")
