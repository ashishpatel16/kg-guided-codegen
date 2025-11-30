from basic_container import SimpleDockerSandbox


def main():
    print("Initializing Sandbox...")
    # Using default python:3.11-slim image
    with SimpleDockerSandbox() as sandbox:
        print("Sandbox started.")

        # 1. Check python version
        print("\n--- Checking Python Version ---")
        exit_code, stdout, stderr = sandbox.run_command("python --version")
        print(f"Exit Code: {exit_code}")
        print(f"Stdout: {stdout.strip()}")
        print(f"Stderr: {stderr.strip()}")

        # 2. Create a file using bash redirect
        print("\n--- Creating a test file ---")
        cmd = "echo 'print(\"Hello from inside the container!\")' > hello.py"
        print(f"Running: {cmd}")
        exit_code, stdout, stderr = sandbox.run_command(cmd)
        print(f"Exit Code: {exit_code}")
        if stderr:
            print(f"Stderr: {stderr.strip()}")

        print("\n--- Installing a package ---")
        cmd = "pip install django"
        print(f"Running: {cmd}")
        exit_code, stdout, stderr = sandbox.run_command(cmd)
        print(
            f"Exit Code: {exit_code}\nStdout: {stdout.strip()}\nStderr: {stderr.strip()}"
        )

        print("\n--- Running pip freeze to verify installation ---")
        cmd = "pip freeze"
        print(f"Running: {cmd}")
        exit_code, stdout, stderr = sandbox.run_command(cmd)
        print(
            f"Exit Code: {exit_code}\nStdout: {stdout.strip()}\nStderr: {stderr.strip()}"
        )

        # 3. Run the file
        print("\n--- Running the test file ---")
        exit_code, stdout, stderr = sandbox.run_command("python hello.py")
        print(f"Exit Code: {exit_code}")
        print(f"Stdout: {stdout.strip()}")
        if stderr:
            print(f"Stderr: {stderr.strip()}")

        print("\n--- Cat test file ---")
        cmd = "cat hello.py"
        print(f"Running: {cmd}")
        exit_code, stdout, stderr = sandbox.run_command(cmd)
        print(
            f"Exit Code: {exit_code}\nStdout: {stdout.strip()}\nStderr: {stderr.strip()}"
        )

        # 4. List files to verify persistence
        print("\n--- Listing files ---")
        exit_code, stdout, stderr = sandbox.run_command("ls")
        print(f"Exit Code: {exit_code}")
        print(f"Stdout:\n{stdout.strip()}")

        print("\n--- Print current directory ---")
        cmd = "pwd"
        print(f"Running: {cmd}")
        exit_code, stdout, stderr = sandbox.run_command(cmd)
        print(
            f"Exit Code: {exit_code}\nStdout: {stdout.strip()}\nStderr: {stderr.strip()}"
        )


if __name__ == "__main__":
    main()
