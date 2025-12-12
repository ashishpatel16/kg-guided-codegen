from swe_bench_container import SWEBenchContainer
import docker
import platform
from pathlib import Path
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_container, build_env_images, setup_logger

def setup_swe():
    # 1. Load Instance Data
    dataset = load_swebench_dataset("princeton-nlp/SWE-bench_Lite", "test")
    # print(dataset[0])
    instance_id = dataset[0]['instance_id']

    instance = dataset[0]

    # # 2. Create Test Spec
    # Detect architecture (important for Apple Silicon)
    arch = "arm64" if platform.machine() in {"arm64", "aarch64"} else "x86_64"
    test_spec = make_test_spec(instance, arch=arch)

    # print(test_spec, arch)

    # # # 3. Build Images & Container 
    client = docker.from_env()
    import os
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), "../../logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{instance_id}_build.log")
    
    logger = setup_logger(instance_id, Path(log_file))

    # Build environment images first
    build_env_images(client, [test_spec], force_rebuild=False, max_workers=1)
    
    # Build and start the instance container
    container = build_container(
        test_spec=test_spec,
        client=client,
        run_id="my_app_run",
        logger=logger,
        nocache=False
    )
    
    container.start()
    print(f"********Container {container.name} is running!")
    # 4. Interact
    output = container.exec_run("ls /testbed")
    print(output.output.decode())
    # 5. Cleanup
    container.stop()
    container.remove()

if __name__ == "__main__":
    # main()
    setup_swe()
