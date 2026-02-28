import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.benchmarks.setup_bugsinpy_docker import BugsInPyDockerSandbox

logger: logging.Logger = logging.getLogger(__name__)

def run_bugsinpy_suspiciousness(
    project_name: str,
    bug_id: str,
    bugsinpy_root: str = "datasets/BugsInPy",
    experiments_dir: str = "experiments",
    output_dir: str = "testgen_artifacts",
) -> bool:
    """
    Spins up the Docker container for a given BugsInPy project and bug_id,
    runs the dynamic tracer, and extracts suspiciousness & coverage into a CSV file.
    """
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"  BugsInPy Suspiciousness: {project_name} #{bug_id}")
    print(f"{'='*60}\n")
    
    try:
        with BugsInPyDockerSandbox(
            project_name,
            bug_id,
            bugsinpy_root=bugsinpy_root,
            experiments_dir=experiments_dir,
        ) as bspy:
            print("[1/3] Checking out buggy version...")
            exit_code, out, err = bspy.checkout(version=0)
            if exit_code != 0:
                print(f"  ✗ Checkout failed: {err}")
                return False

            print("[2/3] Compiling project...")
            exit_code, out, err = bspy.compile(verbose=True)
            if exit_code != 0:
                print(f"  ✗ Compile failed: {err}")
                return False

            output_filename: str = f"call_graph_{project_name}_{bug_id}.json"
            print(f"[3/3] Running dynamic tracer → {output_filename}")
            exit_code, out, err = bspy.run_dynamic_tracer(output_file=output_filename)
            if exit_code != 0:
                print(f"  ✗ Tracer failed: {err}")
                return False
                
            call_graph_path: Path = bspy.host_experiments_dir / project_name / output_filename
            if not call_graph_path.exists():
                print(f"  ✗ Call graph not found at {call_graph_path}")
                return False

            with open(call_graph_path, "r") as f:
                call_graph: Dict[str, Any] = json.load(f)

            nodes: List[Dict[str, Any]] = call_graph.get("nodes", [])
            coverage_matrix: Dict[str, List[str]] = call_graph.get("coverage_matrix", {})

            csv_filename: str = os.path.join(output_dir, f"suspiciousness_{project_name}_{bug_id}.csv")
            os.makedirs(output_dir, exist_ok=True)
            
            with open(csv_filename, "w", newline="") as csvfile:
                fieldnames: List[str] = ["fqn", "suspiciousness", "covered_by_tests"]
                writer: csv.DictWriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for node in nodes:
                    fqn: str = str(node.get("fqn", ""))
                    susp: float = float(node.get("suspiciousness", 0.0))
                    tests: List[str] = coverage_matrix.get(fqn, [])
                    tests_str: str = ";".join(tests)
                    row: Dict[str, Any] = {
                        "fqn": fqn,
                        "suspiciousness": susp,
                        "covered_by_tests": tests_str
                    }
                    writer.writerow(row)
            
            print(f"  ✓ Suspiciousness data saved to {csv_filename}")
            return True

    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
        print(f"\n  ✗ CRASHED: {e}")
        return False


def run_benchmark(
    project_name: str, 
    k: int, 
    bugsinpy_root: str = "datasets/BugsInPy", 
    experiments_dir: str = "experiments", 
    output_dir: str = "artifacts"
) -> None:
    """
    Runs k instances of a specific project and computes suspiciousness metrics.
    """
    print(f"Starting benchmark for {project_name}, running {k} instances.")
    
    successful: int = 0
    bug_id: int = 1
    
    while successful < k:
        success: bool = run_bugsinpy_suspiciousness(
            project_name=project_name,
            bug_id=str(bug_id),
            bugsinpy_root=bugsinpy_root,
            experiments_dir=experiments_dir,
            output_dir=output_dir
        )
        
        if success:
            successful += 1
        else:
            print(f"Warning: Bug ID {bug_id} failed or was not found.")
            
        bug_id += 1
        
        # Failsafe to prevent endless looping if we run out of valid bugs
        # For instance, if k=10 but project only has 5 valid bug IDs
        if bug_id > k * 5:
            print("Stopping: Too many consecutive failures or exhausted bug IDs.")
            break
            
    print(f"Finished. Successfully processed {successful}/{k} requested instances of {project_name}.")

if __name__ == "__main__":
    id = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    run_benchmark(project_name="youtube-dl", k=1, experiments_dir="experiments", output_dir=f"testgen_artifacts/{id}")
