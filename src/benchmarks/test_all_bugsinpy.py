import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.benchmarks.setup_bugsinpy_docker import BugsInPyDockerSandbox

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bugsinpy_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BugsInPyPipeline")

def discover_bugs(bugsinpy_root: str) -> Dict[str, List[str]]:
    """
    Discovers all projects and their bug IDs in the BugsInPy dataset.
    """
    projects_dir = Path(bugsinpy_root) / "projects"
    bugs_data = {}
    
    if not projects_dir.exists():
        logger.error(f"Projects directory not found at {projects_dir}")
        return {}

    for project_path in projects_dir.iterdir():
        if project_path.is_dir():
            project_name = project_path.name
            bugs_dir = project_path / "bugs"
            if bugs_dir.exists() and bugs_dir.is_dir():
                bug_ids = [d.name for d in bugs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                bug_ids.sort(key=int)
                bugs_data[project_name] = bug_ids
    
    return bugs_data

def run_bug_test(project: str, bug_id: str, bugsinpy_root: str, experiments_dir: str) -> Dict[str, Any]:
    """
    Runs a single bug test in a Docker container.
    """
    logger.info(f"Starting test for {project} bug {bug_id}")
    result = {
        "project": project,
        "bug_id": bug_id,
        "status": "FAILED",
        "steps": {},
        "error": None,
        "duration": 0
    }
    
    start_time = datetime.now()
    
    try:
        # keep_alive=False ensures the container is killed even if we don't catch all exceptions
        with BugsInPyDockerSandbox(
            project_name=project,
            bug_id=bug_id,
            bugsinpy_root=bugsinpy_root,
            experiments_dir=experiments_dir,
            keep_alive=False
        ) as bspy:
            
            # 1. Checkout
            logger.info(f"[{project}-{bug_id}] Checking out...")
            exit_code, out, err = bspy.checkout(version=0)
            result["steps"]["checkout"] = {"exit_code": exit_code, "stdout": out, "stderr": err}
            if exit_code != 0:
                result["error"] = "Checkout failed"
                return result

            # 2. Compile
            logger.info(f"[{project}-{bug_id}] Compiling...")
            exit_code, out, err = bspy.compile(verbose=False)
            result["steps"]["compile"] = {"exit_code": exit_code, "stdout": out, "stderr": err}
            if exit_code != 0:
                result["error"] = "Compile failed"
                return result

            # 3. Test
            logger.info(f"[{project}-{bug_id}] Running tests...")
            exit_code, out, err = bspy.test(relevant=True, verbose=False)
            result["steps"]["test"] = {"exit_code": exit_code, "stdout": out, "stderr": err}
            
            # Check for failure file (already handled by bspy.test returns 1 if it exists)
            if exit_code == 0:
                result["status"] = "SUCCESS"
            else:
                result["status"] = "TEST_FAILED"
                # Try to get failure details
                _, fail_details, _ = bspy.sandbox.run_command(
                    f"cat {bspy.container_project_root}/bugsinpy_fail.txt"
                )
                result["fail_details"] = fail_details

    except Exception as e:
        logger.exception(f"Unexpected error testing {project} bug {bug_id}: {e}")
        result["error"] = str(e)
        result["status"] = "CRASHED"
    
    finally:
        result["duration"] = (datetime.now() - start_time).total_seconds()
        logger.info(f"Finished test for {project} bug {bug_id} in {result['duration']:.2f}s. Status: {result['status']}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Run all BugsInPy instances in Docker.")
    parser.add_argument("--bugsinpy-root", default="datasets/BugsInPy", help="Path to BugsInPy root")
    parser.add_argument("--experiments-dir", default="experiments/pipeline_runs", help="Dir for experiment data")
    parser.add_argument("--limit-projects", type=int, help="Limit number of projects to test")
    parser.add_argument("--limit-bugs", type=int, help="Limit number of bugs per project")
    parser.add_argument("--project", help="Run only a specific project")
    parser.add_argument("--bug-id", help="Run only a specific bug ID (requires --project)")
    parser.add_argument("--output", default="bugsinpy_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    bugs_data = discover_bugs(args.bugsinpy_root)
    
    if args.project:
        if args.project not in bugs_data:
            print(f"Project {args.project} not found.")
            return
        if args.bug_id:
            if args.bug_id not in bugs_data[args.project]:
                print(f"Bug {args.bug_id} not found in project {args.project}.")
                return
            bugs_to_run = {args.project: [args.bug_id]}
        else:
            bugs_to_run = {args.project: bugs_data[args.project]}
    else:
        bugs_to_run = bugs_data

    # Apply limits
    final_bugs_to_run = {}
    projects_count = 0
    for project, bugs in bugs_to_run.items():
        if args.limit_projects and projects_count >= args.limit_projects:
            break
        
        subset_bugs = bugs[:args.limit_bugs] if args.limit_bugs else bugs
        final_bugs_to_run[project] = subset_bugs
        projects_count += 1

    total_bugs = sum(len(bugs) for bugs in final_bugs_to_run.values())
    logger.info(f"Planned to run {total_bugs} bugs across {len(final_bugs_to_run)} projects.")

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "results": []
    }

    try:
        current_count = 0
        for project, bugs in final_bugs_to_run.items():
            for bug_id in bugs:
                current_count += 1
                logger.info(f"Progress: {current_count}/{total_bugs}")
                result = run_bug_test(project, bug_id, args.bugsinpy_root, args.experiments_dir)
                report["results"].append(result)
                
                # Save intermediate report
                with open(args.output, "w") as f:
                    json.dump(report, f, indent=2)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving partial report.")
    
    # Final statistics
    success_count = sum(1 for r in report["results"] if r["status"] == "SUCCESS")
    logger.info(f"Pipeline finished. Success: {success_count}/{len(report['results'])}")
    logger.info(f"Final report saved to {args.output}")

if __name__ == "__main__":
    main()
