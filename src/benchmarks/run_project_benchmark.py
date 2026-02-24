"""
Benchmark Script for a Specific BugsInPy Project.

Runs the fault localization agent against all available bug IDs for a given project,
and compiles a summary report indicating which bugs were successfully localized,
which were inconclusive, and which crashed.

Usage:
    uv run python -m src.benchmarks.run_project_benchmark <project_name>
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure logging doesn't spam too much in benchmark loop
from src.agent.fault_localization.tools import configure_logging
configure_logging(level="WARNING")

from src.main_bugsinpy import run_bugsinpy_debugging

def discover_project_bugs(project_name: str, bugsinpy_root: str) -> List[str]:
    """Find all valid bug IDs for a given project in the BugsInPy dataset."""
    project_dir = Path(bugsinpy_root) / "projects" / project_name / "bugs"
    if not project_dir.exists() or not project_dir.is_dir():
        print(f"Error: Project bugs directory not found at {project_dir}")
        return []

    bug_ids = [d.name for d in project_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    bug_ids.sort(key=int)
    return bug_ids

def run_benchmark(
    project_name: str,
    bugsinpy_root: str = "datasets/BugsInPy",
    limit: int = 0,
):
    print(f"Starting Benchmark for project: {project_name}")
    bug_ids = discover_project_bugs(project_name, bugsinpy_root)
    
    if not bug_ids:
        print(f"No bugs found for project {project_name}.")
        return

    if limit > 0:
        bug_ids = bug_ids[:limit]
        
    print(f"Found {len(bug_ids)} bugs to run: {bug_ids}")
    print(f"Outputting silent stream (logs are turned to WARNING level). Check artifacts/ folder for realtime JSON save.")

    results: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"artifacts/benchmark_{project_name}_{timestamp}.md"
    json_file = f"artifacts/benchmark_{project_name}_{timestamp}.json"
    
    os.makedirs("artifacts", exist_ok=True)

    for i, bug_id in enumerate(bug_ids, 1):
        print(f"\n[Benchmarking {i}/{len(bug_ids)}] {project_name} #{bug_id}...")
        
        try:
            res = run_bugsinpy_debugging(
                project_name=project_name,
                bug_id=bug_id,
                bugsinpy_root=bugsinpy_root,
                experiments_dir="experiments",
                artifacts_dir="artifacts",
                recursion_limit=100,
            )
            results.append(res)
            
            # Print brief summary of outcome
            status = res.get("status")
            if status == "LOCALIZED" and res.get("culprit"):
                print(f"   => success: {res['culprit']['fqn']} ({res['culprit']['confidence']:.4f})")
            elif status == "INCONCLUSIVE":
                print(f"   => inconclusive. Top candidate: {res.get('top_candidates', [{}])[0].get('fqn')}")
            else:
                print(f"   => FAILED: {res.get('error', 'unknown error')}")
                
        except Exception as e:
            print(f"FATAL ERROR running bug {bug_id}: {e}")
            results.append({
                "project": project_name,
                "bug_id": bug_id,
                "status": "CRASHED",
                "error": str(e),
                "duration_s": 0.0,
            })
            
        # Save intermediate JSON
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)

    # Compile the final Markdown report
    compile_summary_report(project_name, results, report_file)
    print(f"\nBenchmark complete for {project_name}.")
    print(f"Summary Report saved to: {report_file}")
    print(f"Raw JSON Data saved to: {json_file}")

def compile_summary_report(project_name: str, results: List[Dict[str, Any]], report_file: str):
    total = len(results)
    localized = [r for r in results if r.get("status") == "LOCALIZED"]
    inconclusive = [r for r in results if r.get("status") == "INCONCLUSIVE"]
    crashed = [r for r in results if r.get("status") in ("CRASHED", "FAILED")]
    
    with open(report_file, "w") as f:
        f.write(f"# Fault Localization Benchmark: `{project_name}`\n\n")
        f.write(f"**Total Bugs Run**: {total}\n")
        f.write(f"- **Agent Success (LOCALIZED)**: {len(localized)}\n")
        f.write(f"- **Agent Finished (INCONCLUSIVE)**: {len(inconclusive)}\n")
        f.write(f"- **Pipeline Failure (CRASHED)**: {len(crashed)}\n\n")
        
        f.write("## Outcomes per Bug\n\n")
        f.write("| Bug ID | Status | Identified Culprit (FQN) | Confidence | Time (s) |\n")
        f.write("|---|---|---|---|---|\n")
        
        for r in results:
            b_id = r.get("bug_id", "?")
            status = r.get("status", "UNKNOWN")
            time_s = r.get("duration_s", 0)
            
            if status == "LOCALIZED" and r.get("culprit"):
                culprit_fqn = r["culprit"]["fqn"]
                conf = f"{r['culprit']['confidence']:.4f}"
            else:
                culprit_fqn = "-"
                conf = "-"
                
            if status in ("CRASHED", "FAILED"):
                # For crashed, put the error message in the table
                culprit_fqn = str(r.get("error", "Unknown error")).replace("\n", " ")[:50] + "..."
            
            f.write(f"| {b_id} | {status} | `{culprit_fqn}` | {conf} | {time_s} |\n")
            
        f.write("\n## Detailed Candidates (Inconclusive Runs)\n\n")
        for r in inconclusive:
            b_id = r.get("bug_id")
            f.write(f"### Bug #{b_id}\n")
            top_candidates = r.get("top_candidates", [])
            for c in top_candidates:
                f.write(f"- `{c.get('fqn')}` (conf: {c.get('confidence', 0):.4f})\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete benchmark for a project.")
    parser.add_argument("project_name", help="BugsInPy project (e.g. youtube-dl)")
    parser.add_argument("--bugsinpy-root", default="datasets/BugsInPy", help="Path to BugsInPy dataset")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N bugs")
    
    args = parser.parse_args()
    
    run_benchmark(
        project_name=args.project_name, 
        bugsinpy_root=args.bugsinpy_root,
        limit=args.limit
    )
