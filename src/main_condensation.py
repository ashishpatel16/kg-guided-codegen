"""
Less-is-More MVP: Experiment runner.

Usage:
  # Offline (call_graph.json must already exist in experiments/):
  python -m src.main_condensation --project youtube-dl --bug-ids 1 2 3

  # With Docker tracer (produces call_graph.json then runs condensation):
  python -m src.main_condensation --project youtube-dl --bug-ids 2 3 --run-tracer

  # Single files:
  python -m src.main_condensation --call-graph path/to/cg.json --patch path/to/bug_patch.txt
"""

import argparse
import csv
import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.condensation import ExperimentResult, run_condensation_experiment


def _call_graph_path(experiments_dir: str, project: str, bug_id: str) -> Path:
    return Path(experiments_dir) / project / f"call_graph_{project}_{bug_id}.json"


def load_call_graph(experiments_dir: str, project: str, bug_id: str) -> Optional[Dict[str, Any]]:
    path = _call_graph_path(experiments_dir, project, bug_id)
    if not path.exists():
        print(f"  [WARN] Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_patch(bugsinpy_root: str, project: str, bug_id: str) -> Optional[str]:
    path = Path(bugsinpy_root) / "projects" / project / "bugs" / bug_id / "bug_patch.txt"
    if not path.exists():
        print(f"  [WARN] Not found: {path}")
        return None
    return path.read_text()


def run_tracer(project: str, bug_id: str, bugsinpy_root: str, experiments_dir: str) -> bool:
    """Run the dynamic tracer inside a Docker container to produce call_graph.json."""
    from src.benchmarks.setup_bugsinpy_docker import BugsInPyDockerSandbox

    print(f"  [DOCKER] Starting tracer for {project} #{bug_id}...")
    try:
        with BugsInPyDockerSandbox(
            project, bug_id,
            bugsinpy_root=bugsinpy_root,
            experiments_dir=experiments_dir,
        ) as bspy:
            print("  [1/3] Checkout buggy version...")
            ec, _, err = bspy.checkout(version=0)
            if ec != 0:
                print(f"  [FAIL] Checkout: {err}")
                return False

            print("  [2/3] Compile...")
            ec, _, err = bspy.compile(verbose=True)
            if ec != 0:
                print(f"  [FAIL] Compile: {err}")
                return False

            output_file = f"call_graph_{project}_{bug_id}.json"
            print(f"  [3/3] Dynamic tracer -> {output_file}")
            ec, _, err = bspy.run_dynamic_tracer(output_file=output_file)
            if ec != 0:
                print(f"  [FAIL] Tracer: {err}")
                return False

            cg_path = bspy.host_experiments_dir / project / output_file
            if not cg_path.exists():
                print(f"  [FAIL] Output not found: {cg_path}")
                return False

            print(f"  [OK] Call graph saved: {cg_path}")
            return True
    except Exception as e:
        print(f"  [FAIL] Docker error: {e}")
        return False


def write_csv(csv_path: str, results: List[ExperimentResult]) -> None:
    if not results:
        return
    fields = [f.name for f in dataclasses.fields(ExperimentResult)] + ["rank_improvement", "timestamp"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = dataclasses.asdict(r)
            row["rank_improvement"] = r.rank_full - r.rank_condensed
            row["timestamp"] = datetime.now().isoformat()
            w.writerow(row)


def run_bug(
    project: str, bug_id: str,
    bugsinpy_root: str, experiments_dir: str,
    do_trace: bool = False,
    call_graph: Optional[Dict[str, Any]] = None,
    patch_text: Optional[str] = None,
) -> Optional[ExperimentResult]:
    print(f"\n{'='*60}")
    print(f"  {project} #{bug_id}")
    print(f"{'='*60}")

    # If no pre-loaded call graph, try loading from disk or running the tracer
    if call_graph is None:
        cg_exists = _call_graph_path(experiments_dir, project, bug_id).exists()
        if not cg_exists and do_trace:
            ok = run_tracer(project, bug_id, bugsinpy_root, experiments_dir)
            if not ok:
                return None
        call_graph = load_call_graph(experiments_dir, project, bug_id)
        if call_graph is None:
            return None

    pt = patch_text or load_patch(bugsinpy_root, project, bug_id)
    if pt is None:
        return None

    return run_condensation_experiment(call_graph, pt, project, bug_id)


def main() -> None:
    p = argparse.ArgumentParser(description="Less-is-More: Test Suite Condensation")
    p.add_argument("--project", type=str, help="BugsInPy project name")
    p.add_argument("--bug-ids", nargs="+", type=str, help="Bug IDs (e.g. 1 2 3)")
    p.add_argument("--run-tracer", action="store_true",
                    help="Run Docker tracer if call_graph.json is missing")
    p.add_argument("--call-graph", type=str, help="Direct path to call_graph.json")
    p.add_argument("--patch", type=str, help="Direct path to bug_patch.txt")
    p.add_argument("--bugsinpy-root", default="datasets/BugsInPy")
    p.add_argument("--experiments-dir", default="experiments")
    p.add_argument("--output-dir", default="condensation_artifacts")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Direct file mode
    if args.call_graph and args.patch:
        with open(args.call_graph) as f:
            cg = json.load(f)
        pt = Path(args.patch).read_text()
        result = run_bug("direct", "0", args.bugsinpy_root, args.experiments_dir,
                         call_graph=cg, patch_text=pt)
        if result:
            csv_path = os.path.join(args.output_dir, f"condensation_{ts}.csv")
            write_csv(csv_path, [result])
            print(f"\n  Saved: {csv_path}")
        return

    # Batch mode
    if not args.project or not args.bug_ids:
        p.error("Provide (--project + --bug-ids) or (--call-graph + --patch)")

    results: List[ExperimentResult] = []
    for i, bid in enumerate(args.bug_ids):
        print(f"\n>>> [{i+1}/{len(args.bug_ids)}]")
        r = run_bug(args.project, bid, args.bugsinpy_root, args.experiments_dir,
                     do_trace=args.run_tracer)
        if r:
            results.append(r)

    csv_path = os.path.join(args.output_dir, f"condensation_{args.project}_{ts}.csv")
    write_csv(csv_path, results)

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({len(results)}/{len(args.bug_ids)} bugs)")
    print(f"{'='*60}")
    for r in results:
        delta = r.rank_full - r.rank_condensed
        print(f"  {r.project}#{r.bug_id}: rank {r.rank_full}->{r.rank_condensed} ({delta:+d}), "
              f"tests {r.num_tests_full}->{r.num_tests_condensed}")
    print(f"  Saved: {csv_path}\n")


if __name__ == "__main__":
    main()
