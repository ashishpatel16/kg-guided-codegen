import argparse
import csv
import json
import logging
import os
import re
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.fault_localization.graph import debugging_agent, CONFIDENCE_THRESHOLD
from src.agent.fault_localization.tools import configure_logging
from src.benchmarks.setup_bugsinpy_docker import BugsInPyDockerSandbox

logger = logging.getLogger(__name__)


def run_bugsinpy_debugging(
    project_name: str,
    bug_id: str,
    bugsinpy_root: str = "datasets/BugsInPy",
    experiments_dir: str = "experiments",
    artifacts_dir: str = "artifacts",
    recursion_limit: int = 100,
) -> Dict[str, Any]:
    """
    Full fault-localization pipeline for a single BugsInPy instance.

    Steps:
      1. Start Docker sandbox, checkout buggy version, compile
      2. Run dynamic tracer → call graph JSON
      3. Remap container paths → host paths
      4. Invoke the debugging agent
      5. Save results to artifacts/

    Args:
        project_name: BugsInPy project (e.g. "youtube-dl", "black", "keras")
        bug_id:       Bug number as string (e.g. "1")
        bugsinpy_root: Path to the BugsInPy dataset root
        experiments_dir: Working directory for checked-out projects
        artifacts_dir: Where to write result JSON
        recursion_limit: Max LangGraph recursion depth

    Returns:
        Dict with keys: status, project, bug_id, culprit, top_candidates,
        final_state (partial), duration_s, error.
    """
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result: Dict[str, Any] = {
        "project": project_name,
        "bug_id": bug_id,
        "status": "FAILED",
        "culprit": None,
        "top_candidates": [],
        "error": None,
        "duration_s": 0.0,
    }
    start: datetime = datetime.now()

    print(f"\n{'='*60}")
    print(f"  BugsInPy Fault Localization: {project_name} #{bug_id}")
    print(f"{'='*60}\n")

    host_root: Optional[str] = None
    try:
        with BugsInPyDockerSandbox(
            project_name,
            bug_id,
            bugsinpy_root=bugsinpy_root,
            experiments_dir=experiments_dir,
        ) as bspy:

            # ── Step 1: Checkout ─────────────────────────────────────────
            print("[1/5] Checking out buggy version...")
            exit_code, out, err = bspy.checkout(version=0)
            if exit_code != 0:
                result["error"] = f"Checkout failed: {err}"
                print(f"  ✗ {result['error']}")
                return result
            print("  ✓ Checkout complete")

            # ── Step 2: Compile ──────────────────────────────────────────
            print("[2/5] Compiling project...")
            exit_code, out, err = bspy.compile(verbose=True)
            if exit_code != 0:
                result["error"] = f"Compile failed: {err}"
                print(f"  ✗ {result['error']}")
                return result
            print("  ✓ Compile complete")

            # ── Step 3: Dynamic tracer ───────────────────────────────────
            output_filename: str = f"call_graph_{project_name}_{bug_id}.json"
            print(f"[3/5] Running dynamic tracer → {output_filename}")
            exit_code, out, err = bspy.run_dynamic_tracer(output_file=output_filename)
            if exit_code != 0:
                result["error"] = f"Tracer failed: {err}"
                print(f"  ✗ {result['error']}")
                return result
            print("  ✓ Tracer complete")

            # ── Step 4: Load & remap call graph ──────────────────────────
            print("[4/5] Loading call graph and remapping paths...")
            call_graph_path: Path = bspy.host_experiments_dir / project_name / output_filename
            if not call_graph_path.exists():
                result["error"] = f"Call graph not found at {call_graph_path}"
                print(f"  ✗ {result['error']}")
                return result

            with open(call_graph_path, "r") as f:
                call_graph: Dict[str, Any] = json.load(f)

            container_root: str = bspy.container_project_root
            host_root = os.path.abspath(bspy.host_experiments_dir / project_name)

            for node in call_graph.get("nodes", []):
                file_path: str = node.get("file", "")
                if file_path.startswith(container_root):
                    node["file"] = os.path.abspath(
                        file_path.replace(container_root, host_root)
                    )

            node_count: int = len(call_graph.get("nodes", []))
            edge_count: int = len(call_graph.get("edges", []))
            print(f"  ✓ Call graph loaded: {node_count} nodes, {edge_count} edges")

            # ── Step 5: Run the debugging agent ──────────────────────────
            test_command: str = "bash bugsinpy_run_test.sh"
            initial_state: Dict[str, Any] = {
                "call_graph": call_graph,
                "score_delta": 0.3,
                "test_command": test_command,
                "container_id": bspy.sandbox.container.id,
                "container_workspace": bspy.container_project_root,
                "host_workspace": host_root,
                "use_docker": True,
                "llm_calls": 0,
            }

            print(f"[5/5] Running fault localization agent (threshold={CONFIDENCE_THRESHOLD})...")
            final_state: Dict[str, Any] = debugging_agent.invoke(
                initial_state, config={"recursion_limit": recursion_limit}
            )

            # ── Results ──────────────────────────────────────────────────
            nodes: List[Dict[str, Any]] = final_state["call_graph"]["nodes"]
            sorted_nodes: List[Dict[str, Any]] = sorted(
                nodes, key=lambda n: n.get("confidence_score", 0), reverse=True
            )

            culprit: Optional[Dict[str, Any]] = next(
                (n for n in sorted_nodes if n.get("confidence_score", 0) >= CONFIDENCE_THRESHOLD),
                None,
            )

            # Top 5 candidates
            top_candidates: List[Dict[str, Any]] = [
                {
                    "fqn": n["fqn"],
                    "confidence": round(n.get("confidence_score", 0), 4),
                    "suspiciousness": round(n.get("suspiciousness", 0), 4),
                }
                for n in sorted_nodes[:5]
            ]

            result["top_candidates"] = top_candidates
            result["llm_calls"] = final_state.get("llm_calls", 0)
            result["reflection"] = final_state.get("reflection", "")

            print(f"\n{'='*60}")
            print("  RESULTS")
            print(f"{'='*60}")

            if culprit:
                result["status"] = "LOCALIZED"
                result["culprit"] = {
                    "fqn": culprit["fqn"],
                    "confidence": round(culprit["confidence_score"], 4),
                }
                print(f"  ✓ BUGGY NODE: {culprit['fqn']}")
                print(f"    Confidence: {culprit['confidence_score']:.4f}")
            else:
                result["status"] = "INCONCLUSIVE"
                print(f"  ✗ No node reached threshold {CONFIDENCE_THRESHOLD}")

            print(f"\n  Top 5 candidates:")
            for i, c in enumerate(top_candidates, 1):
                marker: str = " ◀" if culprit and c["fqn"] == culprit["fqn"] else ""
                print(f"    {i}. {c['fqn']}  conf={c['confidence']:.4f}  susp={c['suspiciousness']:.4f}{marker}")

            if final_state.get("reflection"):
                print(f"\n  Last reflection:")
                print(f"    {final_state['reflection'][:200]}")

            if final_state.get("final_diff"):
                result["final_diff"] = final_state["final_diff"]
                print(f"\n  Generated patch diff (first 500 chars):")
                print(f"    {final_state['final_diff'][:500]}")

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "CRASHED"
        logger.exception(f"Pipeline crashed: {e}")
        print(f"\n  ✗ CRASHED: {e}")

    finally:
        result["duration_s"] = round((datetime.now() - start).total_seconds(), 2)
        print(f"\n  Duration: {result['duration_s']}s")
        print(f"{'='*60}\n")

        # Save result JSON
        os.makedirs(artifacts_dir, exist_ok=True)
        result_path: str = os.path.join(
            artifacts_dir, f"{project_name}_{bug_id}.json"
        )
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to {result_path}")

        # Save confidence_evolution JSON if exists
        try:
            if host_root:
                confidence_evolution_path = os.path.join(host_root, "confidence_evolution.json")
                if os.path.exists(confidence_evolution_path):
                    dest_path = os.path.join(artifacts_dir, f"{project_name}_{bug_id}_confidence_evolution.json")
                    shutil.copy2(confidence_evolution_path, dest_path)
                    print(f"  ✓ Confidence evolution saved to {dest_path}")
        except Exception as e:
            print(f"  ✗ Failed to copy confidence_evolution.json: {e}")

    return result



def parse_bug_patch(bugsinpy_root: str, project_name: str, bug_id: str) -> Dict[str, Optional[str]]:
    """
    Extract ground-truth info from bug_patch.txt.

    Returns a dict with:
        - file: relative path of the buggy file (from 'diff --git a/<path> b/<path>')
        - patch: the full raw patch text
    """
    info: Dict[str, Optional[str]] = {"file": None, "patch": None}
    patch_path: Path = Path(bugsinpy_root) / "projects" / project_name / "bugs" / bug_id / "bug_patch.txt"
    if not patch_path.exists():
        return info
    text: str = patch_path.read_text()
    info["patch"] = text

    # Extract file path: 'diff --git a/<path> b/<path>'
    file_match: Optional[re.Match] = re.search(r"diff --git a/(.+?) b/", text)
    if file_match:
        info["file"] = file_match.group(1)

    return info


CSV_COLUMNS: List[str] = [
    "project",
    "bug_id",
    "status",
    "ground_truth_file",
    "culprit_fqn",
    "culprit_confidence",
    "top1_fqn",
    "top1_confidence",
    "top1_suspiciousness",
    "top5_fqns",
    "top5_confidences",
    "llm_calls",
    "node_count",
    "edge_count",
    "recursion_limit",
    "confidence_threshold",
    "reflection",
    "has_patch",
    "agent_patch",
    "ground_truth_patch",
    "duration_s",
    "error",
    "timestamp",
]


def append_result_to_csv(
    csv_path: str,
    result: Dict[str, Any],
    ground_truth: Dict[str, Optional[str]],
    node_count: int,
    edge_count: int,
    recursion_limit: int,
    confidence_threshold: float,
) -> None:
    """
    Append one result row to the CSV file.

    Creates the file with headers on first call.  Uses csv.DictWriter so
    commas and newlines inside fields (e.g. reflections) are safely escaped.
    """
    file_exists: bool = os.path.exists(csv_path)

    top_candidates: List[Dict[str, Any]] = result.get("top_candidates", [])
    top1: Dict[str, Any] = top_candidates[0] if top_candidates else {}

    culprit: Optional[Dict[str, Any]] = result.get("culprit")

    row: Dict[str, Any] = {
        "project": result["project"],
        "bug_id": result["bug_id"],
        "status": result["status"],
        "ground_truth_file": ground_truth.get("file") or "",
        "culprit_fqn": culprit["fqn"] if culprit else "",
        "culprit_confidence": culprit["confidence"] if culprit else "",
        "top1_fqn": top1.get("fqn", ""),
        "top1_confidence": top1.get("confidence", ""),
        "top1_suspiciousness": top1.get("suspiciousness", ""),
        "top5_fqns": ";".join(c.get("fqn", "") for c in top_candidates[:5]),
        "top5_confidences": ";".join(str(c.get("confidence", "")) for c in top_candidates[:5]),
        "llm_calls": result.get("llm_calls", ""),
        "node_count": node_count,
        "edge_count": edge_count,
        "recursion_limit": recursion_limit,
        "confidence_threshold": confidence_threshold,
        "reflection": (result.get("reflection", "") or "")[:500],
        "has_patch": bool(result.get("final_diff")),
        "agent_patch": result.get("final_diff", "") or "",
        "ground_truth_patch": ground_truth.get("patch") or "",
        "duration_s": result.get("duration_s", ""),
        "error": result.get("error", "") or "",
        "timestamp": datetime.now().isoformat(),
    }

    with open(csv_path, "a", newline="") as f:
        writer: csv.DictWriter = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def generate_baseline_call_graphs(project_name: str, bug_ids: List[str], bugsinpy_root: str = "datasets/BugsInPy"):
    """
    Sets up BugsinPy buggy versions for each bug ID, constructs the call graph,
    and exports them to artifacts/call_graphs_bugsinpy_baseline directory.
    Runs the docker sandbox in experiments_baseline folder.
    """
    experiments_dir = "experiments_baseline"
    artifacts_dir = "artifacts/call_graphs_bugsinpy_baseline"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"Starting baseline call graph generation for {project_name} bugs: {bug_ids}")

    for bug_id in bug_ids:
        print(f"\n[{project_name} #{bug_id}] Starting...")
        try:
            with BugsInPyDockerSandbox(
                project_name,
                bug_id,
                bugsinpy_root=bugsinpy_root,
                experiments_dir=experiments_dir,
            ) as bspy:
                print(f"[{project_name} #{bug_id}] Checking out buggy version...")
                exit_code, out, err = bspy.checkout(version=0)
                if exit_code != 0:
                    print(f"[{project_name} #{bug_id}] Checkout failed: {err}")
                    continue
                
                print(f"[{project_name} #{bug_id}] Compiling...")
                exit_code, out, err = bspy.compile(verbose=True)
                if exit_code != 0:
                    print(f"[{project_name} #{bug_id}] Compile failed: {err}")
                    continue
                
                output_filename = f"call_graph_{project_name}_{bug_id}.json"
                print(f"[{project_name} #{bug_id}] Running dynamic tracer...")
                exit_code, out, err = bspy.run_dynamic_tracer(output_file=output_filename)
                if exit_code != 0:
                    print(f"[{project_name} #{bug_id}] Tracer failed: {err}")
                    continue
                
                # Load & remap paths
                print(f"[{project_name} #{bug_id}] Processing & exporting call graph...")
                call_graph_path = bspy.host_experiments_dir / project_name / output_filename
                
                if not call_graph_path.exists():
                    print(f"[{project_name} #{bug_id}] Call graph missing at {call_graph_path}")
                    continue

                with open(call_graph_path, "r") as f:
                    call_graph = json.load(f)
                    
                container_root = bspy.container_project_root
                host_root = os.path.abspath(bspy.host_experiments_dir / project_name)

                for node in call_graph.get("nodes", []):
                    file_path = node.get("file", "")
                    if file_path.startswith(container_root):
                        node["file"] = os.path.abspath(
                            file_path.replace(container_root, host_root)
                        )
                
                out_path = os.path.join(artifacts_dir, output_filename)
                with open(out_path, "w") as f:
                    json.dump(call_graph, f, indent=2)
                    
                print(f"[{project_name} #{bug_id}] Saved call graph to {out_path}")
                
        except Exception as e:
            print(f"[{project_name} #{bug_id}] Exception occurred: {e}")
            logger.exception(f"Exception for {project_name} #{bug_id}")

    print("\nBaseline call graph generation finished.")


if __name__ == "__main__":
    # project_name: str = "youtube-dl"
    # invalid_ids = []
    # end_limit = 10 # Max valid is 44
    # bug_ids_int = [str(num) for num in range(1, end_limit) if num not in invalid_ids]
    # print(bug_ids_int)
    # pass
    # bug_ids: List[str] = bug_ids_int
    # bugsinpy_root: str = "datasets/BugsInPy"
    # experiments_dir: str = "experiments"
    # base_artifacts_dir: str = "testgen_artifacts"
    # recursion_limit: int = 100

    # configure_logging(level="INFO")

    # # Create a dedicated directory for this run
    # timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_dir: str = os.path.join(base_artifacts_dir, f"bugsinpy_{project_name}_{len(bug_ids)}bugs_{timestamp}")
    # os.makedirs(run_dir, exist_ok=True)

    # # Output CSV — one row per bug, written immediately after each instance
    # csv_path: str = os.path.join(run_dir, "results.csv")

    # print(f"\n{'='*60}")
    # print(f"  Running {len(bug_ids)} bugs for project: {project_name}")
    # print(f"  Bug IDs: {bug_ids}")
    # print(f"  Run directory: {run_dir}")
    # print(f"  CSV output: {csv_path}")
    # print(f"{'='*60}\n")

    # summary: Dict[str, int] = {"LOCALIZED": 0, "INCONCLUSIVE": 0, "CRASHED": 0, "FAILED": 0}

    # for i, bug_id in enumerate(bug_ids):
    #     print(f"\n>>> [{i+1}/{len(bug_ids)}] Starting {project_name} bug #{bug_id} <<<\n")

    #     result: Dict[str, Any] = run_bugsinpy_debugging(
    #         project_name=project_name,
    #         bug_id=bug_id,
    #         bugsinpy_root=bugsinpy_root,
    #         experiments_dir=experiments_dir,
    #         artifacts_dir=run_dir,
    #         recursion_limit=recursion_limit,
    #     )

    #     # Extract ground-truth info from the patch
    #     ground_truth: Dict[str, Optional[str]] = parse_bug_patch(bugsinpy_root, project_name, bug_id)

    #     # Count nodes/edges from the call graph if available
    #     node_count: int = 0
    #     edge_count: int = 0
    #     call_graph_path: Path = Path(experiments_dir) / project_name / f"call_graph_{project_name}_{bug_id}.json"
    #     if call_graph_path.exists():
    #         with open(call_graph_path, "r") as f:
    #             cg: Dict[str, Any] = json.load(f)
    #         node_count = len(cg.get("nodes", []))
    #         edge_count = len(cg.get("edges", []))

    #     # Write to CSV immediately after this instance
    #     append_result_to_csv(
    #         csv_path=csv_path,
    #         result=result,
    #         ground_truth=ground_truth,
    #         node_count=node_count,
    #         edge_count=edge_count,
    #         recursion_limit=recursion_limit,
    #         confidence_threshold=CONFIDENCE_THRESHOLD,
    #     )

    #     status: str = result["status"]
    #     summary[status] = summary.get(status, 0) + 1
    #     print(f">>> [{i+1}/{len(bug_ids)}] {project_name} bug #{bug_id}: {status}  (CSV updated) <<<\n")

    # # Final summary
    # print(f"\n{'='*60}")
    # print(f"  BATCH SUMMARY  ({len(bug_ids)} bugs)")
    # print(f"{'='*60}")
    # for s, count in summary.items():
    #     if count > 0:
    #         print(f"  {s}: {count}")
    # print(f"  Results saved to: {csv_path}")
    # print(f"{'='*60}\n")

    bug_ids = [4]
    generate_baseline_call_graphs(project_name="youtube-dl", bug_ids=bug_ids)
