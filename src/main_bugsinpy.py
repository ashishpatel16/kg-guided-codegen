import argparse
import json
import logging
import os
import sys
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
            host_root: str = os.path.abspath(bspy.host_experiments_dir / project_name)

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
            artifacts_dir, f"bugsinpy_{project_name}_{bug_id}_{timestamp}.json"
        )
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to {result_path}")

    return result


if __name__ == "__main__":
    # Specify the project and bug ID here
    project_name = "tqdm"
    bug_id = "1"
    
    configure_logging(level="INFO")

    run_bugsinpy_debugging(
        project_name=project_name,
        bug_id=bug_id,
        bugsinpy_root="datasets/BugsInPy",
        experiments_dir="experiments",
        artifacts_dir="artifacts",
        recursion_limit=100,
    )
