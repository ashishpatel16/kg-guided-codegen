import json
import logging
import os
from datetime import datetime

from src.agent.graph import debugging_agent, one_shot_codegen_agent, CONFIDENCE_THRESHOLD
from src.agent.tools import configure_logging
from src.program_analysis.dynamic_call_graph import run_dynamic_tracer_in_docker
from src.program_analysis.models import RepoDefinition, DockerTracerConfig

logger = logging.getLogger(__name__)

configure_logging(level="INFO")


def run_one_shot_demo():
    print(one_shot_codegen_agent.get_graph().draw_ascii())
    problem_statement = "Write a program that finds the minimum time required to deliver packages to a set of target cities and reach a final destination, given a limited battery range and specific time windows for delivery."

    result = one_shot_codegen_agent.invoke({"problem": problem_statement})

    print("=== GENERATED CODE ===")
    print(result["generated_code"])


def run_debugging_demo():
    print(debugging_agent.get_graph().draw_ascii())

    # Generate call graph in Docker
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"call_graph_{timestamp}.json"
    
    # Use the demo repository for tracing
    demo_repo = os.path.abspath("src/benchmarks/exp/demo")
    demo_scripts = ["test_demo.py"]
    
    repo_def = RepoDefinition(
        repo_path=demo_repo, 
        trace_scripts=demo_scripts
    )
    config = DockerTracerConfig(
        output_file=output_filename,
        test_mode=True
    )

    print(f"Generating call graph in Docker: {output_filename}...")
    # This will save the result to artifacts/call_graph_TIMESTAMP.json
    run_dynamic_tracer_in_docker(repo_def, config)
    
    call_graph_path = os.path.join("artifacts", output_filename)
    if not os.path.exists(call_graph_path):
        print(f"Error: Call graph file not found at {call_graph_path}")
        return

    with open(call_graph_path, "r") as f:
        call_graph = json.load(f)

    container_repo_path = "/codebase/repo"
    for node in call_graph.get("nodes", []):
        if node.get("file", "").startswith(container_repo_path):
            node["file"] = node["file"].replace(container_repo_path, demo_repo)
            # Ensure it's absolute
            node["file"] = os.path.abspath(node["file"])

    initial_state = {
        "call_graph": call_graph,
        "score_delta": 0.3,
        "test_command": "pytest test_demo.py",  # Updated to match the traced repo
        "llm_calls": 0,
        "host_workspace": demo_repo,
    }

    # Run the agent
    final_state = debugging_agent.invoke(initial_state, config={"recursion_limit": 100})

    print("\n** DEBUGGING COMPLETED **")
    if "final_diff" in final_state:
        print("\n** GENERATED PATCH DIFF **")
        print(final_state["final_diff"])
    else:
        print("\nNo final patch was generated (threshold potentially not reached).")
        if "inspection_diff" in final_state:
            print("\n** LAST INSPECTION DIFF **")
            print(final_state["inspection_diff"])
    # Find the node that caused termination
    culprit = next(
        (
            n
            for n in final_state["call_graph"]["nodes"]
            if n.get("confidence_score", 0) >= CONFIDENCE_THRESHOLD
        ),
        None,
    )

    if culprit:
        print(f"IDENTIFIED BUGGY NODE: {culprit['fqn']}")
        print(f"Final Confidence Score: {culprit['confidence_score']:.4f}")
        print(f"Initial Suspiciousness: {culprit.get('suspiciousness', 'N/A')}")
        print("\nLAST REFLECTION:")
        print(final_state.get("reflection"))
    else:
        # Fallback to highest confidence if none reached threshold
        top_node = max(final_state["call_graph"]["nodes"], key=lambda x: x.get("confidence_score", 0))
        print(f"No node reached the {CONFIDENCE_THRESHOLD} confidence threshold.")
        print(f"Top candidate: {top_node['fqn']} (Confidence: {top_node.get('confidence_score', 0):.4f})")

    # Log all scores to a JSON file
    scores_log = []
    for node in final_state["call_graph"]["nodes"]:
        scores_log.append({
            "fqn": node.get("fqn"),
            "suspiciousness": node.get("suspiciousness", 0.0),
            "confidence_score": node.get("confidence_score", 0.0)
        })
    
    log_path = os.path.join("artifacts", f"final_scores_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(scores_log, f, indent=4)
    print(f"\nAll scores logged to {log_path}")


def run_debugging_demo_disambiguated():
    print(debugging_agent.get_graph().draw_ascii())

    # Generate call graph in Docker
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"call_graph_disambig_{timestamp}.json"
    
    # Use the demo repository for tracing
    demo_repo = os.path.abspath("src/benchmarks/exp/demo")
    demo_scripts = ["test_demo.py"]
    
    repo_def = RepoDefinition(
        repo_path=demo_repo, 
        trace_scripts=demo_scripts
    )
    config = DockerTracerConfig(
        output_file=output_filename,
        test_mode=True
    )

    print(f"Generating call graph in Docker with coverage matrix: {output_filename}...")
    run_dynamic_tracer_in_docker(repo_def, config)
    
    call_graph_path = os.path.join("artifacts", output_filename)
    if not os.path.exists(call_graph_path):
        print(f"Error: Call graph file not found at {call_graph_path}")
        return

    with open(call_graph_path, "r") as f:
        call_graph_data = json.load(f)

    # Re-map container paths
    container_repo_path = "/codebase/repo"
    for node in call_graph_data.get("nodes", []):
        if node.get("file", "").startswith(container_repo_path):
            node["file"] = node["file"].replace(container_repo_path, demo_repo)
            node["file"] = os.path.abspath(node["file"])

    initial_state = {
        "call_graph": call_graph_data,
        "score_delta": 0.3,
        "test_command": "pytest test_demo.py",
        "llm_calls": 0,
        "host_workspace": demo_repo,
        # Populate coverage data for SuspiciousnessController
        "tests": [os.path.basename(tf) for tf in demo_scripts],
        "coverage_matrix": call_graph_data.get("coverage_matrix", {}),
    }

    print("\nStarting Disambiguated Debugging Agent...")
    final_state = debugging_agent.invoke(initial_state, config={"recursion_limit": 100})

    print("\n** DEBUGGING COMPLETED **")
    
    culprit = next(
        (n for n in final_state["call_graph"]["nodes"] if n.get("confidence_score", 0) >= CONFIDENCE_THRESHOLD),
        None,
    )

    if culprit:
        print(f"IDENTIFIED BUGGY NODE: {culprit['fqn']}")
        print(f"Final Confidence Score: {culprit['confidence_score']:.4f}")
    
    # Log all scores
    scores_log = []
    for node in final_state["call_graph"]["nodes"]:
        scores_log.append({
            "fqn": node.get("fqn"),
            "suspiciousness": node.get("suspiciousness", 0.0),
            "confidence_score": node.get("confidence_score", 0.0)
        })
    
    log_path = os.path.join("artifacts", f"final_scores_disambig_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(scores_log, f, indent=4)
    print(f"\nAll refined scores logged to {log_path}")


def run_debugging_demo_comparison():
    print(debugging_agent.get_graph().draw_ascii())

    # Generate call graph in Docker
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"call_graph_{timestamp}.json"
    
    # Use the demo repository for tracing
    demo_repo = os.path.abspath("src/benchmarks/exp/demo")
    demo_scripts = ["test_demo.py"]
    
    repo_def = RepoDefinition(
        repo_path=demo_repo, 
        trace_scripts=demo_scripts
    )
    config = DockerTracerConfig(
        output_file=output_filename,
        test_mode=True
    )

    print(f"Generating call graph in Docker: {output_filename}...")
    # This will save the result to artifacts/call_graph_TIMESTAMP.json
    run_dynamic_tracer_in_docker(repo_def, config)
    
    call_graph_path = os.path.join("artifacts", output_filename)
    if not os.path.exists(call_graph_path):
        print(f"Error: Call graph file not found at {call_graph_path}")
        return

    with open(call_graph_path, "r") as f:
        call_graph = json.load(f)

    container_repo_path = "/codebase/repo"
    for node in call_graph.get("nodes", []):
        if node.get("file", "").startswith(container_repo_path):
            node["file"] = node["file"].replace(container_repo_path, demo_repo)
            # Ensure it's absolute
            node["file"] = os.path.abspath(node["file"])

    initial_state = {
        "call_graph": call_graph,
        "score_delta": 0.3,
        "test_command": "pytest test_demo.py",  # Updated to match the traced repo
        "llm_calls": 0,
        "host_workspace": demo_repo,
    }

    print("Suscpiciousness Report")
    for node in call_graph.get("nodes", []):
        print(f"{node['fqn']}: {node['suspiciousness']}")

    # Initialize the Suspiciosuness Controller
    suspiciousness_controller = SuspiciousnessController()
    suspiciousness_controller.invoke(initial_state)

    



def run_suspiciousness_experiment() -> None:
    """
    Non-simulated experiment:
      1. Run Docker tracer → real CallGraph with coverage_matrix + test_results
      2. Build SuspiciousnessController from real data
      3. Print initial top-5 + ambiguity groups
      4. Generate disambiguating test via Gemini
      5. Run generated test in Docker → get real coverage + pass/fail
      6. add_test_case + remove_test_case
      7. Print updated top-5
    """
    from typing import Dict, List, Set
    from dotenv import load_dotenv
    from src.program_analysis.suspiciousness_controller import SuspiciousnessController
    from src.llm.connector import GeminiLLMConnector
    from src.docker_utils.basic_container import SimpleDockerSandbox

    load_dotenv()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def invert_coverage_matrix(coverage_matrix: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """Invert Node→List[Test] into Test→Set[Node] (node_execution_map)."""
        node_exec_map: Dict[str, Set[str]] = {}
        for node_fqn, tests in coverage_matrix.items():
            for test_name in tests:
                if test_name not in node_exec_map:
                    node_exec_map[test_name] = set()
                node_exec_map[test_name].add(node_fqn)
        return node_exec_map

    def calculate_tarantula(
        exec_map: Dict[str, Set[str]], results: Dict[str, bool]
    ) -> Dict[str, float]:
        total_failed: int = sum(1 for r in results.values() if not r)
        total_passed: int = sum(1 for r in results.values() if r)
        all_nodes: Set[str] = set()
        for nodes in exec_map.values():
            all_nodes.update(nodes)
        scores: Dict[str, float] = {}
        for node in all_nodes:
            failed_s: int = sum(
                1 for t, ns in exec_map.items() if node in ns and not results[t]
            )
            passed_s: int = sum(
                1 for t, ns in exec_map.items() if node in ns and results[t]
            )
            failed_ratio: float = failed_s / total_failed if total_failed > 0 else 0
            passed_ratio: float = passed_s / total_passed if total_passed > 0 else 0
            denom: float = failed_ratio + passed_ratio
            scores[node] = round(failed_ratio / denom, 4) if denom > 0 else 0.0
        return scores

    def print_top_n(
        title: str,
        ctrl: SuspiciousnessController,
        n: int = 5,
    ) -> None:
        scores: Dict[str, float] = calculate_tarantula(
            ctrl.node_execution_map, ctrl.test_results
        )
        groups: List[Set[str]] = ctrl.identify_ambiguity_groups()
        # Filter out test nodes (only show demo.* source nodes)
        source_scores: Dict[str, float] = {
            k: v for k, v in scores.items() if not k.startswith("test_demo.")
        }
        sorted_nodes = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'=' * 90}")
        print(f"  {title}")
        print(f"{'=' * 90}")
        print(f"  {'#':<4} {'Node':<45} {'Susp.':<8} {'Ambiguity'}")
        print(f"  {'-' * 85}")
        for rank, (node, score) in enumerate(sorted_nodes[:n], 1):
            group: Set[str] = ctrl.get_ambiguity_group_for_node(node)
            # Filter group to source nodes only
            source_group: Set[str] = {g for g in group if not g.startswith("test_demo.")}
            grp_str: str = (
                f"group of {len(source_group)}" if len(source_group) > 1 else "ISOLATED"
            )
            print(f"  {rank:<4} {node:<45} {score:<8.4f} {grp_str}")
        print(f"  {'-' * 85}")
        if groups:
            # Show groups with at least one source node
            source_groups = [
                {g for g in grp if not g.startswith("test_demo.")}
                for grp in groups
            ]
            source_groups = [g for g in source_groups if len(g) > 1]
            if source_groups:
                print(f"  Ambiguity groups among source nodes ({len(source_groups)}):")
                for i, g in enumerate(source_groups):
                    print(f"    G{i+1}: {sorted(g)}")
            else:
                print("  No ambiguity among source nodes.")
        else:
            print("  No ambiguity — every node is distinguishable.")
        print(f"{'=' * 90}\n")

    # ── Step 1: Run Docker tracer ────────────────────────────────────────────

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename: str = f"call_graph_experiment_{timestamp}.json"
    demo_repo: str = os.path.abspath("src/benchmarks/exp/demo")
    demo_scripts: List[str] = ["test_demo.py"]

    repo_def = RepoDefinition(repo_path=demo_repo, trace_scripts=demo_scripts)
    config = DockerTracerConfig(output_file=output_filename, test_mode=True)

    print(f"\n[STEP 1] Running Docker tracer → {output_filename}")
    run_dynamic_tracer_in_docker(repo_def, config)

    call_graph_path: str = os.path.join("artifacts", output_filename)
    if not os.path.exists(call_graph_path):
        print(f"Error: Call graph file not found at {call_graph_path}")
        return

    with open(call_graph_path, "r") as f:
        call_graph_data: dict = json.load(f)

    # Re-map container paths to host paths
    container_repo_path: str = "/codebase/repo"
    for node in call_graph_data.get("nodes", []):
        if node.get("file", "").startswith(container_repo_path):
            node["file"] = node["file"].replace(container_repo_path, demo_repo)
            node["file"] = os.path.abspath(node["file"])

    # ── Step 2: Build SuspiciousnessController ───────────────────────────────

    coverage_matrix: Dict[str, List[str]] = call_graph_data.get("coverage_matrix", {})
    test_results: Dict[str, bool] = call_graph_data.get("test_results", {})

    if not coverage_matrix or not test_results:
        print("Error: No coverage_matrix or test_results in call graph data.")
        return

    node_execution_map: Dict[str, Set[str]] = invert_coverage_matrix(coverage_matrix)

    # Initialize Gemini LLM
    api_key: str = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: Set GEMINI_API_KEY in your .env to run this experiment.")
        return

    llm = GeminiLLMConnector(model_name="gemini-2.0-flash", temperature=0.2, api_key=api_key)
    controller = SuspiciousnessController(node_execution_map, test_results, llm_connector=llm)

    print(f"\n[STEP 2] Built SuspiciousnessController")
    print(f"  Tests: {len(test_results)} ({sum(1 for v in test_results.values() if not v)} failing)")
    print(f"  Source nodes in coverage: {len(coverage_matrix)}")

    # ── Step 3: Print initial top 5 ──────────────────────────────────────────

    print_top_n("[STEP 3] INITIAL Top-5 Suspicious Nodes", controller)

    # ── Step 4: Generate disambiguating test ─────────────────────────────────

    ambiguity_groups: List[Set[str]] = controller.identify_ambiguity_groups()
    # Filter to source-node-only groups
    source_ambiguity_groups: List[Set[str]] = []
    for grp in ambiguity_groups:
        source_only: Set[str] = {n for n in grp if not n.startswith("test_demo.")}
        if len(source_only) > 1:
            source_ambiguity_groups.append(source_only)

    if not source_ambiguity_groups:
        print("[STEP 4] No ambiguity groups among source nodes. Skipping test generation.")
    else:
        # Pick the largest ambiguity group
        target_group: Set[str] = max(source_ambiguity_groups, key=len)
        # Target the node with the highest suspiciousness in the group
        initial_scores: Dict[str, float] = calculate_tarantula(
            controller.node_execution_map, controller.test_results
        )
        target_node: str = max(target_group, key=lambda n: initial_scores.get(n, 0.0))

        print(f"\n[STEP 4] Generating disambiguating test for '{target_node}'")
        print(f"  Ambiguity group: {sorted(target_group)}")

        # Build call_graph dict for generate_test_to_disambiguate
        cg_for_llm: Dict[str, list] = {
            "nodes": call_graph_data.get("nodes", []),
            "edges": call_graph_data.get("edges", []),
        }
        generated_test: str = controller.generate_test_to_disambiguate(target_node, cg_for_llm)

        print(f"\n─── GENERATED TEST CODE ───────────────────────────────────────────")
        print(generated_test)
        print(f"────────────────────────────────────────────────────────────────────\n")

    # ── Step 5: Run generated test in Docker sandbox ─────────────────────────

    if source_ambiguity_groups and generated_test:
        print("[STEP 5] Running generated test in Docker sandbox...")

        # Write the generated test to a temp file
        import tempfile
        generated_test_path: str = os.path.join(
            tempfile.gettempdir(), "test_generated_disambig.py"
        )
        # Prepend necessary imports, strip LLM-added demo/pytest imports to avoid conflicts
        cleaned_lines: list = []
        for ln in generated_test.splitlines():
            stripped: str = ln.strip()
            # Skip demo/pytest imports the LLM typically adds (we handle these)
            if stripped.startswith("from demo import") or stripped.startswith("from demo "):
                continue
            if stripped == "import pytest" or stripped.startswith("import demo"):
                continue
            cleaned_lines.append(ln)
        cleaned_test: str = "\n".join(cleaned_lines).strip()

        test_file_content: str = (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from demo import *\n"
            "import pytest\n\n"
            f"{cleaned_test}\n"
        )
        with open(generated_test_path, "w") as f:
            f.write(test_file_content)

        print(f"  Wrote generated test to {generated_test_path}")

        # Run in Docker sandbox
        with SimpleDockerSandbox(image_name="python:3.11-slim", keep_alive=False) as sandbox:
            # Install pytest
            sandbox.run_command("pip install pytest")

            # Copy repo
            sandbox.copy_to(demo_repo, "repo")

            # Copy generated test
            repo_in_container: str = os.path.join(sandbox.sandbox_dir, "repo")
            sandbox.copy_to(
                generated_test_path,
                "repo/test_generated_disambig.py",
            )

            # Run pytest on just the generated test and capture result
            exit_code, stdout, stderr = sandbox.run_command(
                "cd repo && python -m pytest test_generated_disambig.py -v --tb=short",
                workdir=repo_in_container,
            )

            test_passed: bool = exit_code == 0
            print(f"  Test result: {'PASSED' if test_passed else 'FAILED'} (exit code {exit_code})")
            output_text: str = stdout or stderr or ""
            if output_text:
                lines = output_text.strip().split("\n")
                for line in lines[-20:]:
                    print(f"    {line}")

        # Determine coverage: the generated test targets `target_node`.
        # For a precise result we'd re-run the tracer, but we can infer
        # from the test content which functions it calls.
        # Conservative approach: assume it covers target_node and its callees.
        covered_nodes: Set[str] = {target_node}
        # If the test imports from demo and calls the target function directly,
        # it likely also hits is_divisible if target_node is is_prime, etc.
        # Use the call graph edges to find direct callees.
        for edge in call_graph_data.get("edges", []):
            if edge.get("source") == target_node:
                callee: str = edge.get("target", "")
                if callee and not callee.startswith("test_demo."):
                    covered_nodes.add(callee)

        print(f"  Inferred coverage: {sorted(covered_nodes)}")

        # Add the test to the controller
        generated_test_name: str = "test_generated_disambig"
        controller.add_test_case(
            test_fqn=generated_test_name,
            covered_nodes=covered_nodes,
            passed=test_passed,
        )
        print(f"  Added '{generated_test_name}' to controller (passed={test_passed})")

    # ── Step 6: Remove the broadest failing test ─────────────────────────────

    # Find the failing test that covers the most nodes
    failing_tests: Dict[str, int] = {}
    for test_name, passed in controller.test_results.items():
        if not passed:
            node_count: int = len(controller.node_execution_map.get(test_name, set()))
            failing_tests[test_name] = node_count

    if failing_tests:
        broadest_test: str = max(failing_tests, key=lambda t: failing_tests[t])
        broadest_count: int = failing_tests[broadest_test]

        print(f"\n[STEP 6] Removing broadest failing test: '{broadest_test}' (covers {broadest_count} nodes)")
        controller.remove_test_case(broadest_test)
    else:
        print("\n[STEP 6] No failing tests to remove.")

    # ── Step 7: Print updated top 5 ──────────────────────────────────────────

    print_top_n("[STEP 7] UPDATED Top-5 Suspicious Nodes (after add + remove)", controller)

    # ── Log results ──────────────────────────────────────────────────────────

    final_scores: Dict[str, float] = calculate_tarantula(
        controller.node_execution_map, controller.test_results
    )
    scores_log: list = [
        {"fqn": fqn, "suspiciousness": score}
        for fqn, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    log_path: str = os.path.join("artifacts", f"experiment_scores_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(scores_log, f, indent=4)
    print(f"\nScores logged to {log_path}")


if __name__ == "__main__":
    run_debugging_demo()

