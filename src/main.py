import json
import logging
import os
from datetime import datetime

from src.agent.fault_localization.graph import debugging_agent, CONFIDENCE_THRESHOLD
from src.agent.fault_localization.tools import configure_logging
from src.program_analysis.dynamic_call_graph import run_dynamic_tracer_in_docker
from src.program_analysis.models import RepoDefinition, DockerTracerConfig

logger = logging.getLogger(__name__)

configure_logging(level="INFO")


def run_debugging_demo():
    print(debugging_agent.get_graph().draw_ascii())

    # Generate call graph in Docker
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"call_graph_{timestamp}.json"
    
    # Use the demo repository for tracing
    demo_repo = os.path.abspath("src/benchmarks/exp/demo")
    demo_scripts = ["tests/test_demo.py"]
    
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
        "test_command": "pytest tests/test_demo.py",  # Updated to match the traced repo
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
    demo_scripts = ["tests/test_demo.py"]
    
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
        "test_command": "pytest tests/test_demo.py",
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


if __name__ == "__main__":
    run_debugging_demo()
