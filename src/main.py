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
    print("=== DEBUGGING AGENT DEMO ===")
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
    }

    # Run the agent
    final_state = debugging_agent.invoke(initial_state, config={"recursion_limit": 100})

    print("\n** DEBUGGING COMPLETED **")
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


if __name__ == "__main__":
    run_debugging_demo()
