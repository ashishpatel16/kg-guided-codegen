import json
import logging
import os

from src.agent.graph import debugging_agent, one_shot_codegen_agent
from src.agent.tools import configure_logging

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

    call_graph_path = "artifacts/demo_call_graph_dynamic_with_suspiciousness.json"
    if not os.path.exists(call_graph_path):
        print(f"Error: Call graph file not found at {call_graph_path}")
        return

    with open(call_graph_path, "r") as f:
        call_graph = json.load(f)

    # Initialize state
    initial_state = {
        "call_graph": call_graph,
        "score_delta": 0.3,
        "test_command": "pytest tests/test_repo_call_graph.py",  # Example test command
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
            if n.get("suspiciousness", 0) > 1.0
        ),
        None,
    )

    if culprit:
        print(f"IDENTIFIED BUGGY NODE: {culprit['fqn']}")
        print(f"Final Suspiciousness: {culprit['suspiciousness']}")
        print("\nLAST REFLECTION:")
        print(final_state.get("reflection"))
    else:
        print("No node reached the suspiciousness threshold > 1.0")


if __name__ == "__main__":
    run_debugging_demo()
