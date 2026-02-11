import json
import logging
import os
from datetime import datetime

from src.agent.graph import debugging_agent, CONFIDENCE_THRESHOLD
from src.agent.tools import configure_logging
from src.benchmarks.setup_bugsinpy_docker import BugsInPyDockerSandbox

logger = logging.getLogger(__name__)

configure_logging(level="INFO")


def run_bugsinpy_debugging(project_name: str, bug_id: str):
    print(f"=== DEBUGGING AGENT BUGS_IN_PY: {project_name} #{bug_id} ===")
    
    # 1. Setup Docker Sandbox
    # This will checkout the project and provide a running container
    with BugsInPyDockerSandbox(project_name, bug_id) as bspy:
        # Checkout buggy version (0)
        exit_code, out, err = bspy.checkout(version=0)
        if exit_code != 0:
            print(f"Error during checkout: {err}")
            return

        # Compile project
        exit_code, out, err = bspy.compile(verbose=True)
        if exit_code != 0:
            print(f"Error during compile: {err}")
            return

        # 2. Run Dynamic Tracer inside the container
        output_filename = f"call_graph_{project_name}_{bug_id}.json"
        print(f"Generating call graph in Docker: {output_filename}...")
        
        exit_code, out, err = bspy.run_dynamic_tracer(output_file=output_filename)
        if exit_code != 0:
            print(f"Error during tracing: {err}")
            return

        # 3. Load the Call Graph
        host_experiments_dir = bspy.host_experiments_dir
        
        
        call_graph_path = host_experiments_dir / project_name / output_filename
        if not os.path.exists(call_graph_path):
            print(f"Error: Call graph file not found at {call_graph_path}")
            return

        with open(call_graph_path, "r") as f:
            call_graph = json.load(f)

        # 4. Map Container Paths to Host Paths for Agent (to read/edit code)
        container_project_root = bspy.container_project_root
        host_project_root = os.path.abspath(host_experiments_dir / project_name)
        
        for node in call_graph.get("nodes", []):
            if node.get("file", "").startswith(container_project_root):
                node["file"] = node["file"].replace(container_project_root, host_project_root)
                # Ensure it's absolute
                node["file"] = os.path.abspath(node["file"])

        # 5. Get Test Command from the container
        # BugsInPy projects have a bugsinpy_run_test.sh script
        test_command = "bash bugsinpy_run_test.sh"
        
        print(f"Using test command: {test_command}")

        # 6. Initialize Agent State
        initial_state = {
            "call_graph": call_graph,
            "score_delta": 0.3,
            "test_command": test_command,
            "container_id": bspy.sandbox.container.id,
            "container_workspace": bspy.container_project_root,
            "host_workspace": host_project_root,
            "use_docker": True,
            "llm_calls": 0,
        }

        # 7. Run the Agent
        print("Invoking debugging agent...")
        final_state = debugging_agent.invoke(initial_state, config={"recursion_limit": 100})

        print("\n** DEBUGGING COMPLETED **")
        # Find the node that reached threshold
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
            print("\nLAST REFLECTION:")
            print(final_state.get("reflection"))
        else:
            top_node = max(final_state["call_graph"]["nodes"], key=lambda x: x.get("confidence_score", 0))
            print(f"No node reached the {CONFIDENCE_THRESHOLD} confidence threshold.")
            print(f"Top candidate: {top_node['fqn']} (Confidence: {top_node.get('confidence_score', 0):.4f})")


if __name__ == "__main__":
    # Example: youtube-dl bug 1
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "youtube-dl"
    bug = sys.argv[2] if len(sys.argv) > 2 else "1"
    
    run_bugsinpy_debugging(project, bug)
