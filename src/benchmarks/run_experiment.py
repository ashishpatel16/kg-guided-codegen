
import sys
import os
import json
import networkx as nx
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add src to sys.path to allow imports if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.benchmarks.setup_bugsinpy_case import setup_bugsinpy_case, run_bugsinpy_case
from src.program_analysis.repo_call_graph import get_repo_call_graph, GraphSlicer

def extract_test_file(repo_dir: Path) -> Optional[str]:
    """
    Extracts the test file path from bugsinpy_run_test.sh
    e.g. "python -m unittest ... test.test_utils..." -> "test/test_utils.py"
    """
    run_test_sh = repo_dir / "bugsinpy_run_test.sh"
    if not run_test_sh.exists():
        return None
    
    try:
        content = run_test_sh.read_text()
        # Expected format: python -m unittest ... test.module.Class.test_method -> We want to resolve to "test/module.py"
        
        # Simple heuristic: find argument starting with 'test.' or 'tests.'
        parts = content.split()
        for part in parts:
            if part.startswith("test.") or part.startswith("tests."):
                # remove Class.method if present to get module
                # logic: keep parts until we find a part that starts with Capital letter (Class)
                # or just look for the file on disk?
                
                # Let's try mapping module path to file path
                module_parts = part.split(".")
                
                # Walk down to find the file
                current_path = repo_dir
                matched_file = None
                
                # Try different lengths of the module path
                for i in range(1, len(module_parts) + 1):
                    # check if subpath + .py exists
                    subpath = "/".join(module_parts[:i]) + ".py"
                    if (repo_dir / subpath).exists():
                        matched_file = subpath
                        # Continue to find the most specific file (longest match)
                
                if matched_file:
                    return matched_file
                    
        return None

    except Exception as e:
        print(f"Error parsing run_test.sh: {e}")
        return None

if __name__ == "__main__":
    project_name = "black"
    bug_id = 1
    version = 0
    experiments_dir = "experiments"
    bugsinpy_root = "datasets/BugsInPy"

    print(f"Running experiment for {project_name} bug {bug_id}...")

    try:
        # 1. Setup
        repo_dir = setup_bugsinpy_case(
            project_name=project_name,
            bug_id=str(bug_id),
            version=version,
            experiments_dir=experiments_dir,
            bugsinpy_root=bugsinpy_root
        )
        print("-" * 20)
        print(f"Repo setup at: {repo_dir}")
        

        # 2. Run tests
        result = run_bugsinpy_case(
            work_dir=repo_dir,
            bugsinpy_root=bugsinpy_root
        )
        
        print(f"Test execution success: {result.success}")
        print("-" * 20)
        print(f"STDOUT:\n{result.stdout}")
        print("-" * 20)
        print(f"STDERR:\n{result.stderr}")
        
        if not result.success:
            print("-" * 20)
            print(f"Failure details: {result.fail_details}")

        # 3. Generate Call Graph
        print("-" * 20)
        print("Generating Call Graph...")
        
        test_file = extract_test_file(repo_dir)
        if test_file:
            print(f"Identified test file: {test_file}")
            
            # Build full graph
            repo_graph = get_repo_call_graph(str(repo_dir))
            print(f"Full graph nodes: {repo_graph.number_of_nodes()}, edges: {repo_graph.number_of_edges()}")
            
            # Slice for test
            slicer = GraphSlicer(repo_graph)
            
            abs_test_file = str(repo_dir / test_file)
            sliced_graph = slicer.slice_for_test(abs_test_file)
            print(f"Sliced graph nodes: {sliced_graph.number_of_nodes()}, edges: {sliced_graph.number_of_edges()}")
            
            # Save artifact
            artifacts_dir = Path("artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%H_%M")
            filename = f"{project_name}_{bug_id}_{timestamp}.json"
            output_path = artifacts_dir / filename
            
            data = nx.node_link_data(sliced_graph)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
                
            print(f"Call graph saved to: {output_path}")

            # Visualize
            print("Visualizing graph...")
            import matplotlib.pyplot as plt
            import collections
            
            plt.figure(figsize=(16, 12))
            
            # Hierarchical Layout Logic (Left-to-Right)
            def hierarchical_layout(G):
                pos = {}
                # 1. Identify roots (nodes with in-degree 0)
                # In our sliced graph, these should be the test functions
                roots = [n for n in G.nodes() if G.in_degree(n) == 0]
                if not roots:
                    # Fallback for cycles
                    if G.nodes():
                        roots = [list(G.nodes())[0]]
                    else:
                        return {}

                # 2. Assign levels (distance from root)
                # Use BFS to assign levels (shortest path from any root)
                # This avoids infinite loops in cyclic graphs
                levels = {}
                queue = collections.deque([(root, 0) for root in roots])
                for root in roots:
                    levels[root] = 0
                
                while queue:
                    u, lvl = queue.popleft()
                    
                    for v in G.neighbors(u):
                        if v not in levels:
                            levels[v] = lvl + 1
                            queue.append((v, lvl + 1))
                            
                # If there are disconnected components or nodes not reachable from roots (shouldn't happen in slice?),
                # assign them level 0 or handle them.
                # In sliced graph, all should be reachable from test roots.
                # However, GraphSlicer does BFS, so technically yes.
                
                # Check for unassigned nodes
                for n in G.nodes():
                    if n not in levels:
                        levels[n] = 0
                
                # 3. Group by level
                level_map = collections.defaultdict(list)
                for node, lvl in levels.items():
                    level_map[lvl].append(node)
                
                # 4. Assign Coordinates
                # X = level, Y = centered index
                for lvl, nodes in level_map.items():
                    # Sort nodes to minimize crossings? 
                    # Simple heuristic: sort by name or degree
                    nodes.sort() 
                    
                    count = len(nodes)
                    # Vertical spacing
                    ys = [i - count/2.0 for i in range(count)]
                    for node, y in zip(nodes, ys):
                        # Scale X to spread out
                        pos[node] = (lvl * 2, -y) # Negative y to match typical top-down logic if we were valid, but here just spreading
                
                return pos

            pos = hierarchical_layout(sliced_graph)
            if not pos:
                # Fallback
                pos = nx.spring_layout(sliced_graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(sliced_graph, pos, node_size=1000, node_color="lightblue", alpha=0.9, node_shape="s")
             # Rounded rectangle not easy in pure nx, use box/square 's' or default
            
            # Draw labels: Function Name Only
            labels = {node: node.split(".")[-1] for node in sliced_graph.nodes()}
            nx.draw_networkx_labels(sliced_graph, pos, labels=labels, font_size=8, font_weight="bold")
            
            # Draw edges with curved lines if possible? 
            # nx doesn't support curved edges easily in basic draw.
            nx.draw_networkx_edges(sliced_graph, pos, arrowstyle="->", arrowsize=15, edge_color="gray", alpha=0.6)
            
            plt.title(f"Call Graph for {project_name} bug {bug_id} ({test_file}) - Left-to-Right Flow")
            plt.axis("off")
            plt.tight_layout()
            
            image_filename = f"{project_name}_{bug_id}_{timestamp}.png"
            image_path = artifacts_dir / image_filename
            plt.savefig(image_path)
            plt.close()
            
            print(f"Graph image saved to: {image_path}")
            
        else:
            print("Could not identify test file from bugsinpy_run_test.sh")

    except SystemExit:
        print("Setup failed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
