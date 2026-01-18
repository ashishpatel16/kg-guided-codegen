
import sys
import os
import json
import networkx as nx
import collections
from pathlib import Path
from datetime import datetime

# Add src to sys.path to allow imports if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.program_analysis.repo_call_graph import get_repo_call_graph, GraphSlicer

def extract_test_file(repo_dir: Path) -> str:
    """
    Extracts the test file path from bugsinpy_run_test.sh
    e.g. "python -m unittest test_calc.TestCalc.test_complex" -> "test_calc.py"
    """
    run_test_sh = repo_dir / "bugsinpy_run_test.sh"
    if not run_test_sh.exists():
        return None
    
    try:
        content = run_test_sh.read_text().strip()
        # Expected format: python -m unittest test_module.TestClass.test_method
        parts = content.split()
        for part in parts:
            if "unittest" in part:
                 continue
            if part == "python" or part == "-m":
                continue
            
            # part is likely the test path
            # split by dot, find module
            module_name = part.split(".")[0]
            if (repo_dir / f"{module_name}.py").exists():
                return f"{module_name}.py"
                
        return None

    except Exception as e:
        print(f"Error parsing run_test.sh: {e}")
        return None

def hierarchical_layout(G):
    pos = {}
    # 1. Identify roots (nodes with in-degree 0)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not roots:
         # Fallback for cycles: pick node with min in-degree
         if G.nodes():
             degrees = dict(G.in_degree())
             min_degree = min(degrees.values())
             roots = [n for n, d in degrees.items() if d == min_degree]
         else:
            return {}

    # 2. Assign levels (distance from root) using BFS
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
                
    # Check for unassigned nodes
    for n in G.nodes():
        if n not in levels:
            levels[n] = 0
    
    # 3. Group by level
    level_map = collections.defaultdict(list)
    for node, lvl in levels.items():
        level_map[lvl].append(node)
    
    # 4. Assign Coordinates
    for lvl, nodes in level_map.items():
        nodes.sort() 
        count = len(nodes)
        ys = [i - count/2.0 for i in range(count)]
        for node, y in zip(nodes, ys):
            pos[node] = (lvl * 2, -y) 
    
    return pos

if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    repo_dir = current_dir / "exp" / "demo"
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    print(f"Running demo experiment in {repo_dir}...")

    # 1. Build Repo Graph
    print("Building repo call graph...")
    repo_graph = get_repo_call_graph(str(repo_dir))
    print(f"Repo graph: {repo_graph.number_of_nodes()} nodes, {repo_graph.number_of_edges()} edges")
    
    # 2. Slice for Test
    print("Slicing for test case...")
    test_file = extract_test_file(repo_dir)
    print(f"Identified test file: {test_file}")
    
    if test_file:
        slicer = GraphSlicer(repo_graph)
        abs_test_file = str(repo_dir / test_file)
        sliced_graph = slicer.slice_for_test(abs_test_file)
        print(f"Sliced graph: {sliced_graph.number_of_nodes()} nodes, {sliced_graph.number_of_edges()} edges")
        
        # 3. Save JSON
        timestamp = datetime.now().strftime("%H_%M")
        json_path = artifacts_dir / f"demo_call_graph_{timestamp}.json"
        data = nx.node_link_data(sliced_graph)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON to {json_path}")
        
        # 4. Visualize
        print("Visualizing...")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        pos = hierarchical_layout(sliced_graph)
        if not pos:
             pos = nx.spring_layout(sliced_graph)
             
        # Draw nodes
        nx.draw_networkx_nodes(sliced_graph, pos, node_size=1200, node_color="lightgreen", alpha=0.9, node_shape="s")
        
        # Labels
        labels = {node: node.split(".")[-1] for node in sliced_graph.nodes()}
        nx.draw_networkx_labels(sliced_graph, pos, labels=labels, font_size=10, font_weight="bold")
        
        # Edges
        nx.draw_networkx_edges(sliced_graph, pos, arrowstyle="->", arrowsize=20, edge_color="gray", alpha=0.6)
        
        plt.title("Demo Call Graph (Hierarchical)")
        plt.axis("off")
        plt.tight_layout()
        
        img_path = artifacts_dir / f"demo_call_graph_{timestamp}.png"
        plt.savefig(img_path)
        plt.close()
        print(f"Saved image to {img_path}")
        
    else:
        print("Could not find test file!")
