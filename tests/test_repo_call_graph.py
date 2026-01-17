import os
import unittest
import sys

# Ensure src is in path for imports if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.program_analysis.repo_call_graph import get_repo_call_graph, GraphSlicer

class TestRepoCallGraph(unittest.TestCase):
    def test_real_repo_tqdm(self):
        # Path relative to project root (assuming run from project root)
        repo_root = os.path.abspath("experiments/tqdm_1/tqdm/tqdm")
        
        # Check if running from tests dir or root
        if not os.path.exists(repo_root):
             # Try adjusting if we are in tests dir
             repo_root = os.path.abspath("../experiments/tqdm_1/tqdm/tqdm")

        if not os.path.exists(repo_root):
            print(f"Skipping tqdm test: {repo_root} not found")
            return

        print(f"\nTesting on real repo: {repo_root}")
        graph = get_repo_call_graph(repo_root)
        
        print(f"Graph nodes: {len(graph.nodes())}")
        print(f"Graph edges: {len(graph.edges())}")
        
        # Verify graph is not empty
        self.assertTrue(len(graph.nodes()) > 0, "Graph should not be empty")
        self.assertTrue(len(graph.edges()) > 0, "Graph should have edges")
        
        slicer = GraphSlicer(graph)
        
        test_files = []
        for root, _, files in os.walk(repo_root):
            for f in files:
                if f.startswith("tests_") or f.endswith("_tests.py"):
                    test_files.append(os.path.join(root, f))
        
        if test_files:
            # Pick a test file that likely has content
            target_test = test_files[0]
            print(f"Slicing for test file: {target_test}")
            subgraph = slicer.slice_for_test(target_test)
            print(f"Subgraph nodes: {len(subgraph.nodes())}")
            print(f"Subgraph edges: {len(subgraph.edges())}")
        else:
            print("No test files found in repo to slice.")

if __name__ == "__main__":
    unittest.main()
