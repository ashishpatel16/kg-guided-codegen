#!/usr/bin/env python3
"""
Analyzes the Black bug using dynamic call graph and Tarantula fault localization.

This script:
1. Traces the execution of failing and passing tests in a subprocess
2. Builds a dynamic call graph
3. Computes Tarantula suspiciousness scores
4. Reports the most suspicious functions

Usage:
    python src/benchmarks/analyze_black_bug.py
    
The script runs tests in the Black virtual environment via subprocess,
so you don't need to activate it first.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from program_analysis.dynamic_call_graph import DynamicCallGraphTracer


def analyze_black_bug(
    repo_root: str,
    python_executable: str,
    failing_test: str,
    passing_tests: list[str],
    output_file: str = "artifacts/black_1_suspiciousness.json",
    top_n: int = 20
):
    """
    Analyzes a bug by comparing failing and passing tests using subprocess execution.
    
    Args:
        repo_root: Root directory of the Black repository
        python_executable: Path to Python executable to use (e.g., venv/bin/python)
        failing_test: Name of the failing test method
        passing_tests: List of passing test method names
        output_file: Where to save the results
        top_n: Number of top suspicious nodes to display
    """
    
    print(f"\n{'='*80}")
    print(f"Black Bug Analysis: Fault Localization with Tarantula")
    print(f"{'='*80}")
    print(f"Repository: {repo_root}")
    print(f"Python: {python_executable}")
    print(f"Failing test: {failing_test}")
    print(f"Passing tests: {', '.join(passing_tests)}")
    print(f"{'='*80}\n")
    
    # Use subprocess method to run tests in target environment
    call_graph = DynamicCallGraphTracer.run_tests_subprocess(
        repo_root=repo_root,
        python_executable=python_executable,
        test_module="tests.test_black",
        test_class="BlackTestCase",
        passing_tests=passing_tests,
        failing_tests=[failing_test],
        include_external=False
    )

    # Display results
    print(f"\n{'='*80}")
    print("Computing Tarantula suspiciousness scores...")
    print(f"{'='*80}\n")

    print(f"Call Graph Statistics:")
    print(f"  • Total nodes: {len(call_graph.nodes)}")
    print(f"  • Total edges: {len(call_graph.edges)}")
    
    # Display suspiciousness scores
    print(f"\n{'='*80}")
    print(f"Top {top_n} Most Suspicious Functions (Likely Bug Locations)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Suspiciousness':<15} {'Function':<50} {'Location'}")
    print(f"{'-'*80}")
    
    top_suspicious = call_graph.get_suspicious_nodes(min_suspiciousness=0.0, limit=top_n)
    for i, node in enumerate(top_suspicious, 1):
        if node.suspiciousness > 0:
            file_name = os.path.basename(node.file) if node.file else "unknown"
            fqn_short = node.fqn if len(node.fqn) <= 48 else node.fqn[:45] + "..."
            print(f"{i:<6} {node.suspiciousness:<15.4f} {fqn_short:<50} {file_name}:{node.start_line}")
    
    # Export as json
    with open(output_file, "w") as f:
        json.dump(call_graph.model_dump(), f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return call_graph


if __name__ == "__main__":
    # Configuration
    repo_root = "/Users/ashish/master-thesis/kg-guided-codegen/experiments/black_1/black/"
    python_executable = "/Users/ashish/master-thesis/kg-guided-codegen/experiments/black_1/black/env/bin/python"
    
    # The failing test that exhibits the bug
    failing_test = "test_works_in_mono_process_only_environment"
    
    # Some passing tests for comparison
    passing_tests = [
        "test_empty",
        "test_empty_ff",
    ]
    
    # Run the analysis
    call_graph = analyze_black_bug(
        repo_root=repo_root,
        python_executable=python_executable,
        failing_test=failing_test,
        passing_tests=passing_tests,
        output_file="artifacts/black_1_suspiciousness.json",
        top_n=20
    )
