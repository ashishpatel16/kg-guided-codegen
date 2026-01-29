import networkx as nx
from typing import Dict, List, Set, Optional, Any
from src.program_analysis.repo_call_graph import build_static_call_graph
class CoverageAnalyzer:
    """
    Analyzes test coverage and code suspiciousness based on a call graph.
    Supports both static reachability analysis and dynamic execution-based coverage.
    """
    def __init__(self, full_graph: nx.DiGraph, is_static: bool = True):
        self.graph = full_graph
        self.is_static = is_static
        # Data for dynamic coverage and suspiciousness
        self.test_results: Dict[str, bool] = {}  # TestFQN -> Pass (True) / Fail (False)
        self.node_execution_map: Dict[str, Set[str]] = {}  # TestFQN -> Set of executed NodeFQNs

    def add_dynamic_test_result(self, test_fqn: str, passed: bool, executed_nodes: Set[str]):
        """
        Adds execution data for a specific test case.
        Required for compute_suspiciousness and dynamic coverage.
        """
        self.test_results[test_fqn] = passed
        self.node_execution_map[test_fqn] = executed_nodes

    def get_covered_nodes(self, test_entry_points: Optional[List[str]] = None) -> Set[str]:
        """
        Returns the set of all covered nodes.
        If is_static=True: Uses reachability from test_entry_points.
        If is_static=False: Uses aggregated data from node_execution_map.
        """
        if self.is_static:
            if not test_entry_points:
                return set()
            
            covered = set()
            for entry in test_entry_points:
                if entry in self.graph:
                    covered.add(entry)
                    descendants = nx.descendants(self.graph, entry)
                    covered.update(descendants)
            return covered
        else:
            # Aggregate all nodes executed by all recorded tests
            all_executed = set()
            for executed in self.node_execution_map.values():
                all_executed.update(executed)
            return all_executed

    def get_uncovered_nodes(self, test_entry_points: Optional[List[str]] = None) -> Set[str]:
        """Returns nodes in the graph that are NOT covered."""
        all_nodes = set(self.graph.nodes())
        covered_nodes = self.get_covered_nodes(test_entry_points)
        return all_nodes - covered_nodes

    def compute_metrics(self, test_entry_points: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Computes summary statistics for coverage.
        """
        covered = self.get_covered_nodes(test_entry_points)
        all_nodes = list(self.graph.nodes())
        total_count = len(all_nodes)
        covered_count = len(covered)
        
        coverage_pct = (covered_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "total_nodes": total_count,
            "covered_nodes": covered_count,
            "uncovered_nodes": total_count - covered_count,
            "coverage_percentage": round(coverage_pct, 2),
            "is_static": self.is_static
        }

    def compute_suspiciousness(self, method: str = "tarantula") -> Dict[str, float]:
        """
        Computes suspiciousness score for each node based on pass/fail test results.
        Currently supports: 'tarantula'
        
        Tarantula Score = (failed(s)/total_failed) / [(passed(s)/total_passed) + (failed(s)/total_failed)]
        """
        if not self.test_results:
            return {}

        total_failed = sum(1 for passed in self.test_results.values() if not passed)
        total_passed = sum(1 for passed in self.test_results.values() if passed)

        if total_failed == 0:
            # If no tests failed, nothing is "suspicious" in the fault localization sense
            return {node: 0.0 for node in self.graph.nodes()}

        suspiciousness = {}
        
        # Count passes/fails per node
        node_stats = {node: {"passed": 0, "failed": 0} for node in self.graph.nodes()}
        
        for test_fqn, executed_nodes in self.node_execution_map.items():
            passed = self.test_results.get(test_fqn, False)
            for node in executed_nodes:
                if node in node_stats:
                    if passed:
                        node_stats[node]["passed"] += 1
                    else:
                        node_stats[node]["failed"] += 1

        for node, stats in node_stats.items():
            failed_s = stats["failed"]
            passed_s = stats["passed"]
            
            # Tarantula formula
            failed_ratio = failed_s / total_failed
            passed_ratio = (passed_s / total_passed) if total_passed > 0 else 0
            
            denominator = passed_ratio + failed_ratio
            if denominator == 0:
                score = 0.0
            else:
                score = failed_ratio / denominator
            
            suspiciousness[node] = round(score, 4)

        return suspiciousness

    def get_test_dependency_map(self) -> Dict[str, List[str]]:
        """
        Returns a mapping from ProductionFunction -> List of TestFunctions that cover it.
        Works for both static and dynamic settings.
        """
        dep_map = {}
        
        if self.is_static:
            # For each test node (heuristic: contains 'test' or in a test file)
            # Find all reachable nodes
            for node, data in self.graph.nodes(data=True):
                # Heuristic for identifying test nodes if not explicitly marked
                is_test = node.split(".")[-1].startswith("test_") or "test" in data.get("file", "").lower()
                if is_test:
                    reachable = nx.descendants(self.graph, node)
                    for r in reachable:
                        if r not in dep_map:
                            dep_map[r] = []
                        dep_map[r].append(node)
        else:
            # Use actual execution data
            for test_fqn, executed_nodes in self.node_execution_map.items():
                for node in executed_nodes:
                    if node not in dep_map:
                        dep_map[node] = []
                    dep_map[node].append(test_fqn)
                    
        return dep_map


if __name__ == "__main__":
    repo_root = "/Users/ashish/master-thesis/kg-guided-codegen/src/benchmarks/exp/demo"

    
    # 1. Build the call graph
    graph = build_static_call_graph(repo_root)
    
    # 2. Initialize analyzer (set is_static=False for fault localization)
    analyzer = CoverageAnalyzer(graph, is_static=False)
    analyzer.add_dynamic_test_result(
        test_fqn="test_calc.TestCalc.test_add", 
        passed=True, 
        executed_nodes={"calc.add"}
    )

    cov_nodes = analyzer.get_covered_nodes(test_entry_points=['test_calc.TestCalc.test_add'])
    print(f"Covered Nodes: {cov_nodes}")

    uncov_nodes = analyzer.get_uncovered_nodes(test_entry_points=['test_calc.TestCalc.test_add'])
    print(f"Uncovered Nodes: {uncov_nodes}\n")

    dependency_map = analyzer.get_test_dependency_map()
    print('Printing Dependency map')
    for key, value in dependency_map.items():
        print(f"{key} -> {value}\n")
    # print(f"\n\nDependency Map: {dependency_map}")

    
    # 3. Simulate adding dynamic test results
    # In a real scenario, you'd get these from a test runner + tracer
    analyzer.add_dynamic_test_result(
        test_fqn="test_calc.TestCalc.test_add", 
        passed=True, 
        executed_nodes={"calc.add"}
    )
    analyzer.add_dynamic_test_result(
        test_fqn="tests.test_demo.test_fail", 
        passed=False, 
        executed_nodes={"math_utils.add", "math_utils.divide"}
    )
    
    # 4. Compute Tarantula scores
    scores = analyzer.compute_suspiciousness(method="tarantula")
    
    print("Tarantula Suspiciousness Scores:")
    for node, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score > 0:
            print(f"  {node}: {score}")

    
    