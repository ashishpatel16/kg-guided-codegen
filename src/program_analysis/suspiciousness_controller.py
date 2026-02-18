import logging
from typing import Dict, List, Set, Any
import networkx as nx

logger = logging.getLogger(__name__)



class SuspiciousnessController:
    """
    Refines and controls suspiciousness scores based on coverage patterns.
    Helps isolate buggy nodes and identify ambiguity groups.
    """
    def __init__(self, node_execution_map: Dict[str, Set[str]], test_results: Dict[str, bool], llm_connector: Any = None):
        self.node_execution_map = node_execution_map
        self.test_results = test_results
        self.node_to_tests = self._invert_execution_map()
        self.llm_connector = llm_connector

    def _invert_execution_map(self) -> Dict[str, Set[str]]:
        """Maps NodeFQN -> Set of TestFQNs that cover it."""
        inverted = {}
        for test_fqn, nodes in self.node_execution_map.items():
            for node in nodes:
                if node not in inverted:
                    inverted[node] = set()
                inverted[node].add(test_fqn)
        return inverted

    def identify_ambiguity_groups(self) -> List[Set[str]]:
        """
        Identifies groups of nodes that are always covered by the exact same set of tests.
        These nodes are indistinguishable using the current test suite.
        """
        groups: Dict[tuple, Set[str]] = {}
        for node, tests in self.node_to_tests.items():
            # Create a stable key from the set of tests
            key = tuple(sorted(list(tests)))
            if key not in groups:
                groups[key] = set()
            groups[key].add(node)
        
        # Return groups with more than one node
        return [g for g in groups.values() if len(g) > 1]

    def get_ambiguity_group_for_node(self, node_fqn: str) -> Set[str]:
        """Returns the set of nodes that are indistinguishable from the given node."""
        tests = self.node_to_tests.get(node_fqn, set())
        key = tuple(sorted(list(tests)))
        
        ambiguous_nodes = set()
        for node, other_tests in self.node_to_tests.items():
            if tuple(sorted(list(other_tests))) == key:
                ambiguous_nodes.add(node)
        return ambiguous_nodes

    def calculate_refinement_potential(self, call_graph: nx.DiGraph) -> Dict[str, float]:
        """
        Calculates which nodes are 'central' to failing coverage but have children 
        that aren't covered by all the same tests.
        Potential is higher for nodes that could help split an ambiguity group.
        """
        potential = {}
        for node in call_graph.nodes():
            group = self.get_ambiguity_group_for_node(node)
            if len(group) <= 1:
                potential[node] = 0.0
                continue
            
            # If the node has children that are NOT in the same ambiguity group,
            # then testing those children specifically could refine this node.
            different_children = 0
            successors = list(call_graph.successors(node))
            for succ in successors:
                if succ not in group:
                    different_children += 1
            
            # Simple metric: ratio of different children to total children
            if successors:
                potential[node] = different_children / len(successors)
            else:
                potential[node] = 0.0
                
        return potential

    def get_refined_scores(self, base_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Refines base suspiciousness scores (e.g. Tarantula) by penalizing nodes 
        in large ambiguity groups if they are 'coincidental' hits (e.g. utility functions).
        """
        refined = base_scores.copy()
        for node, score in base_scores.items():
            group = self.get_ambiguity_group_for_node(node)
            if len(group) > 5:  # Arbitrary threshold for large groups
                # Penalize large ambiguity groups unless the score is very high
                if score < 0.8:
                    refined[node] = score * 0.8  # Slight penalty
        return refined

    def generate_test_to_disambiguate(self, node_fqn: str, call_graph: Any) -> str:
        """
        Uses Gemini LLM to generate a test case that specifically distinguishes the given node
        from its ambiguity group members.
        """
        if not self.llm_connector:
            raise ValueError("LLM connector not initialized in SuspiciousnessController")

        group = self.get_ambiguity_group_for_node(node_fqn)
        if len(group) <= 1:
            return f"# Node {node_fqn} is not ambiguous."

        # Collect source code for all nodes in the group for context
        from src.agent.tools import get_function_source
        
        context_sources = {}
        for member_fqn in group:
            node_data = next((n for n in call_graph.get("nodes", []) if n["fqn"] == member_fqn), None)
            if node_data:
                src = get_function_source(node_data)
                if src:
                    context_sources[member_fqn] = src

        source_info = "\n\n".join([f"Source for {fqn}:\n```python\n{src}\n```" for fqn, src in context_sources.items()])

        prompt = f"""
        You are an expert software tester. Your goal is to write a Python pytest test case that DISAMBIGUATES a specific node from a group of nodes that currently have identical test coverage.
        
        Target Node to isolate: {node_fqn}
        Ambiguity Group: {list(group)}

        CONTEXT (Source code for nodes in the group):
        {source_info}

        INSTRUCTIONS:
        1. Write a NEW test case (using pytest) that executes paths specifically in {node_fqn} while potentially AVOIDING or differently exercising the other nodes in the group.
        2. The test should be designed such that if {node_fqn} is buggy, this test fails, but if the other nodes are buggy and {node_fqn} is not, this test might pass (or vice versa).
        3. Only provide the code for the test function, enclosed in standard markdown code blocks.
        4. Do not include any other text or explanations.
        """

        raw_response = self.llm_connector.generate(prompt)
        # Extract code from the response
        import re
        m = re.search(r"```python\s*([\s\S]*?)\s*```", raw_response)
        if m:
            return m.group(1).strip()
        return raw_response.strip()

    def add_test_case(self, test_fqn: str, covered_nodes: Set[str], passed: bool):
        """
        Manually adds a (hypothetical or real) test case to the internal maps.
        Useful for simulating the impact of a new test without re-running the whole suite.
        """
        self.node_execution_map[test_fqn] = covered_nodes
        self.test_results[test_fqn] = passed
        # Re-invert the map
        self.node_to_tests = self._invert_execution_map()

    def remove_test_case(self, test_fqn: str):
        """Removes a test case from the internal maps."""
        if test_fqn in self.node_execution_map:
            del self.node_execution_map[test_fqn]
        if test_fqn in self.test_results:
            del self.test_results[test_fqn]
        # Re-invert the map
        self.node_to_tests = self._invert_execution_map()

if __name__ == "__main__":
    pass