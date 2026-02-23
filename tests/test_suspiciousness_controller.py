import pytest
import networkx as nx
from src.program_analysis.suspiciousness_controller import SuspiciousnessController

def test_ambiguity_group_detection():
    # Setup mock data where node_A and node_B are always together
    execution_map = {
        "test_fail": {"node_A", "node_B", "bug_node"},
        "test_pass1": {"node_A", "node_B", "helper_node"},
        "test_pass2": {"node_A", "node_B"},
    }
    results = {"test_fail": False, "test_pass1": True, "test_pass2": True}
    
    controller = SuspiciousnessController(execution_map, results)
    
    # node_A and node_B should be ambiguous
    group_a = controller.get_ambiguity_group_for_node("node_A")
    assert "node_A" in group_a
    assert "node_B" in group_a
    assert "bug_node" not in group_a
    
    # bug_node should be alone in its group (only covered by test_fail)
    group_bug = controller.get_ambiguity_group_for_node("bug_node")
    assert group_bug == {"bug_node"}

def test_refinement_potential():
    # Call graph: node_A -> node_B
    cg = nx.DiGraph()
    cg.add_edge("node_A", "node_B")
    
    # node_A and node_B are ambiguous in current tests
    execution_map = {
        "test1": {"node_A", "node_B"}
    }
    controller = SuspiciousnessController(execution_map, {"test1": False})
    
    # node_A has child node_B. Since they are in the same ambiguity group, 
    # potential is initially 0 (simple metric logic).
    # Wait, my logic was: different_children / total.
    # If succ is in group, different_children doesn't increment.
    potential = controller.calculate_refinement_potential(cg)
    assert potential["node_A"] == 0.0
    
    # Now add a test that covers ONLY node_A (not node_B) - hypothetical refinement
    execution_map_refined = {
        "test1": {"node_A", "node_B"},
        "test2": {"node_A"}
    }
    controller_refined = SuspiciousnessController(execution_map_refined, {"test1": False, "test2": True})
    
    # Now node_A and node_B are NOT ambiguous
    assert controller_refined.get_ambiguity_group_for_node("node_A") == {"node_A"}
    assert controller_refined.get_ambiguity_group_for_node("node_B") == {"node_B"}

if __name__ == "__main__":
    pytest.main([__file__])
