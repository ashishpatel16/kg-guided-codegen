
import unittest
from unittest.mock import MagicMock
from src.program_analysis.suspiciousness_controller import SuspiciousnessController

class TestSuspiciousnessControllerLLM(unittest.TestCase):
    def setUp(self):
        self.mock_execution_map = {
            "test_1": {"node_A", "node_B", "node_C"},
            "test_2": {"node_A", "node_B", "node_C"},
        }
        self.mock_results = {"test_1": False, "test_2": False}
        self.mock_llm = MagicMock()
        self.controller = SuspiciousnessController(
            self.mock_execution_map, 
            self.mock_results, 
            llm_connector=self.mock_llm
        )

    def test_identify_ambiguity_groups(self):
        groups = self.controller.identify_ambiguity_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], {"node_A", "node_B", "node_C"})

    def test_add_test_case(self):
        # Adding a test that only covers node_A and node_B, breaking node_C away
        self.controller.add_test_case("test_3", {"node_A", "node_B"}, True)
        
        groups = self.controller.identify_ambiguity_groups()
        # Now {node_A, node_B} should be one group, and node_C should be isolated
        self.assertEqual(len(groups), 1)
        self.assertIn({"node_A", "node_B"}, groups)
        self.assertNotIn("node_C", groups[0])

    def test_remove_test_case(self):
        self.controller.remove_test_case("test_2")
        # Still ambiguous because test_1 covers all 3
        groups = self.controller.identify_ambiguity_groups()
        self.assertEqual(len(groups), 1)

    def test_generate_test_to_disambiguate_mock(self):
        self.mock_llm.generate.return_value = "```python\ndef test_new(): pass\n```"
        call_graph = {"nodes": [
            {"fqn": "node_A", "file": "dummy.py", "start_line": 1, "end_line": 10},
            {"fqn": "node_B", "file": "dummy.py", "start_line": 11, "end_line": 20},
            {"fqn": "node_C", "file": "dummy.py", "start_line": 21, "end_line": 30},
        ]}
        
        # Mock get_function_source to avoid file access
        import src.program_analysis.suspiciousness_controller 
        from unittest.mock import patch
        
        with patch('src.agent.tools.get_function_source', return_value="def dummy(): pass"):
            test_code = self.controller.generate_test_to_disambiguate("node_A", call_graph)
            
        self.assertEqual(test_code, "def test_new(): pass")
        self.mock_llm.generate.assert_called_once()
        prompt = self.mock_llm.generate.call_args[0][0]
        self.assertIn("node_A", prompt)
        self.assertIn("node_B", prompt)
        self.assertIn("node_C", prompt)

if __name__ == "__main__":
    unittest.main()
