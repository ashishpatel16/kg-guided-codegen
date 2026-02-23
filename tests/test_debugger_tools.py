import os
import sys
from typing import Dict, Any

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.nodes import generate_patch
from src.agent.state import DebuggingState

def test_generate_patch_mock():
    mock_call_graph = {
        "nodes": [
            {
                "fqn": "demo.is_prime",
                "file": "demo.py",
                "start_line": 1,
                "end_line": 10,
                "suspiciousness": 1.0,
                "confidence_score": 0.95
            }
        ],
        "edges": []
    }
    
    # Create a dummy demo.py for get_function_source to work
    with open("demo.py", "w") as f:
        f.write("def is_prime(n):\n    if n < 2: return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n")
    
    try:
        state: DebuggingState = {
            "target_node": "demo.is_prime",
            "call_graph": mock_call_graph,
            "reflection": "The function works but is slow for large primes. It should use sqrt(n).",
            "llm_calls": 0,
            "history": []
        }
        
        print("Calling generate_patch...")
        result = generate_patch(state)
        
        assert "final_patch" in result
        assert "final_diff" in result
        assert isinstance(result["final_patch"], str)
        assert isinstance(result["final_diff"], str)
        assert len(result["final_patch"]) > 0
        assert len(result["final_diff"]) > 0
        assert result["llm_calls"] == 1
        
        print(f"SUCCESS: Generated patch:\n{result['final_patch']}")
        print(f"SUCCESS: Generated diff:\n{result['final_diff']}")
    finally:
        # Cleanup
        if os.path.exists("demo.py"):
            os.remove("demo.py")

if __name__ == "__main__":
    test_generate_patch_mock()
