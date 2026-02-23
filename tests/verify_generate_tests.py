import os
import sys
import json
from unittest.mock import MagicMock, patch

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.nodes import generate_tests
from src.agent.state import DebuggingState

def test_generate_tests_mock():
    # Setup mock workspace
    workspace = os.path.abspath("mock_workspace")
    os.makedirs(workspace, exist_ok=True)
    with open(os.path.join(workspace, "test_math.py"), "w") as f:
        f.write("def test_add(): assert 1 + 1 == 2")
    
    state: DebuggingState = {
        "host_workspace": workspace,
        "use_docker": False,
        "llm_calls": 0,
        "history": []
    }
    
    print("Running generate_tests (local mock)...")
    result = generate_tests(state)
    
    assert "tests" in result
    assert "test_math.py" in result["tests"]
    print(f"SUCCESS: Discovered tests: {result['tests']}")
    
    # Clean up
    if os.path.exists(workspace):
        import shutil
        shutil.rmtree(workspace)

if __name__ == "__main__":
    test_generate_tests_mock()
