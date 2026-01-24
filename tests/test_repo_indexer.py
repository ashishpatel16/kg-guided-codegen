import os
import unittest
import tempfile
import shutil
from src.program_analysis.repo_call_graph import RepoIndexer
import pytest

@pytest.fixture
def repo_dir():
    path = "src/benchmarks/exp/demo"
    if not os.path.exists(path):
        pytest.fail(f"Hardcoded repo_dir not found at: {path}")
    return path

@pytest.fixture
def repo_dir_index(repo_dir):
    indexer = RepoIndexer(repo_dir)
    index = indexer.index_repo()
    return index

def test_index(repo_dir_index):
    index = repo_dir_index

    for key, val in index.items():
        print(key," ->" ,val) 
    

    assert isinstance(index, dict)
    assert len(index) > 0

    for fqn, info in index.items():
        assert isinstance(fqn, str)
        assert isinstance(info, dict)
        
        # Verify required keys
        assert "type" in info
        assert "file" in info
        assert "start_line" in info
        
        # Verify value types/contents
        assert info["type"] in ["class_definition", "function_definition"]
        assert os.path.exists(info["file"])
        assert isinstance(info["start_line"], int)
        assert info["start_line"] > 0

def test_index_file(repo_dir_index):
    index = repo_dir_index