import pytest
from unittest.mock import patch
from benchmarks.dataset_loader import load_swebench, load_humaneval
from datasets import Dataset

def test_load_swebench():
    swe = load_swebench()
    assert isinstance(swe, Dataset)
    assert len(swe) > 0
    