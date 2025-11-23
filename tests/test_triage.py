import pytest
from src.benchmarks.triage import TriageEnvironment
from src.benchmarks.dataset_loader import load_swebench
import os

@pytest.fixture
def swe_issue():
    dataset = load_swebench(split='test')
    return dataset[0]

@pytest.fixture
def triage_testing_dir():
    return 'tests/swe-bench-exp'

def test_triage_constructor(swe_issue):
    triage_env = TriageEnvironment(swe_issue)
    assert triage_env.instance_id == swe_issue['instance_id']
    assert triage_env.patch == swe_issue['patch']
    assert triage_env.repo == swe_issue['repo']
    assert triage_env.base_commit == swe_issue['base_commit']
    assert triage_env.version == swe_issue['version']
    assert triage_env.environment_setup_commit == swe_issue['environment_setup_commit']
    assert type(triage_env.fail_to_pass) == list
    assert type(triage_env.pass_to_pass) == list


def test_setup_workspace(swe_issue, triage_testing_dir):
    triage_env = TriageEnvironment(swe_issue)
    repo_dir = triage_env.setup_workspace(work_root=triage_testing_dir)
    assert os.path.exists(triage_env.repo_dir)
    assert repo_dir == triage_env.repo_dir


def test_setup_environment(swe_issue, triage_testing_dir):
    triage_env = TriageEnvironment(swe_issue)
    triage_env.setup_workspace(work_root=triage_testing_dir)
    venv_dir = triage_env.setup_environment()
    assert os.path.exists(triage_env.venv_dir)
    assert venv_dir == triage_env.venv_dir

def test_run_tests(swe_issue, triage_testing_dir):
    triage_env = TriageEnvironment(swe_issue)
    triage_env.setup_workspace(work_root=triage_testing_dir)
    triage_env.setup_environment()
    triage_env.run_tests()
    assert os.path.exists(triage_env.test_results_dir)