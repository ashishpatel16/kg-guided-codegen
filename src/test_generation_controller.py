"""
Controller that bridges BugsInPy Docker sandbox with the test generation agent.

Provides utilities for:
- Extracting source/test code from Docker containers
- Running the LLM-based test gen agent to produce diverse tests
- Injecting generated tests back into the project and re-tracing
- Computing embedding similarity matrices for redundancy analysis
"""

import ast
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agent.test_generation.core.generator import UnitTestGenerator
from src.agent.test_generation.utils.code_processing import extract_unit_tests
from src.benchmarks.setup_bugsinpy_docker import BugsInPyDockerSandbox
from src.condensation import (
    ExperimentResult,
    condense_test_suite,
    compute_tarantula,
    extract_ground_truth_fqns,
    get_rank,
    invert_coverage_matrix,
    run_condensation_experiment,
)


@dataclass
class BugContext:
    source_code: str
    test_code: str
    source_file_container: str
    test_file_container: str
    module_import: str  # e.g. "youtube_dl.jsinterp"
    buggy_fqn: str


def extract_bug_context(
    sandbox: BugsInPyDockerSandbox,
    call_graph: Dict[str, Any],
    patch_text: str,
) -> Optional[BugContext]:
    """Read buggy module source and test file from the running Docker container."""
    nodes = call_graph.get("nodes", [])
    gt_fqns = extract_ground_truth_fqns(patch_text, nodes)
    if not gt_fqns:
        print("  Could not extract ground truth FQN")
        return None

    buggy_fqn = gt_fqns[0]
    buggy_node = next((n for n in nodes if n["fqn"] == buggy_fqn), None)
    if not buggy_node:
        print(f"  Node not found for {buggy_fqn}")
        return None

    source_file = buggy_node["file"]

    test_files = {n["file"] for n in nodes if n.get("is_test")}
    if not test_files:
        test_files = {n["file"] for n in nodes if "test" in n.get("file", "").lower()}
    test_file = sorted(test_files)[0] if test_files else None
    if not test_file:
        print("  No test file found in call graph")
        return None

    ec, source_code, _ = sandbox.sandbox.run_command(f"cat {source_file}")
    if ec != 0:
        print(f"  Failed to read {source_file}")
        return None

    ec, test_code, _ = sandbox.sandbox.run_command(f"cat {test_file}")
    if ec != 0:
        print(f"  Failed to read {test_file}")
        return None

    # Derive module import path: /home/workspace/youtube-dl/youtube_dl/jsinterp.py -> youtube_dl.jsinterp
    project_root = sandbox.container_project_root + "/"
    rel = source_file.replace(project_root, "")
    module_import = rel.replace("/", ".").removesuffix(".py")

    # For large files, extract only the buggy function + surrounding context
    start_line = buggy_node.get("start_line", 0)
    if len(source_code) > 15000 and start_line > 0:
        source_code = _extract_function_context(source_code, start_line)

    return BugContext(
        source_code=source_code,
        test_code=test_code,
        source_file_container=source_file,
        test_file_container=test_file,
        module_import=module_import,
        buggy_fqn=buggy_fqn,
    )


def _extract_function_context(source: str, start_line: int, context_lines: int = 200) -> str:
    """Extract a function and surrounding context from a large source file."""
    lines = source.splitlines()
    begin = max(0, start_line - 30)
    end = min(len(lines), start_line + context_lines)

    # Walk forward from start_line to find the function end via dedent
    indent = None
    for i in range(start_line - 1, min(len(lines), start_line + 500)):
        stripped = lines[i].rstrip()
        if not stripped:
            continue
        cur_indent = len(lines[i]) - len(lines[i].lstrip())
        if indent is None:
            indent = cur_indent
        elif cur_indent <= indent and i > start_line:
            end = i
            break

    # Include imports from the top of the file
    import_lines = []
    for line in lines[:60]:
        if line.startswith(("import ", "from ", "#")) or not line.strip():
            import_lines.append(line)
        else:
            break

    header = "\n".join(import_lines) + "\n\n# ... (truncated) ...\n\n"
    body = "\n".join(lines[begin:end])
    return header + body


def make_agent_config(google_api_key: str) -> Dict[str, Any]:
    """Build config dict for UnitTestGenerator using Gemini models."""
    return {
        "llm": {
            "model_name": "gemini-2.0-flash",
            "max_improvements": 2,
            "similarity_comparison_count": 5,
        },
        "api": {
            "google_api_key": google_api_key,
            "openai_api_key": "",
            "groq_api_key": "",
            "jina_api_key": "",
        },
        "logging": {"enabled": False},
    }


def generate_diverse_tests(
    source_code: str,
    existing_test_code: str,
    n_tests: int,
    agent_config: Dict[str, Any],
) -> List[str]:
    """
    Run the test generation agent N times, accumulating results in the vector
    store so each subsequent test is forced to be diverse from all previous ones.
    """
    logger = logging.getLogger("test_gen_controller")
    loggers = (logger, logger)

    generator = UnitTestGenerator(agent_config, loggers)
    generator.initialize_vector_store(existing_test_code)

    generated: List[str] = []
    for i in range(n_tests):
        print(f"  Generating test {i + 1}/{n_tests} ...")
        try:
            result = generator.generate_test(
                code_to_test=source_code,
                coverage_matrix="Not available (function-level tracing only)",
                uncovered_lines=[],
            )
            test_code = result["generated_test_case"]
            if test_code.strip():
                generated.append(test_code)
                print(f"    ✓ Generated: {test_code.splitlines()[0][:80]}")
            else:
                print(f"    ✗ Empty output")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    return generated


def build_augmented_test_file(
    original_test_code: str,
    generated_tests: List[str],
    module_import: str,
) -> str:
    """
    Append a new AgentGeneratedTests class to the existing test file.
    The class inherits unittest.TestCase and imports from the correct module.
    """
    if not generated_tests:
        return original_test_code

    import_line = f"from {module_import} import *"

    methods = []
    for test in generated_tests:
        indented = "\n".join("    " + line if line.strip() else "" for line in test.splitlines())
        methods.append(indented)

    new_class = (
        f"\n\nimport unittest\n"
        f"{import_line}\n\n"
        f"class AgentGeneratedTests(unittest.TestCase):\n"
        + "\n\n".join(methods)
        + "\n"
    )

    return original_test_code + new_class


def inject_and_retrace(
    sandbox: BugsInPyDockerSandbox,
    augmented_test_content: str,
    test_file_container_path: str,
    output_file: str,
) -> Optional[Dict[str, Any]]:
    """
    Write augmented test file into the Docker container, re-run the dynamic
    tracer, and return the new call graph dict.
    """
    # Write augmented test file via the host-mounted volume
    host_test_path = (
        sandbox.host_experiments_dir
        / sandbox.project_name
        / test_file_container_path.replace(sandbox.container_project_root + "/", "")
    )
    host_test_path.write_text(augmented_test_content)
    print(f"  Injected augmented tests → {host_test_path.name}")

    # Re-run dynamic tracer
    print(f"  Re-running dynamic tracer → {output_file}")
    ec, _, err = sandbox.run_dynamic_tracer(output_file=output_file)
    if ec != 0:
        print(f"  Tracer failed: {err}")
        return None

    cg_path = sandbox.host_experiments_dir / sandbox.project_name / output_file
    if not cg_path.exists():
        print(f"  Output not found: {cg_path}")
        return None

    import json
    with open(cg_path) as f:
        return json.load(f)


def compute_similarity_matrix(
    test_strings: List[str],
    google_api_key: str,
) -> np.ndarray:
    """Embed test strings with Gemini and return a cosine similarity matrix."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key,
    )
    vectors = embeddings_model.embed_documents(test_strings)
    mat = np.array(vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = mat / norms
    return normed @ normed.T


def run_agent_experiment(
    project: str,
    bug_id: str,
    n_tests: int,
    google_api_key: str,
    bugsinpy_root: str = "datasets/BugsInPy",
    experiments_dir: str = "experiments",
    artifacts_dir: str = "condensation_artifacts",
) -> Optional[Dict[str, Any]]:
    """
    End-to-end experiment for one bug:
    1. Docker checkout + compile
    2. Extract source & tests
    3. Generate diverse tests with the agent
    4. Inject + retrace
    5. Run condensation on original and augmented suites
    6. Return structured comparison
    """
    patch_path = Path(bugsinpy_root) / "projects" / project / "bugs" / bug_id / "bug_patch.txt"
    patch_text = patch_path.read_text()

    # Cache paths
    cache_prefix = Path(artifacts_dir) / f"agent_{project}_{bug_id}"
    cached_ctx = cache_prefix.with_suffix(".ctx.json")
    cached_aug_cg = cache_prefix.with_suffix(".aug_cg.json")
    cached_gen_tests = cache_prefix.with_suffix(".gen_tests.json")

    import json

    agent_config = make_agent_config(google_api_key)

    with BugsInPyDockerSandbox(
        project, bug_id,
        bugsinpy_root=bugsinpy_root,
        experiments_dir=experiments_dir,
    ) as sandbox:
        # 1. Checkout + compile
        print(f"  [1/5] Checkout buggy version ...")
        ec, _, err = sandbox.checkout(version=0)
        if ec != 0:
            print(f"  FAIL checkout: {err}")
            return None

        print(f"  [2/5] Compile ...")
        ec, _, err = sandbox.compile(verbose=True)
        if ec != 0:
            print(f"  FAIL compile: {err}")
            return None

        # Run baseline tracer first
        baseline_output = f"call_graph_{project}_{bug_id}.json"
        print(f"  [3/5] Baseline tracer → {baseline_output}")
        ec, _, err = sandbox.run_dynamic_tracer(output_file=baseline_output)
        if ec != 0:
            print(f"  FAIL baseline tracer: {err}")
            return None

        baseline_cg_path = sandbox.host_experiments_dir / project / baseline_output
        with open(baseline_cg_path) as f:
            baseline_cg = json.load(f)

        # Cache baseline call graph
        os.makedirs(artifacts_dir, exist_ok=True)
        shutil.copy2(baseline_cg_path, Path(artifacts_dir) / f"call_graph_{project}_{bug_id}.json")

        # 2. Extract context
        ctx = extract_bug_context(sandbox, baseline_cg, patch_text)
        if not ctx:
            return None
        print(f"  Buggy FQN: {ctx.buggy_fqn}")
        print(f"  Module: {ctx.module_import}")
        print(f"  Source: {len(ctx.source_code)} chars, Tests: {len(ctx.test_code)} chars")

        # 3. Generate diverse tests
        print(f"  [4/5] Generating {n_tests} diverse tests with agent ...")
        generated = generate_diverse_tests(
            ctx.source_code, ctx.test_code, n_tests, agent_config,
        )
        print(f"  Generated {len(generated)} tests")

        if not generated:
            print("  No tests generated, skipping augmented experiment")
            baseline_result = run_condensation_experiment(baseline_cg, patch_text, project, bug_id)
            return {
                "project": project,
                "bug_id": bug_id,
                "context": ctx,
                "baseline_cg": baseline_cg,
                "generated_tests": [],
                "augmented_cg": None,
                "baseline_result": baseline_result,
                "augmented_result": None,
            }

        # 4. Inject + retrace
        print(f"  [5/5] Injecting tests and re-tracing ...")
        augmented_content = build_augmented_test_file(
            ctx.test_code, generated, ctx.module_import,
        )
        aug_output = f"call_graph_{project}_{bug_id}_augmented.json"
        augmented_cg = inject_and_retrace(
            sandbox, augmented_content, ctx.test_file_container, aug_output,
        )

        if augmented_cg:
            shutil.copy2(
                sandbox.host_experiments_dir / project / aug_output,
                Path(artifacts_dir) / f"call_graph_{project}_{bug_id}_augmented.json",
            )

    # 5. Run condensation experiments
    print("\n  Running condensation analysis ...")
    baseline_result = run_condensation_experiment(baseline_cg, patch_text, project, bug_id)

    augmented_result = None
    if augmented_cg:
        print("  --- Augmented suite ---")
        augmented_result = run_condensation_experiment(augmented_cg, patch_text, project, bug_id)

    return {
        "project": project,
        "bug_id": bug_id,
        "context": ctx,
        "baseline_cg": baseline_cg,
        "generated_tests": generated,
        "augmented_cg": augmented_cg,
        "baseline_result": baseline_result,
        "augmented_result": augmented_result,
    }
