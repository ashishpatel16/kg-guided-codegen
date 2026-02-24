"""
Sequential Agent Pipeline for local Python projects.

Runs two agents in sequence:
  1. Dynamic Tracer + Tarantula Suspiciousness
  2. Test Generation Agent (UnitTestGenerator)
  3. Dynamic Tracer (re-run)
  4. Fault Localization Agent (LangGraph)

Usage:
    uv run python -m src.agent.main <project_root> [--dry-run] [--skip-test-gen] [--artifacts-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.fault_localization.graph import debugging_agent, CONFIDENCE_THRESHOLD
from src.agent.fault_localization.tools import configure_logging, get_function_source
from src.agent.test_generation.config.api_config import APIConfig
from src.agent.test_generation.core.generator import UnitTestGenerator
from src.agent.test_generation.utils.logging import setup_logging
from src.program_analysis.dynamic_call_graph import trace_repo
from src.program_analysis.models import CallGraph
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def validate_project(project_root: str) -> Path:
    """Ensure project_root exists and contains a tests/ directory."""
    root: Path = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project root not found: {root}")

    tests_dir: Path = root / "tests"
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"No tests/ directory at: {tests_dir}")

    return root


def discover_test_files(project_root: Path) -> List[str]:
    """Find all test_*.py files inside the tests/ directory."""
    tests_dir: Path = project_root / "tests"
    return [str(p) for p in sorted(tests_dir.rglob("test_*.py"))]


def run_dynamic_tracing(project_root: Path, test_files: List[str], artifacts_dir: str, skip_save: bool = False) -> CallGraph:
    """Run the dynamic tracer with test discovery to produce a CallGraph with suspiciousness scores."""
    call_graph: CallGraph = trace_repo(
        repo_root=str(project_root),
        scripts=test_files,
        test_mode=True,
        include_external=False,
    )

    if not skip_save:
        os.makedirs(artifacts_dir, exist_ok=True)
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        cg_path: str = os.path.join(artifacts_dir, f"call_graph_{project_root.name}_{timestamp}.json")
        with open(cg_path, "w") as f:
            json.dump(call_graph.model_dump(), f, indent=2, default=str)
        logger.info("Saved call graph to %s", cg_path)

    return call_graph


def run_test_generation(project_root: Path, suspicious_nodes: list[Any], artifacts_dir: str) -> Optional[str]:
    """Generate tests for top suspicious nodes using UnitTestGenerator."""
    try:
        # LangChain's Gemini tools expect GOOGLE_API_KEY, but the repo uses GEMINI_API_KEY
        if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

        api_config_obj = APIConfig.from_env()
        api_config = {
            "openai_api_key": api_config_obj.openai_api_key.get_secret_value(),
            "groq_api_key": api_config_obj.groq_api_key.get_secret_value(),
            "jina_api_key": api_config_obj.jina_api_key.get_secret_value(),
            "google_api_key": api_config_obj.google_api_key.get_secret_value(),
        }
    except Exception as e:
        logger.warning(f"Could not load API configs for test generation: {e}")
        return None

    cfg = {
        "llm": {
            "model_name": "gemini-2.5-flash",
            "max_improvements": 1,
            "similarity_comparison_count": 5,
            "max_tests": 10
        },
        "api": api_config
    }
    log_cfg = {"enabled": True, "level": "INFO", "file_logging": False, "console_logging": False}
    loggers = setup_logging(log_cfg)

    generator = UnitTestGenerator(cfg, loggers)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_file = project_root / "tests" / f"test_generated_{timestamp}.py"
    generated_tests_content = []

    for node in suspicious_nodes:
        node_dict = node if isinstance(node, dict) else node.model_dump()
        source_code = get_function_source(node_dict)
        if not source_code:
            continue
            
        print(f"  - Generating test for {node_dict['fqn']}...")
        try:
            result = generator.generate_test(
                code_to_test=source_code,
                coverage_matrix="Not available yet",  # Test generation accepts dummy for now
                uncovered_lines=[],
            )
            raw_test = result.get("generated_test_case", "")
            if not raw_test:
                continue

            # Figure out module path for import so the test actually runs
            rel_path = os.path.relpath(node_dict['file'], project_root)
            module_path = rel_path.replace(".py", "").replace("/", ".")
            import_statement = f"from {module_path} import *\n"

            generated_tests_content.append(import_statement + raw_test)
        except Exception as e:
            print(f"    Failed to generate test: {e}")
            logger.exception("Test gen failed")

    if generated_tests_content:
        # Wrap everything in a valid Python test file
        with open(generated_file, "w") as f:
            f.write("import unittest\nimport pytest\n\n")
            f.write("\n\n".join(generated_tests_content))
            
        print(f"  ✓ Saved generated tests to {generated_file.relative_to(project_root)}")
        return str(generated_file)
    
    return None


def build_initial_state(call_graph: CallGraph, project_root: Path) -> Dict[str, Any]:
    """Convert a CallGraph Pydantic model into the initial dict expected by DebuggingState."""
    return {
        "call_graph": call_graph.model_dump(),
        "score_delta": 0.3,
        "test_command": f"python3 -m pytest {project_root / 'tests'}",
        "host_workspace": str(project_root),
        "use_docker": False,
        "llm_calls": 0,
    }


def extract_results(final_state: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Pull top candidates and culprit from the final agent state."""
    nodes: List[Dict[str, Any]] = final_state["call_graph"]["nodes"]
    sorted_nodes: List[Dict[str, Any]] = sorted(
        nodes, key=lambda n: n.get("confidence_score", 0), reverse=True
    )

    culprit: Optional[Dict[str, Any]] = next(
        (n for n in sorted_nodes if n.get("confidence_score", 0) >= CONFIDENCE_THRESHOLD),
        None,
    )

    top_candidates: List[Dict[str, Any]] = [
        {
            "fqn": n["fqn"],
            "confidence": round(n.get("confidence_score", 0), 4),
            "suspiciousness": round(n.get("suspiciousness", 0), 4),
        }
        for n in sorted_nodes[:5]
    ]

    result: Dict[str, Any] = {
        "project": project_root.name,
        "status": "LOCALIZED" if culprit else "INCONCLUSIVE",
        "culprit": {
            "fqn": culprit["fqn"],
            "confidence": round(culprit["confidence_score"], 4),
        } if culprit else None,
        "top_candidates": top_candidates,
        "llm_calls": final_state.get("llm_calls", 0),
        "reflection": final_state.get("reflection", ""),
    }

    if final_state.get("final_diff"):
        result["final_diff"] = final_state["final_diff"]

    return result


def print_results(result: Dict[str, Any]) -> None:
    """Pretty-print debugging results to stdout."""
    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")

    if result["status"] == "LOCALIZED" and result["culprit"]:
        print(f"  ✓ BUGGY NODE: {result['culprit']['fqn']}")
        print(f"    Confidence: {result['culprit']['confidence']:.4f}")
    else:
        print(f"  ✗ No node reached threshold {CONFIDENCE_THRESHOLD}")

    print(f"\n  Top 5 candidates:")
    for i, c in enumerate(result.get("top_candidates", []), 1):
        marker: str = " ◀" if result["culprit"] and c["fqn"] == result["culprit"]["fqn"] else ""
        print(f"    {i}. {c['fqn']}  conf={c['confidence']:.4f}  susp={c['suspiciousness']:.4f}{marker}")

    if result.get("reflection"):
        print(f"\n  Last reflection:\n    {result['reflection'][:200]}")

    if result.get("final_diff"):
        print(f"\n  Patch diff (first 500 chars):\n    {result['final_diff'][:500]}")

    print(f"{'=' * 60}\n")


def run_local_debugging_pipeline(
    project_root: str,
    *,
    artifacts_dir: str = "artifacts",
    recursion_limit: int = 100,
    dry_run: bool = False,
    skip_test_gen: bool = False,
) -> Dict[str, Any]:
    """
    Full local debugging pipeline.

    Validates the project, discovers tests, runs the dynamic tracer,
    optionally generates new tests for suspicious nodes,
    then invokes the fault localization agent.
    """
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start: datetime = datetime.now()
    result: Dict[str, Any] = {
        "project": "", "status": "FAILED", "culprit": None,
        "top_candidates": [], "error": None, "duration_s": 0.0,
    }

    try:
        root: Path = validate_project(project_root)
        result["project"] = root.name

        print(f"\n{'=' * 60}")
        print(f"  Local Fault Localization: {root.name}")
        print(f"{'=' * 60}\n")

        # Discover tests
        test_files: List[str] = discover_test_files(root)
        if not test_files:
            result["error"] = "No test_*.py files found in tests/"
            print(f"  ✗ {result['error']}")
            return result

        print(f"[Stage 1] Discovered {len(test_files)} test files.")

        # Initial Dynamic tracing
        print(f"\n[Stage 2] Running initial dynamic tracer (test_mode=True)...")
        call_graph: CallGraph = run_dynamic_tracing(root, test_files, artifacts_dir, skip_save=True)
        print(f"  ✓ Call graph: {len(call_graph.nodes)} nodes, {len(call_graph.edges)} edges")

        suspicious: list = call_graph.get_suspicious_nodes(min_suspiciousness=0.0, limit=5)
        if suspicious:
            print(f"\n  Top suspicious nodes (Tarantula):")
            for i, node in enumerate(suspicious, 1):
                print(f"    {i}. {node.fqn}  susp={node.suspiciousness:.4f}")

        # Test Generation
        if suspicious and not skip_test_gen:
            print(f"\n[Stage 3] Generating tests for top suspicious nodes...")
            # Pick max 2 most suspicious nodes with susp > 0
            targets = [n for n in suspicious if (n.suspiciousness or 0) > 0][:2]
            if targets:
                new_test_file = run_test_generation(root, targets, artifacts_dir)
                if new_test_file:
                    test_files.append(new_test_file)
                    
                    print(f"\n[Stage 4] Re-running dynamic tracer with new tests...")
                    call_graph = run_dynamic_tracing(root, test_files, artifacts_dir, skip_save=False)
                    print(f"  ✓ Call graph updated: {len(call_graph.nodes)} nodes, {len(call_graph.edges)} edges")
                    
                    suspicious = call_graph.get_suspicious_nodes(min_suspiciousness=0.0, limit=5)
            else:
                print("  No highly suspicious nodes found. Skipping test generation.")
        else:
            # We skipped test generation, so save the call graph now
            call_graph = run_dynamic_tracing(root, test_files, artifacts_dir, skip_save=False)

        if dry_run:
            print(f"\n  [DRY RUN] Stopping before fault localization.")
            result["status"] = "DRY_RUN"
            result["top_candidates"] = [
                {"fqn": n.fqn, "suspiciousness": round(n.suspiciousness or 0, 4)}
                for n in suspicious
            ]
            return result

        # Fault localization agent
        print(f"\n[Stage 5] Building initial state...")
        initial_state: Dict[str, Any] = build_initial_state(call_graph, root)

        print(f"[Stage 6] Running fault localization agent (threshold={CONFIDENCE_THRESHOLD})...")
        final_state: Dict[str, Any] = debugging_agent.invoke(
            initial_state, config={"recursion_limit": recursion_limit}
        )

        result = extract_results(final_state, root)
        print_results(result)

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "CRASHED"
        logger.exception("Pipeline crashed: %s", e)
        print(f"\n  ✗ CRASHED: {e}")

    finally:
        result["duration_s"] = round((datetime.now() - start).total_seconds(), 2)
        print(f"  Duration: {result['duration_s']}s")

        os.makedirs(artifacts_dir, exist_ok=True)
        result_path: str = os.path.join(artifacts_dir, f"local_{result.get('project', 'unknown')}_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to {result_path}\n{'=' * 60}\n")

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run sequential agent pipeline on a local Python project.")
    parser.add_argument("project_root", help="Path to project directory (must contain tests/).")
    parser.add_argument("--dry-run", action="store_true", help="Stop after dynamic tracing, skip LLM agent.")
    parser.add_argument("--skip-test-gen", action="store_true", help="Skip the LLM test generation phase.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Output directory (default: artifacts).")
    parser.add_argument("--recursion-limit", type=int, default=100, help="Max LangGraph recursion depth.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    configure_logging(level=args.log_level)

    run_local_debugging_pipeline(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        recursion_limit=args.recursion_limit,
        dry_run=args.dry_run,
        skip_test_gen=args.skip_test_gen,
    )


if __name__ == "__main__":
    main()

