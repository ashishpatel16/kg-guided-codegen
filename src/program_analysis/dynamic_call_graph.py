from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import runpy
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

from src.docker_utils.basic_container import SimpleDockerSandbox
from src.program_analysis.coverage import CoverageAnalyzer
from src.program_analysis.models import (
    CallGraph,
    CallGraphEdge,
    CallGraphNode,
    ExecutionRecord,
    RepoDefinition,
    DockerTracerConfig,
    _ActiveCall,
)


def _safe_serialize(value: Any) -> Any:
    """
    Best-effort JSON-friendly serialization.
    Keeps primitives as-is; falls back to repr() for complex objects.
    """
    try:
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        return repr(value)
    except Exception:
        return "<unserializable>"


def _module_path_from_file(repo_root: str, abs_file: str) -> str:
    rel = os.path.relpath(abs_file, repo_root)
    rel = rel[:-3] if rel.endswith(".py") else rel
    mod = rel.replace(os.sep, ".")
    if mod.endswith(".__init__"):
        mod = mod[:-9]
    return mod


def compute_suspiciousness_scores(
    call_graph: CallGraph,
    test_results: Dict[str, bool],
    node_execution_map: Dict[str, Set[str]]
) -> CallGraph:
    """
    Computes Tarantula suspiciousness scores for a dynamic CallGraph.
    
    Args:
        call_graph: The dynamic CallGraph to enrich with suspiciousness scores
        test_results: Dict mapping test FQN to pass/fail status (True = passed, False = failed)
        node_execution_map: Dict mapping test FQN to set of executed node FQNs
    
    Returns:
        Updated CallGraph with suspiciousness scores populated
    """
    if not test_results:
        return call_graph
    
    # Convert CallGraph to NetworkX DiGraph
    graph = nx.DiGraph()
    
    # Add nodes
    for node in call_graph.nodes:
        graph.add_node(
            node.fqn,
            type=node.type,
            file=node.file,
            start_line=node.start_line,
            end_line=node.end_line,
        )
    
    # Add edges
    for edge in call_graph.edges:
        graph.add_edge(edge.source, edge.target)
    
    # Use CoverageAnalyzer to compute suspiciousness
    analyzer = CoverageAnalyzer(graph, is_static=False)
    
    # Add test results to analyzer
    for test_fqn, passed in test_results.items():
        executed_nodes = node_execution_map.get(test_fqn, set())
        analyzer.add_dynamic_test_result(test_fqn, passed, executed_nodes)
    
    # Compute Tarantula scores
    suspiciousness_scores = analyzer.compute_suspiciousness(method="tarantula")
    
    # Update CallGraph nodes with scores
    for node in call_graph.nodes:
        node.suspiciousness = suspiciousness_scores.get(node.fqn, 0.0)
    
    return call_graph


class DynamicCallGraphTracer:
    """
    Builds a dynamic call graph by tracing Python execution in-process.

    Key design point for unification:
    - Node IDs are always `fqn` strings compatible with `CallGraphNode.fqn`.
    - For internal code, `fqn = <repo-relative-module>.<co_qualname>` (or `<module>` -> module node).
    - For external code, `fqn = <module_name>.<co_qualname>` when possible.
    """

    def __init__(
        self,
        repo_root: str,
        *,
        include_external: bool = True,
        max_executions_per_node: Optional[int] = None,
        log_level: int = logging.INFO,
    ):
        self.repo_root = os.path.abspath(repo_root)
        self.include_external = include_external
        self.max_executions_per_node = max_executions_per_node

        self._start_ts = time.perf_counter()
        logging.basicConfig(
            level=log_level,
            format='[%(name)s][%(levelname)s] %(message)s'
        )
        self._logger = logging.getLogger(__name__)

        self._logger.info(f"Initialized DynamicCallGraphTracer for repo: {self.repo_root}")

        # Node accumulators: fqn -> mutable dict
        self._nodes: Dict[str, Dict[str, Any]] = {}

        # Edge accumulators: (source, target) -> {"call_count": int, "total_duration": float}
        self._edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self._stack: List[_ActiveCall] = []
        
        # Test result tracking for fault localization
        self._test_results: Dict[str, bool] = {}  # test_fqn -> passed/failed
        self._node_execution_map: Dict[str, Set[str]] = {}  # test_fqn -> set of executed nodes

    def _is_internal_file(self, abs_file: str) -> bool:
        try:
            rel = os.path.relpath(abs_file, self.repo_root)
        except Exception:
            self._logger.error(f"Error in _is_internal_file: {Exception}")
            return False
        is_internal = not rel.startswith("..")
        self._logger.debug(f"Checking if file is internal: {abs_file} is: {is_internal}")
        return is_internal

    def _frame_to_fqn_and_meta(self, frame) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Returns (fqn, meta) or (None, None) if frame should be ignored.
        """
        abs_file = os.path.abspath(frame.f_code.co_filename)
        is_internal = self._is_internal_file(abs_file)
        if not is_internal and not self.include_external:
            return None, None

        qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)

        if is_internal:
            module_path = _module_path_from_file(self.repo_root, abs_file)
        else:
            # Prefer runtime module name when available; fallback to a stable-ish external bucket.
            module_path = frame.f_globals.get("__name__") or "external"

        if qualname == "<module>":
            fqn = module_path
            node_type = "module"
        else:
            fqn = f"{module_path}.{qualname}"
            node_type = "function_definition" if is_internal else "external"

        meta = {
            "type": node_type,
            "file": abs_file if (is_internal or self.include_external) else "",
            "start_line": int(getattr(frame.f_code, "co_firstlineno", 0) or 0),
            "end_line": 0,  # filled later via static merge if desired
        }
        return fqn, meta

    def _ensure_node(self, fqn: str, meta: Dict[str, Any]) -> None:
        node = self._nodes.get(fqn)
        if node is None:
            self._nodes[fqn] = {
                "fqn": fqn,
                "type": meta.get("type", "external"),
                "file": meta.get("file", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "execution_count": 0,
                "executions": [],
                "total_duration": 0.0,
            }
            return

        # Opportunistically fill missing metadata when a node is seen multiple times.
        # This can happen when: (1) a function is called from multiple callers,
        # (2) external modules are re-entered, (3) recursive calls occur.
        # We prefer non-external type info and populated file paths when available.
        if not node.get("file") and meta.get("file"):
            node["file"] = meta["file"]
        if not node.get("start_line") and meta.get("start_line"):
            node["start_line"] = meta["start_line"]
        if not node.get("end_line") and meta.get("end_line"):
            node["end_line"] = meta["end_line"]
        if node.get("type") == "external" and meta.get("type") != "external":
            node["type"] = meta["type"]

    def _ensure_edge(self, source: str, target: str) -> None:
        key = (source, target)
        if key not in self._edges:
            self._edges[key] = {"call_count": 0, "total_duration": 0.0}

    def add_test_result(self, test_fqn: str, passed: bool, executed_nodes: Set[str]) -> None:
        """
        Records test execution results for fault localization.
        
        Args:
            test_fqn: Fully qualified name of the test function
            passed: True if test passed, False if failed
            executed_nodes: Set of node FQNs that were executed during this test
        """
        self._test_results[test_fqn] = passed
        self._node_execution_map[test_fqn] = executed_nodes
        self._logger.info(f"Added test result: {test_fqn} (passed={passed}, executed {len(executed_nodes)} nodes)")

    def trace_func(self, frame, event: str, arg):
        """
        This is the most important function that is called by the Python tracer. This is responsible to record events necessary for building the call graph.

        # TODO: Currently only interested in recording the function call and funciton return events. Ignored events include: Line and Exception.
        """
        fqn, meta = self._frame_to_fqn_and_meta(frame)
        self._logger.info(f"Tracing frame with fqn: {fqn}, event: {event} arg: {arg}")
        if fqn is None or meta is None:
            return None

        now = time.perf_counter()
        rel_now = now - self._start_ts

        if event == "call":
            # Caller is current stack top (if any)
            caller_fqn = self._stack[-1].fqn if self._stack else None

            # Args snapshot
            args: Dict[str, Any] = {}
            try:
                arg_info = inspect.getargvalues(frame)
                for name in arg_info.args:
                    if name in arg_info.locals:
                        args[name] = _safe_serialize(arg_info.locals[name])
            except Exception:
                # Can fail on exotic frames (generators, coroutines, C extensions)
                args = {}

            self._ensure_node(fqn, meta)
            if caller_fqn:
                self._ensure_edge(caller_fqn, fqn)
                self._edges[(caller_fqn, fqn)]["call_count"] += 1

            self._stack.append(
                _ActiveCall(
                    fqn=fqn,
                    caller_fqn=caller_fqn,
                    start_ts=now,
                    start_rel_ts=rel_now,
                    args=args,
                )
            )
            return self.trace_func

        if event == "return":
            if not self._stack:
                return self.trace_func

            active = self._stack.pop()
            if active.fqn != fqn:
                # Stack mismatch can occur with generators, coroutines, or exception unwinding.
                # Since we can't reliably attribute this return, skip it.
                # TODO: Consider adding optional warning via logging for debugging.
                return self.trace_func

            duration = now - active.start_ts
            node = self._nodes.get(fqn)
            if node is not None:
                node["execution_count"] += 1
                node["total_duration"] += float(duration)

                if self.max_executions_per_node is None or len(node["executions"]) < self.max_executions_per_node:
                    node["executions"].append(
                        ExecutionRecord(
                            args=active.args,
                            return_value=_safe_serialize(arg),
                            duration=float(duration),
                            timestamp=float(active.start_rel_ts),
                            coverage=0.0,
                        )
                    )

            if active.caller_fqn:
                edge_key = (active.caller_fqn, fqn)
                if edge_key in self._edges:
                    self._edges[edge_key]["total_duration"] += float(duration)

            return self.trace_func

        # Ignore other events for call graph purposes.
        return self.trace_func

    def run_callable(self, fn: Callable[[], Any]) -> Any:
        """
        Traces a callable in-process.
        """
        old = sys.gettrace()
        sys.settrace(self.trace_func)
        try:
            return fn()
        finally:
            sys.settrace(old)

    def run_script(self, file_path: str) -> Any:
        """
        Traces execution of a Python script file via `runpy.run_path`.

        NOTE: This runs in the current Python process (important for tracing).
        """
        abs_path = os.path.abspath(file_path)
        script_dir = os.path.dirname(abs_path)

        old_sys_path = list(sys.path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)

        def _run():
            return runpy.run_path(abs_path, run_name="__main__")

        try:
            return self.run_callable(_run)
        finally:
            sys.path = old_sys_path

    def export_call_graph(self) -> CallGraph:
        """
        Exports collected dynamic data into the unified CallGraph model.
        Automatically computes suspiciousness scores if test results are available.
        """
        nodes: List[CallGraphNode] = []
        for fqn, data in self._nodes.items():
            execution_count = int(data.get("execution_count") or 0)
            total_duration = float(data.get("total_duration") or 0.0)
            avg_duration = (total_duration / execution_count) if execution_count > 0 else 0.0

            nodes.append(
                CallGraphNode(
                    fqn=fqn,
                    type=data.get("type", "external"),
                    file=data.get("file", ""),
                    start_line=int(data.get("start_line") or 0),
                    end_line=int(data.get("end_line") or 0),
                    description="",
                    execution_count=execution_count,
                    executions=list(data.get("executions") or []),
                    total_duration=total_duration,
                    avg_duration=float(avg_duration),
                    covered=execution_count > 0,
                    suspiciousness=0.0,
                    analysis_type="dynamic",
                )
            )

        edges: List[CallGraphEdge] = []
        for (source, target), edata in self._edges.items():
            call_count = int(edata.get("call_count") or 0)
            total_d = float(edata.get("total_duration") or 0.0)
            avg_d = (total_d / call_count) if call_count > 0 else 0.0
            edges.append(
                CallGraphEdge(
                    source=source,
                    target=target,
                    call_count=call_count,
                    avg_call_duration=float(avg_d),
                    analysis_type="dynamic",
                )
            )

        call_graph = CallGraph(static=False, dynamic=True, nodes=nodes, edges=edges)
        
        # Compute suspiciousness scores if test results are available
        if self._test_results:
            self._logger.info(f"Computing suspiciousness scores for {len(self._test_results)} tests")
            call_graph = compute_suspiciousness_scores(
                call_graph,
                self._test_results,
                self._node_execution_map
            )
        
        return call_graph


def build_dynamic_call_graph_for_script(
    repo_root: str,
    file_path: str,
    *,
    include_external: bool = True,
    max_executions_per_node: Optional[int] = None,
) -> CallGraph:
    tracer = DynamicCallGraphTracer(
        repo_root,
        include_external=include_external,
        max_executions_per_node=max_executions_per_node,
    )
    tracer.run_script(file_path)
    return tracer.export_call_graph()


def build_dynamic_call_graph(
    repo_root: str,
    runner: Callable[[], Any],
    *,
    include_external: bool = True,
    max_executions_per_node: Optional[int] = None,
) -> CallGraph:
    tracer = DynamicCallGraphTracer(
        repo_root,
        include_external=include_external,
        max_executions_per_node=max_executions_per_node,
    )
    tracer.run_callable(runner)
    return tracer.export_call_graph()


def build_experiment_primes():
    repo_root = "/Users/ashish/master-thesis/kg-guided-codegen/src/benchmarks/exp/demo"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import test_demo  # type: ignore
        import demo  # type: ignore
    except ImportError as e:
        print(f"Error: Could not import demo or test_demo. Make sure {repo_root} is correct.")
        print(f"Details: {e}")
        sys.exit(1)

    # Initialize tracer
    tracer = DynamicCallGraphTracer(
        repo_root,
        include_external=False,
        max_executions_per_node=25,
        log_level=logging.WARNING  # Reduce verbosity for demo
    )

    def run_test_and_track(test_name, test_func):
        # snapshot execution counts to see what was executed
        counts_before = {fqn: data["execution_count"] for fqn, data in tracer._nodes.items()}
        
        passed = True
        try:
            # We use run_callable to trace the execution of the test
            tracer.run_callable(test_func)
        except AssertionError:
            passed = False
        except Exception as e:
            # print(f"  Test {test_name} raised an exception: {e}")
            passed = False
        
        # Determine executed nodes during this specific test
        # A node was executed if its execution_count increased
        executed_nodes = set()
        for fqn, data in tracer._nodes.items():
            if data["execution_count"] > counts_before.get(fqn, 0):
                # EXCLUDE test code from suspiciousness calculation
                # We only want to rank the actual source code
                filename = os.path.basename(data.get("file", ""))
                if not (filename.startswith("test_") or fqn.startswith("test_")):
                    executed_nodes.add(fqn)
        
        tracer.add_test_result(test_name, passed=passed, executed_nodes=executed_nodes)
        return passed

    # Define the tests we want to run
    passing_tests = [
        ("test_fib", test_demo.test_fib),
        ("test_is_divisible", test_demo.test_is_divisible),
        ("test_is_prime_correct", test_demo.test_is_prime_correct),
        ("test_is_prime_composites", test_demo.test_is_prime_composites),
        ("test_is_prime_negatives", test_demo.test_is_prime_negatives),
        ("test_fib_sequence", test_demo.test_fib_sequence),
    ]

    failing_tests = [
        ("test_is_prime_bug", test_demo.test_is_prime_bug),
        ("test_get_prime_fibs_bug", test_demo.test_get_prime_fibs_bug),
        ("test_count_primes_up_to_10", test_demo.test_count_primes_up_to_10),
        ("test_get_primes_in_range_1_to_10", test_demo.test_get_primes_in_range_1_to_10),
        ("test_nth_prime_first", test_demo.test_nth_prime_first),
        ("test_sum_of_primes", test_demo.test_sum_of_primes),
    ]

    print(f"\n{'='*60}")
    print("Running Tests and Building Dynamic Call Graph")
    print(f"{'='*60}")

    print("\nRunning passing tests...")
    for name, func in passing_tests:
        res = run_test_and_track(name, func)
        print(f"  {name}: {'PASSED' if res else 'FAILED (unexpected)'}")

    print("\nRunning failing tests...")
    for name, func in failing_tests:
        res = run_test_and_track(name, func)
        print(f"  {name}: {'PASSED (unexpected)' if res else 'FAILED'}")

    # Export call graph with suspiciousness scores
    g = tracer.export_call_graph()

    print(f"\n{'='*60}")
    print("Dynamic Call Graph with Tarantula Suspiciousness Scores")
    print(f"{'='*60}")
    print(f"Total nodes: {len(g.nodes)}")
    print(f"Total edges: {len(g.edges)}")
    print(f"Tests tracked: {len(tracer._test_results)}")
    
    # Display suspiciousness scores using the new method
    print(f"\n{'='*60}")
    print("Suspiciousness Scores (sorted by score, descending)")
    print(f"{'='*60}")
    
    # We want to see nodes that have some suspiciousness
    suspicious_nodes = g.get_suspicious_nodes(min_suspiciousness=0.01)
    if not suspicious_nodes:
        print("No suspicious nodes found (all scores are 0).")
    else:
        for node in suspicious_nodes:
            print(f"  {node.fqn:40} : {node.suspiciousness:.4f} (executed {node.execution_count} times)")
    
    # Example: Get top 3 most suspicious nodes
    print(f"\n{'='*60}")
    print("Top 10 Most Suspicious Nodes")
    print(f"{'='*60}")
    top_suspicious = g.get_suspicious_nodes(min_suspiciousness=0.0, limit=10)
    for i, node in enumerate(top_suspicious, 1):
        print(f"  {i}. {node.fqn:40} : {node.suspiciousness:.4f}")
    
    # Export as json
    output_path = "artifacts/demo_call_graph_dynamic_with_suspiciousness.json"
    with open(output_path, "w") as f:
         json.dump(g.model_dump(), f, indent=4)
         print(f"\nSaved full graph to {output_path}")


def build_experiment_library():
    repo_root = "/Users/ashish/master-thesis/kg-guided-codegen/src/benchmarks/exp/library"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import test_catalog  # type: ignore
        import test_borrowing  # type: ignore
        import catalog  # type: ignore
        import borrowing  # type: ignore
    except ImportError as e:
        print(f"Error: Could not import library modules or tests. Make sure {repo_root} is correct.")
        print(f"Details: {e}")
        sys.exit(1)

    # Initialize tracer
    tracer = DynamicCallGraphTracer(
        repo_root,
        include_external=False,
        max_executions_per_node=25,
        log_level=logging.WARNING
    )

    def run_test_and_track(test_name, test_func):
        # snapshot execution counts to see what was executed
        counts_before = {fqn: data["execution_count"] for fqn, data in tracer._nodes.items()}
        
        passed = True
        try:
            # Create an instance of the test class if needed, or just run the test function
            # Since these are unittest.TestCase subclasses, we should ideally use a runner
            # but for simplicity in this pipeline, we can instantiate and call the test methods.
            
            if hasattr(test_func, "__self__") or (inspect.isfunction(test_func) and "." in test_func.__qualname__):
                # It's a method, we might need to handle it differently if it was passed as Class.method
                pass
            
            tracer.run_callable(test_func)
        except AssertionError:
            passed = False
        except Exception as e:
            passed = False
        
        # Determine executed nodes during this specific test
        executed_nodes = set()
        for fqn, data in tracer._nodes.items():
            if data["execution_count"] > counts_before.get(fqn, 0):
                filename = os.path.basename(data.get("file", ""))
                if not (filename.startswith("test_") or fqn.startswith("test_")):
                    executed_nodes.add(fqn)
        
        tracer.add_test_result(test_name, passed=passed, executed_nodes=executed_nodes)
        return passed

    # Instantiate test classes to get bound methods
    catalog_tests = test_catalog.TestCatalog()
    catalog_tests.setUp()
    
    borrowing_tests = test_borrowing.TestBorrowing()
    borrowing_tests.setUp()

    passing_tests = [
        ("test_get_book", catalog_tests.test_get_book),
        ("test_search_by_title", catalog_tests.test_search_by_title),
        ("test_search_by_author", catalog_tests.test_search_by_author),
        ("test_mark_as_borrowed_and_returned", catalog_tests.test_mark_as_borrowed_and_returned),
        ("test_borrow_success", borrowing_tests.test_borrow_success),
        ("test_borrow_non_existent", borrowing_tests.test_borrow_non_existent),
        ("test_return_on_time", borrowing_tests.test_return_on_time),
        ("test_return_late", borrowing_tests.test_return_late),
    ]

    failing_tests = [
        ("test_return_exactly_on_due_date", borrowing_tests.test_return_exactly_on_due_date),
    ]

    print(f"\n{'='*60}")
    print("Running Library Experiment Tests and Building Dynamic Call Graph")
    print(f"{'='*60}")

    print("\nRunning passing tests...")
    for name, func in passing_tests:
        res = run_test_and_track(name, func)
        print(f"  {name}: {'PASSED' if res else 'FAILED (unexpected)'}")

    print("\nRunning failing tests...")
    for name, func in failing_tests:
        res = run_test_and_track(name, func)
        print(f"  {name}: {'PASSED (unexpected)' if res else 'FAILED'}")

    # Export call graph with suspiciousness scores
    g = tracer.export_call_graph()

    print(f"\n{'='*60}")
    print("Library Dynamic Call Graph with Tarantula Suspiciousness Scores")
    print(f"{'='*60}")
    print(f"Total nodes: {len(g.nodes)}")
    print(f"Total edges: {len(g.edges)}")
    print(f"Tests tracked: {len(tracer._test_results)}")
    
    # Display top most suspicious nodes
    print(f"\n{'='*60}")
    print("Top 10 Most Suspicious Nodes (Library)")
    print(f"{'='*60}")
    top_suspicious = g.get_suspicious_nodes(min_suspiciousness=0.0, limit=10)
    for i, node in enumerate(top_suspicious, 1):
        print(f"  {i}. {node.fqn:40} : {node.suspiciousness:.4f}")
    
    # Export as json
    output_path = "artifacts/library_call_graph_dynamic_with_suspiciousness.json"
    with open(output_path, "w") as f:
         json.dump(g.model_dump(), f, indent=4)
         print(f"\nSaved full graph to {output_path}")


def run_dynamic_tracer_in_docker(
    repo_def: RepoDefinition,
    config: DockerTracerConfig = DockerTracerConfig()
) -> CallGraph:
    """
    Standardized pipeline to run dynamic tracing in a Docker container.
    It handles sandbox setup, dependency installation, source copying, 
    tracer execution, and result retrieval.
    """
    with SimpleDockerSandbox(image_name=config.image_name, keep_alive=config.keep_alive) as sandbox:
        try:
            # Install debugger dependencies (minimum needed for the tracer)
            print("Sandbox: Installing debugger dependencies...")
            sandbox.run_command("pip install networkx pydantic pytest tree-sitter tree-sitter-python docker")

            # Copy the debugger itself to the container
            debugger_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            print(f"Sandbox: Copying debugger source from {debugger_root}...")
            sandbox.copy_to(debugger_root, "debugger")

            # Copy the target repo to /repo
            print(f"Sandbox: Copying target repo from {repo_def.repo_path} to /repo...")
            sandbox.copy_to(repo_def.repo_path, "repo")

            # Install target repo requirements
            if repo_def.install_command:
                print(f"Sandbox: Running install command: {repo_def.install_command}")
                sandbox.run_command(repo_def.install_command, workdir=os.path.join(sandbox.sandbox_dir, "repo"))
            else:
                print("Sandbox: Checking for requirements.txt...")
                sandbox.run_command("if [ -f requirements.txt ]; then pip install -r requirements.txt; fi", workdir=os.path.join(sandbox.sandbox_dir, "repo"))

            # Run the tracer inside the container
            output_in_container = os.path.join(sandbox.sandbox_dir, config.output_file)
            print(f"Sandbox: Running tracer for script {repo_def.trace_script}...")
            
            debugger_path = os.path.join(sandbox.sandbox_dir, "debugger")
            repo_path = os.path.join(sandbox.sandbox_dir, "repo")
            
            run_cmd = (
                f"export PYTHONPATH=$PYTHONPATH:{debugger_path}:{repo_path} && "
                f"python3 -m src.program_analysis.dynamic_call_graph "
                f"--repo {repo_path} --script {repo_def.trace_script} --output {output_in_container}"
            )
            exit_code, stdout, stderr = sandbox.run_command(run_cmd)
            
            if exit_code != 0:
                print(f"Error: Tracer failed with exit code {exit_code}")
                if stdout: print(f"STDOUT: {stdout}")
                if stderr: print(f"STDERR: {stderr}")
                raise RuntimeError("Dynamic tracing failed inside container")

            # Retrieve the results
            host_output_dir = os.path.abspath("artifacts")
            os.makedirs(host_output_dir, exist_ok=True)
            host_output_path = os.path.join(host_output_dir, config.output_file)
            
            print(f"Sandbox: Retrieving results to {host_output_path}...")
            # SimpleDockerSandbox.copy_from extracts into a directory, so we handle that
            temp_extract_dir = os.path.join(host_output_dir, "_temp_extract")
            os.makedirs(temp_extract_dir, exist_ok=True)
            sandbox.copy_from(output_in_container, temp_extract_dir)
            
            # The file should be at temp_extract_dir/config.output_file
            extracted_file = os.path.join(temp_extract_dir, config.output_file)
            if os.path.exists(extracted_file):
                if os.path.exists(host_output_path):
                    os.remove(host_output_path)
                os.rename(extracted_file, host_output_path)
            
            # Clean up temp dir
            shutil.rmtree(temp_extract_dir)

            # Load and return CallGraph
            with open(host_output_path, "r") as f:
                data = json.load(f)
                return CallGraph.model_validate(data)

        except Exception as e:
            print(f"Error in Dockerized pipeline: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Call Graph Tracer")
    parser.add_argument("--repo", help="Path to the repository root")
    parser.add_argument("--script", help="Path to the script to trace (relative to repo root)")
    parser.add_argument("--output", help="Path to save the output JSON", default="call_graph.json")
    
    args = parser.parse_args()
    
    if args.repo and args.script:
        # Running INSIDE container (or directly via CLI), equivalent to 'uv run -m src.program_analysis.dynamic_call_graph'
        print(f"Tracing repo: {args.repo} with script: {args.script}")
        
        repo_abs = os.path.abspath(args.repo)
        if repo_abs not in sys.path:
            sys.path.insert(0, repo_abs)
            
        script_path = os.path.join(repo_abs, args.script)
        if not os.path.exists(script_path):
            print(f"Error: Script not found at {script_path}")
            sys.exit(1)
            
        # Run tracing
        call_graph = build_dynamic_call_graph_for_script(repo_abs, script_path)
        
        # Save output
        with open(args.output, "w") as f:
            json.dump(call_graph.model_dump(), f, indent=4)
        print(f"Saved call graph to {args.output}")
        
    else:
        # Running OUTSIDE container to trigger the pipeline
        print("No CLI arguments provided. Running pipeline demo...")
        
        # Use absolute path for the host repo
        demo_repo = os.path.abspath("src/benchmarks/exp/demo")
        demo_script = "test_demo.py"
        
        repo_def = RepoDefinition(repo_path=demo_repo, trace_script=demo_script)
        config = DockerTracerConfig(keep_alive=False, output_file="demo_docker_call_graph.json")
        
        try:
            cg = run_dynamic_tracer_in_docker(repo_def, config)
            print(f"\nSUCCESS: Built call graph with {len(cg.nodes)} nodes and {len(cg.edges)} edges.")
            print(f"Result saved to artifacts/{config.output_file}")
        except Exception as e:
            print(f"\nPIPELINE FAILED: {e}")
            sys.exit(1)

   

    

