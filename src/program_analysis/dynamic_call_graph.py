import sys
import os
import inspect
import json
import time
from typing import Any, Dict, List, Optional
import networkx as nx
import matplotlib.pyplot as plt
from src.program_analysis.file_utils import dump_to_json


class DynamicTracer:
    def __init__(self, target_file: str):
        self.target_file = os.path.abspath(target_file)
        self.events: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def _serialize_value(self, value: Any) -> str:
        """Helper to safely serialize values to string representation."""
        try:
            # For basic types, keep them as is? JSON only supports simple types.
            # safe option: always repr(), or try specific types.
            if isinstance(value, (int, float, str, bool, type(None))):
                return value
            return repr(value)
        except Exception:
            return "<unserializable>"

    def _get_arg_values(self, frame) -> Dict[str, Any]:
        """Extracts argument values from the frame."""
        args = {}
        # inspect.getargvalues returns (args, varargs, keywords, locals)
        arg_info = inspect.getargvalues(frame)
        for arg_name in arg_info.args:
            if arg_name in arg_info.locals:
                args[arg_name] = self._serialize_value(arg_info.locals[arg_name])
        return args

    def _get_locals(self, frame) -> Dict[str, Any]:
        """Extracts local variables from the frame."""
        local_vars = {}
        for k, v in frame.f_locals.items():
            if not k.startswith("__"):  # Filter out dunder vars if desired
                local_vars[k] = self._serialize_value(v)
        return local_vars

    def trace_func(self, frame, event, arg):
        # Filter: Only trace events belonging to the target file
        # We need to handle the case where the code is executed via exec()
        # The filename in frame might be absolute or relative.

        frame_file = os.path.abspath(frame.f_code.co_filename)

        # When running via exec(), the filename is often what we passed to compile or the file itself.
        if frame_file != self.target_file:
            return None  # Don't trace other files/libraries

        timestamp = time.time() - self.start_time

        func_name = frame.f_code.co_name
        line_no = frame.f_lineno

        record = {
            "event": event,
            "func_name": func_name,
            "line": line_no,
            "timestamp": timestamp,
        }

        if event == "call":
            # Function called
            code_obj = frame.f_code
            record["args"] = self._get_arg_values(frame)
            self.events.append(record)

        elif event == "line":
            # Executing a line
            record["locals"] = self._get_locals(frame)
            self.events.append(record)

        elif event == "return":
            # Returning from function
            record["return_value"] = self._serialize_value(arg)
            # Capture final local state too?
            record["locals"] = self._get_locals(frame)
            self.events.append(record)

        elif event == "exception":
            exc_type, exc_value, exc_traceback = arg
            record["exception"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "value": str(exc_value),
            }
            self.events.append(record)

        return self.trace_func


def trace_execution(file_path: str) -> List[Dict[str, Any]]:
    """
    Executes the python file at file_path with tracing enabled.
    """
    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")

    with open(abs_path, "r") as f:
        source_code = f.read()

    tracer = DynamicTracer(abs_path)

    # Prepare global context
    # We want to mimic running the script as main
    global_context = {
        "__name__": "__main__",
        "__file__": abs_path,
        "__doc__": None,
        "__package__": None,
    }

    # Add directory of script to sys.path so imports work
    script_dir = os.path.dirname(abs_path)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Set trace
    sys.settrace(tracer.trace_func)

    try:
        # Compile first to ensure filename is associated correctly
        code_obj = compile(source_code, abs_path, "exec")
        exec(code_obj, global_context)
    except Exception as e:
        print(f"Execution failed with error: {e}")
        # We might still want to return the events captured so far
    finally:
        sys.settrace(None)

    return tracer.events


def build_dynamic_graph(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Converts a flat list of trace events into a graph structure.

    Structure:
    {
      "nodes": [
        { "id": "fib", "executions": [ { "args": {"n": 5}, "return_value": 5, ... } ] }
      ],
      "edges": [
        { "source": "main", "target": "fib", "count": 1 }
      ]
    }
    """
    nodes = {}  # id -> {id, executions: []}
    edges = {}  # (source, target) -> count

    # stack of (func_name, start_time, args)
    # We initialize with a root caller
    call_stack = [("<module>", 0.0, {})]

    # Ensure root node exists
    nodes["<module>"] = {"id": "<module>", "executions": []}

    for event in events:
        evt_type = event["event"]

        if evt_type == "call":
            func_name = event["func_name"]
            caller = call_stack[-1][0]

            # Edge
            edge_key = (caller, func_name)
            edges[edge_key] = edges.get(edge_key, 0) + 1

            # Node
            if func_name not in nodes:
                nodes[func_name] = {"id": func_name, "executions": []}

            # Push to stack
            call_stack.append((func_name, event["timestamp"], event.get("args", {})))

        elif evt_type == "return":
            if not call_stack:
                continue

            func_name, start_ts, args = call_stack.pop()

            # Verify we are popping what we expect (simple check)
            # Python trace guarantees correct nesting mostly.
            # But specific edge cases (generators etc) might vary.
            # For this simple tracer, we assume consistency or just log matching.

            if func_name in nodes:
                exec_data = {
                    "args": args,
                    "return_value": event.get("return_value"),
                    "duration": event["timestamp"] - start_ts,
                    "timestamp": start_ts,
                }
                nodes[func_name]["executions"].append(exec_data)

    # Format output
    graph_out = {
        "nodes": list(nodes.values()),
        "edges": [
            {"source": s, "target": t, "count": c} for (s, t), c in edges.items()
        ],
    }
    return graph_out


def visualize_dynamic_graph(
    graph_data: Dict[str, Any], output_path: str = "dynamic_call_graph.png"
):
    G = nx.DiGraph()

    for node in graph_data["nodes"]:
        G.add_node(node["id"])

    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], count=edge["count"])

    plt.figure(figsize=(8, 6))

    # Try hierarchical layout
    pos = nx.spring_layout(G, k=0.9)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_weight="bold")

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")

    # Edge labels (counts)
    edge_labels = {(u, v): d["count"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Dynamic Call Graph (Edge Weights = Call Counts)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Graph visualization saved to {output_path}")


import csv


def export_variable_matrix(events: List[Dict[str, Any]], output_csv: str):
    """
    Exports the trace events to a CSV matrix.
    Rows: Events (Time)
    Colums: Metadata + All seen variables (scoped by function)
    """
    # Pass 1: Collect all unique variable names (scoped)
    all_vars = set()

    for event in events:
        func_name = event["func_name"]

        # Args
        if "args" in event:
            for k in event["args"]:
                all_vars.add(f"{func_name}.{k}")

        # Locals
        if "locals" in event:
            for k in event["locals"]:
                all_vars.add(f"{func_name}.{k}")

        # Return value
        if "return_value" in event:
            all_vars.add(f"{func_name}.return")

    sorted_vars = sorted(list(all_vars))
    headers = ["timestamp", "event", "func_name", "line"] + sorted_vars

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, restval="null")
        writer.writeheader()

        for event in events:
            row = {
                "timestamp": event["timestamp"],
                "event": event["event"],
                "func_name": event["func_name"],
                "line": event["line"],
            }

            func_name = event["func_name"]

            # Populate vars for this row
            if "args" in event:
                for k, v in event["args"].items():
                    row[f"{func_name}.{k}"] = v

            if "locals" in event:
                for k, v in event["locals"].items():
                    row[f"{func_name}.{k}"] = v

            if "return_value" in event:
                row[f"{func_name}.return"] = event["return_value"]

            writer.writerow(row)

    print(f"Variable matrix exported to {output_csv}")


if __name__ == "__main__":
    target_py = "src/program_analysis/demo.py"
    print(f"Tracing execution of {target_py}...")

    events = trace_execution(target_py)

    graph = build_dynamic_graph(events)

    output_graph_file = "dynamic_call_graph.json"
    dump_to_json(output_graph_file, graph)
    print(
        f"Dynamic call graph saved to {output_graph_file} with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges."
    )

    visualize_dynamic_graph(graph, "artifacts/dynamic_call_graph.png")

    export_variable_matrix(events, "artifacts/variable_matrix.csv")
