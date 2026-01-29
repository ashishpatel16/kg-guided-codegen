from __future__ import annotations

from typing import Dict, List, Tuple

from src.program_analysis.models import CallGraph, CallGraphEdge, CallGraphNode


def merge_call_graphs(static_graph: CallGraph, dynamic_graph: CallGraph) -> CallGraph:
    """
    Merge a static and dynamic call graph into a hybrid graph using `fqn` as the node key
    and (source, target) as the edge key.

    Merge policy (high level):
    - Prefer static metadata for file/line/type when available.
    - Prefer dynamic metrics for execution_count/executions/durations when available.
    - If a node/edge exists in both, mark it as `analysis_type="hybrid"`.
    """

    # Nodes
    nodes_by_fqn: Dict[str, Dict] = {}

    def _node_to_dict(n: CallGraphNode) -> Dict:
        # Pydantic v1/v2 compatible
        if hasattr(n, "model_dump"):
            return n.model_dump()
        return n.dict()

    for n in static_graph.nodes or []:
        d = _node_to_dict(n)
        nodes_by_fqn[d["fqn"]] = d

    for n in dynamic_graph.nodes or []:
        d_dyn = _node_to_dict(n)
        fqn = d_dyn["fqn"]

        if fqn not in nodes_by_fqn:
            nodes_by_fqn[fqn] = d_dyn
            continue

        d = nodes_by_fqn[fqn]

        # Prefer static metadata when present.
        if d.get("file") and not d_dyn.get("file"):
            d_dyn["file"] = d["file"]
        if d.get("start_line") and not d_dyn.get("start_line"):
            d_dyn["start_line"] = d["start_line"]
        if d.get("end_line") and not d_dyn.get("end_line"):
            d_dyn["end_line"] = d["end_line"]

        # Prefer static type if dynamic marked it external.
        if d.get("type") and d_dyn.get("type") in {"external"}:
            d_dyn["type"] = d["type"]

        # Merge dynamic metrics.
        exec_count = int(d.get("execution_count") or 0) + int(d_dyn.get("execution_count") or 0)
        total_dur = float(d.get("total_duration") or 0.0) + float(d_dyn.get("total_duration") or 0.0)
        executions: List = []
        executions.extend(list(d.get("executions") or []))
        executions.extend(list(d_dyn.get("executions") or []))

        d_dyn["execution_count"] = exec_count
        d_dyn["total_duration"] = total_dur
        d_dyn["avg_duration"] = (total_dur / exec_count) if exec_count > 0 else 0.0
        d_dyn["executions"] = executions

        # Hybrid flags.
        d_dyn["covered"] = bool(d.get("covered") or False) or bool(d_dyn.get("covered") or False) or exec_count > 0
        d_dyn["analysis_type"] = "hybrid"

        # Suspiciousness: keep if already set; otherwise default 0.
        d_dyn["suspiciousness"] = float(d_dyn.get("suspiciousness") or d.get("suspiciousness") or 0.0)

        nodes_by_fqn[fqn] = d_dyn

    merged_nodes = [CallGraphNode(**d) for d in nodes_by_fqn.values()]

    # Edges
    edges_by_key: Dict[Tuple[str, str], Dict] = {}

    def _edge_to_dict(e: CallGraphEdge) -> Dict:
        if hasattr(e, "model_dump"):
            return e.model_dump()
        return e.dict()

    for e in static_graph.edges or []:
        d = _edge_to_dict(e)
        edges_by_key[(d["source"], d["target"])] = d

    for e in dynamic_graph.edges or []:
        d_dyn = _edge_to_dict(e)
        key = (d_dyn["source"], d_dyn["target"])

        if key not in edges_by_key:
            edges_by_key[key] = d_dyn
            continue

        d = edges_by_key[key]

        c1 = int(d.get("call_count") or 0)
        c2 = int(d_dyn.get("call_count") or 0)
        a1 = float(d.get("avg_call_duration") or 0.0)
        a2 = float(d_dyn.get("avg_call_duration") or 0.0)

        c = c1 + c2
        if c > 0:
            avg = ((a1 * c1) + (a2 * c2)) / c
        else:
            avg = 0.0

        d_dyn["call_count"] = c
        d_dyn["avg_call_duration"] = avg
        d_dyn["analysis_type"] = "hybrid"

        edges_by_key[key] = d_dyn

    merged_edges = [CallGraphEdge(**d) for d in edges_by_key.values()]

    return CallGraph(
        static=bool(static_graph.static) or bool(static_graph.nodes),
        dynamic=bool(dynamic_graph.dynamic) or bool(dynamic_graph.nodes),
        nodes=merged_nodes,
        edges=merged_edges,
    )

