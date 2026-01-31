from __future__ import annotations
from src.agent.state import DebuggingState

import logging

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    generate_code,
    generate_evidence,
    generate_hypothesis,
    evaluate_evidence,
    generate_reflection,
    select_target_node,
    generate_inspection_patch,
    execute_inspection,
    update_suspiciousness_and_reflect,
)
from src.agent.state import OneShotCodeGenState, DebuggingState

logger = logging.getLogger(__name__)


def should_stop(state: OneShotCodeGenState) -> bool:
    """
    Stopping condition.
    For now this always evaluates to True (one-shot generation).
    """
    evidence_evaluation = float(state.get("evidence_evaluation") or 0)
    if evidence_evaluation >= 7:
        return True
    return False


def build_one_shot_codegen_agent():
    """
    Graph:
      START -> generate_code -> END
    """
    logger.info("Building one-shot codegen agent graph (START -> generate_code -> END)")
    builder = StateGraph(OneShotCodeGenState)

    # Nodes
    builder.add_node("generate_code", generate_code)
    builder.add_node("generate_hypothesis", generate_hypothesis)
    builder.add_node("generate_evidence", generate_evidence)
    builder.add_node("evaluate_evidence", evaluate_evidence)
    builder.add_node("generate_reflection", generate_reflection)
    # Edges
    builder.add_edge(START, "generate_hypothesis")
    builder.add_edge("generate_hypothesis", "generate_code")
    builder.add_edge("generate_code", "generate_evidence")
    builder.add_edge("generate_evidence", "evaluate_evidence")
    builder.add_edge("generate_reflection", "generate_hypothesis")
    builder.add_conditional_edges(
        "evaluate_evidence", should_stop, {False: "generate_reflection", True: END}
    )

    compiled = builder.compile()
    logger.info("Compiled one-shot codegen agent graph")
    return compiled


def should_continue_debugging(state: DebuggingState) -> bool:
    """
    Check if any node's suspiciousness has exceeded 1.0.
    """
    call_graph = state.get("call_graph")
    for node in call_graph["nodes"]:
        if node.get("suspiciousness", 0) > 1.0:
            logger.info(f"Target found: {node['fqn']} has suspiciousness > 1.0")
            return True
    return False


def build_debugging_agent():
    """
    Graph:
      START -> select_target_node -> generate_inspection_patch -> execute_inspection -> update_suspiciousness_and_reflect -> select_target_node (if score <= 1.0)
    """
    logger.info("Building debugging agent graph")
    builder = StateGraph[DebuggingState, None, DebuggingState, DebuggingState](DebuggingState)

    # Nodes
    builder.add_node("select_target_node", select_target_node)
    builder.add_node("generate_inspection_patch", generate_inspection_patch)
    builder.add_node("execute_inspection", execute_inspection)
    builder.add_node("update_suspiciousness_and_reflect", update_suspiciousness_and_reflect)

    # Edges
    builder.add_edge(START, "select_target_node")
    builder.add_edge("select_target_node", "generate_inspection_patch")
    builder.add_edge("generate_inspection_patch", "execute_inspection")
    builder.add_edge("execute_inspection", "update_suspiciousness_and_reflect")

    builder.add_conditional_edges(
        "update_suspiciousness_and_reflect",
        should_continue_debugging,
        {
            True: END,
            False: "select_target_node",
        },
    )

    compiled = builder.compile()
    logger.info("Compiled debugging agent graph")
    return compiled


one_shot_codegen_agent = build_one_shot_codegen_agent()
debugging_agent = build_debugging_agent()
