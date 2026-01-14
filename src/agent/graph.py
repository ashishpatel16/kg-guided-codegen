from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    generate_code,
    generate_evidence,
    generate_hypothesis,
    evaluate_evidence,
    generate_reflection,
)
from agent.state import OneShotCodeGenState

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


one_shot_codegen_agent = build_one_shot_codegen_agent()
